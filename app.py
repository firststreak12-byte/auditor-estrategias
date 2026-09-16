"""
app.py
Interfaz web (MVP) en Streamlit para el motor de auditoría de Prop Firms.

Estructura modular deliberada, en 3 capas dentro del mismo archivo:
  1. Recolección de inputs (sidebar + file_uploader) -> objetos tipados
     de prop_audit.models.
  2. Ejecución del pipeline matemático (engine -> metrics -> montecarlo
     -> report), aislada en `ejecutar_pipeline()` para que no dependa de
     ningún widget de Streamlit — así se puede testear o reusar desde el
     futuro backend FastAPI sin arrastrar código de UI.
  3. Renderización de resultados en 4 pestañas, aislada en `renderizar_resultados()`.

Streamlit re-ejecuta TODO el script en cada interacción del usuario, así
que el resultado del pipeline se guarda en `st.session_state` para
sobrevivir al rerun que dispara el propio `st.download_button`.
"""

from __future__ import annotations
import io
import json
import traceback
from datetime import date
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from prop_audit.models import (
    ActivoConfig, ReglasEstrategia, PropFirmProfile,
    Direccion, TipoDrawdown, ModoPosicion,
)
from prop_audit.engine import ejecutar_backtest, trades_a_dataframe
from prop_audit.strategy import CruceMediasMoviles, RupturaDeCanal, BaseStrategy
from prop_audit.metrics import calcular_drawdown, calcular_metricas_rendimiento
from prop_audit.montecarlo import simular_montecarlo
from prop_audit.report import generar_diccionario_auditoria, ReporteAuditoriaDict
from prop_audit.data_loader import (
    cargar_desde_csv, cargar_desde_yfinance, cargar_desde_alpaca, COLUMNAS_MINIMAS,
)


FUENTES_DATOS = ("Archivo CSV", "Datos Públicos (yfinance)", "Conexión Broker (Alpaca)")


# ============================================================
# Capa 1 — Recolección de inputs
# ============================================================

def _sidebar_configuracion() -> Tuple[dict, dict, dict, str, dict, str]:
    """Renderiza los widgets de la barra lateral y regresa los valores
    crudos agrupados por sección (sin construir todavía los dataclasses,
    para poder mostrar errores de validación en el panel principal en vez
    de reventar dentro del sidebar)."""

    st.sidebar.header("📡 Fuente de Datos")
    fuente_datos = st.sidebar.selectbox("Fuente de Datos", FUENTES_DATOS, index=0)

    st.sidebar.header("⚙️ Configuración del Activo")
    activo = {
        "ticker": st.sidebar.text_input("Ticker / Activo", value="QQQ"),
        "temporalidad": st.sidebar.selectbox("Temporalidad", ["1m", "5m", "15m", "1h", "1D"], index=3),
    }
    with st.sidebar.expander("Configuración avanzada del instrumento"):
        activo["tick_value"] = st.number_input(
            "Valor monetario por punto (tick_value)", min_value=0.0001, value=1.0, step=0.1,
            help="Cuánto vale 1 punto de movimiento de precio por unidad. 1.0 para acciones/ETFs típicos.",
        )
        activo["contract_size"] = st.number_input(
            "Unidades por lote (contract_size)", min_value=0.0001, value=1.0, step=1.0,
            help="100000 para un lote estándar de FX; 1 para acciones/ETFs comprados por unidad.",
        )

    st.sidebar.header("📐 Reglas de la Estrategia")
    reglas = {
        "riesgo_por_operacion_pct": st.sidebar.number_input(
            "Riesgo por operación (%)", min_value=0.01, max_value=5.0, value=0.5, step=0.05
        ),
        "stop_loss_pct": st.sidebar.number_input("Stop Loss (%)", min_value=0.01, value=0.3, step=0.05),
        "take_profit_pct": st.sidebar.number_input("Take Profit (%)", min_value=0.01, value=0.6, step=0.05),
        "direccion": st.sidebar.selectbox("Dirección permitida", ["ambas", "long", "short"], index=0),
    }

    with st.sidebar.expander("Estrategia de señales"):
        tipo_estrategia = st.selectbox("Tipo de estrategia", ["Cruce de Medias Móviles", "Ruptura de Canal"])
        parametros_estrategia = {}
        if tipo_estrategia == "Cruce de Medias Móviles":
            parametros_estrategia["ventana_rapida"] = st.number_input("SMA rápida", min_value=2, value=10)
            parametros_estrategia["ventana_lenta"] = st.number_input("SMA lenta", min_value=3, value=30)
        else:
            parametros_estrategia["ventana"] = st.number_input("Ventana del canal (barras)", min_value=2, value=20)

    st.sidebar.header("🏦 Perfil de la Prop Firm")
    prop = {
        "nombre_firma": st.sidebar.text_input("Nombre de la firma", value="Generic Prop Firm"),
        "capital_inicial": st.sidebar.number_input(
            "Capital inicial ($)", min_value=1000.0, value=100000.0, step=1000.0
        ),
        "limite_perdida_diaria_pct": st.sidebar.number_input(
            "Límite pérdida diaria (%)", min_value=0.1, value=5.0, step=0.5
        ),
        "limite_perdida_total_pct": st.sidebar.number_input(
            "Límite pérdida total (%)", min_value=0.1, value=10.0, step=0.5
        ),
        "meta_beneficio_pct": st.sidebar.number_input(
            "Meta de beneficio (%)", min_value=0.1, value=8.0, step=0.5
        ),
        "tipo_drawdown": st.sidebar.selectbox(
            "Tipo de Drawdown", [t.value for t in TipoDrawdown], index=1
        ),
    }

    return activo, reglas, prop, tipo_estrategia, parametros_estrategia, fuente_datos


def _construir_objetos_dominio(
    df: pd.DataFrame,
    activo: dict,
    reglas: dict,
    prop: dict,
    tipo_estrategia: str,
    parametros_estrategia: dict,
) -> Tuple[ActivoConfig, ReglasEstrategia, PropFirmProfile, BaseStrategy]:
    """Construye los dataclasses tipados a partir de los inputs crudos del
    sidebar + el rango de fechas real del CSV cargado. Aquí es donde las
    validaciones fail-fast de models.py (mínimo 3 años de histórico, rangos
    de riesgo, etc.) se disparan — se dejan burbujear como excepciones para
    que el caller las muestre con st.error()."""

    fecha_inicio: date = df["timestamp"].min().date()
    fecha_fin: date = df["timestamp"].max().date()

    config = ActivoConfig(
        ticker=activo["ticker"],
        temporalidad=activo["temporalidad"],
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        tick_value=activo["tick_value"],
        contract_size=activo["contract_size"],
    )

    reglas_estrategia = ReglasEstrategia(
        direccion=Direccion(reglas["direccion"]),
        take_profit_pct=reglas["take_profit_pct"],
        stop_loss_pct=reglas["stop_loss_pct"],
        modo_posicion=ModoPosicion.RIESGO_PORCENTAJE,
        riesgo_por_operacion_pct=reglas["riesgo_por_operacion_pct"],
    )

    prop_profile = PropFirmProfile(
        nombre_firma=prop["nombre_firma"],
        capital_inicial=prop["capital_inicial"],
        limite_perdida_diaria_pct=prop["limite_perdida_diaria_pct"],
        limite_perdida_total_pct=prop["limite_perdida_total_pct"],
        meta_beneficio_pct=prop["meta_beneficio_pct"],
        tipo_drawdown=TipoDrawdown(prop["tipo_drawdown"]),
    )

    if tipo_estrategia == "Cruce de Medias Móviles":
        estrategia: BaseStrategy = CruceMediasMoviles(
            ventana_rapida=int(parametros_estrategia["ventana_rapida"]),
            ventana_lenta=int(parametros_estrategia["ventana_lenta"]),
        )
    else:
        estrategia = RupturaDeCanal(ventana=int(parametros_estrategia["ventana"]))

    return config, reglas_estrategia, prop_profile, estrategia


def _cargar_datos_segun_fuente(
    fuente_datos: str,
    ticker: str,
    temporalidad: str,
) -> Optional[pd.DataFrame]:
    """
    Dispatcher de UI: renderiza los widgets específicos de cada fuente
    (dinámicos según la selección del sidebar) y, al confirmarse la carga,
    delega en la función correspondiente de data_loader.py. Devuelve None
    mientras no haya datos disponibles todavía (nada subido / no
    descargado), nunca lanza excepción hacia el caller — los errores de
    red o de datos se capturan aquí y se muestran con st.error().

    El resultado se persiste en st.session_state["df_cargado"] para
    sobrevivir a los reruns que Streamlit dispara en cada interacción
    (incluido el propio st.download_button de los resultados).
    """
    if st.session_state.get("fuente_datos_anterior") != fuente_datos:
        # Cambiar de fuente invalida cualquier dato cargado previamente:
        # evita auditar por accidente un CSV viejo tras cambiar a yfinance.
        st.session_state.pop("df_cargado", None)
        st.session_state["fuente_datos_anterior"] = fuente_datos

    if fuente_datos == "Archivo CSV":
        archivo_subido = st.file_uploader(
            "Sube tu histórico OHLCV en CSV",
            type=["csv"],
            help=f"Columnas obligatorias: {', '.join(sorted(COLUMNAS_MINIMAS))}",
        )
        if archivo_subido is not None:
            try:
                st.session_state["df_cargado"] = cargar_desde_csv(archivo_subido)
            except ValueError as e:
                st.error(f"Error al cargar el CSV: {e}")
                st.session_state.pop("df_cargado", None)

    elif fuente_datos == "Datos Públicos (yfinance)":
        st.info(
            "📊 **Límites del tier gratuito de yfinance:** 1 minuto (máx. 7 días), "
            "5m a 30m (máx. 60 días). Para probar con años de histórico y mayor rigor "
            "estadístico, conecta tus llaves de **Alpaca** o sube un **CSV** con datos "
            "institucionales."
        )
        col1, col2 = st.columns(2)
        fecha_inicio = col1.date_input("Fecha de inicio", value=date(2022, 1, 1), key="yf_inicio")
        fecha_fin = col2.date_input("Fecha de fin", value=date.today(), key="yf_fin")
        if st.button("📥 Descargar de yfinance"):
            try:
                with st.spinner(f"Descargando {ticker} desde yfinance..."):
                    st.session_state["df_cargado"] = cargar_desde_yfinance(
                        ticker, temporalidad, fecha_inicio, fecha_fin
                    )
            except ValueError as e:
                st.error(f"Error al descargar de yfinance: {e}")
                st.session_state.pop("df_cargado", None)
            except Exception as e:  # red u otro error inesperado no cubierto por el loader
                st.error(f"Error inesperado al conectar con yfinance: {e}")
                st.session_state.pop("df_cargado", None)

    else:  # "Conexión Broker (Alpaca)"
        col1, col2 = st.columns(2)
        fecha_inicio = col1.date_input("Fecha de inicio", value=date(2022, 1, 1), key="al_inicio")
        fecha_fin = col2.date_input("Fecha de fin", value=date.today(), key="al_fin")
        api_key = st.text_input("Alpaca API Key", type="password", key="al_api_key")
        secret_key = st.text_input("Alpaca Secret Key", type="password", key="al_secret_key")
        if st.button("📥 Descargar de Alpaca"):
            try:
                with st.spinner(f"Conectando con Alpaca para {ticker}..."):
                    st.session_state["df_cargado"] = cargar_desde_alpaca(
                        api_key, secret_key, ticker, temporalidad, fecha_inicio, fecha_fin
                    )
            except ValueError as e:
                st.error(f"Error al conectar con Alpaca: {e}")
                st.session_state.pop("df_cargado", None)
            except Exception as e:  # timeout/conexión no cubiertos explícitamente por el loader
                st.error(f"Error inesperado al conectar con Alpaca: {e}")
                st.session_state.pop("df_cargado", None)

    return st.session_state.get("df_cargado")


# ============================================================
# Capa 2 — Pipeline matemático (sin dependencias de Streamlit)
# ============================================================

def ejecutar_pipeline(
    df: pd.DataFrame,
    config: ActivoConfig,
    reglas: ReglasEstrategia,
    prop: PropFirmProfile,
    estrategia: BaseStrategy,
    capital_inicial: float,
    n_simulaciones_mc: int = 1000,
) -> Tuple[ReporteAuditoriaDict, pd.DataFrame]:
    """Corre engine -> metrics -> montecarlo -> report en secuencia.
    No usa st.* en absoluto: es una función pura de dominio, reusable desde
    un test, un notebook o el futuro backend FastAPI."""
    trades = ejecutar_backtest(df, estrategia.generate_signals, reglas, prop, config, capital_inicial)
    trades_df = trades_a_dataframe(trades)

    if len(trades_df) < 10:
        raise ValueError(
            f"La estrategia solo generó {len(trades_df)} operaciones en el histórico cargado. "
            f"Se requieren al menos 10 trades para que las métricas y Monte Carlo sean válidos: "
            f"revisa los parámetros de la estrategia o carga más histórico."
        )

    drawdown = calcular_drawdown(
        trades_df, df, prop, capital_inicial,
        multiplicador_monetario=config.contract_size * config.tick_value,
    )
    metricas = calcular_metricas_rendimiento(trades_df, capital_inicial)
    monte_carlo = simular_montecarlo(trades_df, prop, capital_inicial, n_simulaciones=n_simulaciones_mc)

    reporte = generar_diccionario_auditoria(
        config, reglas, prop, capital_inicial, metricas, drawdown, monte_carlo
    )
    return reporte, trades_df


# ============================================================
# Capa 3 — Renderización de resultados
# ============================================================

def _renderizar_tab_configuracion(reporte: ReporteAuditoriaDict) -> None:
    st.subheader("Configuración base utilizada en esta auditoría")
    st.json(reporte["configuracion_base"])


def _renderizar_tab_rendimiento(reporte: ReporteAuditoriaDict) -> None:
    rp = reporte["rendimiento_puro"]
    st.subheader("Rendimiento puro de la estrategia")

    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{rp['win_rate_pct']:.2f}%")
    col2.metric("Profit Factor", f"{rp['profit_factor']:.2f}")
    col3.metric("Total de Operaciones", f"{rp['total_trades']}")

    col4, col5, col6 = st.columns(3)
    col4.metric("EV por dólar arriesgado", f"${rp['ev_por_dolar']:.4f}")
    col5.metric("Sharpe", f"{rp['sharpe']:.2f}")
    col6.metric("Sortino", f"{rp['sortino']:.2f}")

    st.divider()
    st.markdown(
        f"**Peor racha de pérdidas consecutivas:** {rp['racha_perdidas_max']} operaciones "
        f"(**${rp['peor_racha_usd']:,.2f}**)"
    )
    with st.expander("Ver desglose estructurado (JSON)"):
        st.json(rp)


def _renderizar_tab_riesgo(reporte: ReporteAuditoriaDict) -> None:
    ar = reporte["auditoria_riesgo"]
    st.subheader("Auditoría de riesgo frente a las reglas de la firma")

    if ar["excedio_limite"]:
        st.error(
            f"🔴 CUENTA QUEMADA — El Max Drawdown ({ar['max_drawdown_pct']:.2f}%) superó el "
            f"límite permitido por la firma ({ar['limite_permitido_pct']:.2f}%)."
        )
    else:
        st.success(
            f"🟢 La estrategia se mantuvo dentro del límite de drawdown "
            f"({ar['max_drawdown_pct']:.2f}% vs. límite de {ar['limite_permitido_pct']:.2f}%)."
        )

    col1, col2 = st.columns(2)
    col1.metric("Max Drawdown ($)", f"${ar['max_drawdown_usd']:,.2f}")
    col2.metric("Max Drawdown (%)", f"{ar['max_drawdown_pct']:.2f}%")

    st.divider()
    if ar["fecha_max_drawdown"]:
        st.markdown(f"**Fecha del máximo drawdown:** {ar['fecha_max_drawdown']}")
    st.markdown(f"**Peor día operativo:** {ar['peor_dia_drawdown_pct']:.2f}%")
    if ar["fecha_peor_dia"]:
        st.markdown(f"**Fecha del peor día:** {ar['fecha_peor_dia']}")


def _renderizar_tab_montecarlo(reporte: ReporteAuditoriaDict) -> None:
    mc = reporte["prueba_estres_montecarlo"]
    st.subheader("Prueba de estrés — Simulación de Monte Carlo")

    prob = mc["prob_quiebra_antes_de_meta_pct"]
    if prob >= 50:
        st.error(f"🔴 Probabilidad de quiebra antes de la meta: {prob:.2f}% (alto riesgo de varianza)")
    elif prob >= 20:
        st.warning(f"🟡 Probabilidad de quiebra antes de la meta: {prob:.2f}% (riesgo moderado)")
    else:
        st.success(f"🟢 Probabilidad de quiebra antes de la meta: {prob:.2f}% (riesgo bajo)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Simulaciones ejecutadas", f"{mc['n_simulaciones']:,}")
    col2.metric("Drawdown mediana simulado", f"{mc['drawdown_max_mediana_pct']:.2f}%")
    col3.metric("Percentil 95 (peor escenario)", f"{mc['drawdown_max_percentil_95_pct']:.2f}%")


def renderizar_resultados(reporte: ReporteAuditoriaDict) -> None:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Configuración", "📈 Rendimiento Puro", "⚠️ Auditoría de Riesgo", "🎲 Monte Carlo"]
    )
    with tab1:
        _renderizar_tab_configuracion(reporte)
    with tab2:
        _renderizar_tab_rendimiento(reporte)
    with tab3:
        _renderizar_tab_riesgo(reporte)
    with tab4:
        _renderizar_tab_montecarlo(reporte)

    st.divider()
    st.download_button(
        label="⬇️ Descargar informe completo (JSON)",
        data=json.dumps(reporte, indent=2, ensure_ascii=False),
        file_name=f"auditoria_{reporte['configuracion_base']['ticker']}.json",
        mime="application/json",
    )
    st.caption(reporte["metadata"]["descargo_legal"])


# ============================================================
# main
# ============================================================

def main() -> None:
    st.set_page_config(page_title="Auditor de Prop Firms", page_icon="🛡️", layout="wide")
    st.title("🛡️ Auditor Cuantitativo de Estrategias para Prop Firms")
    st.caption(
        "Valida tu estrategia contra las reglas de una evaluación de fondeo "
        "ANTES de pagar el examen."
    )

    activo, reglas, prop, tipo_estrategia, parametros_estrategia, fuente_datos = _sidebar_configuracion()

    st.header("1. Carga de datos históricos")
    df: Optional[pd.DataFrame] = _cargar_datos_segun_fuente(fuente_datos, activo["ticker"], activo["temporalidad"])

    if df is not None:
        st.success(f"Datos disponibles: {len(df):,} filas, "
                   f"del {df['timestamp'].min().date()} al {df['timestamp'].max().date()}.")
        with st.expander("Previsualizar primeros registros"):
            st.dataframe(df.head(20), use_container_width=True)

    st.header("2. Ejecutar auditoría")
    ejecutar = st.button("🚀 Ejecutar Auditoría Completa", type="primary", disabled=df is None)

    if ejecutar and df is not None:
        with st.spinner("Corriendo backtest event-driven, métricas, drawdown y Monte Carlo..."):
            try:
                config, reglas_estrategia, prop_profile, estrategia = _construir_objetos_dominio(
                    df, activo, reglas, prop, tipo_estrategia, parametros_estrategia
                )
                reporte, _trades_df = ejecutar_pipeline(
                    df, config, reglas_estrategia, prop_profile, estrategia,
                    capital_inicial=prop["capital_inicial"],
                )
                st.session_state["reporte_auditoria"] = reporte
                st.success("Auditoría completada.")
            except ValueError as e:
                st.error(f"No se pudo completar la auditoría: {e}")
                st.session_state.pop("reporte_auditoria", None)
            except Exception as e:  # fail-fast pero sin tumbar la app con un traceback crudo al usuario
                st.error(f"Error inesperado en el pipeline: {e}")
                with st.expander("Detalle técnico"):
                    st.code(traceback.format_exc())
                st.session_state.pop("reporte_auditoria", None)

    st.header("3. Resultados")
    if "reporte_auditoria" in st.session_state:
        renderizar_resultados(st.session_state["reporte_auditoria"])
    else:
        st.info("Carga un CSV y ejecuta la auditoría para ver resultados aquí.")


if __name__ == "__main__":
    main()
