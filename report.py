"""
report.py
Orquestador final del pipeline de auditoría. No recalcula nada: toma los
objetos YA producidos por engine.py, metrics.py y montecarlo.py, los
aplana en un diccionario tipado (TypedDict) listo para `json.dumps` o para
insertar en una base de datos documental, y opcionalmente lo imprime en
consola en 4 bloques.

Separación de responsabilidades deliberada:
  - generar_diccionario_auditoria(): SOLO transforma datos, no imprime,
    no decide pass/fail. Es la función que debe llamar cualquier capa que
    sirva esto por API/webhook (ver arquitectura FastAPI + Stripe).
  - imprimir_reporte(): SOLO consume el diccionario y formatea texto. No
    conoce engine.py, metrics.py ni montecarlo.py — así se puede reemplazar
    por un render a HTML/PDF sin tocar la lógica de negocio.
"""

from __future__ import annotations
import json
from datetime import datetime, date
from typing import Any, Dict, Optional, TypedDict, Union

import pandas as pd

from prop_audit.models import ActivoConfig, ReglasEstrategia, PropFirmProfile
from prop_audit.metrics import ResultadoDrawdown
from prop_audit.montecarlo import ResultadoMonteCarlo


DESCARGO_LEGAL = (
    "Análisis cuantitativo sobre datos históricos. No constituye asesoría "
    "financiera ni garantía de rendimientos o aprobación en exámenes reales."
)


# ============================================================
# Tipado estricto del diccionario de salida (TypedDict)
# ============================================================

class ConfiguracionBaseDict(TypedDict):
    ticker: str
    temporalidad: str
    fecha_inicio: str
    fecha_fin: str
    capital_inicial: float
    riesgo_por_operacion_pct: float
    firma_fondeo: str
    limite_perdida_diaria_pct: float
    limite_perdida_total_pct: float
    meta_beneficio_pct: float
    tipo_drawdown: str


class RendimientoPuroDict(TypedDict):
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    ev_por_dolar: float
    sharpe: float
    sortino: float
    racha_perdidas_max: int
    peor_racha_usd: float


class AuditoriaRiesgoDict(TypedDict):
    max_drawdown_usd: float
    max_drawdown_pct: float
    limite_permitido_pct: float
    excedio_limite: bool
    fecha_max_drawdown: Optional[str]
    peor_dia_drawdown_pct: float
    fecha_peor_dia: Optional[str]


class PruebaEstresMontecarloDict(TypedDict):
    n_simulaciones: int
    prob_quiebra_antes_de_meta_pct: float
    drawdown_max_mediana_pct: float
    drawdown_max_percentil_95_pct: float  # peor escenario relativo (cola alta de drawdown)


class MetadataDict(TypedDict):
    fecha_generacion: str
    descargo_legal: str


class ReporteAuditoriaDict(TypedDict):
    configuracion_base: ConfiguracionBaseDict
    rendimiento_puro: RendimientoPuroDict
    auditoria_riesgo: AuditoriaRiesgoDict
    prueba_estres_montecarlo: PruebaEstresMontecarloDict
    metadata: MetadataDict


# ============================================================
# Helpers de normalización (fail-fast ante tipos no serializables)
# ============================================================

def _a_iso(valor: Union[pd.Timestamp, datetime, date, None]) -> Optional[str]:
    """Normaliza cualquier tipo de fecha/timestamp a str ISO-8601 o None.
    Se hace explícito aquí para que el diccionario resultante sea
    json.dumps-able sin necesidad de un encoder custom en el caller."""
    if valor is None:
        return None
    if isinstance(valor, pd.Timestamp):
        return valor.isoformat()
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    # Último recurso: si llega un tipo no reconocido, se fuerza a str en vez
    # de fallar silenciosamente con un objeto no serializable en el dict.
    return str(valor)


# ============================================================
# Función 1: construcción del diccionario tipado
# ============================================================

def generar_diccionario_auditoria(
    config: ActivoConfig,
    reglas: ReglasEstrategia,
    prop: PropFirmProfile,
    capital_inicial: float,
    metricas: Dict[str, Any],          # salida de metrics.calcular_metricas_rendimiento
    drawdown: ResultadoDrawdown,       # salida de metrics.calcular_drawdown
    monte_carlo: ResultadoMonteCarlo,  # salida de montecarlo.simular_montecarlo
) -> ReporteAuditoriaDict:
    """
    Aplana los objetos de las 3 etapas previas del pipeline en un único
    diccionario tipado, listo para `json.dumps(reporte, indent=2)` o para
    insertar en Mongo/Postgres-JSONB sin transformación adicional.

    Fail-fast: si `metricas` no trae alguna de las llaves esperadas (porque
    el caller pasó un dict de otra función por error), revienta aquí con un
    KeyError claro en vez de generar un reporte con campos faltantes o en
    None silenciosamente.
    """
    llaves_requeridas = {
        "total_trades", "win_rate", "profit_factor", "ev_por_dolar",
        "sharpe", "sortino", "racha_perdidas_max", "peor_racha_usd",
    }
    faltantes = llaves_requeridas - set(metricas.keys())
    if faltantes:
        raise KeyError(
            f"El diccionario de métricas no trae las llaves esperadas: {faltantes}. "
            f"¿Se pasó la salida de metrics.calcular_metricas_rendimiento()?"
        )

    configuracion_base: ConfiguracionBaseDict = {
        "ticker": config.ticker,
        "temporalidad": config.temporalidad,
        "fecha_inicio": _a_iso(config.fecha_inicio),
        "fecha_fin": _a_iso(config.fecha_fin),
        "capital_inicial": float(capital_inicial),
        "riesgo_por_operacion_pct": float(reglas.riesgo_por_operacion_pct),
        "firma_fondeo": prop.nombre_firma,
        "limite_perdida_diaria_pct": float(prop.limite_perdida_diaria_pct),
        "limite_perdida_total_pct": float(prop.limite_perdida_total_pct),
        "meta_beneficio_pct": float(prop.meta_beneficio_pct),
        "tipo_drawdown": prop.tipo_drawdown.value,
    }

    rendimiento_puro: RendimientoPuroDict = {
        "total_trades": int(metricas["total_trades"]),
        "win_rate_pct": float(metricas["win_rate"]),
        "profit_factor": float(metricas["profit_factor"]),
        "ev_por_dolar": float(metricas["ev_por_dolar"]),
        "sharpe": float(metricas["sharpe"]),
        "sortino": float(metricas["sortino"]),
        "racha_perdidas_max": int(metricas["racha_perdidas_max"]),
        "peor_racha_usd": float(metricas["peor_racha_usd"]),
    }

    auditoria_riesgo: AuditoriaRiesgoDict = {
        "max_drawdown_usd": float(drawdown.max_drawdown_usd),
        "max_drawdown_pct": float(drawdown.max_drawdown_pct),
        "limite_permitido_pct": float(prop.limite_perdida_total_pct),
        "excedio_limite": bool(drawdown.excedio_limite),
        "fecha_max_drawdown": _a_iso(drawdown.fecha_max_drawdown),
        "peor_dia_drawdown_pct": float(drawdown.peor_dia_drawdown_pct),
        "fecha_peor_dia": _a_iso(drawdown.fecha_peor_dia),
    }

    prueba_estres_montecarlo: PruebaEstresMontecarloDict = {
        "n_simulaciones": int(monte_carlo.n_simulaciones),
        "prob_quiebra_antes_de_meta_pct": float(monte_carlo.prob_quiebra_antes_de_meta),
        "drawdown_max_mediana_pct": float(monte_carlo.drawdown_max_mediana),
        "drawdown_max_percentil_95_pct": float(monte_carlo.drawdown_max_percentil_5),
        # ^ ojo de nomenclatura: ResultadoMonteCarlo.drawdown_max_percentil_5 ya
        # representa el percentil-95 de la DISTRIBUCIÓN de drawdown (el peor
        # 5% de escenarios), que es exactamente lo pedido aquí. Ver docstring
        # de montecarlo.py para la razón del nombre del campo origen.
    }

    metadata: MetadataDict = {
        "fecha_generacion": datetime.now().isoformat(timespec="seconds"),
        "descargo_legal": DESCARGO_LEGAL,
    }

    return {
        "configuracion_base": configuracion_base,
        "rendimiento_puro": rendimiento_puro,
        "auditoria_riesgo": auditoria_riesgo,
        "prueba_estres_montecarlo": prueba_estres_montecarlo,
        "metadata": metadata,
    }


# ============================================================
# Función 2: impresión en consola, 4 bloques, sin librerías externas
# ============================================================

_ANCHO = 70


def _linea(caracter: str = "=") -> str:
    return caracter * _ANCHO


def _fila(etiqueta: str, valor: str) -> str:
    """Alinea etiqueta a la izquierda y valor a la derecha dentro de _ANCHO
    columnas, usando solo f-strings nativos (sin rich/tabulate)."""
    espacio_disponible = _ANCHO - len(etiqueta)
    return f"{etiqueta}{valor.rjust(max(espacio_disponible, 1))}"


def imprimir_reporte(reporte: ReporteAuditoriaDict) -> None:
    """
    Imprime `reporte` (la salida de generar_diccionario_auditoria) en
    consola, en 4 bloques legibles. No realiza NINGÚN cálculo: si un valor
    parece incorrecto, el bug está en la etapa que generó el diccionario,
    no aquí.
    """
    cb = reporte["configuracion_base"]
    rp = reporte["rendimiento_puro"]
    ar = reporte["auditoria_riesgo"]
    mc = reporte["prueba_estres_montecarlo"]
    meta = reporte["metadata"]

    print(_linea())
    print("INFORME DE AUDITORÍA Y SIMULACIÓN DE PROP FIRM")
    print(_linea())
    print(f"Generado: {meta['fecha_generacion']}")
    print()

    # --- Bloque 1: Configuración Base ---
    print(_linea("-"))
    print("1. CONFIGURACIÓN BASE")
    print(_linea("-"))
    print(_fila("Activo:", cb["ticker"]))
    print(_fila("Temporalidad:", cb["temporalidad"]))
    print(_fila("Rango histórico:", f"{cb['fecha_inicio']} a {cb['fecha_fin']}"))
    print(_fila("Capital inicial:", f"${cb['capital_inicial']:,.2f}"))
    print(_fila("Riesgo por operación:", f"{cb['riesgo_por_operacion_pct']:.2f}%"))
    print(_fila("Firma de fondeo:", cb["firma_fondeo"]))
    print(_fila("Límite pérdida diaria:", f"{cb['limite_perdida_diaria_pct']:.2f}%"))
    print(_fila("Límite pérdida total:", f"{cb['limite_perdida_total_pct']:.2f}%"))
    print(_fila("Meta de beneficio:", f"{cb['meta_beneficio_pct']:.2f}%"))
    print(_fila("Tipo de drawdown:", cb["tipo_drawdown"].capitalize()))
    print()

    # --- Bloque 2: Rendimiento Puro ---
    print(_linea("-"))
    print("2. RENDIMIENTO PURO")
    print(_linea("-"))
    print(_fila("Operaciones totales:", f"{rp['total_trades']}"))
    print(_fila("Win Rate:", f"{rp['win_rate_pct']:.2f}%"))
    print(_fila("Profit Factor:", f"{rp['profit_factor']:.2f}"))
    print(_fila("Esperanza Matemática (EV):", f"${rp['ev_por_dolar']:.4f} por trade"))
    print(_fila("Sharpe:", f"{rp['sharpe']:.2f}"))
    print(_fila("Sortino:", f"{rp['sortino']:.2f}"))
    print(_fila("Peor racha de pérdidas:", f"{rp['racha_perdidas_max']} ops / ${rp['peor_racha_usd']:,.2f}"))
    print()

    # --- Bloque 3: Auditoría de Riesgo ---
    print(_linea("-"))
    print("3. AUDITORÍA DE RIESGO")
    print(_linea("-"))
    print(_fila("Max Drawdown registrado:", f"${ar['max_drawdown_usd']:,.2f} ({ar['max_drawdown_pct']:.2f}%)"))
    print(_fila("Límite permitido:", f"{ar['limite_permitido_pct']:.2f}%"))
    estado_limite = "SÍ — CUENTA QUEMADA" if ar["excedio_limite"] else "No"
    print(_fila("¿Excedió el límite?:", estado_limite))
    if ar["fecha_max_drawdown"]:
        print(_fila("Fecha del max drawdown:", ar["fecha_max_drawdown"]))
    print(_fila("Peor día operativo:", f"{ar['peor_dia_drawdown_pct']:.2f}%"))
    if ar["fecha_peor_dia"]:
        print(_fila("Fecha del peor día:", ar["fecha_peor_dia"]))
    print()

    # --- Bloque 4: Prueba de Estrés (Monte Carlo) ---
    print(_linea("-"))
    print("4. PRUEBA DE ESTRÉS (MONTE CARLO)")
    print(_linea("-"))
    print(_fila("Simulaciones ejecutadas:", f"{mc['n_simulaciones']:,}"))
    print(_fila("Prob. de quiebra antes de meta:", f"{mc['prob_quiebra_antes_de_meta_pct']:.2f}%"))
    print(_fila("Drawdown mediana simulado:", f"{mc['drawdown_max_mediana_pct']:.2f}%"))
    print(_fila("Percentil 95 (peor escenario):", f"{mc['drawdown_max_percentil_95_pct']:.2f}%"))
    print()

    print(_linea())
    print(meta["descargo_legal"])
    print(_linea())


# ============================================================
# Utilidad opcional: export directo a JSON en disco
# ============================================================

def exportar_json(reporte: ReporteAuditoriaDict, ruta: str) -> None:
    """El diccionario ya está 100% compuesto de tipos nativos serializables
    (str/float/int/bool/None), así que json.dump no necesita encoder custom."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
