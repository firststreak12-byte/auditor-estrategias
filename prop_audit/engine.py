"""
engine.py
Motor de backtest "event-driven" estricto:

  - La señal de entrada generada en la barra t solo puede haberse calculado
    con información disponible hasta el cierre de la barra t (el usuario es
    responsable de que su función de señal use únicamente rolling windows
    hacia atrás; el motor además fuerza la ejecución en la APERTURA de la
    barra t+1, nunca en el cierre de la misma barra donde nació la señal).
  - Una vez abierta la posición, se recorre barra a barra comprobando si
    high/low de esa barra tocan el Stop Loss, Take Profit, Break Even o el
    cierre por tiempo. Se prioriza el SL sobre el TP dentro de la misma
    barra (supuesto conservador: si ambos niveles caben en el rango de la
    vela, se asume que el precio tocó primero el nivel adverso).
  - Comisión y slippage se descuentan en CADA operación (entrada y salida).
  - El lotaje y el multiplicador monetario del PnL usan `ActivoConfig`
    (tick_value, contract_size), NUNCA campos de PropFirmProfile: el valor
    de un punto de precio es una propiedad del INSTRUMENTO, no de la firma
    de fondeo.
  - El equity es dinámico: cada trade cerrado actualiza el balance usado
    para calcular el riesgo en % de la siguiente operación (interés
    compuesto real, positivo o negativo).
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, List

from prop_audit.models import ActivoConfig, ReglasEstrategia, PropFirmProfile, Direccion, ModoPosicion


@dataclass
class Trade:
    fecha_entrada: pd.Timestamp
    fecha_salida: pd.Timestamp
    direccion: str
    precio_entrada: float
    precio_salida: float
    lotes: float
    pnl_bruto: float
    comision: float
    slippage_costo: float
    pnl_neto: float
    motivo_salida: str
    fecha_operativa: object
    equity_tras_cierre: float


def _costo_slippage(puntos_slippage: float) -> float:
    """Slippage expresado directamente en unidades de precio del activo."""
    return puntos_slippage


def _calcular_lotes(
    equity: float,
    riesgo_dist: float,
    reglas: ReglasEstrategia,
    config: ActivoConfig,
) -> float:
    """
    Determina el tamaño de posición en lotes.

    - RIESGO_PORCENTAJE: lotes = (equity * riesgo%) / (distancia_al_SL_en_precio * contract_size * tick_value)
      Es decir: cuántos lotes hacen que, si el precio recorre exactamente la
      distancia al stop, la pérdida monetaria sea igual al riesgo permitido.
    - LOTE_FIJO: usa el lote fijo configurado por el usuario, sin importar el equity.
    """
    if reglas.modo_posicion == ModoPosicion.LOTE_FIJO:
        return reglas.lote_fijo

    riesgo_usd = equity * reglas.riesgo_por_operacion_pct / 100.0
    valor_por_punto_por_lote = riesgo_dist * config.contract_size * config.tick_value
    if valor_por_punto_por_lote <= 0:
        raise ValueError(
            "valor_por_punto_por_lote <= 0: revisa stop_loss_pct, contract_size y tick_value "
            "en ActivoConfig; con estos parámetros no es posible dimensionar la posición."
        )
    return max(riesgo_usd / valor_por_punto_por_lote, 0.01)


def _registrar_cierre(
    trades: List[Trade],
    df: pd.DataFrame,
    idx_entrada: int,
    idx_salida: int,
    dir_actual: str,
    precio_entrada: float,
    precio_entrada_puro: float,
    salida_precio: float,
    motivo: str,
    equity: float,
    reglas: ReglasEstrategia,
    prop: PropFirmProfile,
    config: ActivoConfig,
    riesgo_dist: float,
) -> float:
    """Aplica slippage/comisión de salida, calcula PnL con contract_size/tick_value,
    agrega el Trade a la lista y devuelve el nuevo equity.

    Semántica de PnL (sin solapamiento):
      - pnl_bruto: mide la ventaja pura de la ESTRATEGIA sobre el mercado,
        usando precios OHLCV puros (sin slippage, sin comisión) tanto en
        entrada como en salida. Responde: "¿la señal en sí tiene edge?".
      - pnl_neto: lo que realmente queda en la cuenta — usa los precios ya
        ajustados por slippage (precio_entrada / salida_ajustada) y además
        descuenta la comisión. Responde: "¿cuánto dinero real gana/pierde
        la cuenta?". slippage_costo aísla el costo de fricción entre ambos.
    """
    fila_salida = df.iloc[idx_salida]

    slip = _costo_slippage(prop.slippage_puntos)
    salida_ajustada = salida_precio - slip if dir_actual == "long" else salida_precio + slip

    lotes = _calcular_lotes(equity, riesgo_dist, reglas, config)
    multiplicador = config.contract_size * config.tick_value

    mov_bruto = (salida_precio - precio_entrada_puro) if dir_actual == "long" else (precio_entrada_puro - salida_precio)
    pnl_bruto = mov_bruto * lotes * multiplicador

    mov_neto = (salida_ajustada - precio_entrada) if dir_actual == "long" else (precio_entrada - salida_ajustada)
    comision = prop.comision_por_lote * lotes
    pnl_neto = (mov_neto * lotes * multiplicador) - comision

    slippage_costo = pnl_bruto - (mov_neto * lotes * multiplicador)  # costo puro de fricción de mercado

    equity += pnl_neto  # <-- equity dinámico: habilita riesgo compuesto en el siguiente trade

    trades.append(Trade(
        fecha_entrada=df.iloc[idx_entrada]["timestamp"],
        fecha_salida=fila_salida["timestamp"],
        direccion=dir_actual,
        precio_entrada=precio_entrada,
        precio_salida=salida_ajustada,
        lotes=lotes,
        pnl_bruto=pnl_bruto,
        comision=comision,
        slippage_costo=slippage_costo,
        pnl_neto=pnl_neto,
        motivo_salida=motivo,
        fecha_operativa=fila_salida["fecha_operativa"],
        equity_tras_cierre=equity,
    ))
    return equity


def ejecutar_backtest(
    df: pd.DataFrame,
    señal_fn: Callable[[pd.DataFrame], pd.Series],
    reglas: ReglasEstrategia,
    prop: PropFirmProfile,
    config: ActivoConfig,
    capital_inicial: float,
) -> List[Trade]:
    """
    señal_fn recibe el DataFrame completo y debe regresar una Serie con valores
    en {"long", "short", None} indexada igual que df. Es responsabilidad del
    caller garantizar que señal_fn solo use pd.Series.rolling/expanding hacia
    atrás (no funciones centradas ni acceso a df.shift(-n)).

    Fail-fast: si el DataFrame no trae las columnas OHLCV+fecha_operativa
    esperadas, revienta aquí en vez de producir un backtest corrupto.
    """
    columnas_requeridas = {"timestamp", "open", "high", "low", "close", "fecha_operativa"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"df no contiene las columnas requeridas para el backtest: {faltantes}")

    señales = señal_fn(df).shift(1)  # fuerza ejecución en la barra siguiente (anti look-ahead)

    trades: List[Trade] = []
    equity = capital_inicial
    en_posicion = False
    dir_actual: Optional[str] = None
    precio_entrada: Optional[float] = None       # ejecutado: incluye slippage de entrada
    precio_entrada_puro: Optional[float] = None  # OHLCV puro (fila["open"]), sin fricción
    barras_en_trade = 0
    break_even_activo = False
    idx_entrada: Optional[int] = None
    riesgo_dist_actual: Optional[float] = None

    for i in range(1, len(df)):
        fila = df.iloc[i]

        if en_posicion:
            barras_en_trade += 1
            salida_precio = None
            motivo = None

            riesgo_dist_actual = precio_entrada * reglas.stop_loss_pct / 100.0
            sl_precio = (precio_entrada - riesgo_dist_actual) if dir_actual == "long" else (precio_entrada + riesgo_dist_actual)

            if reglas.take_profit_pct is not None:
                tp_dist = precio_entrada * reglas.take_profit_pct / 100.0
                tp_precio = (precio_entrada + tp_dist) if dir_actual == "long" else (precio_entrada - tp_dist)
            else:
                tp_precio = None

            if reglas.break_even_activacion_pct is not None and not break_even_activo:
                be_dist = precio_entrada * reglas.break_even_activacion_pct / 100.0
                nivel_be = (precio_entrada + be_dist) if dir_actual == "long" else (precio_entrada - be_dist)
                tocó_be = (fila["high"] >= nivel_be) if dir_actual == "long" else (fila["low"] <= nivel_be)
                if tocó_be:
                    break_even_activo = True
                    sl_precio = precio_entrada  # mueve el stop a break even

            # Prioridad conservadora: SL antes que TP si ambos caben en el rango de la vela
            if dir_actual == "long":
                if fila["low"] <= sl_precio:
                    salida_precio, motivo = sl_precio, ("Break Even" if break_even_activo else "Stop Loss")
                elif tp_precio is not None and fila["high"] >= tp_precio:
                    salida_precio, motivo = tp_precio, "Take Profit"
            else:
                if fila["high"] >= sl_precio:
                    salida_precio, motivo = sl_precio, ("Break Even" if break_even_activo else "Stop Loss")
                elif tp_precio is not None and fila["low"] <= tp_precio:
                    salida_precio, motivo = tp_precio, "Take Profit"

            if salida_precio is None and reglas.cierre_por_tiempo_barras is not None:
                if barras_en_trade >= reglas.cierre_por_tiempo_barras:
                    salida_precio, motivo = fila["close"], "Cierre por tiempo"

            if salida_precio is not None:
                equity = _registrar_cierre(
                    trades, df, idx_entrada, i, dir_actual, precio_entrada, precio_entrada_puro,
                    salida_precio, motivo, equity, reglas, prop, config, riesgo_dist_actual,
                )
                en_posicion = False
                break_even_activo = False
                barras_en_trade = 0

        if not en_posicion:
            señal = señales.iloc[i]
            if señal in ("long", "short") and reglas.direccion in (Direccion.AMBAS, Direccion[señal.upper()]):
                en_posicion = True
                dir_actual = señal
                slip = _costo_slippage(prop.slippage_puntos)
                precio_entrada_puro = fila["open"]
                precio_entrada = fila["open"] + slip if señal == "long" else fila["open"] - slip
                idx_entrada = i
                barras_en_trade = 0

    # --- Cierre forzado del trade abierto al final de la serie histórica ---
    # Sin esto, una posición viva en la última barra desaparece silenciosamente
    # del reporte: ni cuenta como ganadora, perdedora, ni afecta el drawdown.
    if en_posicion:
        ultima_fila = df.iloc[-1]
        riesgo_dist_actual = precio_entrada * reglas.stop_loss_pct / 100.0
        equity = _registrar_cierre(
            trades, df, idx_entrada, len(df) - 1, dir_actual, precio_entrada, precio_entrada_puro,
            ultima_fila["close"], "Cierre forzado (fin de histórico)",
            equity, reglas, prop, config, riesgo_dist_actual,
        )

    return trades


def trades_a_dataframe(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
    return pd.DataFrame([t.__dict__ for t in trades])
