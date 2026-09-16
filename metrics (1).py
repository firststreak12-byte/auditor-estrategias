"""
metrics.py
Evaluador de curva de equity y drawdown para el DataFrame de trades
producido por engine.ejecutar_backtest().

Tres nociones de drawdown, porque cada Prop Firm mide "romper la cuenta"
de forma distinta (esto es la causa #1 de auditorías mal hechas):

  - ESTATICO: el piso de pérdida máxima permitida es fijo desde el día 1
    (capital_inicial * (1 - limite_perdida_total_pct/100)) y NUNCA sube,
    ni siquiera si la cuenta genera profit. Es el más común en fase 1/2.
  - FLOTANTE: el piso se mide contra el equity flotante intradía (incluye
    la Excursión Máxima Adversa -MAE- de trades que siguen abiertos), no
    solo contra el equity ya realizado al cierre de cada trade.
  - TRAILING: el piso SUBE con cada nuevo high-water mark de equity
    (como una cuenta FTMO clásica): drawdown = high_water_mark * (1 - pct).

Como este motor no simula tick-by-tick dentro del trade (solo O/H/L/C por
barra), la MAE intradía se estima de forma conservadora: se asume que,
mientras el trade estuvo abierto, el precio pudo llegar al extremo más
adverso registrado entre high/low de TODAS las barras que abarcó ese
trade (peor escenario, no el promedio) — esto es intencionalmente
pesimista para no subestimar el riesgo real frente al kill switch diario.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from prop_audit.models import PropFirmProfile, TipoDrawdown


@dataclass
class ResultadoDrawdown:
    max_drawdown_pct: float
    max_drawdown_usd: float
    fecha_max_drawdown: Optional[pd.Timestamp]
    excedio_limite: bool
    peor_dia_drawdown_pct: float
    fecha_peor_dia: Optional[object]
    curva_equity: pd.DataFrame


def calcular_mae_por_trade(trades_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.Series:
    """
    Excursión Máxima Adversa por trade: para cada operación, busca el
    peor precio (mínimo si es long, máximo si es short) entre TODAS las
    barras del histórico comprendidas entre fecha_entrada y fecha_salida.
    Devuelve la MAE en unidades monetarias por lote (ya multiplicada por
    contract_size*tick_value queda fuera de este módulo: aquí se regresa
    en puntos de precio, para no acoplar metrics.py a ActivoConfig).
    """
    if trades_df.empty:
        return pd.Series(dtype=float)

    ohlcv_indexado = ohlcv_df.set_index("timestamp")
    maes = []
    for _, t in trades_df.iterrows():
        ventana = ohlcv_indexado.loc[t["fecha_entrada"]: t["fecha_salida"]]
        if ventana.empty:
            maes.append(0.0)
            continue
        if t["direccion"] == "long":
            peor_precio = ventana["low"].min()
            mae = max(t["precio_entrada"] - peor_precio, 0.0)
        else:
            peor_precio = ventana["high"].max()
            mae = max(peor_precio - t["precio_entrada"], 0.0)
        maes.append(mae)
    return pd.Series(maes, index=trades_df.index)


def construir_curva_equity(trades_df: pd.DataFrame, capital_inicial: float) -> pd.DataFrame:
    """Curva de equity REALIZADO (post-cierre de cada trade), base para
    el drawdown estático y trailing."""
    if trades_df.empty:
        return pd.DataFrame({"fecha": [], "equity": []})

    curva = trades_df[["fecha_salida", "equity_tras_cierre"]].copy()
    curva = curva.rename(columns={"fecha_salida": "fecha", "equity_tras_cierre": "equity"})
    fila_inicial = pd.DataFrame({"fecha": [trades_df["fecha_entrada"].iloc[0]], "equity": [capital_inicial]})
    curva = pd.concat([fila_inicial, curva], ignore_index=True)
    return curva


def calcular_drawdown(
    trades_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    prop: PropFirmProfile,
    capital_inicial: float,
    multiplicador_monetario: float = 1.0,
) -> ResultadoDrawdown:
    """
    multiplicador_monetario = config.contract_size * config.tick_value del
    ActivoConfig usado en el backtest. Se pasa explícito en vez de importar
    ActivoConfig aquí para mantener metrics.py desacoplado del módulo de
    configuración de activos; es responsabilidad del caller (report.py)
    proveer el valor correcto.
    """
    curva = construir_curva_equity(trades_df, capital_inicial)
    if curva.empty:
        return ResultadoDrawdown(0.0, 0.0, None, False, 0.0, None, curva)

    limite_pct = prop.limite_perdida_total_pct

    if prop.tipo_drawdown == TipoDrawdown.ESTATICO:
        piso = capital_inicial * (1 - limite_pct / 100.0)
        curva["drawdown_usd"] = capital_inicial - curva["equity"]
        curva["piso"] = piso

    elif prop.tipo_drawdown == TipoDrawdown.TRAILING:
        curva["high_water_mark"] = curva["equity"].cummax().clip(lower=capital_inicial)
        curva["piso"] = curva["high_water_mark"] * (1 - limite_pct / 100.0)
        curva["drawdown_usd"] = curva["high_water_mark"] - curva["equity"]

    else:  # FLOTANTE: incorpora la MAE del trade que estaba abierto en cada momento
        # Para cada trade, el equity flotante MÍNIMO alcanzado mientras estuvo
        # abierto = equity al momento de la entrada - MAE monetaria de ese trade.
        # Esto es estrictamente peor (más conservador) que solo mirar el
        # equity ya realizado al cierre, que es lo que exige un kill switch
        # de tipo "flotante" real en una prop firm.
        mae_puntos = calcular_mae_por_trade(trades_df, ohlcv_df) if not trades_df.empty else pd.Series(dtype=float)
        high_water_mark = curva["equity"].cummax().clip(lower=capital_inicial)
        curva["high_water_mark"] = high_water_mark

        drawdown_usd_flotante = (high_water_mark - curva["equity"]).copy()
        if not trades_df.empty and len(mae_puntos) == len(trades_df):
            equity_previo_a_trade = curva["equity"].iloc[:-1].reset_index(drop=True)  # equity antes de cada cierre
            mae_usd = mae_puntos.reset_index(drop=True) * trades_df["lotes"].reset_index(drop=True) * multiplicador_monetario
            hwm_en_trade = high_water_mark.iloc[:-1].reset_index(drop=True)
            dd_intra_trade_usd = (hwm_en_trade - (equity_previo_a_trade - mae_usd)).clip(lower=0)
            # Combina, barra a barra del vector de trades, el drawdown ya
            # realizado al cierre con el peor drawdown flotante visto MIENTRAS
            # el trade estuvo abierto; se toma el máximo de ambos por posición.
            drawdown_usd_flotante.iloc[1:] = np.maximum(
                drawdown_usd_flotante.iloc[1:].values, dd_intra_trade_usd.values
            )

        curva["piso"] = high_water_mark * (1 - limite_pct / 100.0)
        curva["drawdown_usd"] = drawdown_usd_flotante

    curva["drawdown_pct"] = (curva["drawdown_usd"] / capital_inicial) * 100.0
    idx_peor = curva["drawdown_pct"].idxmax()
    max_dd_pct = float(curva["drawdown_pct"].max())
    max_dd_usd = float(curva["drawdown_usd"].max())
    fecha_max_dd = curva.loc[idx_peor, "fecha"] if pd.notna(idx_peor) else None
    excedio = max_dd_pct >= limite_pct

    # Peor día operativo (para el kill switch de drawdown DIARIO, distinto del total)
    if "fecha_operativa" in trades_df.columns and not trades_df.empty:
        pnl_por_dia = trades_df.groupby("fecha_operativa")["pnl_neto"].sum()
        peor_dia_usd = pnl_por_dia.min()
        peor_dia_pct = abs(min(peor_dia_usd, 0.0)) / capital_inicial * 100.0
        fecha_peor_dia = pnl_por_dia.idxmin() if peor_dia_usd < 0 else None
    else:
        peor_dia_pct, fecha_peor_dia = 0.0, None

    return ResultadoDrawdown(
        max_drawdown_pct=max_dd_pct,
        max_drawdown_usd=max_dd_usd,
        fecha_max_drawdown=fecha_max_dd,
        excedio_limite=excedio,
        peor_dia_drawdown_pct=peor_dia_pct,
        fecha_peor_dia=fecha_peor_dia,
        curva_equity=curva,
    )


def calcular_metricas_rendimiento(trades_df: pd.DataFrame, capital_inicial: float) -> dict:
    """EV, Win Rate, Profit Factor, Sharpe y Sortino (por operación, no anualizados
    salvo que se le pase una frecuencia; aquí se reportan por-trade, que es lo
    relevante para evaluar la ventaja estadística de una estrategia de scalping)."""
    if trades_df.empty:
        return {
            "ev_por_dolar": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "total_trades": 0,
            "racha_perdidas_max": 0, "peor_racha_usd": 0.0,
        }

    ganadoras = trades_df[trades_df["pnl_neto"] > 0]
    perdedoras = trades_df[trades_df["pnl_neto"] <= 0]

    win_rate = len(ganadoras) / len(trades_df)
    ganancia_prom = ganadoras["pnl_neto"].mean() if not ganadoras.empty else 0.0
    perdida_prom = abs(perdedoras["pnl_neto"].mean()) if not perdedoras.empty else 0.0

    ev = (win_rate * ganancia_prom) - ((1 - win_rate) * perdida_prom)

    bruto_ganado = ganadoras["pnl_neto"].sum()
    bruto_perdido = abs(perdedoras["pnl_neto"].sum())
    profit_factor = bruto_ganado / bruto_perdido if bruto_perdido > 0 else float("inf")

    retornos = trades_df["pnl_neto"] / capital_inicial
    sharpe = (retornos.mean() / retornos.std()) * np.sqrt(len(retornos)) if retornos.std() > 0 else 0.0

    # Sortino: la desviación a la baja se mide contra 0 (el objetivo mínimo
    # aceptable), NO contra la media del subconjunto de retornos negativos,
    # y se promedia sobre TODA la población de trades (los positivos aportan
    # 0 en np.minimum(0, r)**2). Este es el estándar correcto: castiga sólo
    # la varianza adversa sin descartar información de tamaño de muestra.
    downside_deviation = np.sqrt(np.mean(np.minimum(0, retornos.to_numpy()) ** 2))
    sortino = (retornos.mean() / downside_deviation) * np.sqrt(len(retornos)) if downside_deviation > 0 else 0.0

    # Peor racha de pérdidas consecutivas
    signo = np.where(trades_df["pnl_neto"].values > 0, 1, -1)
    racha_actual = 0
    racha_max = 0
    pnl_racha = 0.0
    peor_pnl_racha = 0.0
    for i, s in enumerate(signo):
        if s < 0:
            racha_actual += 1
            pnl_racha += trades_df["pnl_neto"].values[i]
            if racha_actual > racha_max:
                racha_max = racha_actual
                peor_pnl_racha = pnl_racha
        else:
            racha_actual = 0
            pnl_racha = 0.0

    return {
        "ev_por_dolar": float(ev),
        "win_rate": float(win_rate * 100),
        "profit_factor": float(profit_factor),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "total_trades": int(len(trades_df)),
        "racha_perdidas_max": int(racha_max),
        "peor_racha_usd": float(peor_pnl_racha),
    }
