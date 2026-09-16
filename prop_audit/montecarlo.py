"""
montecarlo.py
Simulación de estrés por bootstrapping: toma la secuencia REAL de pnl_neto
de los trades ya cerrados y la reordena (con reemplazo) N veces para
responder una pregunta que el backtest lineal no puede responder por sí
solo: "¿qué tan sensible es esta cuenta al ORDEN en que llegaron las
rachas de pérdidas?".

Una estrategia con EV positivo puede seguir teniendo una probabilidad no
trivial de romper el drawdown máximo ANTES de alcanzar la meta de
beneficio, simplemente por mala suerte en la secuencia (varianza). Esto es
precisamente lo que rompe cuentas de fondeo con estrategias "rentables en
papel".

Nota de diseño: el bootstrap con reemplazo asume trades aproximadamente
independientes entre sí (sin autocorrelación fuerte de rachas). Para
estrategias con dependencia serial fuerte (ej. martingala, grid), esto
subestima el riesgo de cola; en ese caso se recomienda block-bootstrap,
fuera del alcance de esta primera versión.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from prop_audit.models import PropFirmProfile, TipoDrawdown


@dataclass
class ResultadoMonteCarlo:
    n_simulaciones: int
    prob_quiebra_antes_de_meta: float          # % de simulaciones que tocan el límite de DD antes de la meta
    drawdown_max_percentil_5: float             # peor escenario (percentil 5, en % del capital)
    drawdown_max_mediana: float                 # escenario típico (percentil 50)
    drawdown_max_percentil_95: float             # mejor escenario relativo (percentil 95)
    dias_estimados_meta_mediana: Optional[float]


def simular_montecarlo(
    trades_df: pd.DataFrame,
    prop: PropFirmProfile,
    capital_inicial: float,
    n_simulaciones: int = 1000,
    semilla: Optional[int] = 42,
) -> ResultadoMonteCarlo:
    if trades_df.empty or len(trades_df) < 10:
        raise ValueError(
            "Se requieren al menos 10 trades cerrados para que la simulación de "
            "Monte Carlo tenga algún valor estadístico. Backtest insuficiente."
        )

    rng = np.random.default_rng(semilla)
    pnl = trades_df["pnl_neto"].to_numpy()
    n_trades = len(pnl)

    limite_dd_usd = capital_inicial * prop.limite_perdida_total_pct / 100.0
    meta_usd = capital_inicial * prop.meta_beneficio_pct / 100.0

    drawdowns_max_pct = np.empty(n_simulaciones)
    quiebras = np.zeros(n_simulaciones, dtype=bool)
    trades_hasta_meta = np.full(n_simulaciones, np.nan)

    for s in range(n_simulaciones):
        orden = rng.integers(0, n_trades, size=n_trades)  # muestreo CON reemplazo (bootstrap clásico)
        secuencia_pnl = pnl[orden]
        equity_curva = capital_inicial + np.cumsum(secuencia_pnl)

        high_water_mark = np.maximum.accumulate(np.concatenate(([capital_inicial], equity_curva)))[1:]
        if prop.tipo_drawdown == TipoDrawdown.ESTATICO:
            piso_referencia = np.full(n_trades, capital_inicial)
        else:  # trailing y flotante se aproximan aquí con high-water mark (sin datos intrabarra en el bootstrap)
            piso_referencia = high_water_mark

        drawdown_usd = piso_referencia - equity_curva
        drawdown_max_usd = max(drawdown_usd.max(), 0.0)
        drawdowns_max_pct[s] = drawdown_max_usd / capital_inicial * 100.0

        tocó_limite_idx = np.argmax(drawdown_usd >= limite_dd_usd) if np.any(drawdown_usd >= limite_dd_usd) else -1
        tocó_meta_idx = np.argmax((equity_curva - capital_inicial) >= meta_usd) if np.any((equity_curva - capital_inicial) >= meta_usd) else -1

        if tocó_limite_idx != -1 and (tocó_meta_idx == -1 or tocó_limite_idx <= tocó_meta_idx):
            quiebras[s] = True
        if tocó_meta_idx != -1:
            trades_hasta_meta[s] = tocó_meta_idx + 1

    return ResultadoMonteCarlo(
        n_simulaciones=n_simulaciones,
        prob_quiebra_antes_de_meta=float(quiebras.mean() * 100.0),
        drawdown_max_percentil_5=float(np.percentile(drawdowns_max_pct, 95)),  # peor caso = cola alta de DD
        drawdown_max_mediana=float(np.percentile(drawdowns_max_pct, 50)),
        drawdown_max_percentil_95=float(np.percentile(drawdowns_max_pct, 5)),  # mejor caso = cola baja de DD
        dias_estimados_meta_mediana=float(np.nanmedian(trades_hasta_meta)) if not np.all(np.isnan(trades_hasta_meta)) else None,
    )
