"""
colab_orchestrator.py
Script pensado para pegarse en una celda (o varias) de Google Colab.

Flujo:
  1. Monta/copia los 6 módulos del proyecto (prop_audit.*) al runtime.
  2. Carga un DataFrame OHLCV histórico (subido por el usuario o generado).
  3. Define una grilla de parámetros sobre una subclase de BaseStrategy.
  4. Para cada combinación: corre engine.ejecutar_backtest -> metrics ->
     montecarlo, y junta todo en una tabla comparativa.
  5. Grafica la curva de equity de la mejor combinación y el histograma
     de Monte Carlo del drawdown máximo, en un solo panel matplotlib.

Requisito en Colab: subir los módulos del proyecto a /content/prop_audit/
(o !git clone tu repo) ANTES de correr este script, ya que Colab no
comparte el filesystem de este sandbox.
"""

# ============================================================
# CELDA 1 — Setup (ejecutar una sola vez por sesión de Colab)
# ============================================================
# !pip install pandas numpy matplotlib --quiet
# (si el proyecto vive en GitHub) !git clone https://github.com/tu-usuario/prop_audit_engine.git
# import sys; sys.path.insert(0, "/content/prop_audit_engine")

import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prop_audit.models import ActivoConfig, ReglasEstrategia, PropFirmProfile, Direccion, TipoDrawdown, ModoPosicion
from prop_audit.engine import ejecutar_backtest, trades_a_dataframe
from prop_audit.strategy import CruceMediasMoviles, BaseStrategy
from prop_audit.metrics import calcular_drawdown, calcular_metricas_rendimiento
from prop_audit.montecarlo import simular_montecarlo


# ============================================================
# CELDA 2 — Carga de datos históricos
# ============================================================
def cargar_datos_colab(csv_path: str) -> pd.DataFrame:
    """
    En Colab: sube el CSV con el panel lateral (Files) o usa
    `from google.colab import files; uploaded = files.upload()`.
    El CSV debe traer: timestamp, open, high, low, close, volume.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["fecha_operativa"] = df["timestamp"].dt.date
    faltantes = {"open", "high", "low", "close"} - set(df.columns)
    if faltantes:
        raise ValueError(f"CSV incompleto, faltan columnas: {faltantes}")
    return df


# ============================================================
# CELDA 3 — Barrido de parámetros (grid search vectorizado por combinación)
# ============================================================
def barrido_parametros(
    df: pd.DataFrame,
    config: ActivoConfig,
    prop: PropFirmProfile,
    grilla_sma: dict,       # ej. {"ventana_rapida": [10,20], "ventana_lenta": [50,100]}
    grilla_riesgo: dict,    # ej. {"stop_loss_pct": [0.2,0.3], "take_profit_pct": [0.4,0.6]}
    capital_inicial: float,
) -> pd.DataFrame:
    """
    Itera TODAS las combinaciones de grilla_sma x grilla_riesgo. Cada
    combinación corre un backtest independiente y completo (no hay atajo
    vectorizado válido aquí sin romper el motor event-driven anti
    look-ahead — el barrido se paraleliza a nivel de "cuántas corridas",
    no dentro del bucle barra-a-barra).
    """
    filas_resultado = []
    combinaciones_sma = list(itertools.product(*grilla_sma.values()))
    combinaciones_riesgo = list(itertools.product(*grilla_riesgo.values()))

    for combo_sma in combinaciones_sma:
        params_sma = dict(zip(grilla_sma.keys(), combo_sma))
        if params_sma.get("ventana_rapida", 0) >= params_sma.get("ventana_lenta", 1):
            continue  # combinación inválida, se descarta sin frenar el barrido

        for combo_riesgo in combinaciones_riesgo:
            params_riesgo = dict(zip(grilla_riesgo.keys(), combo_riesgo))

            estrategia = CruceMediasMoviles(**params_sma)
            reglas = ReglasEstrategia(
                direccion=Direccion.AMBAS,
                riesgo_por_operacion_pct=0.5,
                **params_riesgo,
            )

            try:
                trades = ejecutar_backtest(df, estrategia.generate_signals, reglas, prop, config, capital_inicial)
                tdf = trades_a_dataframe(trades)
                if len(tdf) < 10:
                    continue  # sin muestra suficiente, no vale la pena auditar

                dd = calcular_drawdown(tdf, df, prop, capital_inicial,
                                        multiplicador_monetario=config.contract_size * config.tick_value)
                met = calcular_metricas_rendimiento(tdf, capital_inicial)
                mc = simular_montecarlo(tdf, prop, capital_inicial, n_simulaciones=500)

                filas_resultado.append({
                    **params_sma, **params_riesgo,
                    "n_trades": met["total_trades"],
                    "win_rate": met["win_rate"],
                    "profit_factor": met["profit_factor"],
                    "ev_por_dolar": met["ev_por_dolar"],
                    "sharpe": met["sharpe"],
                    "sortino": met["sortino"],
                    "max_dd_pct": dd.max_drawdown_pct,
                    "excedio_dd": dd.excedio_limite,
                    "prob_quiebra_mc": mc.prob_quiebra_antes_de_meta,
                    "dd_mc_p95": mc.drawdown_max_percentil_5,
                    "_trades_df": tdf,  # se conserva para poder graficar la mejor combinación después
                })
            except ValueError as e:
                print(f"[SKIP] combo {params_sma}|{params_riesgo} -> {e}")
                continue

    return pd.DataFrame(filas_resultado)


# ============================================================
# CELDA 4 — Visualización integrada: equity curve + histograma Monte Carlo
# ============================================================
def graficar_resultado(tdf: pd.DataFrame, dd, mc, capital_inicial: float, titulo: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel izquierdo: curva de equity + piso de drawdown
    curva = dd.curva_equity
    axes[0].plot(curva["fecha"], curva["equity"], label="Equity", linewidth=1.5)
    if "piso" in curva.columns:
        axes[0].plot(curva["fecha"], curva["piso"], label="Piso de Drawdown", linestyle="--", color="red")
    axes[0].axhline(capital_inicial, color="gray", linestyle=":", linewidth=1)
    axes[0].set_title(f"Curva de Equity {titulo}")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    # Panel derecho: distribución Monte Carlo del drawdown máximo (reconstruida a partir de percentiles)
    etiquetas = ["P5 (mejor)", "Mediana", "P95 (peor)"]
    valores = [mc.drawdown_max_percentil_95, mc.drawdown_max_mediana, mc.drawdown_max_percentil_5]
    axes[1].bar(etiquetas, valores, color=["#2ca02c", "#1f77b4", "#d62728"])
    axes[1].axhline(dd.max_drawdown_pct, color="black", linestyle="--", label="DD real observado")
    axes[1].set_title(f"Monte Carlo ({mc.n_simulaciones} sims) — Prob. quiebra: {mc.prob_quiebra_antes_de_meta:.1f}%")
    axes[1].set_ylabel("Drawdown máximo (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# CELDA 5 — Ejecución de ejemplo
# ============================================================
if __name__ == "__main__":
    # df = cargar_datos_colab("/content/qqq_1min_3y.csv")
    # config = ActivoConfig("QQQ", "1m", date(2022,1,1), date(2025,1,1), tick_value=1.0, contract_size=1)
    # prop = PropFirmProfile("FTMO", 100000, 5, 10, 8, TipoDrawdown.TRAILING)
    #
    # resultados = barrido_parametros(
    #     df, config, prop,
    #     grilla_sma={"ventana_rapida": [10, 20, 30], "ventana_lenta": [50, 100]},
    #     grilla_riesgo={"stop_loss_pct": [0.15, 0.3], "take_profit_pct": [0.3, 0.6]},
    #     capital_inicial=100000,
    # )
    # mejor = resultados.sort_values("ev_por_dolar", ascending=False).iloc[0]
    # print(resultados.drop(columns="_trades_df").sort_values("ev_por_dolar", ascending=False).head(10))
    #
    # dd_mejor = calcular_drawdown(mejor["_trades_df"], df, prop, 100000,
    #                               multiplicador_monetario=config.contract_size * config.tick_value)
    # mc_mejor = simular_montecarlo(mejor["_trades_df"], prop, 100000, n_simulaciones=1000)
    # graficar_resultado(mejor["_trades_df"], dd_mejor, mc_mejor, 100000, titulo="(mejor combinación)")
    pass
