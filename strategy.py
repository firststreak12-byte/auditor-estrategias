"""
strategy.py
Clase base orientada a objetos para reemplazar la función cruda señal_fn.

Diseño:
  - `BaseStrategy.generate_signals(df)` es el único método que las
    subclases DEBEN implementar. Debe regresar una pd.Series indexada
    igual que `df`, con valores en {"long", "short", None}.
  - Los indicadores se calculan 100% vectorizados (rolling/ewm de pandas),
    nunca con bucles fila a fila — el bucle event-driven de la gestión de
    la posición ya vive en engine.py; strategy.py solo decide CUÁNDO
    entrar, no CÓMO gestionar el trade.
  - engine.ejecutar_backtest() ya aplica .shift(1) sobre lo que regrese
    generate_signals(), así que las subclases pueden escribir su lógica
    de forma "natural" (comparando el indicador en la barra actual) sin
    preocuparse por el shift: el anti-look-ahead es responsabilidad del
    motor, no de la estrategia. Lo único que la estrategia debe garantizar
    es no usar .shift(-n) ni funciones centradas (ej. rolling con center=True).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseStrategy(ABC):
    """Clase base. Subclasea e implementa generate_signals()."""

    nombre: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df trae al menos: timestamp, open, high, low, close, volume.
        Debe regresar pd.Series alineada al índice de df con valores en
        {"long", "short", None}.
        """
        raise NotImplementedError

    # --- Utilidades vectorizadas reutilizables por las subclases ---

    @staticmethod
    def sma(serie: pd.Series, ventana: int) -> pd.Series:
        return serie.rolling(window=ventana, min_periods=ventana).mean()

    @staticmethod
    def ema(serie: pd.Series, ventana: int) -> pd.Series:
        return serie.ewm(span=ventana, adjust=False, min_periods=ventana).mean()

    @staticmethod
    def rsi(serie: pd.Series, ventana: int = 14) -> pd.Series:
        delta = serie.diff()
        ganancia = delta.clip(lower=0)
        pérdida = -delta.clip(upper=0)
        avg_ganancia = ganancia.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()
        avg_pérdida = pérdida.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()
        rs = avg_ganancia / avg_pérdida.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # neutro mientras no hay suficiente historia

    @staticmethod
    def atr(df: pd.DataFrame, ventana: int = 14) -> pd.Series:
        high, low, close_prev = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()


class CruceMediasMoviles(BaseStrategy):
    """
    Estrategia de ejemplo: cruce de SMA rápida sobre SMA lenta = long;
    cruce hacia abajo = short. Sirve como referencia de implementación
    y como estrategia por defecto para probar el motor end-to-end.
    """

    nombre = "Cruce SMA"

    def __init__(self, ventana_rapida: int = 20, ventana_lenta: int = 50, filtro_rsi: Optional[int] = None):
        if ventana_rapida >= ventana_lenta:
            raise ValueError("ventana_rapida debe ser menor que ventana_lenta")
        self.ventana_rapida = ventana_rapida
        self.ventana_lenta = ventana_lenta
        self.filtro_rsi = filtro_rsi  # si se define, evita entradas con RSI extremo

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma_rapida = self.sma(df["close"], self.ventana_rapida)
        sma_lenta = self.sma(df["close"], self.ventana_lenta)

        cruce_arriba = (sma_rapida > sma_lenta) & (sma_rapida.shift(1) <= sma_lenta.shift(1))
        cruce_abajo = (sma_rapida < sma_lenta) & (sma_rapida.shift(1) >= sma_lenta.shift(1))

        señales = pd.Series(None, index=df.index, dtype=object)

        if self.filtro_rsi is not None:
            rsi = self.rsi(df["close"], 14)
            cruce_arriba &= rsi < (100 - self.filtro_rsi)
            cruce_abajo &= rsi > self.filtro_rsi

        señales[cruce_arriba] = "long"
        señales[cruce_abajo] = "short"
        return señales


class RupturaDeCanal(BaseStrategy):
    """Estrategia de ejemplo tipo breakout: entra en la ruptura del máximo/mínimo de N barras."""

    nombre = "Ruptura de Canal"

    def __init__(self, ventana: int = 20):
        self.ventana = ventana

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        max_n = df["high"].rolling(self.ventana, min_periods=self.ventana).max()
        min_n = df["low"].rolling(self.ventana, min_periods=self.ventana).min()

        rompe_arriba = df["close"] > max_n.shift(1)
        rompe_abajo = df["close"] < min_n.shift(1)

        señales = pd.Series(None, index=df.index, dtype=object)
        señales[rompe_arriba] = "long"
        señales[rompe_abajo] = "short"
        return señales
