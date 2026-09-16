"""
models.py
Dataclasses de entrada para el motor de auditoría.
Todas las validaciones de rango ocurren en __post_init__ para que el
sistema falle rápido si el usuario manda parámetros inconsistentes,
en lugar de producir un backtest silenciosamente incorrecto.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class Direccion(str, Enum):
    LONG = "long"
    SHORT = "short"
    AMBAS = "ambas"


class TipoDrawdown(str, Enum):
    ESTATICO = "estatico"      # el piso de equity nunca sube
    FLOTANTE = "flotante"      # el piso se mide contra equity intradía (incluye flotante)
    TRAILING = "trailing"      # el piso sube con cada nuevo high-water mark


class ModoPosicion(str, Enum):
    LOTE_FIJO = "lote_fijo"
    RIESGO_PORCENTAJE = "riesgo_pct"


@dataclass
class ActivoConfig:
    ticker: str
    temporalidad: str  # "1m","5m","15m","1h","1D"
    fecha_inicio: date
    fecha_fin: date
    tick_value: float = 1.0     # valor monetario por punto/tick, depende del instrumento
    contract_size: float = 1.0  # unidades por lote (FX=100000, acciones=1, futuros=valor del contrato)

    def __post_init__(self):
        temporalidades_validas = {"1m", "5m", "15m", "1h", "1D"}
        if self.temporalidad not in temporalidades_validas:
            raise ValueError(f"Temporalidad inválida: {self.temporalidad}")
        if self.fecha_fin <= self.fecha_inicio:
            raise ValueError("fecha_fin debe ser posterior a fecha_inicio")
        dias_totales = (self.fecha_fin - self.fecha_inicio).days
        if dias_totales < 365 * 3 * 0.9:
            raise ValueError(
                "Se requiere un mínimo aproximado de 3 años de histórico OHLCV "
                "para que las métricas de drawdown y Monte Carlo sean estadísticamente válidas."
            )


@dataclass
class ReglasEstrategia:
    direccion: Direccion
    take_profit_pct: Optional[float]   # None si no aplica TP fijo
    stop_loss_pct: float               # obligatorio: sin SL no se puede auditar riesgo
    break_even_activacion_pct: Optional[float] = None  # mueve SL a entrada tras X% a favor
    cierre_por_tiempo_barras: Optional[int] = None     # cierra tras N barras si no tocó TP/SL
    modo_posicion: ModoPosicion = ModoPosicion.RIESGO_PORCENTAJE
    riesgo_por_operacion_pct: float = 0.5   # % del capital si modo_posicion = RIESGO_PORCENTAJE
    lote_fijo: float = 0.01                 # usado si modo_posicion = LOTE_FIJO

    def __post_init__(self):
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct debe ser > 0 (el sistema no audita estrategias sin SL)")
        if self.modo_posicion == ModoPosicion.RIESGO_PORCENTAJE and not (0 < self.riesgo_por_operacion_pct <= 5):
            raise ValueError("riesgo_por_operacion_pct fuera de rango razonable (0-5%)")


@dataclass
class PropFirmProfile:
    nombre_firma: str
    capital_inicial: float                  # 10000, 50000, 100000...
    limite_perdida_diaria_pct: float         # ej. 4 o 5
    limite_perdida_total_pct: float          # ej. 8 o 10
    meta_beneficio_pct: float                # ej. 8 o 10
    tipo_drawdown: TipoDrawdown
    comision_por_lote: float = 4.0           # USD, entre 2 y 7 típico
    slippage_puntos: float = 0.5             # slippage estimado por operación en puntos
    max_dias_operativos: Optional[int] = None  # límite de días de la fase, si aplica

    def __post_init__(self):
        if self.capital_inicial <= 0:
            raise ValueError("capital_inicial debe ser positivo")
        if not (0 < self.limite_perdida_diaria_pct < self.limite_perdida_total_pct):
            raise ValueError("limite_perdida_diaria_pct debe ser menor que limite_perdida_total_pct")
        if not (2.0 <= self.comision_por_lote <= 7.0):
            raise ValueError("comision_por_lote fuera del rango realista (2-7 USD por lote)")
