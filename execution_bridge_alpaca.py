"""
execution_bridge_alpaca.py
BOCETO de puente de ejecución en vivo. NO es un sistema production-ready:
falta manejo de reconexión de websocket, persistencia de estado tras un
crash, y reconciliación de posiciones al arrancar. Está pensado como punto
de partida para validar la tubería señal -> orden antes de construir la
infraestructura completa.

Flujo:
  1. Polling de velas de 1 minuto vía la API REST de Alpaca (alpaca-py).
  2. Se acumulan en un buffer -> mismo DataFrame OHLCV que usa el backtest.
  3. Se llama a BaseStrategy.generate_signals(buffer) — la MISMA estrategia
     que ya fue auditada en el backtest, sin reescribir lógica.
  4. Si la última señal generada es "long"/"short" y no hay posición
     abierta en ese símbolo, se arma el payload de orden y se envía.

ADVERTENCIA DE RIESGO — LEE ESTO ANTES DE CONECTAR UNA CUENTA REAL:
  - Corre esto SIEMPRE primero contra el entorno paper de Alpaca
    (https://paper-api.alpaca.markets) durante semanas, no minutos.
  - Las claves de API nunca deben vivir hardcodeadas en el script: usa
    variables de entorno (os.environ) o un gestor de secretos.
  - Este puente NO implementa el kill switch de drawdown diario/total del
    Prop Firm Profile — eso debe vivir en una capa de riesgo separada que
    corra ANTES de que la orden llegue a Alpaca, no después.
  - Ningún backtest garantiza resultados futuros. Ver descargo legal del
    informe de auditoría.
"""

from __future__ import annotations
import os
import time
import logging
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from prop_audit.strategy import BaseStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("execution_bridge")


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"   # SIEMPRE paper por defecto
    data_url: str = "https://data.alpaca.markets"

    @classmethod
    def desde_entorno(cls) -> "AlpacaConfig":
        """Lee credenciales de variables de entorno. Fail-fast si faltan:
        preferible reventar aquí que enviar una orden sin autenticación válida."""
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise EnvironmentError(
                "Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY en variables de entorno. "
                "Nunca hardcodees las claves en el script."
            )
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url)


class BufferVelas:
    """Mantiene una ventana móvil de velas de 1m en memoria, con el mismo
    esquema de columnas que usa el motor de backtest, para que la estrategia
    audite exactamente la misma estructura de datos en vivo y en histórico."""

    def __init__(self, max_barras: int = 500):
        self.max_barras = max_barras
        self.df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "fecha_operativa"])

    def agregar_vela(self, vela: dict) -> None:
        fila = pd.DataFrame([{
            "timestamp": pd.Timestamp(vela["t"]),
            "open": float(vela["o"]),
            "high": float(vela["h"]),
            "low": float(vela["l"]),
            "close": float(vela["c"]),
            "volume": float(vela.get("v", 0)),
        }])
        fila["fecha_operativa"] = fila["timestamp"].dt.date
        self.df = pd.concat([self.df, fila], ignore_index=True).drop_duplicates(
            subset="timestamp", keep="last"
        ).tail(self.max_barras).reset_index(drop=True)


def obtener_ultima_vela_1m(simbolo: str, config: AlpacaConfig) -> Optional[dict]:
    """
    Boceto de llamada REST a Alpaca Market Data v2 (barras de 1 minuto).
    En producción, usar el SDK oficial `alpaca-py` en vez de requests crudo:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    Aquí se deja la forma del request para no acoplar el boceto a una
    versión específica del SDK que pueda no estar instalada.
    """
    import requests  # import local: este boceto no debe forzar la dependencia si solo se lee el archivo

    headers = {
        "APCA-API-KEY-ID": config.api_key,
        "APCA-API-SECRET-KEY": config.secret_key,
    }
    ahora = datetime.now(timezone.utc)
    desde = ahora - timedelta(minutes=5)
    params = {
        "symbols": simbolo,
        "timeframe": "1Min",
        "start": desde.isoformat(),
        "end": ahora.isoformat(),
        "limit": 5,
        "adjustment": "raw",
    }
    resp = requests.get(f"{config.data_url}/v2/stocks/bars", headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        logger.error(f"Error obteniendo velas de Alpaca: {resp.status_code} {resp.text}")
        return None

    barras = resp.json().get("bars", {}).get(simbolo, [])
    return barras[-1] if barras else None


def construir_payload_orden(
    simbolo: str,
    señal: str,           # "long" o "short"
    cantidad: float,
    tipo_orden: str = "market",
    time_in_force: str = "day",
) -> dict:
    """
    Estructura el payload EXACTO que espera el endpoint POST /v2/orders de
    Alpaca. "long" -> side "buy"; "short" -> side "sell" (venta en corto,
    requiere que la cuenta tenga margen habilitado).
    """
    if señal not in ("long", "short"):
        raise ValueError(f"Señal inválida para construir orden: {señal!r}")
    if cantidad <= 0:
        raise ValueError("cantidad debe ser > 0")

    return {
        "symbol": simbolo,
        "qty": str(cantidad),
        "side": "buy" if señal == "long" else "sell",
        "type": tipo_orden,
        "time_in_force": time_in_force,
    }


def enviar_orden(payload: dict, config: AlpacaConfig, dry_run: bool = True) -> Optional[dict]:
    """dry_run=True (default) SOLO loguea el payload sin enviar la orden.
    Cambiar a False es una decisión explícita del operador, nunca el default."""
    logger.info(f"Payload de orden construido: {payload}")
    if dry_run:
        logger.info("[DRY RUN] Orden NO enviada. Cambia dry_run=False para enviar de verdad.")
        return None

    import requests
    headers = {
        "APCA-API-KEY-ID": config.api_key,
        "APCA-API-SECRET-KEY": config.secret_key,
        "Content-Type": "application/json",
    }
    resp = requests.post(f"{config.base_url}/v2/orders", headers=headers, json=payload, timeout=10)
    if resp.status_code not in (200, 201):
        logger.error(f"Alpaca rechazó la orden: {resp.status_code} {resp.text}")
        return None
    logger.info(f"Orden aceptada: {resp.json().get('id')}")
    return resp.json()


def loop_ejecucion(
    simbolo: str,
    estrategia: BaseStrategy,
    config: AlpacaConfig,
    cantidad_por_orden: float,
    dry_run: bool = True,
    intervalo_segundos: int = 60,
    max_iteraciones: Optional[int] = None,
) -> None:
    """
    Loop principal. max_iteraciones existe solo para poder probarlo en un
    entorno de test sin quedar corriendo indefinidamente; en producción se
    deja en None y se corre bajo un supervisor de proceso (systemd, Docker
    con restart policy, etc.), NUNCA como script suelto en una terminal.
    """
    buffer = BufferVelas(max_barras=500)
    posicion_abierta: Optional[str] = None
    iteraciones = 0

    while max_iteraciones is None or iteraciones < max_iteraciones:
        vela = obtener_ultima_vela_1m(simbolo, config)
        if vela is not None:
            buffer.agregar_vela(vela)

        if len(buffer.df) >= 50:  # mínimo de barras para que los indicadores no sean NaN
            señales = estrategia.generate_signals(buffer.df)
            señal_actual = señales.iloc[-1]

            if señal_actual in ("long", "short") and señal_actual != posicion_abierta:
                payload = construir_payload_orden(simbolo, señal_actual, cantidad_por_orden)
                enviar_orden(payload, config, dry_run=dry_run)
                posicion_abierta = señal_actual

        iteraciones += 1
        if max_iteraciones is None or iteraciones < max_iteraciones:
            time.sleep(intervalo_segundos)


if __name__ == "__main__":
    # Ejemplo de uso (paper trading, dry_run=True por defecto):
    # from prop_audit.strategy import CruceMediasMoviles
    # config = AlpacaConfig.desde_entorno()
    # estrategia = CruceMediasMoviles(ventana_rapida=10, ventana_lenta=30)
    # loop_ejecucion("QQQ", estrategia, config, cantidad_por_orden=1, dry_run=True, max_iteraciones=3)
    pass
