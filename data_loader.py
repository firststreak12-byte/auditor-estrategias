"""
data_loader.py
Gestor de datos: tres fuentes distintas, un único contrato de salida.

Toda función pública de este módulo devuelve un DataFrame OHLCV
estandarizado con, como mínimo, las columnas:
    timestamp, open, high, low, close, fecha_operativa
(volume se incluye cuando la fuente lo provee; puede venir en 0 si no).

`fecha_operativa` se calcula SIEMPRE aquí, de forma interna — ya no es un
requisito que el usuario deba traer en su CSV (ese era el bug reportado:
antes se exigía como columna obligatoria del archivo, cuando es un dato
derivado de `timestamp`, no una entrada independiente).

Cada fuente es responsable de sus propios errores de red/parseo, pero
todas terminan pasando por `_finalizar_dataframe()`, que es el único lugar
donde vive la lógica de:
  - cast numérico + parseo de fechas (con manejo de errores explícito)
  - orden cronológico y dedupe
  - cómputo de fecha_operativa
  - validación de integridad básica (precios > 0)
Así, un bug de "datos corruptos" se corrige en un solo sitio para las
tres fuentes en vez de tres veces.
"""

from __future__ import annotations
import pandas as pd
from datetime import date, datetime
from typing import List

COLUMNAS_MINIMAS = ["timestamp", "open", "high", "low", "close"]

# Límites conocidos del tier gratuito de yfinance (Yahoo Finance), en días,
# para las temporalidades intradía que soporta este proyecto.
LIMITES_YFINANCE_DIAS = {"1m": 7, "5m": 60, "15m": 60}

TIMEFRAME_ALPACA = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1D": "1Day"}
INTERVALO_YFINANCE = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1D": "1d"}


# ============================================================
# Núcleo compartido de estandarización / validación
# ============================================================

def _finalizar_dataframe(df: pd.DataFrame, columnas_precio: List[str] = ("open", "high", "low", "close")) -> pd.DataFrame:
    """
    Aplica el mismo contrato de calidad a cualquier DataFrame OHLCV, venga
    de donde venga. Lanza ValueError limpio (nunca deja que un TypeError o
    ParserError crudo de pandas llegue al caller) si los datos están
    corruptos o incompletos.
    """
    faltantes = set(COLUMNAS_MINIMAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias en los datos: {sorted(faltantes)}")

    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=False)
        for col in columnas_precio:
            df[col] = pd.to_numeric(df[col], errors="raise")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0.0
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Datos corruptos: no fue posible convertir timestamp/columnas de precio "
            f"a los tipos esperados ({e})"
        ) from e

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    if df.empty:
        raise ValueError("El DataFrame resultante está vacío tras la limpieza.")

    for col in columnas_precio:
        if (df[col] <= 0).any():
            raise ValueError(f"La columna '{col}' contiene valores <= 0: datos corruptos.")

    df["fecha_operativa"] = df["timestamp"].dt.date
    return df[["timestamp", "open", "high", "low", "close", "volume", "fecha_operativa"]]


# ============================================================
# Fuente A — CSV subido por el usuario
# ============================================================

def cargar_desde_csv(archivo) -> pd.DataFrame:
    """
    `archivo` es cualquier objeto compatible con pd.read_csv (ruta, o el
    UploadedFile que entrega st.file_uploader). ÚNICO requisito de columnas:
    timestamp, open, high, low, close. `fecha_operativa`, si el usuario la
    trae, se ignora y se recalcula — nunca se exige (bug corregido).
    """
    try:
        df = pd.read_csv(archivo)
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo como CSV: {e}") from e

    df.columns = [c.strip().lower() for c in df.columns]
    return _finalizar_dataframe(df)


# ============================================================
# Fuente B — yfinance (datos públicos, gratuitos, con límites de rango)
# ============================================================

def cargar_desde_yfinance(
    ticker: str,
    temporalidad: str,
    fecha_inicio: date,
    fecha_fin: date,
) -> pd.DataFrame:
    """
    Descarga OHLCV público vía yfinance. Valida ANTES de golpear la red que
    el rango solicitado no exceda los límites conocidos del tier gratuito
    para temporalidades intradía (1m: 7 días; 5m/15m: 60 días) — evita una
    llamada de red que Yahoo respondería truncada o vacía sin avisar.
    """
    if temporalidad not in INTERVALO_YFINANCE:
        raise ValueError(f"Temporalidad no soportada por yfinance en este proyecto: {temporalidad}")

    limite_dias = LIMITES_YFINANCE_DIAS.get(temporalidad)
    if limite_dias is not None:
        dias_solicitados = (fecha_fin - fecha_inicio).days
        if dias_solicitados > limite_dias:
            raise ValueError(
                f"yfinance (tier gratuito) no permite más de {limite_dias} días de histórico "
                f"para la temporalidad '{temporalidad}'. Solicitaste {dias_solicitados} días. "
                f"Reduce el rango, usa una temporalidad mayor, conecta Alpaca o sube un CSV."
            )

    try:
        import yfinance as yf
    except ImportError as e:
        raise ValueError(
            "El paquete 'yfinance' no está instalado. Instálalo con: pip install yfinance"
        ) from e

    try:
        datos = yf.download(
            tickers=ticker,
            start=fecha_inicio,
            end=fecha_fin,
            interval=INTERVALO_YFINANCE[temporalidad],
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        raise ValueError(f"Error de red/consulta a yfinance para '{ticker}': {e}") from e

    if datos is None or datos.empty:
        raise ValueError(
            f"yfinance no devolvió datos para '{ticker}' en el rango {fecha_inicio} a {fecha_fin}. "
            f"Verifica el ticker o que el rango tenga sesiones de mercado."
        )

    # yfinance puede devolver columnas MultiIndex cuando se piden varios tickers;
    # aquí se pide uno solo, pero se aplana por robustez.
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = [c[0] for c in datos.columns]

    datos = datos.reset_index()
    datos.columns = [str(c).strip().lower() for c in datos.columns]
    columna_fecha = "datetime" if "datetime" in datos.columns else "date"
    datos = datos.rename(columns={columna_fecha: "timestamp"})

    return _finalizar_dataframe(datos)


# ============================================================
# Fuente C — Alpaca (broker, requiere API key propia del usuario)
# ============================================================

def cargar_desde_alpaca(
    api_key: str,
    secret_key: str,
    ticker: str,
    temporalidad: str,
    fecha_inicio: date,
    fecha_fin: date,
    feed: str = "iex",
    max_paginas: int = 200,
) -> pd.DataFrame:
    """
    Descarga histórico OHLCV directo del broker vía la API de Market Data
    de Alpaca (paginada, porque Alpaca limita bars por respuesta). Con
    `feed="iex"` (por defecto) porque es lo que cubre una cuenta gratuita;
    "sip" requiere suscripción de datos de pago.

    Maneja explícitamente:
      - credenciales inválidas (401/403) -> ValueError claro, sin traceback crudo
      - timeout / error de conexión -> ValueError claro
      - símbolo sin datos en el rango -> ValueError claro
    """
    if temporalidad not in TIMEFRAME_ALPACA:
        raise ValueError(f"Temporalidad no soportada por Alpaca en este proyecto: {temporalidad}")
    if not api_key or not secret_key:
        raise ValueError("Debes proporcionar api_key y secret_key de Alpaca.")

    import requests

    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    params = {
        "timeframe": TIMEFRAME_ALPACA[temporalidad],
        "start": datetime.combine(fecha_inicio, datetime.min.time()).isoformat() + "Z",
        "end": datetime.combine(fecha_fin, datetime.min.time()).isoformat() + "Z",
        "limit": 10000,
        "adjustment": "raw",
        "feed": feed,
    }

    barras: List[dict] = []
    pagina = 0
    try:
        while pagina < max_paginas:
            resp = requests.get(url, headers=headers, params=params, timeout=15)

            if resp.status_code in (401, 403):
                raise ValueError(
                    "Credenciales de Alpaca inválidas o sin permiso para este símbolo/feed "
                    f"(HTTP {resp.status_code})."
                )
            if resp.status_code != 200:
                raise ValueError(f"Alpaca respondió con error HTTP {resp.status_code}: {resp.text[:200]}")

            cuerpo = resp.json()
            barras.extend(cuerpo.get("bars", []) or [])

            token_siguiente = cuerpo.get("next_page_token")
            if not token_siguiente:
                break
            params["page_token"] = token_siguiente
            pagina += 1

    except requests.exceptions.Timeout as e:
        raise ValueError(f"Timeout consultando la API de Alpaca para '{ticker}'.") from e
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error de red consultando la API de Alpaca: {e}") from e

    if not barras:
        raise ValueError(
            f"Alpaca no devolvió velas para '{ticker}' en el rango {fecha_inicio} a {fecha_fin} "
            f"con temporalidad '{temporalidad}'. Verifica el símbolo, el rango y el feed."
        )

    df = pd.DataFrame(barras).rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    })
    return _finalizar_dataframe(df)
