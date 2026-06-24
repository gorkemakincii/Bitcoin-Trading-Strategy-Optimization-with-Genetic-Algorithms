"""Teknik indikator hesaplama modulu."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram, signal_line


def _bollinger_bands(
    close: pd.Series,
    length: int = 20,
    std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=length, min_periods=length).mean()
    deviation = close.rolling(window=length, min_periods=length).std()
    lower = middle - std * deviation
    upper = middle + std * deviation
    return lower, middle, upper


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window=length, min_periods=length).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=length, min_periods=length).min()
    highest_high = high.rolling(window=length, min_periods=length).max()
    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return stoch_k, stoch_d


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisine temel teknik indikatorleri ekler."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Teknik indikatorler icin eksik sutunlar: {missing}")

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    df["EMA_12"] = close.ewm(span=12, min_periods=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, min_periods=26, adjust=False).mean()

    macd_line, macd_hist, macd_signal = _macd(close)
    df["MACD_12_26_9"] = macd_line
    df["MACDh_12_26_9"] = macd_hist
    df["MACDs_12_26_9"] = macd_signal

    df["RSI_14"] = _rsi(close)
    stoch_k, stoch_d = _stoch(high, low, close)
    df["STOCHk_14_3_3"] = stoch_k
    df["STOCHd_14_3_3"] = stoch_d

    bb_lower, bb_middle, bb_upper = _bollinger_bands(close)
    df["BBL_20_2.0"] = bb_lower
    df["BBM_20_2.0"] = bb_middle
    df["BBU_20_2.0"] = bb_upper
    df["ATR_14"] = _atr(high, low, close)
    df["OBV"] = _obv(close, volume)

    return df
