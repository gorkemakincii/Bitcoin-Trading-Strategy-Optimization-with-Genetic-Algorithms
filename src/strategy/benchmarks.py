"""
Benchmark stratejileri -- GA optimize stratejisinin karsilastirilmasi icin.

Faz 4.5: Dört temel benchmark strateji:
  1. Buy & Hold: Baslangicta al, sonuna kadar tut
  2. Rastgele Strateji: Rastgele alis/satis sinyalleri
  3. Basit SMA Crossover: Sabit parametreli (50/200 gunluk)
  4. Basit RSI Stratejisi: Sabit esikli (14 periyot, 30/70)

Tum fonksiyonlar numpy sinyal dizisi dondurur (+1, -1, 0)
ve run_backtest_fast() ile uyumludur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.backtest import run_backtest_fast
from src.strategy.signals import _compute_rsi, _compute_sma


def buy_and_hold_signals(n: int) -> np.ndarray:
    """
    Buy & Hold sinyali: ilk gun AL, sonuna kadar tut.

    Args:
        n: Veri uzunlugu.

    Returns:
        Sinyal dizisi: ilk eleman 1, gerisi 0.
    """
    signals = np.zeros(n, dtype=np.float64)
    signals[0] = 1.0  # ilk gun alis
    return signals


def random_signals(n: int, seed: int = 42, trade_prob: float = 0.05) -> np.ndarray:
    """
    Rastgele strateji: her gun trade_prob olasilikla alis veya satis.

    Args:
        n: Veri uzunlugu.
        seed: Tekrarlanabilirlik icin rastgele tohum.
        trade_prob: Her gunde sinyal uretme olasiligi.

    Returns:
        Sinyal dizisi (+1, -1, 0).
    """
    rng = np.random.RandomState(seed)
    signals = np.zeros(n, dtype=np.float64)

    in_position = False
    for i in range(n):
        if rng.random() < trade_prob:
            if not in_position:
                signals[i] = 1.0   # AL
                in_position = True
            else:
                signals[i] = -1.0  # SAT
                in_position = False

    return signals


def sma_crossover_signals(
    close: np.ndarray, short_period: int = 50, long_period: int = 200
) -> np.ndarray:
    """
    SMA Crossover sinyali: kisa SMA uzun SMA'yi yukari keserse AL,
    asagi keserse SAT.

    Args:
        close: Kapanis fiyatlari (1-D numpy).
        short_period: Kisa SMA periyodu (default: 50).
        long_period: Uzun SMA periyodu (default: 200).

    Returns:
        Sinyal dizisi (+1, -1, 0).
    """
    close_s = pd.Series(close)
    sma_short = _compute_sma(close_s, short_period).values
    sma_long = _compute_sma(close_s, long_period).values

    n = len(close)
    signals = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if np.isnan(sma_short[i]) or np.isnan(sma_long[i]):
            continue
        if np.isnan(sma_short[i - 1]) or np.isnan(sma_long[i - 1]):
            continue

        # Golden Cross: kisa SMA yukari kesiyor
        if sma_short[i] > sma_long[i] and sma_short[i - 1] <= sma_long[i - 1]:
            signals[i] = 1.0
        # Death Cross: kisa SMA asagi kesiyor
        elif sma_short[i] < sma_long[i] and sma_short[i - 1] >= sma_long[i - 1]:
            signals[i] = -1.0

    return signals


def rsi_signals(
    close: np.ndarray, period: int = 14, oversold: float = 30.0, overbought: float = 70.0
) -> np.ndarray:
    """
    RSI sinyali: RSI < oversold ise AL, RSI > overbought ise SAT.

    Args:
        close: Kapanis fiyatlari (1-D numpy).
        period: RSI periyodu (default: 14).
        oversold: Asiri satim esigi (default: 30).
        overbought: Asiri alim esigi (default: 70).

    Returns:
        Sinyal dizisi (+1, -1, 0).
    """
    close_s = pd.Series(close)
    rsi = _compute_rsi(close_s, period).values

    signals = np.where(
        rsi < oversold, 1.0,
        np.where(rsi > overbought, -1.0, 0.0)
    )
    signals = np.nan_to_num(signals, nan=0.0)
    return signals


def run_all_benchmarks(
    close: np.ndarray,
    initial_capital: float = 10_000.0,
    position_size: float = 0.95,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> dict[str, dict]:
    """
    Tum benchmark stratejilerini calistirir ve sonuclari dondurur.

    Args:
        close: Kapanis fiyatlari (1-D numpy).
        initial_capital: Baslangic sermayesi.
        position_size: Pozisyon buyuklugu orani.
        commission_rate: Komisyon orani.
        slippage_rate: Kayma orani.

    Returns:
        {strateji_adi: backtest_sonuclari} seklinde dictionary.
    """
    backtest_kwargs = dict(
        initial_capital=initial_capital,
        position_size=position_size,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    n = len(close)
    benchmarks = {
        "Buy & Hold": buy_and_hold_signals(n),
        "Rastgele": random_signals(n),
        "SMA 50/200": sma_crossover_signals(close),
        "RSI 30/70": rsi_signals(close),
    }

    results = {}
    for name, signals in benchmarks.items():
        metrics = run_backtest_fast(close, signals, **backtest_kwargs)
        results[name] = {
            k: v for k, v in metrics.items()
            if k not in ("equity_curve", "buy_and_hold")
        }

    return results
