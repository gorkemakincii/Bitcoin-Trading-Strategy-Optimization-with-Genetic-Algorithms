"""
Sinyal uretim modulu — GA tabanli BTC trading botu icin.

GA'nin her nesilde binlerce kez cagirmasi icin optimize edilmistir:
  - Tamamen vektorel islemler (for dongusu / df.apply() YOK)
  - pandas-ta bagimliliginden bagimsiz, saf pandas/numpy hesaplamalari
  - Look-ahead bias (veri sizintisi) YOK

Akis:
  1. GA kromozomundan gelen StrategyParams ile indikatorler yeniden hesaplanir
  2. Her indikator icin alt sinyal (1=AL, -1=SAT, 0=NOTR) uretilir
  3. Alt sinyaller GA agirliklariyia birlestirilip Composite_Signal olusturulur
"""
import pandas as pd
import numpy as np

# ── StrategyParams: chromosome.py'de zaten tanimli — duplikasyondan kaciniyoruz ──
# Mimari olarak StrategyParams strateji katmanina ait, ancak mevcut kodda
# ga/chromosome.py icinde tanimli. Iki yerde ayni dataclass'i tanimlamak yerine
# oradan import edip re-export ediyoruz.
from src.ga.chromosome import StrategyParams


# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI: HIZLI VEKTOREL INDIKATOR HESAPLAMALARI
# pandas-ta kullanmak yerine saf pandas/numpy ile hesapliyoruz.
# Bu sayede her fitness cagrisinda pandas-ta overhead'indan kacinilir.
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    RSI (Relative Strength Index) — Wilder's smoothing (EWM) ile.

    Formul:
        delta  = close.diff()
        gain   = max(delta, 0)  →  ewm(alpha=1/period)
        loss   = max(-delta, 0) →  ewm(alpha=1/period)
        RS     = avg_gain / avg_loss
        RSI    = 100 - 100 / (1 + RS)
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_bbands(
    close: pd.Series, period: int, num_std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns:
        (lower_band, middle_band, upper_band)
    """
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return lower, middle, upper


def _compute_sma(close: pd.Series, period: int) -> pd.Series:
    """Basit Hareketli Ortalama (SMA)."""
    return close.rolling(window=period, min_periods=period).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# ANA SINYAL URETIM FONKSIYONU
# ═══════════════════════════════════════════════════════════════════════════════


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """
    GA parametreleriyle indikatorleri yeniden hesaplar ve bilisik sinyal uretir.

    Her indikator icin alt sinyal (sub-signal) uretilir:
        +1 = AL (Buy)
        -1 = SAT (Sell)
         0 = NOTR (Neutral)

    Alt sinyaller GA agirliklariyia (weight_rsi, weight_macd, ...) carpilip
    toplanir. Toplam > 0 ise nihai sinyal AL, < 0 ise SAT, == 0 ise NOTR.

    Args:
        df: En az 'Close' sutununu iceren DataFrame.
            (OHLCV tamami mevcut olmasa bile Close yeterlidir.)
        params: GA'dan gelen 16 parametrelik StrategyParams dataclass'i.

    Returns:
        Asagidaki sutunlar eklenmis DataFrame kopyasi:
            - RSI           : Hesaplanan RSI degerleri
            - MACD_Line     : MACD cizgisi
            - MACD_Signal   : MACD sinyal cizgisi
            - MACD_Hist     : MACD histogram
            - BB_Lower      : Bollinger alt bant
            - BB_Middle     : Bollinger orta bant
            - BB_Upper      : Bollinger ust bant
            - SMA_Short     : Kisa periyot SMA
            - SMA_Long      : Uzun periyot SMA
            - Signal_RSI    : RSI alt sinyali  (+1 / -1 / 0)
            - Signal_MACD   : MACD alt sinyali (+1 / -1 / 0)
            - Signal_BB     : BB alt sinyali   (+1 / -1 / 0)
            - Signal_SMA    : SMA alt sinyali  (+1 / -1 / 0)
            - Composite_Signal : Agirlikli bilisik sinyal (+1 / -1 / 0)

    Raises:
        ValueError: 'Close' sutunu eksikse.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame'de 'Close' sutunu bulunamadi.")

    df = df.copy()
    close = df["Close"]

    # ──────────────────────────────────────────────────────────────
    # 1. INDIKATOR HESAPLAMALARI (GA parametreleriyle)
    # ──────────────────────────────────────────────────────────────

    # RSI
    df["RSI"] = _compute_rsi(close, period=params.rsi_period)

    # MACD
    macd_line, signal_line, histogram = _compute_macd(
        close, fast=params.macd_fast, slow=params.macd_slow, signal=params.macd_signal
    )
    df["MACD_Line"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = histogram

    # Bollinger Bands
    bb_lower, bb_middle, bb_upper = _compute_bbands(
        close, period=params.bb_period, num_std=params.bb_std
    )
    df["BB_Lower"] = bb_lower
    df["BB_Middle"] = bb_middle
    df["BB_Upper"] = bb_upper

    # SMA
    df["SMA_Short"] = _compute_sma(close, period=params.sma_short)
    df["SMA_Long"] = _compute_sma(close, period=params.sma_long)

    # ──────────────────────────────────────────────────────────────
    # 2. ALT SINYAL URETIMI (tamamen vektorel — np.where)
    # ──────────────────────────────────────────────────────────────

    # RSI Sinyali
    # Asiri satim (oversold) → AL, Asiri alim (overbought) → SAT
    rsi_values = df["RSI"].values
    df["Signal_RSI"] = np.where(
        rsi_values < params.rsi_oversold, 1,
        np.where(rsi_values > params.rsi_overbought, -1, 0)
    )

    # MACD Sinyali
    # Histogram > 0 (MACD cizgisi sinyal cizgisinin ustunde) → AL
    # Histogram < 0 (MACD cizgisi sinyal cizgisinin altinda) → SAT
    hist_values = df["MACD_Hist"].values
    df["Signal_MACD"] = np.where(
        hist_values > 0, 1,
        np.where(hist_values < 0, -1, 0)
    )

    # Bollinger Bands Sinyali
    # Fiyat alt bandın altinda → AL (asiri satim bolgesi)
    # Fiyat ust bandın ustunde → SAT (asiri alim bolgesi)
    close_values = close.values
    bb_lower_values = df["BB_Lower"].values
    bb_upper_values = df["BB_Upper"].values
    df["Signal_BB"] = np.where(
        close_values < bb_lower_values, 1,
        np.where(close_values > bb_upper_values, -1, 0)
    )

    # SMA Kesisim (Crossover) Sinyali
    # Kisa SMA > Uzun SMA → AL (Golden Cross yonu)
    # Kisa SMA < Uzun SMA → SAT (Death Cross yonu)
    sma_short_values = df["SMA_Short"].values
    sma_long_values = df["SMA_Long"].values
    df["Signal_SMA"] = np.where(
        sma_short_values > sma_long_values, 1,
        np.where(sma_short_values < sma_long_values, -1, 0)
    )

    # ──────────────────────────────────────────────────────────────
    # 3. BILESIK SINYAL (Composite Signal)
    # ──────────────────────────────────────────────────────────────

    # Agirlikli toplam — vektorel carpim ve toplama
    weighted_score = (
        params.weight_rsi  * df["Signal_RSI"].values
        + params.weight_macd * df["Signal_MACD"].values
        + params.weight_bb   * df["Signal_BB"].values
        + params.weight_sma  * df["Signal_SMA"].values
    )

    # Toplam > 0 → AL (1), < 0 → SAT (-1), == 0 → NOTR (0)
    df["Composite_Signal"] = np.where(
        weighted_score > 0, 1,
        np.where(weighted_score < 0, -1, 0)
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# HIZLI SINYAL URETIMI (sadece Composite_Signal dondurur — fitness icin)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_signals_fast(close: np.ndarray, params: StrategyParams) -> np.ndarray:
    """
    Yuksek performansli sinyal uretimi — sadece numpy, DataFrame overhead'i yok.

    GA fitness fonksiyonunun her cagrisinda kullanilmak uzere tasarlandi.
    generate_signals() ile ayni mantigi kullanir, ancak sadece nihai
    Composite_Signal dizisini (np.ndarray) dondurur.

    Args:
        close: Kapanis fiyatlarinin numpy dizisi (1-D).
        params: GA'dan gelen StrategyParams.

    Returns:
        Composite sinyal dizisi: +1 (AL), -1 (SAT), 0 (NOTR).
        Uzunluk = len(close). Ilk satirlar NaN yerine 0 ile doldurulur.
    """
    n = len(close)
    close_s = pd.Series(close)

    # ── RSI ───────────────────────────────────────────────────────
    rsi = _compute_rsi(close_s, params.rsi_period).values
    sig_rsi = np.where(
        rsi < params.rsi_oversold, 1.0,
        np.where(rsi > params.rsi_overbought, -1.0, 0.0)
    )
    # NaN bolgeleri (baslangic) notr olarak isle
    sig_rsi = np.nan_to_num(sig_rsi, nan=0.0)

    # ── MACD ──────────────────────────────────────────────────────
    _, _, histogram = _compute_macd(
        close_s, params.macd_fast, params.macd_slow, params.macd_signal
    )
    hist_vals = histogram.values
    sig_macd = np.where(
        hist_vals > 0, 1.0,
        np.where(hist_vals < 0, -1.0, 0.0)
    )
    sig_macd = np.nan_to_num(sig_macd, nan=0.0)

    # ── Bollinger Bands ───────────────────────────────────────────
    bb_lower, _, bb_upper = _compute_bbands(
        close_s, params.bb_period, params.bb_std
    )
    bb_l = bb_lower.values
    bb_u = bb_upper.values
    sig_bb = np.where(
        close < bb_l, 1.0,
        np.where(close > bb_u, -1.0, 0.0)
    )
    sig_bb = np.nan_to_num(sig_bb, nan=0.0)

    # ── SMA Crossover ─────────────────────────────────────────────
    sma_s = _compute_sma(close_s, params.sma_short).values
    sma_l = _compute_sma(close_s, params.sma_long).values
    sig_sma = np.where(
        sma_s > sma_l, 1.0,
        np.where(sma_s < sma_l, -1.0, 0.0)
    )
    sig_sma = np.nan_to_num(sig_sma, nan=0.0)

    # ── Composite ─────────────────────────────────────────────────
    weighted = (
        params.weight_rsi  * sig_rsi
        + params.weight_macd * sig_macd
        + params.weight_bb   * sig_bb
        + params.weight_sma  * sig_sma
    )

    composite = np.where(weighted > 0, 1, np.where(weighted < 0, -1, 0))
    return composite
