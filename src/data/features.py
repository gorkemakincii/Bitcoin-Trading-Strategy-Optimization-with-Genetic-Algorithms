"""
Feature engineering modulu - GA tabanli BTC trading botu icin.

Tamamen vektorel (for dongusu yok) pandas/numpy islemleri kullanir.
Look-ahead bias (veri sizintisi) icermez.
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV + teknik indikator iceren DataFrame'den turetilmis ozellikler uretir.

    Turetilen ozellikler:
        - Return_1d, Return_3d, Return_7d  : Fiyat degisim oranlari
        - Log_Return                        : Gunluk logaritmik getiri
        - Volatility_20d                    : 20 gunluk rolling standart sapma
        - ROC_14                            : 14 gunluk Rate of Change
        - RSI_MACD_Diff                     : RSI ve MACD uyum/fark ozelligi
        - RSI_MACD_Signal_Diff              : RSI ile MACD sinyal farki
        - MACD_Hist_Momentum                : MACD histogram degisim hizi
        - BB_Position                       : Bollinger Bands icerisindeki pozisyon (0-1)
        - BB_Width                          : Bollinger Bands genisligi (volatilite)
        - Price_SMA20_Ratio                 : Fiyatin SMA20'ye orani
        - Price_SMA50_Ratio                 : Fiyatin SMA50'ye orani
        - SMA_Cross_Diff                    : SMA20 - SMA50 farki (trend yonu)
        - Volume_SMA20                      : 20 gunluk hacim hareketli ortalamasi
        - Volume_Ratio                      : Hacmin ortalamaya orani
        - ATR_Ratio                         : ATR'nin fiyata orani (normalize volatilite)
        - OBV_Change                        : OBV degisim orani

    Args:
        df: OHLCV ve teknik indikatorleri iceren DataFrame.
            Beklenen sutunlar: Close, High, Low, Open, Volume,
            RSI_14, MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9,
            SMA_20, SMA_50, BBL_20_2.0, BBM_20_2.0, BBU_20_2.0,
            ATR_14, OBV

    Returns:
        Turetilmis ozellikler eklenmis ve NaN satirlari temizlenmis DataFrame.

    Raises:
        ValueError: Gerekli sutunlar eksikse.
    """
    # ── Girdi dogrulama ──────────────────────────────────────────────
    _required_base = ["Close", "Volume"]
    _missing = [c for c in _required_base if c not in df.columns]
    if _missing:
        raise ValueError(
            f"Gerekli sutunlar eksik: {_missing}. "
            f"Lutfen OHLCV verisi iceren bir DataFrame girin."
        )

    df = df.copy()
    close = df["Close"]

    # ═══════════════════════════════════════════════════════════════
    # 1. GETIRILER (Returns)
    # ═══════════════════════════════════════════════════════════════

    # Basit fiyat degisim oranlari — sadece gecmis veriye bakar (shift(+n))
    df["Return_1d"] = close.pct_change(periods=1)
    df["Return_3d"] = close.pct_change(periods=3)
    df["Return_7d"] = close.pct_change(periods=7)

    # ═══════════════════════════════════════════════════════════════
    # 2. LOGARITMIK GETIRI
    # ═══════════════════════════════════════════════════════════════

    # log(P_t / P_{t-1}) — vektorel, gelecege bakmaz
    df["Log_Return"] = np.log(close / close.shift(1))

    # ═══════════════════════════════════════════════════════════════
    # 3. VOLATILITE
    # ═══════════════════════════════════════════════════════════════

    # Logaritmik getirilerin 20 gunluk hareketli standart sapmasi
    df["Volatility_20d"] = df["Log_Return"].rolling(window=20).std()

    # ═══════════════════════════════════════════════════════════════
    # 4. ROC (Rate of Change)
    # ═══════════════════════════════════════════════════════════════

    # (P_t - P_{t-14}) / P_{t-14} * 100  →  14 gunluk momentum
    df["ROC_14"] = ((close - close.shift(14)) / close.shift(14)) * 100.0

    # ═══════════════════════════════════════════════════════════════
    # 5. INDIKATOR ILISKILERI (Interaction Features)
    # ═══════════════════════════════════════════════════════════════

    # RSI ve MACD uyumu — RSI (0-100) normalize edilip MACD ile farki alinir
    if "RSI_14" in df.columns and "MACD_12_26_9" in df.columns:
        # RSI'yi -1 ile +1 arasina olcekle: (RSI - 50) / 50
        rsi_normalized = (df["RSI_14"] - 50.0) / 50.0
        # MACD'yi kendi std'si ile normalize et (rolling, look-ahead yok)
        macd_std = df["MACD_12_26_9"].rolling(window=50, min_periods=20).std()
        macd_mean = df["MACD_12_26_9"].rolling(window=50, min_periods=20).mean()
        macd_normalized = (df["MACD_12_26_9"] - macd_mean) / macd_std.replace(0, np.nan)

        # Uyum ozelligi: iki sinyalin farki — uyumlu ise 0'a yakin
        df["RSI_MACD_Diff"] = rsi_normalized - macd_normalized

    # RSI ile MACD sinyal cizgisi farki
    if "RSI_14" in df.columns and "MACDs_12_26_9" in df.columns:
        rsi_norm = (df["RSI_14"] - 50.0) / 50.0
        macds_std = df["MACDs_12_26_9"].rolling(window=50, min_periods=20).std()
        macds_mean = df["MACDs_12_26_9"].rolling(window=50, min_periods=20).mean()
        macds_norm = (df["MACDs_12_26_9"] - macds_mean) / macds_std.replace(0, np.nan)
        df["RSI_MACD_Signal_Diff"] = rsi_norm - macds_norm

    # MACD histogram degisim hizi (momentum'un momentumu)
    if "MACDh_12_26_9" in df.columns:
        df["MACD_Hist_Momentum"] = df["MACDh_12_26_9"].diff(1)

    # ═══════════════════════════════════════════════════════════════
    # 6. BOLLINGER BANDS POZISYON OZELLIKLERI
    # ═══════════════════════════════════════════════════════════════

    if all(c in df.columns for c in ["BBL_20_2.0", "BBU_20_2.0", "BBM_20_2.0"]):
        bb_range = df["BBU_20_2.0"] - df["BBL_20_2.0"]
        # Fiyatin bant icerisindeki konumu (0 = alt bant, 1 = ust bant)
        df["BB_Position"] = (close - df["BBL_20_2.0"]) / bb_range.replace(0, np.nan)
        # Bant genisligi — volatilite proxy'si
        df["BB_Width"] = bb_range / df["BBM_20_2.0"].replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════
    # 7. FIYAT / HAREKETLI ORTALAMA ILISKILERI
    # ═══════════════════════════════════════════════════════════════

    if "SMA_20" in df.columns:
        df["Price_SMA20_Ratio"] = close / df["SMA_20"].replace(0, np.nan)

    if "SMA_50" in df.columns:
        df["Price_SMA50_Ratio"] = close / df["SMA_50"].replace(0, np.nan)

    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        # SMA cross farki — trend yonu ve gucu
        df["SMA_Cross_Diff"] = df["SMA_20"] - df["SMA_50"]

    # ═══════════════════════════════════════════════════════════════
    # 8. HACIM (Volume) OZELLIKLERI
    # ═══════════════════════════════════════════════════════════════

    vol = df["Volume"]
    df["Volume_SMA20"] = vol.rolling(window=20).mean()
    # Anlik hacmin ortalamaya orani — yuksek degerler guclü hareket gosterir
    df["Volume_Ratio"] = vol / df["Volume_SMA20"].replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════
    # 9. ATR NORMALIZE
    # ═══════════════════════════════════════════════════════════════

    if "ATR_14" in df.columns:
        # ATR'yi fiyata oranlayarak farkli fiyat seviyelerinde karsilastirma sagla
        df["ATR_Ratio"] = df["ATR_14"] / close.replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════
    # 10. OBV DEGISIM ORANI
    # ═══════════════════════════════════════════════════════════════

    if "OBV" in df.columns:
        obv_shifted = df["OBV"].shift(1).replace(0, np.nan)
        df["OBV_Change"] = (df["OBV"] - df["OBV"].shift(1)) / obv_shifted.abs()

    # ═══════════════════════════════════════════════════════════════
    # NaN TEMIZLIGI — hareketli ortalamalar ve kaydirmalardan kaynaklanan bos satirlar
    # ═══════════════════════════════════════════════════════════════
    df = df.dropna()
    df = df.reset_index(drop=False) if df.index.name else df

    return df


def normalize_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Belirtilen sutunlari Z-Score ile standardize eder (ortalama=0, std=1).

    GA'nin farkli olceklerdeki ozellikleri esit agirliklarda
    degerlendirmesi icin kritik bir adimdir.

    Args:
        df: Ozellik sutunlarini iceren DataFrame.
        feature_columns: Normalize edilecek sutun listesi.
            None ise, sayisal (numeric) tum sutunlari otomatik secer
            (OHLCV ham sutunlari haric).
        method: Normalizasyon yontemi. Su an yalnizca "zscore" desteklenir.

    Returns:
        Belirtilen sutunlari normalize edilmis DataFrame (kopya).

    Raises:
        ValueError: Gecersiz method veya eksik sutunlar verilirse.
    """
    if method != "zscore":
        raise ValueError(
            f"Desteklenmeyen normalizasyon yontemi: '{method}'. "
            f"Yalnizca 'zscore' desteklenmektedir."
        )

    df = df.copy()

    # Sutun secimi
    if feature_columns is None:
        # OHLCV ham sutunlari ve tarih/index haric sayisal sutunlari sec
        _exclude = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        feature_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in _exclude
        ]

    # Eksik sutun kontrolu
    _missing = [c for c in feature_columns if c not in df.columns]
    if _missing:
        raise ValueError(f"DataFrame'de bulunamayan sutunlar: {_missing}")

    if not feature_columns:
        return df

    # ── Z-Score: (x - μ) / σ ─────────────────────────────────────
    # Tamamen vektorel pandas islemleri — scikit-learn bagimliligina gerek yok
    means = df[feature_columns].mean()
    stds = df[feature_columns].std()

    # Standart sapma 0 olan sutunlarda bolmeye sifir hatasini onle
    stds_safe = stds.replace(0, np.nan)

    df[feature_columns] = (df[feature_columns] - means) / stds_safe

    # std=0 olan sutunlari (sabit degerli) sifirla
    zero_std_cols = stds[stds == 0].index.tolist()
    if zero_std_cols:
        df[zero_std_cols] = 0.0

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    DataFrame'deki turetilmis ozellik (feature) sutun isimlerini dondurur.

    generate_features() tarafindan eklenen sutunlari otomatik tespit eder.
    GA fitness fonksiyonunda hangi sutunlarin kullanilacagini belirlemek icin.

    Args:
        df: generate_features() ile islenmis DataFrame.

    Returns:
        Turetilmis ozellik sutun isimleri listesi.
    """
    _generated_features = [
        # Getiriler
        "Return_1d", "Return_3d", "Return_7d", "Log_Return",
        # Volatilite & Momentum
        "Volatility_20d", "ROC_14",
        # Indikator Iliskileri
        "RSI_MACD_Diff", "RSI_MACD_Signal_Diff", "MACD_Hist_Momentum",
        # Bollinger Bands
        "BB_Position", "BB_Width",
        # Fiyat / SMA
        "Price_SMA20_Ratio", "Price_SMA50_Ratio", "SMA_Cross_Diff",
        # Hacim
        "Volume_SMA20", "Volume_Ratio",
        # Diger
        "ATR_Ratio", "OBV_Change",
    ]
    return [c for c in _generated_features if c in df.columns]
