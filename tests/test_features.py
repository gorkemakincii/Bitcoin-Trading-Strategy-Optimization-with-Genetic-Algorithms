"""
features.py modul dogrulama scripti.

Sentetik OHLCV verisi ile generate_features ve normalize_features
fonksiyonlarinin dogru calistigini kontrol eder.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

# Sentetik BTC-benzeri veri olustur (100 gun)
np.random.seed(42)
n = 200
dates = pd.date_range("2023-01-01", periods=n, freq="D")
close = 25000 + np.cumsum(np.random.randn(n) * 500)
high = close + np.abs(np.random.randn(n) * 300)
low = close - np.abs(np.random.randn(n) * 300)
open_ = close + np.random.randn(n) * 100
volume = np.random.randint(1_000_000, 50_000_000, size=n).astype(float)

df = pd.DataFrame({
    "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
}, index=dates)

# Teknik indikatorleri ekle (pandas-ta ile)
try:
    from src.indicators.technical import calculate_indicators
    df = calculate_indicators(df)
    print("[OK] Teknik indikatorler hesaplandi.")
except ImportError:
    print("[UYARI] pandas-ta bulunamadi, sentetik indikatorler ekleniyor...")
    # Sentetik indikator sutunlari
    df["RSI_14"] = 30 + np.random.rand(n) * 40
    df["MACD_12_26_9"] = np.random.randn(n) * 100
    df["MACDh_12_26_9"] = np.random.randn(n) * 50
    df["MACDs_12_26_9"] = np.random.randn(n) * 80
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["BBL_20_2.0"] = df["SMA_20"] - 2 * df["Close"].rolling(20).std()
    df["BBM_20_2.0"] = df["SMA_20"]
    df["BBU_20_2.0"] = df["SMA_20"] + 2 * df["Close"].rolling(20).std()
    df["ATR_14"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()

# ─── generate_features testi ───────────────────────────────────
from src.data.features import generate_features, normalize_features, get_feature_columns

print(f"\n[INFO] Girdi DataFrame: {df.shape[0]} satir, {df.shape[1]} sutun")

df_feat = generate_features(df)
print(f"[INFO] Cikti DataFrame: {df_feat.shape[0]} satir, {df_feat.shape[1]} sutun")
print(f"[INFO] NaN sayisi: {df_feat.isnull().sum().sum()}")

feature_cols = get_feature_columns(df_feat)
print(f"\n[INFO] Turetilen ozellikler ({len(feature_cols)} adet):")
for col in feature_cols:
    print(f"  - {col}: min={df_feat[col].min():.4f}, max={df_feat[col].max():.4f}")

# ─── normalize_features testi ──────────────────────────────────
df_norm = normalize_features(df_feat, feature_cols)
print(f"\n[INFO] Normalizasyon sonrasi:")
for col in feature_cols[:5]:
    print(f"  - {col}: mean={df_norm[col].mean():.6f}, std={df_norm[col].std():.6f}")

# ─── Dogrulama kontrolleri ─────────────────────────────────────
assert df_feat.isnull().sum().sum() == 0, "HATA: NaN degerleri temizlenmemis!"
assert "Return_1d" in df_feat.columns, "HATA: Return_1d eksik!"
assert "Log_Return" in df_feat.columns, "HATA: Log_Return eksik!"
assert "Volatility_20d" in df_feat.columns, "HATA: Volatility_20d eksik!"
assert "ROC_14" in df_feat.columns, "HATA: ROC_14 eksik!"
assert "RSI_MACD_Diff" in df_feat.columns, "HATA: RSI_MACD_Diff eksik!"

# Z-Score dogrulama: ortalama ≈ 0, std ≈ 1
for col in feature_cols:
    mean_val = abs(df_norm[col].mean())
    std_val = df_norm[col].std()
    assert mean_val < 1e-10, f"HATA: {col} ortalamasi sifir degil: {mean_val}"
    assert abs(std_val - 1.0) < 1e-10, f"HATA: {col} std'si 1 degil: {std_val}"

print("\n" + "=" * 50)
print("[BASARILI] Tum dogrulama kontrolleri gecti!")
print("=" * 50)
