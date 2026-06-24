"""
signals.py modul dogrulama scripti.

Sentetik BTC verisi uzerinde generate_signals ve generate_signals_fast
fonksiyonlarinin dogrulugunu, vektorelligini ve tutarliligini test eder.
"""
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from src.ga.chromosome import StrategyParams
from src.strategy.signals import generate_signals, generate_signals_fast

# ═══════════════════════════════════════════════════════════════════
# SENTETIK VERI OLUSTUR
# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)
n = 500
dates = pd.date_range("2022-01-01", periods=n, freq="D")
close = 25000 + np.cumsum(np.random.randn(n) * 500)
high = close + np.abs(np.random.randn(n) * 300)
low = close - np.abs(np.random.randn(n) * 300)
open_ = close + np.random.randn(n) * 100
volume = np.random.randint(1_000_000, 50_000_000, size=n).astype(float)

df = pd.DataFrame({
    "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
}, index=dates)

# ═══════════════════════════════════════════════════════════════════
# ORNEK GA PARAMETRELERI
# ═══════════════════════════════════════════════════════════════════

params = StrategyParams(
    rsi_period=14,
    rsi_oversold=30.0,
    rsi_overbought=70.0,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_period=20,
    bb_std=2.0,
    sma_short=10,
    sma_long=50,
    stop_loss=0.05,
    take_profit=0.10,
    weight_rsi=0.3,
    weight_macd=0.3,
    weight_bb=0.2,
    weight_sma=0.2,
)

# ═══════════════════════════════════════════════════════════════════
# TEST 1: generate_signals cikti sutunlari
# ═══════════════════════════════════════════════════════════════════

print("=" * 60)
print("TEST 1: generate_signals sutun kontrolu")
print("=" * 60)

df_sig = generate_signals(df, params)

expected_cols = [
    "RSI", "MACD_Line", "MACD_Signal", "MACD_Hist",
    "BB_Lower", "BB_Middle", "BB_Upper",
    "SMA_Short", "SMA_Long",
    "Signal_RSI", "Signal_MACD", "Signal_BB", "Signal_SMA",
    "Composite_Signal",
]

for col in expected_cols:
    assert col in df_sig.columns, f"HATA: '{col}' sutunu eksik!"
    print(f"  [OK] {col}")

print(f"\n  Toplam satir: {len(df_sig)}, eklenen sutun: {len(expected_cols)}")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: Sinyal degerleri sadece {-1, 0, 1} olmali
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 2: Sinyal deger araligi kontrolu")
print("=" * 60)

valid_values = {-1, 0, 1}
signal_cols = ["Signal_RSI", "Signal_MACD", "Signal_BB", "Signal_SMA", "Composite_Signal"]

for col in signal_cols:
    # NaN olabilecek ilk satirlari atla
    non_nan = df_sig[col].dropna()
    unique_vals = set(non_nan.unique().astype(int))
    assert unique_vals.issubset(valid_values), \
        f"HATA: {col} icinde gecersiz degerler: {unique_vals - valid_values}"
    counts = non_nan.value_counts().to_dict()
    print(f"  [OK] {col:20s} --> dagilim: {counts}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: generate_signals_fast tutarliligi
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 3: generate_signals vs generate_signals_fast tutarliligi")
print("=" * 60)

composite_df = df_sig["Composite_Signal"].values
composite_fast = generate_signals_fast(close, params)

# NaN bolgelerini (baslangic warm-up) maskeliyoruz
valid_mask = ~np.isnan(df_sig["RSI"].values)
match_count = np.sum(composite_df[valid_mask] == composite_fast[valid_mask])
total_valid = np.sum(valid_mask)
match_pct = match_count / total_valid * 100

print(f"  Eslesme: {match_count}/{total_valid} ({match_pct:.1f}%)")
assert match_pct > 99.0, f"HATA: Tutarlilik cok dusuk: {match_pct:.1f}%"
print(f"  [OK] Iki fonksiyon tutarli calisiyyor.")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: Look-ahead bias kontrolu
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 4: Look-ahead bias kontrolu")
print("=" * 60)

# Veriyi yariya kes — ilk yari ile uretilen sinyaller ayni kalmali
half = n // 2
df_half = df.iloc[:half].copy()
df_sig_half = generate_signals(df_half, params)

# Ilk yarinin sinyalleri tam veri ile ayni olmali
df_sig_full_first_half = df_sig.iloc[:half]

for col in signal_cols:
    half_vals = df_sig_half[col].dropna().values
    full_vals = df_sig_full_first_half[col].dropna().values
    min_len = min(len(half_vals), len(full_vals))
    if min_len > 0:
        match = np.array_equal(half_vals[:min_len], full_vals[:min_len])
        assert match, f"HATA: {col} look-ahead bias iceriyor!"
        print(f"  [OK] {col:20s} --> gelecek veriden bagimsiz")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: Performans (hiz) testi
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TEST 5: Performans testi (1000 cagri)")
print("=" * 60)

iterations = 1000
close_arr = close.copy()

# generate_signals_fast zamanlama
start = time.perf_counter()
for _ in range(iterations):
    _ = generate_signals_fast(close_arr, params)
elapsed_fast = time.perf_counter() - start

print(f"  generate_signals_fast: {iterations} cagri --> {elapsed_fast:.2f}s "
      f"(ortalama {elapsed_fast/iterations*1000:.2f}ms/cagri)")

# generate_signals zamanlama
start = time.perf_counter()
for _ in range(iterations):
    _ = generate_signals(df, params)
elapsed_full = time.perf_counter() - start

print(f"  generate_signals:      {iterations} cagri --> {elapsed_full:.2f}s "
      f"(ortalama {elapsed_full/iterations*1000:.2f}ms/cagri)")

print(f"  Hiz kazanimi (fast): {elapsed_full/elapsed_fast:.1f}x")

# ═══════════════════════════════════════════════════════════════════
# SONUC
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("[BASARILI] Tum dogrulama testleri gecti!")
print("=" * 60)
