"""
BTC GA Trading - Ana Calistirma Scripti

Tum pipeline'i birlestirir:
  1. Veri toplama (yfinance)
  2. Veri temizleme ve on isleme
  3. Teknik indikatorler
  4. GA optimizasyonu
  5. Test seti degerlendirmesi
  6. Sonuclarin kaydedilmesi
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from config.config import DataConfig, GAConfig, StrategyConfig
from src.data.collector import collect_btc_data
from src.data.preprocessor import clean_data, get_data_report, split_train_test
from src.indicators.technical import calculate_indicators
from src.ga.engine import run_ga
from src.ga.fitness import evaluate_individual


def main() -> None:
    print("=" * 80)
    print("  BTC GA Trading Stratejisi Optimizasyonu")
    print("=" * 80)

    data_cfg = DataConfig()
    ga_cfg = GAConfig()
    strat_cfg = StrategyConfig()

    # ─── 1. Veri Toplama ─────────────────────────────────────────────────
    raw_path = "data/raw/btc_daily.csv"

    if os.path.exists(raw_path):
        print(f"\n[1/5] Mevcut veri yukleniyor: {raw_path}")
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    else:
        print(f"\n[1/5] BTC verisi indiriliyor ({data_cfg.start_date} ~ {data_cfg.end_date})...")
        df = collect_btc_data(data_cfg.start_date, data_cfg.end_date, data_cfg.interval)
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(raw_path)
        print(f"  Kaydedildi: {raw_path}")

    # MultiIndex varsa duzelt (yfinance bazen boyle dondurur)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  Satir: {len(df)} | Aralik: {df.index[0].date()} ~ {df.index[-1].date()}")

    # ─── 2. Veri Temizleme ───────────────────────────────────────────────
    print("\n[2/5] Veri temizleniyor ve indikatörler hesaplaniyor...")
    df = clean_data(df)
    df = calculate_indicators(df)

    report = get_data_report(df)
    print(f"  Temiz satir: {report['total_rows']}")
    print(f"  Eksik deger: {sum(v for v in report['missing_values'].values() if isinstance(v, (int, float)))}")

    # ─── 3. Train/Test Ayirimi ───────────────────────────────────────────
    print(f"\n[3/5] Train/Test ayirimi...")
    train_df, test_df = split_train_test(df, data_cfg.train_end_date, data_cfg.test_start_date)

    # Close verisini yardimci hesaplar icin hazirla; GA OHLC DataFrame kullanir.
    test_close = test_df["Close"].to_numpy(dtype=np.float64)

    print(f"  Train: {len(train_df)} gun ({train_df.index[0].date()} ~ {train_df.index[-1].date()})")
    print(f"  Test:  {len(test_df)} gun ({test_df.index[0].date()} ~ {test_df.index[-1].date()})")

    # ─── 4. GA Optimizasyonu (Train Seti) ────────────────────────────────
    print(f"\n[4/5] GA Optimizasyonu basliyor...")
    print(f"  Populasyon: {ga_cfg.population_size} | Nesil: {ga_cfg.num_generations}")
    print()

    result = run_ga(
        train_df,
        population_size=ga_cfg.population_size,
        num_generations=ga_cfg.num_generations,
        crossover_rate=ga_cfg.crossover_rate,
        mutation_rate=ga_cfg.mutation_rate,
        tournament_size=ga_cfg.tournament_size,
        elite_size=ga_cfg.elite_size,
        hof_size=ga_cfg.hof_size,
        initial_capital=strat_cfg.initial_capital,
        position_size=strat_cfg.position_size,
        commission_rate=strat_cfg.commission_rate,
        slippage_rate=strat_cfg.slippage_rate,
        seed=42,
        verbose=True,
    )

    # ─── 5. Test Seti Degerlendirmesi ────────────────────────────────────
    print(f"\n[5/5] Test seti degerlendirmesi...")

    test_metrics = evaluate_individual(
        result.best_individual,
        test_df,
        initial_capital=strat_cfg.initial_capital,
        position_size=strat_cfg.position_size,
        commission_rate=strat_cfg.commission_rate,
        slippage_rate=strat_cfg.slippage_rate,
    )

    # Buy & Hold hesapla
    bh_return = (test_close[-1] / test_close[0]) - 1

    print(f"\n{'=' * 80}")
    print(f"  SONUCLAR")
    print(f"{'=' * 80}")
    print(f"\n  {'':30s} {'TRAIN':>12s} {'TEST':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'Total Return':30s} {result.best_metrics.get('total_return', 0):>10.2f}% {test_metrics.get('total_return', 0):>10.2f}%")
    print(f"  {'Sharpe Ratio':30s} {result.best_metrics.get('sharpe_ratio', 0):>12.4f} {test_metrics.get('sharpe_ratio', 0):>12.4f}")
    print(f"  {'Max Drawdown':30s} {result.best_metrics.get('max_drawdown', 0):>10.2f}% {test_metrics.get('max_drawdown', 0):>10.2f}%")
    print(f"  {'Win Rate':30s} {result.best_metrics.get('win_rate', 0):>11.1f}% {test_metrics.get('win_rate', 0):>11.1f}%")
    print(f"  {'Trade Sayisi':30s} {result.best_metrics.get('total_trades', 0):>12d} {test_metrics.get('total_trades', 0):>12d}")
    print(f"  {'Buy & Hold Return (Test)':30s} {'':>12s} {bh_return:>11.2%}")
    print(f"\n  GA Sure: {result.total_time:.1f}s")

    # ─── Sonuclari Kaydet ────────────────────────────────────────────────
    os.makedirs("results/tables", exist_ok=True)

    # Optimal parametreler
    params_out = {
        "optimal_params": result.best_params,
        "train_metrics": {
            k: v for k, v in result.best_metrics.items()
            if k not in ("equity_curve", "params", "repaired_individual", "validation_errors")
        },
        "test_metrics": {
            k: v for k, v in test_metrics.items()
            if k not in ("equity_curve", "params", "repaired_individual", "validation_errors")
        },
        "buy_hold_return_test": float(bh_return),
        "ga_config": result.config,
    }

    # float/int donusumu (json uyumu)
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    with open("results/tables/ga_results.json", "w") as f:
        json.dump(params_out, f, indent=2, default=_convert)
    print(f"\n  Sonuclar kaydedildi: results/tables/ga_results.json")

    # Logbook CSV
    logbook_df = pd.DataFrame(result.logbook)
    logbook_df.to_csv("results/tables/logbook.csv", index=False)
    print(f"  Logbook kaydedildi: results/tables/logbook.csv")

    print(f"\n{'=' * 80}")
    print(f"  TAMAMLANDI")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
