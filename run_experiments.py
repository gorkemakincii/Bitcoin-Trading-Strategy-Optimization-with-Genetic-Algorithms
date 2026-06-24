"""
Faz 5: Deneysel Calismalar ve Sonuc Tablolari

Tum deneyleri sistematik olarak calistirir ve sonuclari kaydeder:
  5.1  Temel Deney (Baseline)
  5.2  Parametre Duyarlilik Analizi
  5.3  Walk-Forward Validation
  5.4  Benchmark Karsilastirmasi
  5.5  Gorsellesitirme (plots.py)
  5.6  Sonuc Tablolari (CSV)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from config.config import DataConfig, GAConfig, StrategyConfig
from src.data.preprocessor import clean_data, split_train_test
from src.ga.engine import run_ga
from src.ga.fitness import evaluate_individual
from src.indicators.technical import calculate_indicators
from src.strategy.benchmarks import run_all_benchmarks
from src.strategy.signals import generate_signals_fast
from src.ga.chromosome import decode_chromosome
from src.strategy.backtest import run_backtest_fast

RESULTS_DIR = "results/tables"
SEEDS = [42, 123, 456]  # 3 bagimsiz calistirma


def _load_data():
    """Veriyi yukle, temizle, indikatorleri hesapla."""
    data_cfg = DataConfig()
    df = pd.read_csv("data/raw/btc_daily.csv", index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = clean_data(df)
    df = calculate_indicators(df)
    train_df, test_df = split_train_test(df, data_cfg.train_end_date, data_cfg.test_start_date)
    _train_close = train_df["Close"].to_numpy(dtype=np.float64)  # noqa: F841 (unused, kept for reference)
    test_close = test_df["Close"].to_numpy(dtype=np.float64)
    return df, train_df, test_df, test_close


def _run_single_experiment(
    train_data, test_data, strat_cfg, seed, **ga_overrides
):
    """Tek bir GA deneyini calistirir, train+test metriklerini dondurur."""
    ga_cfg = GAConfig()
    ga_kwargs = dict(
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
        seed=seed,
        verbose=False,
    )
    ga_kwargs.update(ga_overrides)

    result = run_ga(train_data, **ga_kwargs)

    test_metrics = evaluate_individual(
        result.best_individual, test_data,
        initial_capital=strat_cfg.initial_capital,
        position_size=strat_cfg.position_size,
        commission_rate=strat_cfg.commission_rate,
        slippage_rate=strat_cfg.slippage_rate,
    )

    return {
        "train_return": result.best_metrics.get("total_return", 0),
        "train_sharpe": result.best_metrics.get("sharpe_ratio", 0),
        "train_max_dd": result.best_metrics.get("max_drawdown", 0),
        "train_win_rate": result.best_metrics.get("win_rate", 0),
        "train_trades": result.best_metrics.get("total_trades", 0),
        "test_return": test_metrics.get("total_return", 0),
        "test_sharpe": test_metrics.get("sharpe_ratio", 0),
        "test_max_dd": test_metrics.get("max_drawdown", 0),
        "test_win_rate": test_metrics.get("win_rate", 0),
        "test_trades": test_metrics.get("total_trades", 0),
        "best_fitness": result.best_fitness,
        "ga_time": result.total_time,
        "best_individual": result.best_individual,
        "best_params": result.best_params,
        "logbook": result.logbook,
        "hall_of_fame": result.hall_of_fame,
    }


def _multi_seed_experiment(train_data, test_data, strat_cfg, label, **ga_overrides):
    """Ayni konfigurasyonu 3 farkli seed ile calistirir, ortalama ve std dondurur."""
    all_results = []
    for seed in SEEDS:
        r = _run_single_experiment(train_data, test_data, strat_cfg, seed, **ga_overrides)
        all_results.append(r)

    metrics_keys = [
        "train_return", "train_sharpe", "train_max_dd", "train_win_rate", "train_trades",
        "test_return", "test_sharpe", "test_max_dd", "test_win_rate", "test_trades",
        "best_fitness", "ga_time",
    ]

    row = {"experiment": label}
    for k in metrics_keys:
        vals = [r[k] for r in all_results]
        row[f"{k}_mean"] = np.mean(vals)
        row[f"{k}_std"] = np.std(vals)

    return row, all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5.1 TEMEL DENEY
# ═══════════════════════════════════════════════════════════════════════════════


def run_baseline(train_data, test_data, strat_cfg):
    """Varsayilan parametrelerle baseline deney."""
    print("\n[5.1] Temel Deney (Baseline)...")
    row, results = _multi_seed_experiment(
        train_data, test_data, strat_cfg, "Baseline"
    )

    df = pd.DataFrame([row])
    df.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    print(f"  Kaydedildi: {RESULTS_DIR}/baseline_results.csv")
    print(f"  Train Sharpe: {row['train_sharpe_mean']:.4f} +/- {row['train_sharpe_std']:.4f}")
    print(f"  Test Sharpe:  {row['test_sharpe_mean']:.4f} +/- {row['test_sharpe_std']:.4f}")

    # En iyi seed'in parametrelerini kaydet
    best_idx = max(range(len(results)), key=lambda i: results[i]["best_fitness"])
    best = results[best_idx]
    params_df = pd.DataFrame([best["best_params"]])
    params_df.to_csv(os.path.join(RESULTS_DIR, "best_individual_params.csv"), index=False)
    print(f"  Kaydedildi: {RESULTS_DIR}/best_individual_params.csv")

    return results[best_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2 PARAMETRE DUYARLILIK ANALIZI
# ═══════════════════════════════════════════════════════════════════════════════


def run_parameter_sensitivity(train_data, test_data, strat_cfg):
    """Farkli GA konfigurasyonlari ile parametre duyarlilik analizi."""
    print("\n[5.2] Parametre Duyarlilik Analizi...")
    rows = []

    # Deney 1: Populasyon Boyutu
    print("  Deney 1: Populasyon Boyutu...")
    for pop in [50, 100, 200, 300]:
        label = f"Pop={pop}"
        print(f"    {label}...", end=" ", flush=True)
        row, _ = _multi_seed_experiment(
            train_data, test_data, strat_cfg, label,
            population_size=pop,
        )
        rows.append(row)
        print(f"Sharpe: {row['test_sharpe_mean']:.4f}")

    # Deney 2: Nesil Sayisi
    print("  Deney 2: Nesil Sayisi...")
    for gen in [25, 50, 100, 200]:
        label = f"Gen={gen}"
        print(f"    {label}...", end=" ", flush=True)
        row, _ = _multi_seed_experiment(
            train_data, test_data, strat_cfg, label,
            num_generations=gen,
        )
        rows.append(row)
        print(f"Sharpe: {row['test_sharpe_mean']:.4f}")

    # Deney 3: Caprazlama ve Mutasyon Orani
    print("  Deney 3: CX/MUT Oranlari...")
    for cx, mut in [(0.6, 0.1), (0.8, 0.2), (0.9, 0.3), (0.7, 0.4)]:
        label = f"CX={cx}/MUT={mut}"
        print(f"    {label}...", end=" ", flush=True)
        row, _ = _multi_seed_experiment(
            train_data, test_data, strat_cfg, label,
            crossover_rate=cx, mutation_rate=mut,
        )
        rows.append(row)
        print(f"Sharpe: {row['test_sharpe_mean']:.4f}")

    # Deney 4: Turnuva Boyutu
    print("  Deney 4: Turnuva Boyutu...")
    for ts in [3, 5, 7]:
        label = f"Tournament={ts}"
        print(f"    {label}...", end=" ", flush=True)
        row, _ = _multi_seed_experiment(
            train_data, test_data, strat_cfg, label,
            tournament_size=ts,
        )
        rows.append(row)
        print(f"Sharpe: {row['test_sharpe_mean']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "parameter_sensitivity.csv"), index=False)
    print(f"  Kaydedildi: {RESULTS_DIR}/parameter_sensitivity.csv")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.3 WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_walk_forward(df_full, strat_cfg):
    """Walk-forward validation: kayan pencere ile overfitting kontrolu."""
    print("\n[5.3] Walk-Forward Validation...")

    periods = [
        ("2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    rows = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(periods, 1):
        train_mask = (df_full.index >= train_start) & (df_full.index <= train_end)
        test_mask = (df_full.index >= test_start) & (df_full.index <= test_end)

        train_data = df_full.loc[train_mask]
        test_data = df_full.loc[test_mask]
        train_close = train_data["Close"].to_numpy(dtype=np.float64)
        test_close = test_data["Close"].to_numpy(dtype=np.float64)

        if len(train_close) < 100 or len(test_close) < 10:
            print(f"  Donem {i}: Yetersiz veri, atlaniyor.")
            continue

        label = f"WF{i}: Train[{train_start[:4]}-{train_end[:4]}] Test[{test_start[:4]}]"
        print(f"  {label}...", end=" ", flush=True)

        r = _run_single_experiment(
            train_data, test_data, strat_cfg, seed=42,
            population_size=100, num_generations=50,
        )

        row = {
            "period": label,
            "train_years": f"{train_start[:4]}-{train_end[:4]}",
            "test_year": test_start[:4],
            "train_return": r["train_return"],
            "train_sharpe": r["train_sharpe"],
            "train_max_dd": r["train_max_dd"],
            "train_trades": r["train_trades"],
            "test_return": r["test_return"],
            "test_sharpe": r["test_sharpe"],
            "test_max_dd": r["test_max_dd"],
            "test_trades": r["test_trades"],
            "overfit_ratio": (
                r["train_sharpe"] / r["test_sharpe"]
                if r["test_sharpe"] != 0 else float("inf")
            ),
        }
        rows.append(row)
        print(f"Train Sharpe: {r['train_sharpe']:.4f} | Test Sharpe: {r['test_sharpe']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "walkforward_results.csv"), index=False)
    print(f"  Kaydedildi: {RESULTS_DIR}/walkforward_results.csv")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.4 BENCHMARK KARSILASTIRMASI
# ═══════════════════════════════════════════════════════════════════════════════


def run_benchmark_comparison(test_data, test_close, strat_cfg, ga_best_individual):
    """GA optimize stratejisi vs benchmark stratejileri."""
    print("\n[5.4] Benchmark Karsilastirmasi...")

    backtest_kwargs = dict(
        initial_capital=strat_cfg.initial_capital,
        position_size=strat_cfg.position_size,
        commission_rate=strat_cfg.commission_rate,
        slippage_rate=strat_cfg.slippage_rate,
    )

    # GA stratejisi
    ga_metrics = evaluate_individual(
        ga_best_individual, test_data, **backtest_kwargs
    )

    # Benchmark stratejileri
    bench = run_all_benchmarks(test_close, **backtest_kwargs)

    # Birlestir
    all_results = {"GA Optimize": {
        "total_return": ga_metrics.get("total_return", 0),
        "sharpe_ratio": ga_metrics.get("sharpe_ratio", 0),
        "max_drawdown": ga_metrics.get("max_drawdown", 0),
        "win_rate": ga_metrics.get("win_rate", 0),
        "total_trades": ga_metrics.get("total_trades", 0),
    }}
    all_results.update(bench)

    rows = []
    for name, m in all_results.items():
        rows.append({"strategy": name, **m})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "benchmark_comparison.csv"), index=False)
    print(f"  Kaydedildi: {RESULTS_DIR}/benchmark_comparison.csv")

    print(f"\n  {'Strateji':15s} {'Return':>12s} {'Sharpe':>10s} {'MaxDD':>10s} {'WinRate':>10s} {'Trades':>8s}")
    print(f"  {'-'*65}")
    for name, m in all_results.items():
        print(
            f"  {name:15s} {m['total_return']:>11.2f}% {m['sharpe_ratio']:>10.4f} "
            f"{m['max_drawdown']:>9.2f}% {m['win_rate']:>9.1f}% {m['total_trades']:>8d}"
        )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5.5 GORSELLESITIRME
# ═══════════════════════════════════════════════════════════════════════════════


def run_visualization(
    baseline_result, test_df, test_close, benchmark_results, sensitivity_df
):
    """Tum grafikleri olusturur."""
    print("\n[5.5] Gorsellesitirme...")

    from src.visualization.plots import (
        plot_fitness_evolution,
        plot_equity_curve,
        plot_signals_on_price,
        plot_drawdown,
        plot_trade_distribution,
        plot_benchmark_comparison,
        plot_correlation_matrix,
        plot_parameter_sensitivity,
    )
    from src.strategy.backtest import _ffill_numpy

    # Sinyalleri uret
    params = decode_chromosome(baseline_result["best_individual"])
    signals = generate_signals_fast(test_close, params)
    bt = run_backtest_fast(
        test_close,
        signals,
        high=test_df["High"].to_numpy(dtype=np.float64) if "High" in test_df.columns else None,
        low=test_df["Low"].to_numpy(dtype=np.float64) if "Low" in test_df.columns else None,
        stop_loss=params.stop_loss,
        take_profit=params.take_profit,
    )

    # 1. Fitness evrimi
    plot_fitness_evolution(baseline_result["logbook"])

    # 2. Equity curve
    plot_equity_curve(
        bt["equity_curve"], bt["buy_and_hold"],
        dates=test_df.index, title="GA Stratejisi vs Buy & Hold (Test Seti)"
    )

    # 3. Sinyaller
    plot_signals_on_price(test_close, signals, dates=test_df.index)

    # 4. Drawdown
    plot_drawdown(bt["equity_curve"], dates=test_df.index)

    # 5. Trade dagilimi
    raw_position = np.where(signals == 1, 1.0, np.where(signals == -1, 0.0, np.nan))
    raw_position = _ffill_numpy(raw_position, fill_value=0.0)
    raw_change = np.empty(len(raw_position), dtype=np.float64)
    raw_change[0] = raw_position[0]
    raw_change[1:] = raw_position[1:] - raw_position[:-1]
    entry_prices = test_close[raw_change == 1.0]
    exit_prices = test_close[raw_change == -1.0]
    plot_trade_distribution(entry_prices, exit_prices)

    # 6. Korelasyon matrisi
    plot_correlation_matrix(baseline_result["hall_of_fame"])

    # 7. Benchmark karsilastirma
    plot_benchmark_comparison(benchmark_results, metric="sharpe_ratio")
    plot_benchmark_comparison(benchmark_results, metric="total_return")

    # 8. Parametre duyarlilik grafikleri
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        # Populasyon boyutu
        pop_rows = sensitivity_df[sensitivity_df["experiment"].str.startswith("Pop=")]
        if len(pop_rows) > 0:
            pop_vals = [int(e.split("=")[1]) for e in pop_rows["experiment"]]
            plot_parameter_sensitivity(
                pop_vals, pop_rows["test_sharpe_mean"].tolist(),
                "Populasyon Boyutu"
            )

        # Nesil sayisi
        gen_rows = sensitivity_df[sensitivity_df["experiment"].str.startswith("Gen=")]
        if len(gen_rows) > 0:
            gen_vals = [int(e.split("=")[1]) for e in gen_rows["experiment"]]
            plot_parameter_sensitivity(
                gen_vals, gen_rows["test_sharpe_mean"].tolist(),
                "Nesil Sayisi"
            )

    print("  Gorsellesitirme tamamlandi.")


# ═══════════════════════════════════════════════════════════════════════════════
# ANA CALISTIRMA
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 80)
    print("  FAZ 5: DENEYSEL CALISMALAR")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    strat_cfg = StrategyConfig()

    # Veri yukle
    print("\nVeri yukleniyor...")
    df_full, train_df, test_df, test_close = _load_data()
    print(f"  Train: {len(train_df)} gun | Test: {len(test_close)} gun")

    t0 = time.time()

    # 5.1 Baseline
    baseline_result = run_baseline(train_df, test_df, strat_cfg)

    # 5.2 Parametre Duyarlilik
    sensitivity_df = run_parameter_sensitivity(train_df, test_df, strat_cfg)

    # 5.3 Walk-Forward Validation
    wf_df = run_walk_forward(df_full, strat_cfg)

    # 5.4 Benchmark Karsilastirmasi
    benchmark_results = run_benchmark_comparison(
        test_df, test_close, strat_cfg, baseline_result["best_individual"]
    )

    # 5.5 Gorsellesitirme
    run_visualization(
        baseline_result, test_df, test_close, benchmark_results, sensitivity_df
    )

    total = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  FAZ 5 TAMAMLANDI | Toplam Sure: {total:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
