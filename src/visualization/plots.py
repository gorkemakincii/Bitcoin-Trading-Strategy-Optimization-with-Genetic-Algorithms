"""
Gorsellesitirme modulu -- GA tabanli BTC trading botu icin.

Faz 5.6: Asagidaki grafikleri olusturur:
  1. Fitness Evrimi (nesil bazli avg/max/min)
  2. Equity Curve (GA strateji vs Buy & Hold)
  3. Alis/Satis Sinyalleri (fiyat grafigi uzerinde)
  4. Drawdown Grafigi
  5. Parametre Duyarlilik Grafikleri
  6. Trade Dagilim Histogrami
  7. Benchmark Karsilastirma Bar Grafigi
  8. Korelasyon Matrisi (Hall of Fame parametreleri)
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # GUI olmadan kaydetmek icin

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

FIGURES_DIR = "results/figures"


def _ensure_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FITNESS EVRIMI
# ═══════════════════════════════════════════════════════════════════════════════


def plot_fitness_evolution(
    logbook: pd.DataFrame | list[dict],
    save_path: str | None = None,
) -> None:
    """
    Nesil bazli fitness evrimi grafigi.

    Args:
        logbook: Nesil istatistiklerini iceren DataFrame veya dict listesi.
                 Sutunlar: generation, avg_fitness, max_fitness, min_fitness
        save_path: Kaydedilecek dosya yolu. None ise varsayilan yol kullanilir.
    """
    _ensure_dir()
    if isinstance(logbook, list):
        logbook = pd.DataFrame(logbook)

    gen = logbook["generation"]
    avg = logbook["avg_fitness"]
    max_f = logbook["max_fitness"]
    min_f = logbook["min_fitness"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gen, max_f, label="En Iyi (Max)", color="green", linewidth=2)
    ax.plot(gen, avg, label="Ortalama (Avg)", color="blue", linewidth=1.5)
    ax.plot(gen, min_f, label="En Kotu (Min)", color="red", linewidth=1, alpha=0.7)
    ax.fill_between(gen, min_f, max_f, alpha=0.1, color="blue")

    ax.set_xlabel("Nesil", fontsize=12)
    ax.set_ylabel("Fitness (Sharpe Ratio)", fontsize=12)
    ax.set_title("GA Fitness Evrimi", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = save_path or os.path.join(FIGURES_DIR, "fitness_evolution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EQUITY CURVE
# ═══════════════════════════════════════════════════════════════════════════════


def plot_equity_curve(
    equity: np.ndarray | pd.Series,
    buy_and_hold: np.ndarray | pd.Series,
    dates: pd.DatetimeIndex | None = None,
    title: str = "GA Stratejisi vs Buy & Hold",
    save_path: str | None = None,
) -> None:
    """
    Portfoy degeri karsilastirma grafigi.

    Args:
        equity: GA strateji equity curve.
        buy_and_hold: Buy & Hold equity curve.
        dates: Tarih indeksi (opsiyonel).
        title: Grafik basligi.
        save_path: Kaydedilecek dosya yolu.
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(14, 7))

    x = dates if dates is not None else np.arange(len(equity))

    ax.plot(x, equity, label="GA Stratejisi", color="blue", linewidth=1.5)
    ax.plot(x, buy_and_hold, label="Buy & Hold", color="orange", linewidth=1.5)

    ax.set_xlabel("Tarih" if dates is not None else "Gun", fontsize=12)
    ax.set_ylabel("Portfoy Degeri ($)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

    path = save_path or os.path.join(FIGURES_DIR, "equity_curve.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ALIS/SATIS SINYALLERI
# ═══════════════════════════════════════════════════════════════════════════════


def plot_signals_on_price(
    close: np.ndarray,
    signals: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
    title: str = "BTC Fiyat ve Alis/Satis Sinyalleri",
    save_path: str | None = None,
) -> None:
    """
    BTC fiyat grafigi uzerinde alis (yesil) ve satis (kirmizi) noktlari.

    Args:
        close: Kapanis fiyatlari.
        signals: Sinyal dizisi (+1, -1, 0).
        dates: Tarih indeksi.
        save_path: Kaydedilecek dosya yolu.
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(16, 7))

    x = dates if dates is not None else np.arange(len(close))

    ax.plot(x, close, label="BTC Fiyat", color="gray", linewidth=0.8, alpha=0.8)

    buy_mask = signals == 1
    sell_mask = signals == -1
    if buy_mask.any():
        ax.scatter(
            x[buy_mask] if dates is not None else np.where(buy_mask)[0],
            close[buy_mask], marker="^", color="green", s=80, label="AL", zorder=5
        )
    if sell_mask.any():
        ax.scatter(
            x[sell_mask] if dates is not None else np.where(sell_mask)[0],
            close[sell_mask], marker="v", color="red", s=80, label="SAT", zorder=5
        )

    ax.set_xlabel("Tarih" if dates is not None else "Gun", fontsize=12)
    ax.set_ylabel("Fiyat ($)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

    path = save_path or os.path.join(FIGURES_DIR, "signals_on_price.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DRAWDOWN GRAFIGI
# ═══════════════════════════════════════════════════════════════════════════════


def plot_drawdown(
    equity: np.ndarray | pd.Series,
    dates: pd.DatetimeIndex | None = None,
    save_path: str | None = None,
) -> None:
    """
    Portfoyun tepe noktasindan dusus yuzdesi (drawdown) grafigi.

    Args:
        equity: Equity curve.
        dates: Tarih indeksi.
        save_path: Kaydedilecek dosya yolu.
    """
    _ensure_dir()
    equity_arr = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = np.where(peak > 0, (equity_arr - peak) / peak * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(14, 5))

    x = dates if dates is not None else np.arange(len(equity_arr))
    ax.fill_between(x, drawdown, 0, color="red", alpha=0.3)
    ax.plot(x, drawdown, color="red", linewidth=0.8)

    ax.set_xlabel("Tarih" if dates is not None else "Gun", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title("Maximum Drawdown", fontsize=14)
    ax.grid(True, alpha=0.3)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

    path = save_path or os.path.join(FIGURES_DIR, "drawdown.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PARAMETRE DUYARLILIK
# ═══════════════════════════════════════════════════════════════════════════════


def plot_parameter_sensitivity(
    param_values: list[int | float],
    fitness_values: list[float],
    param_name: str,
    save_path: str | None = None,
) -> None:
    """
    Tek bir GA parametresinin fitness uzerindeki etkisi.

    Args:
        param_values: Test edilen parametre degerleri.
        fitness_values: Her deger icin elde edilen en iyi fitness.
        param_name: Parametre adi (grafik basligi icin).
        save_path: Kaydedilecek dosya yolu.
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(param_values, fitness_values, "o-", color="blue", linewidth=2, markersize=8)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("En Iyi Fitness (Sharpe)", fontsize=12)
    ax.set_title(f"Parametre Duyarliligi: {param_name}", fontsize=14)
    ax.grid(True, alpha=0.3)

    slug = param_name.lower().replace(" ", "_").replace(".", "")
    path = save_path or os.path.join(FIGURES_DIR, f"sensitivity_{slug}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRADE DAGILIM HISTOGRAMI
# ═══════════════════════════════════════════════════════════════════════════════


def plot_trade_distribution(
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    save_path: str | None = None,
) -> None:
    """
    Tamamlanan islemlerin kar/zarar dagilim histogrami.

    Args:
        entry_prices: Giris fiyatlari.
        exit_prices: Cikis fiyatlari.
        commission_rate: Komisyon orani.
        slippage_rate: Kayma orani.
        save_path: Kaydedilecek dosya yolu.
    """
    _ensure_dir()
    n_trades = min(len(entry_prices), len(exit_prices))
    if n_trades == 0:
        print("  Trade yok, histogram olusturulamadi.")
        return

    ep = entry_prices[:n_trades]
    xp = exit_prices[:n_trades]
    total_cost = commission_rate + slippage_rate
    pnl_pct = ((xp * (1 - total_cost)) / (ep * (1 + total_cost)) - 1) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["green" if p > 0 else "red" for p in pnl_pct]
    ax.bar(range(n_trades), pnl_pct, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Trade No", fontsize=12)
    ax.set_ylabel("Kar/Zarar (%)", fontsize=12)
    ax.set_title(f"Trade Kar/Zarar Dagilimi ({n_trades} trade)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    win_count = int(np.sum(pnl_pct > 0))
    ax.text(0.02, 0.95, f"Win: {win_count}/{n_trades} ({win_count/n_trades*100:.0f}%)",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    path = save_path or os.path.join(FIGURES_DIR, "trade_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BENCHMARK KARSILASTIRMA
# ═══════════════════════════════════════════════════════════════════════════════


def plot_benchmark_comparison(
    results: dict[str, dict],
    metric: str = "sharpe_ratio",
    save_path: str | None = None,
) -> None:
    """
    Benchmark stratejilerinin bar grafik karsilastirmasi.

    Args:
        results: {strateji_adi: {metric: value, ...}} seklinde sonuclar.
        metric: Karsilastirilacak metrik adi.
        save_path: Kaydedilecek dosya yolu.
    """
    _ensure_dir()
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    metric_label = {
        "sharpe_ratio": "Sharpe Ratio",
        "total_return": "Total Return (%)",
        "max_drawdown": "Max Drawdown (%)",
        "win_rate": "Win Rate (%)",
        "total_trades": "Trade Sayisi",
    }.get(metric, metric)

    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"Strateji Karsilastirmasi: {metric_label}", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.xticks(rotation=15)

    path = save_path or os.path.join(FIGURES_DIR, f"benchmark_{metric}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. KORELASYON MATRISI (Hall of Fame parametreleri)
# ═══════════════════════════════════════════════════════════════════════════════


def plot_correlation_matrix(
    hall_of_fame: list[dict],
    save_path: str | None = None,
) -> None:
    """
    Hall of Fame bireylerinin parametre korelasyon matrisi.

    Args:
        hall_of_fame: [{"individual": [...], "metrics": {...}}, ...] listesi.
        save_path: Kaydedilecek dosya yolu.
    """
    _ensure_dir()
    from src.ga.chromosome import GENE_NAMES, decode_chromosome

    rows = []
    for entry in hall_of_fame:
        ind = entry["individual"]
        params = decode_chromosome(ind)
        rows.append(params.__dict__)

    if len(rows) < 2:
        print("  Korelasyon matrisi icin en az 2 birey gerekli.")
        return

    df = pd.DataFrame(rows)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Korelasyon")
    ax.set_title("Optimal Parametreler Korelasyon Matrisi", fontsize=14)

    path = save_path or os.path.join(FIGURES_DIR, "correlation_matrix.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TOPLU CALISTIRMA
# ═══════════════════════════════════════════════════════════════════════════════


def generate_all_plots(
    logbook: pd.DataFrame | list[dict],
    equity_curve: np.ndarray,
    buy_and_hold: np.ndarray,
    close: np.ndarray,
    signals: np.ndarray,
    hall_of_fame: list[dict],
    benchmark_results: dict[str, dict] | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> None:
    """
    Tum grafikleri tek seferde olusturur.

    Args:
        logbook: Nesil istatistikleri.
        equity_curve: GA strateji equity curve.
        buy_and_hold: Buy & Hold equity curve.
        close: Kapanis fiyatlari.
        signals: Composite sinyal dizisi.
        hall_of_fame: Hall of Fame bireyleri.
        benchmark_results: Benchmark sonuclari (opsiyonel).
        dates: Tarih indeksi (opsiyonel).
    """
    print("\nGrafikler olusturuluyor...")

    # 1. Fitness evrimi
    plot_fitness_evolution(logbook)

    # 2. Equity curve
    plot_equity_curve(equity_curve, buy_and_hold, dates=dates)

    # 3. Alis/Satis sinyalleri
    plot_signals_on_price(close, signals, dates=dates)

    # 4. Drawdown
    plot_drawdown(equity_curve, dates=dates)

    # 5. Trade dagilimi -- entry/exit fiyatlarini sinyallerden cikar
    from src.strategy.backtest import _ffill_numpy
    raw_position = np.where(signals == 1, 1.0, np.where(signals == -1, 0.0, np.nan))
    raw_position = _ffill_numpy(raw_position, fill_value=0.0)
    raw_change = np.empty(len(raw_position), dtype=np.float64)
    raw_change[0] = raw_position[0]
    raw_change[1:] = raw_position[1:] - raw_position[:-1]
    entry_prices = close[raw_change == 1.0]
    exit_prices = close[raw_change == -1.0]
    plot_trade_distribution(entry_prices, exit_prices)

    # 6. Korelasyon matrisi
    plot_correlation_matrix(hall_of_fame)

    # 7. Benchmark karsilastirma
    if benchmark_results:
        plot_benchmark_comparison(benchmark_results, metric="sharpe_ratio")
        plot_benchmark_comparison(benchmark_results, metric="total_return")

    print("Tum grafikler olusturuldu.\n")
