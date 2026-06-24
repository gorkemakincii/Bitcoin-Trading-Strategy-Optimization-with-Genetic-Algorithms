"""GA fitness fonksiyonlari ve OHLC backtesting entegrasyonu."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.ga.chromosome import (
    clone_individual,
    decode_chromosome,
    repair_individual,
    validate_individual,
)


def _get_strategy_imports():
    """Lazy import -- circular import onlemi."""
    from src.strategy.backtest import run_backtest_fast
    from src.strategy.signals import generate_signals_fast
    return generate_signals_fast, run_backtest_fast


DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_POSITION_SIZE = 0.95
DEFAULT_COMMISSION_RATE = 0.001
DEFAULT_SLIPPAGE_RATE = 0.0005
DEFAULT_MIN_TRADES = 1
DEFAULT_PENALTY = 1_000.0
FAILED_FITNESS = -1_000_000.0


def _as_1d_array(values, name: str) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Fitness icin {name} 1 boyutlu olmali.")
    if len(arr) < 2:
        raise ValueError(f"Fitness icin en az 2 {name} degeri gerekli.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Fitness verisi NaN veya sonsuz {name} iceremez.")
    return arr


def _extract_ohlc(
    data: pd.DataFrame | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if isinstance(data, pd.DataFrame):
        if "Close" not in data.columns:
            raise ValueError("Fitness icin DataFrame'de 'Close' sutunu bulunmali.")
        close = _as_1d_array(data["Close"], "kapanis fiyati")
        high = _as_1d_array(data["High"], "en yuksek fiyat") if "High" in data.columns else None
        low = _as_1d_array(data["Low"], "en dusuk fiyat") if "Low" in data.columns else None
        return close, high, low

    return _as_1d_array(data, "kapanis fiyati"), None, None


def _failed_metrics(reason: str) -> dict[str, Any]:
    return {
        "fitness_score": FAILED_FITNESS,
        "total_return": FAILED_FITNESS,
        "sharpe_ratio": FAILED_FITNESS,
        "max_drawdown": abs(FAILED_FITNESS),
        "max_drawdown_risk": abs(FAILED_FITNESS),
        "win_rate": 0.0,
        "total_trades": 0,
        "penalty": abs(FAILED_FITNESS),
        "is_penalized": True,
        "reason": reason,
    }


def evaluate_individual(
    individual: Sequence[float | int],
    data: pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    position_size: float = DEFAULT_POSITION_SIZE,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    min_trades: int = DEFAULT_MIN_TRADES,
    repair: bool = True,
    penalty: float = DEFAULT_PENALTY,
) -> dict[str, Any]:
    """Bir kromozomu sinyal + backtest hattindan gecirip metrikleri dondurur."""
    candidate = clone_individual(individual)
    validation_errors = validate_individual(candidate)
    if validation_errors and not repair:
        return _failed_metrics("; ".join(validation_errors))

    try:
        if repair:
            repair_individual(candidate)
        params = decode_chromosome(candidate, repair=False)
        generate_signals_fast, run_backtest_fast = _get_strategy_imports()
        close, high, low = _extract_ohlc(data)
        signals = generate_signals_fast(close, params)
        metrics = run_backtest_fast(
            close=close,
            signals=signals,
            high=high,
            low=low,
            initial_capital=initial_capital,
            position_size=position_size,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            stop_loss=params.stop_loss,
            take_profit=params.take_profit,
        )
    except Exception as exc:  # Fitness dongusunde hatali birey tum GA'yi durdurmasin.
        return _failed_metrics(str(exc))

    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    total_return = float(metrics.get("total_return", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    total_trades = int(metrics.get("total_trades", 0))

    if not np.isfinite([sharpe, total_return, max_drawdown]).all():
        return _failed_metrics("Backtest metrikleri sonlu degil.")

    constraint_penalty = penalty * len(validation_errors)
    trade_penalty = penalty * max(0, min_trades - total_trades)
    total_penalty = constraint_penalty + trade_penalty

    result = dict(metrics)
    result.update({
        "fitness_score": sharpe - total_penalty,
        "max_drawdown_risk": abs(max_drawdown),
        "penalty": total_penalty,
        "is_penalized": total_penalty > 0,
        "validation_errors": validation_errors,
        "repaired_individual": candidate,
        "params": params,
    })
    return result


def fitness_single(
    individual: Sequence[float | int],
    data: pd.DataFrame | Sequence[float] | np.ndarray,
    **kwargs: Any,
) -> tuple[float]:
    """Tek amacli fitness: cezalar uygulanmis Sharpe Ratio skorunu maksimize eder."""
    metrics = evaluate_individual(individual, data, **kwargs)
    return (float(metrics["fitness_score"]),)


def fitness_multi(
    individual: Sequence[float | int],
    data: pd.DataFrame | Sequence[float] | np.ndarray,
    **kwargs: Any,
) -> tuple[float, float, float]:
    """Cok amacli fitness: Sharpe ve getiriyi maksimize, drawdown riskini minimize eder."""
    metrics = evaluate_individual(individual, data, **kwargs)
    penalty = float(metrics.get("penalty", 0.0))
    return (
        float(metrics["sharpe_ratio"]) - penalty,
        float(metrics["total_return"]) - penalty,
        float(metrics["max_drawdown_risk"]) + penalty,
    )
