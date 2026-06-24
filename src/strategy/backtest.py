"""
Stateful backtest module for the GA-based BTC trading project.

The engine is long-only and keeps the existing anti look-ahead rule:
signals from day t-1 are applied on day t. When High/Low data is available,
open positions can be closed by stop-loss or take-profit before a signal exit.
If both risk levels are touched on the same candle, stop-loss is applied first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ffill_numpy(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Forward-fill NaN values in a one-dimensional numpy array."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr.copy()

    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    result = arr[idx]

    first_valid = np.argmax(~mask)
    if first_valid > 0:
        result[:first_valid] = fill_value

    return result


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "Composite_Signal",
    initial_capital: float = 10_000.0,
    position_size: float = 0.95,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> dict:
    """
    Run a long-only backtest from a DataFrame.

    High/Low columns are used for intraday stop-loss and take-profit checks
    when available. If they are missing, Close is used as a backward-compatible
    fallback.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame'de 'Close' sutunu bulunamadi.")
    if signal_col not in df.columns:
        raise ValueError(f"DataFrame'de '{signal_col}' sinyal sutunu bulunamadi.")

    close = _as_1d_array(df["Close"], "Close")
    high = _as_1d_array(df["High"], "High") if "High" in df.columns else close
    low = _as_1d_array(df["Low"], "Low") if "Low" in df.columns else close
    signals = _as_1d_array(df[signal_col], signal_col)

    result = _backtest_core(
        close=close,
        signals=signals,
        high=high,
        low=low,
        initial_capital=initial_capital,
        position_size=position_size,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    result["equity_curve"] = pd.Series(
        result["equity_curve"], index=df.index, name="Equity"
    )
    result["buy_and_hold"] = pd.Series(
        result["buy_and_hold"], index=df.index, name="Buy_and_Hold"
    )
    return result


def run_backtest_fast(
    close: np.ndarray,
    signals: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    initial_capital: float = 10_000.0,
    position_size: float = 0.95,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> dict:
    """
    Run the same backtest from numpy arrays.

    Passing only close/signals remains supported. In that mode, Close is used
    as the High/Low fallback, so stop-loss/take-profit can still work on a
    close-only approximation.
    """
    close_arr = _as_1d_array(close, "close")
    return _backtest_core(
        close=close_arr,
        signals=_as_1d_array(signals, "signals"),
        high=_as_1d_array(high, "high") if high is not None else close_arr,
        low=_as_1d_array(low, "low") if low is not None else close_arr,
        initial_capital=initial_capital,
        position_size=position_size,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def _as_1d_array(values, name: str) -> np.ndarray:
    """Convert Series/DataFrame/array-like input to a finite 1-D float array."""
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} 1 boyutlu olmali.")
    if len(arr) == 0:
        raise ValueError(f"{name} bos olamaz.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} NaN veya sonsuz deger iceremez.")
    return arr


def _validate_core_inputs(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signals: np.ndarray,
    initial_capital: float,
    position_size: float,
    commission_rate: float,
    slippage_rate: float,
    stop_loss: float | None,
    take_profit: float | None,
) -> None:
    n = len(close)
    if len(high) != n or len(low) != n or len(signals) != n:
        raise ValueError("close, high, low ve signals uzunluklari ayni olmali.")
    if initial_capital <= 0:
        raise ValueError("initial_capital pozitif olmali.")
    if not (0.0 <= position_size <= 1.0):
        raise ValueError("position_size 0 ile 1 arasinda olmali.")
    if commission_rate < 0 or slippage_rate < 0:
        raise ValueError("Komisyon ve slippage negatif olamaz.")
    if stop_loss is not None and stop_loss <= 0:
        raise ValueError("stop_loss pozitif olmali.")
    if take_profit is not None and take_profit <= 0:
        raise ValueError("take_profit pozitif olmali.")


def _backtest_core(
    close: np.ndarray,
    signals: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    initial_capital: float,
    position_size: float,
    commission_rate: float,
    slippage_rate: float,
    stop_loss: float | None,
    take_profit: float | None,
) -> dict:
    """
    Stateful long-only backtest.

    A signal on index t-1 is applied to the return path of index t. Entry and
    signal-exit prices use the previous close. Stop-loss and take-profit exits
    use their threshold prices and are checked with the current candle's Low
    and High.
    """
    close = _as_1d_array(close, "close")
    high = _as_1d_array(high, "high")
    low = _as_1d_array(low, "low")
    signals = np.nan_to_num(_as_1d_array(signals, "signals"), nan=0.0)
    _validate_core_inputs(
        close,
        high,
        low,
        signals,
        initial_capital,
        position_size,
        commission_rate,
        slippage_rate,
        stop_loss,
        take_profit,
    )

    n = len(close)
    total_cost = commission_rate + slippage_rate
    net_return = np.zeros(n, dtype=np.float64)
    equity = np.empty(n, dtype=np.float64)
    equity[0] = initial_capital

    in_position = False
    entry_price = 0.0
    trade_returns: list[float] = []
    stop_loss_exits = 0
    take_profit_exits = 0
    signal_exits = 0

    for t in range(1, n):
        prev_close = close[t - 1]
        day_return = 0.0
        day_cost = 0.0
        signal = signals[t - 1]

        if in_position and signal == -1:
            exit_price = prev_close
            day_cost += total_cost * position_size
            trade_returns.append(_trade_return(entry_price, exit_price, total_cost))
            signal_exits += 1
            in_position = False
            entry_price = 0.0

        if not in_position and signal == 1:
            entry_price = prev_close
            day_cost += total_cost * position_size
            in_position = True

        if in_position:
            exit_price = None
            if stop_loss is not None and low[t] <= entry_price * (1.0 - stop_loss):
                exit_price = entry_price * (1.0 - stop_loss)
                stop_loss_exits += 1
            elif take_profit is not None and high[t] >= entry_price * (1.0 + take_profit):
                exit_price = entry_price * (1.0 + take_profit)
                take_profit_exits += 1

            if exit_price is not None:
                day_return = (exit_price / prev_close - 1.0) * position_size
                day_cost += total_cost * position_size
                trade_returns.append(_trade_return(entry_price, exit_price, total_cost))
                in_position = False
                entry_price = 0.0
            else:
                day_return = (close[t] / prev_close - 1.0) * position_size

        net_return[t] = day_return - day_cost
        equity[t] = equity[t - 1] * (1.0 + net_return[t])

    buy_and_hold = _buy_and_hold_equity(close, initial_capital, position_size)
    total_return = (
        (equity[-1] / initial_capital - 1.0) * 100.0
        if np.isfinite(equity[-1])
        else 0.0
    )
    sharpe_ratio = _sharpe_ratio(net_return)
    max_drawdown = _max_drawdown(equity)
    total_trades = len(trade_returns)
    win_rate = (
        float(np.sum(np.asarray(trade_returns) > 0.0)) / total_trades * 100.0
        if total_trades > 0
        else 0.0
    )

    return {
        "total_return": round(float(total_return), 4),
        "sharpe_ratio": round(float(sharpe_ratio), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "win_rate": round(float(win_rate), 2),
        "total_trades": int(total_trades),
        "stop_loss_exits": int(stop_loss_exits),
        "take_profit_exits": int(take_profit_exits),
        "signal_exits": int(signal_exits),
        "equity_curve": equity,
        "buy_and_hold": buy_and_hold,
    }


def _trade_return(entry_price: float, exit_price: float, total_cost: float) -> float:
    return (exit_price * (1.0 - total_cost)) / (entry_price * (1.0 + total_cost)) - 1.0


def _buy_and_hold_equity(
    close: np.ndarray,
    initial_capital: float,
    position_size: float,
) -> np.ndarray:
    daily_return = np.zeros(len(close), dtype=np.float64)
    daily_return[1:] = (close[1:] - close[:-1]) / close[:-1] * position_size
    return initial_capital * np.cumprod(1.0 + daily_return)


def _sharpe_ratio(net_return: np.ndarray) -> float:
    finite_returns = net_return[np.isfinite(net_return)]
    if len(finite_returns) <= 1:
        return 0.0
    std_ret = np.std(finite_returns, ddof=1)
    if std_ret <= 1e-12:
        return 0.0
    return float(np.mean(finite_returns) / std_ret * np.sqrt(365.0))


def _max_drawdown(equity: np.ndarray) -> float:
    valid_equity = np.where(np.isfinite(equity), equity, equity[0])
    peak = np.maximum.accumulate(valid_equity)
    drawdown = np.where(peak > 0, (valid_equity - peak) / peak, 0.0)
    return float(np.min(drawdown)) * 100.0
