"""Causal monthly inputs shared by training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Actual UTC halving dates, not a predicted date for the next halving.
# https://en.bitcoin.it/wiki/Controlled_supply
HALVING_EVENTS = (
    ("2012-11-28", 25.0),
    ("2016-07-09", 12.5),
    ("2020-05-11", 6.25),
    ("2024-04-20", 3.125),
)
HALVING_COLUMNS = ["months_since_halving", "block_reward_btc"]
FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_6m",
    "price_to_ma3",
    "price_to_ma6",
    "range_fraction",
    "realized_volatility",
    "volume_change_1m",
]


def make_features(monthly: pd.DataFrame, include_halving: bool = True) -> pd.DataFrame:
    """Return inputs available at each month end, with warmup NaNs preserved.

    A row at month t only uses candles at or before t. The first six rows have
    incomplete lag history and must be excluded from training. The caller adds
    the future target independently; target months are never read here.
    """
    if not isinstance(monthly.index, pd.DatetimeIndex):
        raise ValueError("O índice mensal deve conter datas.")
    if monthly.empty or not monthly.index.is_monotonic_increasing or monthly.index.has_duplicates:
        raise ValueError("Os meses devem ser únicos, ordenados e não vazios.")
    periods = monthly.index.to_period("M")
    if not monthly.index.is_month_end.all() or not periods.equals(
        pd.period_range(periods[0], periods[-1], freq="M")
    ):
        raise ValueError("As entradas devem ser meses completos e consecutivos.")
    close = monthly["close"]
    result = pd.DataFrame(index=monthly.index)
    for months in (1, 3, 6):
        result[f"return_{months}m"] = close.pct_change(months, fill_method=None)
    for window in (3, 6):
        result[f"price_to_ma{window}"] = close / close.rolling(window).mean() - 1
    result["range_fraction"] = (monthly["high"] - monthly["low"]) / close
    result["realized_volatility"] = monthly["volatility"]
    result["volume_change_1m"] = np.log1p(monthly["volume_btc"]).diff()
    if include_halving:
        event_dates = pd.to_datetime([event[0] for event in HALVING_EVENTS])
        event_rewards = np.array([event[1] for event in HALVING_EVENTS])
        latest = event_dates.searchsorted(monthly.index, side="right") - 1
        if (latest < 0).any():
            raise ValueError("Histórico de halving disponível somente desde 2012-11-28.")
        result["months_since_halving"] = (
            monthly.index - event_dates[latest]
        ).days / 30.4375
        result["block_reward_btc"] = event_rewards[latest]
    return result[FEATURE_COLUMNS + (HALVING_COLUMNS if include_halving else [])]
