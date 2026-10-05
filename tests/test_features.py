import numpy as np
import pandas as pd
import pytest

from bitcoin.features import FEATURE_COLUMNS, HALVING_COLUMNS, make_features


def monthly_frame():
    dates = pd.date_range("2019-12-31", "2024-06-30", freq="ME", name="date")
    close = np.arange(len(dates), dtype=float) + 100
    return pd.DataFrame(
        {"close": close, "high": close + 5, "low": close - 5, "volume_btc": 1000.0, "volatility": 0.1},
        index=dates,
    )


def test_features_do_not_change_when_future_candles_change():
    monthly = monthly_frame()
    cutoff = pd.Timestamp("2022-06-30")
    original = make_features(monthly)
    altered = monthly.copy()
    altered.loc[altered.index > cutoff] *= 5
    pd.testing.assert_frame_equal(original.loc[:cutoff], make_features(altered).loc[:cutoff])
    pd.testing.assert_frame_equal(original.loc[:cutoff], make_features(monthly.loc[:cutoff]))


def test_halving_changes_only_when_event_already_happened():
    features = make_features(monthly_frame())
    assert features.loc["2020-04-30", "block_reward_btc"] == 12.5
    assert features.loc["2020-05-31", "block_reward_btc"] == 6.25
    assert features.loc["2024-03-31", "block_reward_btc"] == 6.25
    assert features.loc["2024-04-30", "block_reward_btc"] == 3.125
    assert features.loc["2020-05-31", "months_since_halving"] == pytest.approx(20 / 30.4375)
    assert features.loc["2024-04-30", "months_since_halving"] == pytest.approx(10 / 30.4375)


def test_warmup_and_ablation_have_explicit_columns():
    monthly = monthly_frame()
    features = make_features(monthly)
    assert features.index.equals(monthly.index)
    assert features.dropna().index[0] == monthly.index[6]
    assert list(features.columns) == FEATURE_COLUMNS + HALVING_COLUMNS
    assert list(make_features(monthly, include_halving=False).columns) == FEATURE_COLUMNS


def test_missing_month_does_not_become_a_one_month_lag():
    monthly = monthly_frame().drop(pd.Timestamp("2021-01-31"))
    with pytest.raises(ValueError, match="consecutivos"):
        make_features(monthly)
