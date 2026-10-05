"""O experimento antigo respeita o corte sem gravar artefatos da API."""

import numpy as np
import pandas as pd

from bitcoin import compare


def test_comparison_ignores_future_data_and_creates_no_files(monkeypatch, tmp_path):
    dates = pd.date_range("2020-01-01", "2026-10-04", freq="D")
    phase = np.arange(len(dates))
    close = 10000 * np.exp(0.0004 * phase + 0.15 * np.sin(phase / 60))
    daily = pd.DataFrame({"open": close, "high": close * 1.03, "low": close * 0.97,
                          "close": close, "volume_btc": 1000 + phase / 10}, index=dates)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare, "read_daily", lambda path: daily)
    aggregate = compare.aggregate_monthly
    cutoffs = []

    def aggregate_before_features(frame, as_of):
        cutoffs.append(frame.index.max())
        return aggregate(frame, as_of=as_of)

    def candidates(samples, columns, horizon):
        return [
            {"model": "persistence", "validation_mae": 1.0},
            {"model": "ridge_intercept_True_halving_True", "intercept": True,
             "halving": True, "alpha": 1.0, "validation_mae": 2.0},
        ]

    def estimate(samples, inputs, base, choice):
        assert samples["target_time"].max() <= inputs.index[0] <= pd.Timestamp("2025-08-31")
        return base

    monkeypatch.setattr(compare, "aggregate_monthly", aggregate_before_features)
    monkeypatch.setattr(compare, "validation_candidates", candidates)
    monkeypatch.setattr(compare, "estimate_price", estimate)

    before = compare.compare()
    daily.loc[daily.index > "2025-08-31", ["open", "high", "low", "close"]] *= 10
    after = compare.compare()

    assert before == after
    assert cutoffs == [pd.Timestamp("2025-08-31"), pd.Timestamp("2025-08-31")]
    for horizon in ("1", "2"):
        assert before["horizons"][horizon]["last_target_month"] == "2025-08"
        assert before["horizons"][horizon]["test_samples"] == 12
    assert list(tmp_path.iterdir()) == []
