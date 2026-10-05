"""Verifica a avaliação temporal e a decisão de pesquisa, sem dados de mercado novos."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from bitcoin import research
from bitcoin.features import make_features
from bitcoin.train import available_training_rows, supervised, tune_and_fit


def monthly_history():
    dates = pd.date_range("2020-01-31", periods=81, freq="ME")
    phase = np.arange(len(dates))
    close = 10000 * np.exp(0.01 * phase + 0.2 * np.sin(phase / 3))
    return pd.DataFrame({"close": close, "high": close * 1.15, "low": close * 0.85,
                         "volume_btc": 1000 + phase * 5, "volatility": 0.03}, index=dates)


def test_quantile_uses_price_weights_and_simple_return():
    # Uma observação com preço de origem maior deve dominar a mediana ponderada.
    samples = pd.DataFrame({"x": [2., 2., 2.], "base_close": [100., 100., 1000.],
                            "target_close": [80., 110., 1300.]})
    samples["target_log_return"] = np.log(samples.target_close / samples.base_close)
    model = research.fit_model("quantile", 0.1, samples, ["x"])
    inputs = pd.DataFrame({"x": [2.], "base_close": [200.]})
    prices = research.estimate_prices(model, "quantile", inputs, ["x"])
    assert prices[0] == pytest.approx(260.)
    assert model.named_steps["standardscaler"].mean_[0] == 2.


@pytest.mark.parametrize("value", [-1., -1.1, np.nan])
def test_invalid_quantile_price_is_rejected_without_clipping(value):
    model = SimpleNamespace(predict=lambda inputs: np.array([value]))
    inputs = pd.DataFrame({"x": [1.], "base_close": [100.]})
    with pytest.raises(ValueError, match="finita e positiva"):
        research.estimate_prices(model, "quantile", inputs, ["x"])


def test_ridge_reference_matches_production_and_window_is_applied_before_cv():
    monthly = monthly_history()
    features = make_features(monthly).dropna()
    samples = supervised(monthly, features, 2)
    origin = samples.index[-13]
    historical = available_training_rows(samples, origin)
    inputs = samples.loc[[origin]]
    columns = list(features)
    for name, window in (("current_ridge", historical), ("ridge_window_36", historical.tail(36))):
        price, details = research.tune_predict(name, historical, inputs, columns, 2)
        expected, alpha, scores = tune_and_fit(window, columns, 2)
        assert price == pytest.approx(inputs.base_close.iloc[0] * np.exp(expected.predict(inputs[columns])[0]))
        assert details["selected_alpha"] == alpha
        assert details["training_samples"] == len(window)
        assert {item["alpha"]: item["validation_mae_usd"] for item in details["validation_scores"]} == pytest.approx(scores)

    contaminated = historical.copy()
    contaminated.loc[contaminated.index[-1], "target_time"] = origin + pd.offsets.MonthEnd(1)
    with pytest.raises(ValueError, match="Rótulo futuro"):
        research.tune_predict("current_ridge", contaminated, inputs, columns, 2)


def test_promotion_requires_global_gain_and_three_joint_block_wins():
    # Huber ganha na média, mas das DUAS referências em apenas dois blocos.
    errors = {
        "current_ridge": [10, 10, 10, 10, 50],
        "persistence": [20, 20, 20, 1, 40],
        "huber": [1, 1, 11, 2, 40],
        "quantile": [5, 5, 5, 5, 50],
    }
    rows = []
    for index, month in enumerate(pd.period_range("2023-03", periods=30, freq="M")):
        block = index // 6
        prices = {name: float(100 + errors.get(name, [100] * 5)[block]) for name in research.MODEL_NAMES}
        rows.append({"target_month": str(month), "actual_close_usd": 100.,
                     "base_close_usd": prices["persistence"], "predictions_usd": prices})
    summary = research.summarize(rows, research.MODEL_NAMES)
    selection = research.select_candidate(summary)
    assert selection["best_candidate"] == "huber"
    assert summary["huber"]["blocks_beating_both"] == 2
    assert selection["selected_model"] == "quantile"
    assert selection["qualified_candidates"] == ["quantile"]

    # Um mês inválido impede promover o candidato; os 29 restantes não o salvam.
    rows[0]["predictions_usd"]["quantile"] = None
    summary = research.summarize(rows, research.MODEL_NAMES)
    assert summary["quantile"]["status"] == "failed"
    assert summary["quantile"]["metrics"] is None
    assert research.select_candidate(summary)["selected_model"] == "current_ridge"


def test_failed_parameters_are_reported_and_never_selected(monkeypatch):
    monthly = monthly_history()
    features = make_features(monthly).dropna()
    samples = supervised(monthly, features, 2)
    inputs = samples.iloc[[-1]]
    historical = available_training_rows(samples, inputs.index[0])

    def sometimes_fails(name, alpha, train, columns):
        if alpha == 1:
            raise ConvergenceWarning("falha simulada")
        return SimpleNamespace(predict=lambda frame: np.zeros(len(frame)))

    monkeypatch.setattr(research, "fit_model", sometimes_fails)
    price, details = research.tune_predict("current_ridge", historical, inputs, list(features), 2)
    assert price == inputs.base_close.iloc[0]
    assert details["selected_alpha"] == 10
    assert details["failures"] == [{"alpha": 1., "phase": "validation", "error": "falha simulada"}]

    def always_fails(*args):
        raise ConvergenceWarning("falha simulada")

    monkeypatch.setattr(research, "fit_model", always_fails)
    price, details = research.tune_predict("current_ridge", historical, inputs, list(features), 2)
    assert price is None
    assert len(details["failures"]) == 4
    assert details["validation_scores"] == []


def test_development_ignores_future_and_does_not_write_files(monkeypatch, tmp_path):
    dates = pd.date_range("2020-01-01", "2026-10-04", freq="D")
    phase = np.arange(len(dates))
    close = 10000 * np.exp(0.0004 * phase + 0.15 * np.sin(phase / 60))
    daily = pd.DataFrame({"open": close, "high": close * 1.03, "low": close * .97,
                          "close": close, "volume_btc": 1000 + phase / 10}, index=dates)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(research, "read_daily", lambda path: daily)

    def fake_tuning(name, historical, inputs, columns, horizon):
        assert historical.target_time.max() <= inputs.index[0]
        assert len(columns) == {1: 8, 2: 10}[horizon]
        return float(inputs.base_close.iloc[0]), {}

    monkeypatch.setattr(research, "tune_predict", fake_tuning)
    before = research.research()
    daily.loc[daily.index > "2025-08-31", ["open", "high", "low", "close"]] *= 10
    after = research.research(retrospective=True)
    assert before["development"] == after["development"]
    for horizon in ("1", "2"):
        development = after["development"]["horizons"][horizon]
        assert development["test_samples"] == 30
        assert development["first_target_month"] == "2023-03"
        assert development["last_target_month"] == "2025-08"
        assert development["selection"]["selected_model"] == "current_ridge"
        assert after["retrospective"]["horizons"][horizon]["test_samples"] == 12
    assert list(tmp_path.iterdir()) == []
