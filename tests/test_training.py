"""Dados sintéticos somente nos testes; nunca usados como histórico real."""

import joblib
import json
import numpy as np
import pandas as pd
import pytest

from bitcoin.features import HALVING_COLUMNS, make_features
from bitcoin.forecast import load_artifact, predict
from bitcoin.train import available_training_rows, evaluate, supervised, train


def monthly_fixture():
    dates = pd.date_range("2020-01-31", periods=64, freq="ME")
    phase = np.arange(len(dates))
    close = 10000 * np.exp(0.01 * phase + 0.2 * np.sin(phase / 3))
    return pd.DataFrame({"close": close, "high": close * 1.15, "low": close * 0.85,
                         "volume_btc": 1000 + phase * 5, "volatility": 0.03}, index=dates)


def test_horizon_two_labels_available_only_after_target_month():
    monthly = monthly_fixture()
    samples = supervised(monthly, make_features(monthly), horizon=2)
    origin = monthly.index[40]
    past = available_training_rows(samples, origin)
    assert past.index.max() == monthly.index[38]
    assert past["target_time"].max() == origin
    assert past.iloc[-1]["target_close"] == monthly.loc[origin, "close"]
    assert samples.loc[origin, "target_close"] == monthly.iloc[42]["close"]


def test_future_prices_do_not_change_first_backtest_prediction():
    monthly = monthly_fixture()
    samples = supervised(monthly, make_features(monthly), horizon=2)
    columns = list(make_features(monthly).columns)
    first_origin = samples.tail(12).index[0]
    before = evaluate(samples, columns, horizon=2).iloc[0]
    changed = monthly.copy()
    changed.loc[changed.index > first_origin, ["close", "high", "low"]] *= 3
    changed_samples = supervised(changed, make_features(changed), horizon=2)
    after = evaluate(changed_samples, columns, horizon=2).iloc[0]
    assert before["actual_close_usd"] != after["actual_close_usd"]
    assert before["ridge_halving_usd"] == pytest.approx(after["ridge_halving_usd"])
    assert before["ridge_without_halving_usd"] == pytest.approx(after["ridge_without_halving_usd"])
    assert before["alpha"] == after["alpha"]
    assert before["alpha_without_halving"] == after["alpha_without_halving"]
    assert before["last_training_target"] == after["last_training_target"]
    assert before["last_training_target"] <= first_origin.strftime("%Y-%m")


def test_training_export_roundtrip_and_next_calendar_month(tmp_path):
    dates = pd.date_range("2020-01-01", "2025-06-04", freq="D")
    phase = np.arange(len(dates))
    close = 10000 * np.exp(0.0004 * phase + 0.15 * np.sin(phase / 60))
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": close,
                          "high": close * 1.03, "low": close * 0.97, "close": close,
                          "volume_btc": 1000 + phase / 10})
    data = tmp_path / "synthetic_test_only.csv"
    frame.to_csv(data, index=False)
    output = tmp_path / "artifacts"
    result = train(data, output, as_of="2025-06-05")
    artifact = load_artifact(output / "model.joblib")
    assert predict(artifact) == result
    report = json.loads((output / "results.json").read_text())
    assert set(report) == {"metadata", "metrics", "forecast"}
    assert report["forecast"] == result
    assert report["metrics"] == artifact["metrics"]
    assert report["metadata"] == json.loads(json.dumps(artifact["metadata"]))
    assert {path.name for path in output.iterdir()} == {
        "model.joblib", "results.json", "monthly.csv", "backtest.csv", "evaluation.png"
    }
    assert result["last_complete_month"] == "2025-05"
    assert result["target_month"] == "2025-07"
    assert result["horizon_months"] == 2
    first_month = predict(artifact, "2025-06")
    assert first_month["horizon_months"] == 1
    with pytest.raises(ValueError, match="não suportado"):
        predict(artifact, "2025-08")
    audit = pd.read_csv(output / "backtest.csv")
    assert (audit["last_training_target"] <= audit["origin_month"]).all()
    assert len(audit) == 24
    for horizon, prediction, selected in (
        (1, first_month, "ridge_without_halving"),
        (2, result, "ridge_halving"),
    ):
        pipeline = artifact["models"][horizon]
        expected_columns = [name for name in artifact["feature_names"]
                            if horizon == 2 or name not in HALVING_COLUMNS]
        assert len(expected_columns) == {1: 8, 2: 10}[horizon]
        assert list(pipeline.feature_names_in_) == expected_columns
        fit = artifact["metadata"]["fit"][str(horizon)]
        assert fit["feature_names"] == expected_columns
        assert fit["last_training_target"] == "2025-05"
        metrics = artifact["metrics"][str(horizon)]
        assert metrics["selected_model"] == selected
        assert {"ridge_halving", "ridge_without_halving", "persistence"} <= metrics.keys()

        history = audit.loc[audit["horizon_months"] == horizon]
        expected_error = np.quantile(
            np.abs(np.log(history["actual_close_usd"] / history[selected + "_usd"])),
            0.8, method="higher",
        )
        assert artifact["error_quantiles"][horizon] == pytest.approx(expected_error)
        inputs = pd.DataFrame([artifact["latest_features"]], columns=expected_columns)
        log_return = pipeline.predict(inputs)[0]
        band = prediction["empirical_error_band_usd"]
        assert band["lower"] == round(artifact["last_close"] * np.exp(log_return - expected_error), 2)
        assert band["upper"] == round(artifact["last_close"] * np.exp(log_return + expected_error), 2)
        assert set(prediction["halving_features"]) == (set(HALVING_COLUMNS) if horizon == 2 else set())
        assert ("com halving" if horizon == 2 else "sem halving") in prediction["model"]

    # O artefato ainda guarda dez features, mas h1 não deve consumir as de halving.
    changed = {**artifact, "latest_features": {
        **artifact["latest_features"], "months_since_halving": 10000.0, "block_reward_btc": 1000.0,
    }}
    assert predict(changed, "2025-06") == first_month
    assert (output / "evaluation.png").stat().st_size > 1000
    with pytest.raises(ValueError, match="desatualizado"):
        train(data, output, as_of="2025-10-05")
    (tmp_path / "source.json").write_text('{"snapshot_sha256": "incorrect"}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash do CSV"):
        train(data, output, as_of="2025-06-05")


def test_rejects_unknown_artifact_format(tmp_path):
    path = tmp_path / "invalid.joblib"
    joblib.dump({"schema_version": 0}, path)
    with pytest.raises(ValueError, match="incompatível"):
        load_artifact(path)
