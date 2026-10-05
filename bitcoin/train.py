"""Treina horizontes mensais diretos, avalia no tempo e exporta evidências."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bitcoin.data import aggregate_monthly, read_daily
from bitcoin.features import HALVING_COLUMNS, make_features
from bitcoin.forecast import SCHEMA_VERSION, predict

ALPHAS = (1.0, 10.0, 100.0, 1000.0)
HORIZONS = (1, 2)
MIN_TRAIN_SAMPLES = 24
TEST_MONTHS = 12


def supervised(monthly: pd.DataFrame, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Amostra em t: entradas até t, rótulo log(close[t+h] / close[t])."""
    frame = features.copy()
    frame["base_close"] = monthly["close"]
    frame["target_close"] = monthly["close"].shift(-horizon)
    frame["target_time"] = (frame.index.to_period("M") + horizon).to_timestamp("M")
    frame["target_log_return"] = np.log(frame["target_close"] / frame["base_close"])
    return frame.dropna()


def available_training_rows(samples: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Inclui somente rótulos que já estavam observáveis no mês de origem."""
    return samples.loc[samples["target_time"] <= origin]


def tune_and_fit(samples: pd.DataFrame, columns: list[str], horizon: int):
    if len(samples) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"Treino requer pelo menos {MIN_TRAIN_SAMPLES} amostras; recebeu {len(samples)}.")
    # gap h-1 impede que o rótulo de uma linha de treino ultrapasse a origem
    # da validação. O scaler é ajustado dentro de cada fold, nunca na série toda.
    splitter = TimeSeriesSplit(n_splits=3, test_size=4, gap=horizon - 1)
    scores = {}
    for alpha in ALPHAS:
        errors = []
        for train_idx, validation_idx in splitter.split(samples):
            train, validation = samples.iloc[train_idx], samples.iloc[validation_idx]
            if train["target_time"].max() > validation.index.min():
                raise ValueError("Divisão temporal inválida: rótulo futuro entrou no treino.")
            pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipeline.fit(train[columns], train["target_log_return"])
            estimated = validation["base_close"].to_numpy() * np.exp(pipeline.predict(validation[columns]))
            errors.extend(np.abs(estimated - validation["target_close"].to_numpy()))
        scores[alpha] = float(np.mean(errors))
    best_alpha = min(scores, key=scores.get)
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    pipeline.fit(samples[columns], samples["target_log_return"])
    return pipeline, best_alpha, scores


def regression_metrics(actual: np.ndarray, estimated: np.ndarray, base: np.ndarray) -> dict:
    return {
        "mae_usd": round(float(mean_absolute_error(actual, estimated)), 2),
        "rmse_usd": round(float(np.sqrt(mean_squared_error(actual, estimated))), 2),
        "mape_pct": round(float(np.mean(np.abs((actual - estimated) / actual)) * 100), 2),
        "direction_accuracy_pct": round(float(np.mean(np.sign(actual - base) == np.sign(estimated - base))) * 100, 2),
    }


def evaluate(samples: pd.DataFrame, columns: list[str], horizon: int) -> pd.DataFrame:
    if len(samples) < MIN_TRAIN_SAMPLES + TEST_MONTHS + horizon - 1:
        raise ValueError("Histórico insuficiente para treino e teste cronológico de 12 meses.")
    without_halving = [name for name in columns if name not in HALVING_COLUMNS]
    rows = []
    for origin, sample in samples.tail(TEST_MONTHS).iterrows():
        historical = available_training_rows(samples, origin)
        with_model, alpha, _ = tune_and_fit(historical, columns, horizon)
        without_model, no_halving_alpha, _ = tune_and_fit(historical, without_halving, horizon)
        inputs = samples.loc[[origin]]
        base = float(sample["base_close"])
        rows.append({
            "horizon_months": horizon,
            "origin_month": origin.strftime("%Y-%m"),
            "target_month": sample["target_time"].strftime("%Y-%m"),
            "train_samples": len(historical),
            "last_training_target": historical["target_time"].max().strftime("%Y-%m"),
            "actual_close_usd": float(sample["target_close"]),
            "ridge_halving_usd": base * float(np.exp(with_model.predict(inputs[columns])[0])),
            "ridge_without_halving_usd": base * float(np.exp(without_model.predict(inputs[without_halving])[0])),
            "persistence_usd": base,
            "alpha": alpha,
            "alpha_without_halving": no_halving_alpha,
        })
    return pd.DataFrame(rows)


def draw_evaluation(monthly: pd.DataFrame, backtest: pd.DataFrame, forecast: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    axes[0].plot(monthly.index, monthly["close"], color="#172c50", label="Fechamento mensal observado")
    for event in ("2020-05-11", "2024-04-20"):
        axes[0].axvline(pd.Timestamp(event), color="#cf8b19", linestyle="--", alpha=0.8,
                        label="Halving" if event.startswith("2020") else None)
    target = pd.Timestamp(forecast["target_date"])
    point = forecast["predicted_close_usd"]
    band = forecast["empirical_error_band_usd"]
    axes[0].errorbar(target, point, yerr=[[point - band["lower"]], [band["upper"] - point]],
                     fmt="o", color="#bd4428", capsize=6, label="Previsão e faixa empírica de erro")
    axes[0].set(title=f"BTC/USD desde 2020 · previsão para {forecast['target_month']}", ylabel="USD")
    axes[0].legend(loc="upper left", fontsize=8)
    subset = backtest.loc[backtest["horizon_months"] == forecast["horizon_months"]]
    dates = pd.to_datetime(subset["target_month"])
    for column, label, color in (("actual_close_usd", "Observado", "#172c50"),
                                  ("ridge_halving_usd", "Ridge + halving", "#bd4428"),
                                  ("ridge_without_halving_usd", "Ridge sem halving", "#6b7d32"),
                                  ("persistence_usd", "Último preço (baseline)", "#888888")):
        axes[1].plot(dates, subset[column], marker=".", label=label, color=color)
    axes[1].set(title=f"Teste temporal · 12 meses · horizonte {forecast['horizon_months']} meses", ylabel="USD")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def train(data_path: str | Path, output: str | Path, as_of: str) -> dict:
    data_path, output = Path(data_path), Path(output)
    as_of_date = pd.Timestamp(as_of)
    if as_of_date != as_of_date.normalize() or as_of_date.tzinfo is not None:
        raise ValueError("Use --as-of como data YYYY-MM-DD, sem horário.")
    source_path = data_path.with_name("source.json")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
    if source is not None and source.get("snapshot_sha256") != digest:
        raise ValueError("O hash do CSV diverge de source.json. Baixe novamente os dados e sua procedência.")
    daily = read_daily(data_path)
    if daily.index[0] != pd.Timestamp("2020-01-01"):
        raise ValueError("O histórico deve começar em 2020-01-01.")
    monthly = aggregate_monthly(daily, as_of=as_of)
    features = make_features(monthly).dropna()
    if features.empty:
        raise ValueError("Histórico mensal insuficiente para construir as entradas.")
    origin = monthly.index[-1].to_period("M")
    target = as_of_date.to_period("M") + 1
    if target.ordinal - origin.ordinal not in HORIZONS:
        raise ValueError("CSV desatualizado: baixe os dados recentes ou use --as-of compatível com o snapshot. O modelo suporta 1 ou 2 meses após o último mês completo.")
    if features.index[-1] != monthly.index[-1]:
        raise ValueError("Entradas inválidas no último mês completo; verifique os dados.")
    columns = list(features.columns)
    metrics, models, errors, backtests, fit_details = {}, {}, {}, [], {}
    for horizon in HORIZONS:
        # Escolha por horizonte no desenvolvimento até agosto/2025:
        # retirar halving ajudou em 1 mês, mas piorou em 2 meses.
        selected_model = "ridge_without_halving" if horizon == 1 else "ridge_halving"
        training_columns = [name for name in columns if name not in HALVING_COLUMNS] if horizon == 1 else columns
        samples = supervised(monthly, features, horizon)
        backtest = evaluate(samples, columns, horizon)
        actual = backtest["actual_close_usd"].to_numpy()
        base = backtest["persistence_usd"].to_numpy()
        metrics[str(horizon)] = {
            "selected_model": selected_model,
            "test_samples": len(backtest),
            "first_target_month": backtest["target_month"].iloc[0],
            "last_target_month": backtest["target_month"].iloc[-1],
            "ridge_halving": regression_metrics(actual, backtest["ridge_halving_usd"].to_numpy(), base),
            "ridge_without_halving": regression_metrics(actual, backtest["ridge_without_halving_usd"].to_numpy(), base),
            "persistence": regression_metrics(actual, base, base),
        }
        pipeline, alpha, validation_scores = tune_and_fit(samples, training_columns, horizon)
        models[horizon] = pipeline
        selected_predictions = backtest[selected_model + "_usd"].to_numpy()
        errors[horizon] = float(np.quantile(np.abs(np.log(actual / selected_predictions)), 0.8, method="higher"))
        fit_details[str(horizon)] = {"training_samples": len(samples), "alpha": alpha,
                                     "feature_names": training_columns,
                                     "validation_mae_by_alpha": validation_scores,
                                     "last_training_target": samples["target_time"].max().strftime("%Y-%m")}
        backtests.append(backtest)
    metadata = {
        "pair": "BTC/USD", "target": "monthly_close", "currency": "USD",
        "as_of": as_of_date.date().isoformat(), "target_month": str(target),
        "last_complete_month": str(origin), "data_start": monthly.index[0].replace(day=1).date().isoformat(),
        "snapshot_last_daily_date": daily.index[-1].date().isoformat(),
        "complete_months": len(monthly), "feature_names": columns,
        "horizons": list(HORIZONS), "fit": fit_details,
        "csv_sha256": digest, "source": source,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "evaluation": "12 alvos finais já conhecidos: comparação retrospectiva, janela expansiva e ajuste interno cronológico; rótulos disponíveis na origem.",
        "model_selection": "Features por horizonte escolhidas no desenvolvimento até 2025-08: 1 mês sem halving, 2 meses com halving. A comparação final é retrospectiva, não um novo teste independente.",
    }
    artifact = {
        "schema_version": SCHEMA_VERSION, "metadata": metadata, "models": models,
        "feature_names": columns, "latest_features": {key: float(value) for key, value in features.iloc[-1].items()},
        "last_close": float(monthly["close"].iloc[-1]), "origin_month": str(origin),
        "target_month": str(target), "metrics": metrics, "error_quantiles": errors,
    }
    forecast = predict(artifact)
    output.mkdir(parents=True, exist_ok=True)
    # Substituição atômica evita a API observar um joblib parcialmente escrito.
    temporary = output / "model.joblib.tmp"
    joblib.dump(artifact, temporary)
    temporary.replace(output / "model.joblib")
    monthly.to_csv(output / "monthly.csv", index_label="date")
    backtest_frame = pd.concat(backtests, ignore_index=True)
    backtest_frame.to_csv(output / "backtest.csv", index=False)
    results = {"metadata": metadata, "metrics": metrics, "forecast": forecast}
    (output / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    draw_evaluation(monthly, backtest_frame, forecast, output / "evaluation.png")
    return forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/btcusd_daily.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).date().isoformat(), help="Data de referência UTC YYYY-MM-DD; ignora o mês em andamento.")
    args = parser.parse_args()
    try:
        result = train(args.data, args.output, args.as_of)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Falha no treinamento: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
