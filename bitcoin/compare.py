"""Compara hipóteses em um recorte antigo, sem alterar o modelo da API.

Execute: python -m bitcoin.compare. O relatório JSON aparece somente no terminal.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from bitcoin.data import aggregate_monthly, read_daily
from bitcoin.features import HALVING_COLUMNS, make_features
from bitcoin.train import ALPHAS, available_training_rows, regression_metrics, supervised

CUTOFF = "2025-08-31"
AS_OF = "2025-09-01"
BOOST_PARAMETERS = {
    "loss": "absolute_error", "max_leaf_nodes": 3, "min_samples_leaf": 10,
    "learning_rate": 0.05, "max_iter": 100, "l2_regularization": 10,
    "early_stopping": False, "random_state": 42,
}


def validation_candidates(samples, columns, horizon):
    """Mesmos folds para todas as opções; scaler aprende somente com o treino."""
    folds = list(TimeSeriesSplit(n_splits=3, test_size=4, gap=horizon - 1).split(samples))
    persistence_errors = []
    for train_idx, validation_idx in folds:
        train, validation = samples.iloc[train_idx], samples.iloc[validation_idx]
        if train["target_time"].max() > validation.index.min():
            raise ValueError("Rótulo futuro entrou no treinamento da validação.")
        persistence_errors.extend(abs(validation["base_close"] - validation["target_close"]))
    candidates = [{"model": "persistence", "validation_mae": float(np.mean(persistence_errors))}]
    for intercept in (True, False):
        for halving in (True, False):
            selected_columns = [name for name in columns if halving or name not in HALVING_COLUMNS]
            for alpha in ALPHAS:
                errors = []
                for train_idx, validation_idx in folds:
                    train, validation = samples.iloc[train_idx], samples.iloc[validation_idx]
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=intercept))
                    model.fit(train[selected_columns], train["target_log_return"])
                    prices = validation["base_close"].to_numpy() * np.exp(model.predict(validation[selected_columns]))
                    errors.extend(abs(prices - validation["target_close"].to_numpy()))
                candidates.append({"model": f"ridge_intercept_{intercept}_halving_{halving}",
                                   "intercept": intercept, "halving": halving, "alpha": alpha,
                                   "validation_mae": float(np.mean(errors))})
    return candidates


def select_models(candidates):
    """Empates preferem persistência; nas Ridges, intercepto e ordem de ALPHAS."""
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["model"], []).append(candidate)
    with_halving = [item for item in candidates if item.get("halving")]
    groups["A_alpha_and_intercept"] = with_halving
    groups["B_original_ridge_or_persistence"] = [candidates[0]] + [item for item in with_halving if item["intercept"]]
    groups["C_alpha_intercept_or_persistence"] = [candidates[0]] + with_halving
    return {name: min(options, key=lambda item: item["validation_mae"]) for name, options in groups.items()}


def estimate_price(samples, inputs, base, choice):
    if choice["model"] == "persistence":
        return base
    if choice["model"] == "hist_gradient_fixed":
        model = HistGradientBoostingRegressor(**BOOST_PARAMETERS)
        columns = list(inputs.columns)
    else:
        columns = [name for name in inputs if choice["halving"] or name not in HALVING_COLUMNS]
        model = make_pipeline(StandardScaler(), Ridge(alpha=choice["alpha"], fit_intercept=choice["intercept"]))
    model.fit(samples[columns], samples["target_log_return"])
    return float(base * np.exp(model.predict(inputs[columns])[0]))


def compare(data_path=Path("data/btcusd_daily.csv")):
    # Corte antes de agregar, criar features ou construir os rótulos futuros.
    daily = read_daily(data_path)
    monthly = aggregate_monthly(daily.loc[:CUTOFF], as_of=AS_OF)
    if monthly.empty or monthly.index[-1] != pd.Timestamp(CUTOFF):
        raise ValueError(f"O experimento requer meses completos até {CUTOFF}.")
    features = make_features(monthly).dropna()
    report = {
        "cutoff": CUTOFF, "as_of": AS_OF, "complete_months": len(monthly),
        "protocol": "12 alvos anteriores ao corte, treino expansivo; validação interna com 3 divisões de 4 e gap=h-1.",
        "note": "Experimento de desenvolvimento; não altera a API nem avalia o período posterior ao corte.",
        "alphas": list(ALPHAS), "boost_parameters": BOOST_PARAMETERS,
        "feature_names": list(features), "horizons": {},
    }
    # Evita centenas de threads em um experimento com poucas dezenas de linhas.
    with threadpool_limits(limits=1):
        for horizon in (1, 2):
            samples = supervised(monthly, features, horizon)
            if len(samples) < 24 + 12 + horizon - 1:
                raise ValueError("Histórico insuficiente para o experimento temporal.")
            rows = []
            for origin, sample in samples.tail(12).iterrows():
                historical = available_training_rows(samples, origin)
                choices = select_models(validation_candidates(historical, list(features), horizon))
                choices["hist_gradient_fixed"] = {"model": "hist_gradient_fixed"}
                for name, choice in choices.items():
                    selected = choice["model"] + (f"_alpha_{choice['alpha']}" if "alpha" in choice else "")
                    rows.append({"model": name, "selected": selected, "actual": float(sample["target_close"]),
                                 "base": float(sample["base_close"]),
                                 "predicted": estimate_price(historical, features.loc[[origin]], float(sample["base_close"]), choice)})
            frame = pd.DataFrame(rows)
            metrics = {}
            for name, group in frame.groupby("model", sort=False):
                metrics[name] = regression_metrics(group.actual.to_numpy(), group.predicted.to_numpy(), group.base.to_numpy())
                metrics[name]["selection_counts"] = group.selected.value_counts().to_dict()
            report["horizons"][str(horizon)] = {
                "test_samples": 12, "first_target_month": samples.tail(12)["target_time"].iloc[0].strftime("%Y-%m"),
                "last_target_month": samples["target_time"].iloc[-1].strftime("%Y-%m"), "metrics": metrics,
            }
    return report


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2, ensure_ascii=False, allow_nan=False))
