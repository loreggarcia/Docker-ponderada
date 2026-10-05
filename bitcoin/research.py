"""Pesquisa temporal exploratória: compara oito abordagens sem alterar a API.

Execute ``python -m bitcoin.research``; ``--retrospective`` também confere o
período posterior, depois de fixar a escolha no desenvolvimento. Só imprime JSON.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, QuantileRegressor, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from bitcoin.data import aggregate_monthly, read_daily
from bitcoin.features import HALVING_COLUMNS, make_features
from bitcoin.train import ALPHAS, MIN_TRAIN_SAMPLES, available_training_rows, regression_metrics, supervised

CUTOFF = "2025-08-31"
AS_OF = "2025-09-01"
FIRST_TARGET = "2023-03"
LAST_TARGET = "2025-08"
MODEL_NAMES = (
    "current_ridge", "persistence", "ridge_window_36", "ridge_persistence_50_50",
    "huber", "elastic_net", "bayesian_ridge", "quantile",
)
PARAMETERS = {
    "current_ridge": {"alpha": list(ALPHAS)},
    "persistence": {},
    "ridge_window_36": {"alpha": list(ALPHAS), "training_window": 36},
    "ridge_persistence_50_50": {"ridge_price_weight": 0.5},
    "huber": {"alpha": [0.1, 1.0, 10.0], "epsilon": 1.35, "max_iter": 2000, "tol": 1e-7},
    "elastic_net": {"alpha": [0.001, 0.01, 0.1], "l1_ratio": 0.5, "max_iter": 10000, "tol": 1e-7},
    "bayesian_ridge": {"parameters": "scikit-learn defaults"},
    "quantile": {"alpha": [0.001, 0.01, 0.1], "quantile": 0.5, "solver": "highs"},
}


def fit_model(name, alpha, samples, columns):
    """Scaler e pesos aprendem somente com as linhas recebidas neste treino."""
    if name in ("current_ridge", "ridge_window_36"):
        estimator = Ridge(alpha=alpha)
    elif name == "huber":
        estimator = HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=2000, tol=1e-7)
    elif name == "elastic_net":
        estimator = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000, tol=1e-7)
    elif name == "bayesian_ridge":
        estimator = BayesianRidge()
    elif name == "quantile":
        estimator = QuantileRegressor(alpha=alpha, quantile=0.5, solver="highs")
    else:
        raise ValueError(f"Modelo desconhecido: {name}.")
    model = make_pipeline(StandardScaler(), estimator)
    target = samples["target_log_return"]
    fit_options = {}
    if name == "quantile":
        # |preço real - preço previsto| = preço de origem * |retorno real - previsto|.
        target = samples["target_close"] / samples["base_close"] - 1
        fit_options["quantileregressor__sample_weight"] = samples["base_close"] / samples["base_close"].mean()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(samples[columns], target, **fit_options)
    return model


def estimate_prices(model, name, inputs, columns):
    """Converte o retorno em preço; resultados inválidos falham sem clipping."""
    returns = model.predict(inputs[columns])
    with np.errstate(over="raise", invalid="raise"):
        multiplier = 1 + returns if name == "quantile" else np.exp(returns)
        prices = inputs["base_close"].to_numpy() * multiplier
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("A previsão de preço deve ser finita e positiva.")
    return prices


def tune_predict(name, historical, inputs, columns, horizon):
    """Escolhe alpha por MAE em dólares, com três validações cronológicas."""
    samples = historical.tail(36) if name == "ridge_window_36" else historical
    if len(samples) < MIN_TRAIN_SAMPLES:
        raise ValueError("Histórico insuficiente para a validação temporal.")
    folds = list(TimeSeriesSplit(n_splits=3, test_size=4, gap=horizon - 1).split(samples))
    for train_idx, validation_idx in folds:
        if samples.iloc[train_idx]["target_time"].max() > samples.iloc[validation_idx].index.min():
            raise ValueError("Rótulo futuro entrou no treinamento da validação.")
    if samples["target_time"].max() > inputs.index.min():
        raise ValueError("Rótulo futuro entrou no treinamento da previsão.")
    scores, failures = [], []
    for alpha in PARAMETERS[name].get("alpha", [None]):
        phase = "validation"
        try:
            errors = []
            for train_idx, validation_idx in folds:
                train, validation = samples.iloc[train_idx], samples.iloc[validation_idx]
                model = fit_model(name, alpha, train, columns)
                prices = estimate_prices(model, name, validation, columns)
                errors.extend(np.abs(prices - validation["target_close"].to_numpy()))
            score = float(np.mean(errors))
            if not np.isfinite(score):
                raise ValueError("O MAE de validação deve ser finito.")
            phase = "fit_and_predict"
            model = fit_model(name, alpha, samples, columns)
            price = float(estimate_prices(model, name, inputs, columns)[0])
            scores.append({"alpha": alpha, "validation_mae_usd": score, "price": price})
        except (ConvergenceWarning, ValueError, FloatingPointError, ArithmeticError) as exc:
            failures.append({"alpha": alpha, "phase": phase, "error": str(exc)})
    details = {
        "training_samples": len(samples),
        "last_training_target": samples["target_time"].max().strftime("%Y-%m"),
        "validation_scores": [{key: value for key, value in row.items() if key != "price"} for row in scores],
        "failures": failures,
    }
    if not scores:
        return None, details
    best = min(scores, key=lambda row: row["validation_mae_usd"])
    details["selected_alpha"] = best["alpha"]
    return best["price"], details


def summarize(rows, names):
    """Um mês com falha invalida o modelo no período inteiro, sem excluir erros."""
    actual = np.array([row["actual_close_usd"] for row in rows])
    base = np.array([row["base_close_usd"] for row in rows])
    persistence_mae = float(np.mean(np.abs(actual - base)))
    current = [row["predictions_usd"]["current_ridge"] for row in rows]
    current_valid = all(price is not None for price in current)
    result = {}
    for name in names:
        prices = [row["predictions_usd"][name] for row in rows]
        if any(price is None for price in prices):
            result[name] = {"status": "failed", "metrics": None, "blocks": [],
                            "failed_target_months": [row["target_month"] for row in rows if row["predictions_usd"][name] is None]}
            continue
        prices = np.array(prices)
        mae = float(np.mean(np.abs(actual - prices)))
        blocks = []
        for start in range(0, len(rows), 6):
            stop = start + 6
            block_mae = float(np.mean(np.abs(actual[start:stop] - prices[start:stop])))
            persistence_block = float(np.mean(np.abs(actual[start:stop] - base[start:stop])))
            current_block = float(np.mean(np.abs(actual[start:stop] - np.array(current[start:stop])))) if current_valid else None
            blocks.append({
                "first_target_month": rows[start]["target_month"],
                "last_target_month": rows[min(stop, len(rows)) - 1]["target_month"],
                "mae_usd": block_mae, "persistence_mae_usd": persistence_block,
                "current_ridge_mae_usd": current_block,
                "mae_over_persistence": block_mae / persistence_block if persistence_block > 0 else None,
                "beats_both": bool(current_block is not None and block_mae < current_block and block_mae < persistence_block),
            })
        result[name] = {
            "status": "ok", "metrics": regression_metrics(actual, prices, base),
            "mae_usd_unrounded": mae,
            "mae_over_persistence": mae / persistence_mae if persistence_mae > 0 else None,
            "blocks": blocks, "blocks_beating_both": sum(block["beats_both"] for block in blocks),
        }
    return result


def select_candidate(summary):
    """Regra congelada: ganhar das duas referências no total e em 3/5 blocos."""
    alternatives = [name for name in MODEL_NAMES[2:] if summary[name]["status"] == "ok"]
    best = min(alternatives, key=lambda name: summary[name]["mae_usd_unrounded"]) if alternatives else None
    qualified = []
    if summary["current_ridge"]["status"] == "ok":
        for name in alternatives:
            candidate = summary[name]
            if (len(candidate["blocks"]) == 5
                    and candidate["mae_usd_unrounded"] < summary["current_ridge"]["mae_usd_unrounded"]
                    and candidate["mae_usd_unrounded"] < summary["persistence"]["mae_usd_unrounded"]
                    and candidate["blocks_beating_both"] >= 3):
                qualified.append(name)
    selected = min(qualified, key=lambda name: summary[name]["mae_usd_unrounded"]) if qualified else "current_ridge"
    return {"selected_model": selected, "best_candidate": best, "qualified_candidates": qualified,
            "note": "Escolha exploratória de desenvolvimento; este comando não modifica a produção."}


def evaluate_period(monthly, features, first_target, last_target, names_by_horizon):
    """Todas as abordagens recebem os mesmos meses, alvos e entradas por horizonte."""
    expected = pd.period_range(first_target, last_target, freq="M").astype(str).tolist()
    report = {"horizons": {}, "predictions": []}
    for horizon in (1, 2):
        names = names_by_horizon[horizon]
        columns = [name for name in features if horizon == 2 or name not in HALVING_COLUMNS]
        samples = supervised(monthly, features, horizon)
        targets = samples.loc[samples["target_time"].dt.strftime("%Y-%m").isin(expected)]
        if targets["target_time"].dt.strftime("%Y-%m").tolist() != expected:
            raise ValueError("O período deve conter todos os meses previstos no protocolo.")
        rows = []
        for origin, sample in targets.iterrows():
            historical = available_training_rows(samples, origin)
            inputs = samples.loc[[origin]]
            predictions, fits = {}, {}
            for name in names:
                if name == "persistence":
                    predictions[name] = float(sample["base_close"])
                elif name == "ridge_persistence_50_50":
                    price = predictions["current_ridge"]
                    predictions[name] = 0.5 * price + 0.5 * float(sample["base_close"]) if price is not None else None
                    fits[name] = {"reuses": "current_ridge", "ridge_price_weight": 0.5}
                else:
                    predictions[name], fits[name] = tune_predict(name, historical, inputs, columns, horizon)
            rows.append({"horizon_months": horizon, "origin_month": origin.strftime("%Y-%m"),
                         "target_month": sample["target_time"].strftime("%Y-%m"),
                         "base_close_usd": float(sample["base_close"]),
                         "actual_close_usd": float(sample["target_close"]),
                         "predictions_usd": predictions, "fit": fits})
        report["horizons"][str(horizon)] = {"test_samples": len(rows), "feature_names": columns,
            "first_target_month": first_target, "last_target_month": last_target, "models": summarize(rows, names)}
        report["predictions"].extend(rows)
    return report


def research(data_path=Path("data/btcusd_daily.csv"), retrospective=False):
    daily = read_daily(data_path)
    # Nenhum dado posterior entra na agregação, nas features ou nos rótulos de desenvolvimento.
    monthly = aggregate_monthly(daily.loc[:CUTOFF], as_of=AS_OF)
    if monthly.empty or monthly.index[-1] != pd.Timestamp(CUTOFF):
        raise ValueError(f"O desenvolvimento requer meses completos até {CUTOFF}.")
    features = make_features(monthly).dropna()
    report = {
        "cutoff": CUTOFF, "as_of": AS_OF, "parameters": PARAMETERS,
        "protocol": {"first_target_month": FIRST_TARGET, "last_target_month": LAST_TARGET,
            "development_months": 30, "block_months": 6, "blocks": 5,
            "validation_splits": 3, "validation_months": 4, "validation_gap": "horizon - 1",
            "selection": "MAE estritamente menor que produção e persistência no total e simultaneamente em pelo menos 3/5 blocos; desempate pela ordem declarada.",
            "features": "Mesmos atributos da produção por horizonte: 8 em um mês; 10 em dois meses.",
            "quantile_target": "Retorno simples, pesos = preço de origem / média dos preços de origem do treino.",
            "other_targets": "Retorno logarítmico; mistura 50/50 aritmética dos preços.",
            "failure_policy": "Warning de convergência ou preço inválido descarta o parâmetro; sem opção válida, modelo falha no período inteiro.",
            "note": "Pesquisa exploratória: este histórico já foi reutilizado. A comparação retrospectiva não é teste independente e não muda a seleção. Não altera artefatos nem a API."},
    }
    with threadpool_limits(limits=1):
        development = evaluate_period(monthly, features, FIRST_TARGET, LAST_TARGET, {1: MODEL_NAMES, 2: MODEL_NAMES})
        for horizon in (1, 2):
            summary = development["horizons"][str(horizon)]
            summary["selection"] = select_candidate(summary["models"])
        report["development"] = development
        if retrospective:
            # As escolhas acima estão fixadas antes de ler os meses posteriores para avaliação.
            names = {}
            for horizon in (1, 2):
                best = development["horizons"][str(horizon)]["selection"]["best_candidate"]
                names[horizon] = ["current_ridge", "persistence"] + ([best] if best is not None else [])
            recent_monthly = aggregate_monthly(daily.loc[:"2026-10-04"], as_of="2026-10-05")
            recent_features = make_features(recent_monthly).dropna()
            report["retrospective"] = evaluate_period(recent_monthly, recent_features, "2025-10", "2026-09", names)
            report["retrospective"]["note"] = "Somente produção, persistência e best_candidate do desenvolvimento. Não participa da seleção."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/btcusd_daily.csv")
    parser.add_argument("--retrospective", action="store_true")
    args = parser.parse_args()
    print(json.dumps(research(args.data, args.retrospective), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
