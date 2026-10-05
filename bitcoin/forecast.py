"""Carregamento de artefato local confiável e inferência sem treinar na API."""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def load_artifact(path: str | Path) -> dict:
    # joblib/pickle executa código: somente o artefato produzido pelo projeto.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Formato de modelo incompatível. Execute o treinamento novamente.")
    required = {"metadata", "models", "feature_names", "latest_features", "last_close",
                "origin_month", "target_month", "metrics", "error_quantiles"}
    if required - artifact.keys():
        raise ValueError("Artefato incompleto. Execute o treinamento novamente.")
    # Não anunciar prontidão para um arquivo que não consegue prever.
    predict(artifact)
    return artifact


def predict(artifact: dict, target_month: str | None = None) -> dict:
    target_month = target_month or artifact["target_month"]
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", target_month):
        raise ValueError("target_month deve usar YYYY-MM.")
    target = pd.Period(target_month, freq="M")
    origin = pd.Period(artifact["origin_month"], freq="M")
    horizon = target.ordinal - origin.ordinal
    if horizon not in artifact["models"]:
        valid = [str(origin + h) for h in sorted(artifact["models"])]
        raise ValueError(f"Mês não suportado pelo artefato. Escolha: {', '.join(valid)}.")
    model = artifact["models"][horizon]
    columns = list(model.feature_names_in_)
    features = pd.DataFrame([artifact["latest_features"]], columns=columns)
    log_return = float(model.predict(features)[0])
    halving_features = {key: float(artifact["latest_features"][key])
                       for key in ("months_since_halving", "block_reward_btc") if key in columns}
    base = float(artifact["last_close"])
    error = float(artifact["error_quantiles"][horizon])
    with np.errstate(over="raise", invalid="raise"):
        point, lower, upper = base * np.exp([log_return, log_return - error, log_return + error])
    if not np.all(np.isfinite([point, lower, upper])) or min(point, lower, upper) <= 0:
        raise ValueError("O modelo produziu uma previsão numérica inválida.")
    return {
        "pair": "BTC/USD",
        "currency": "USD",
        "target": "monthly_close",
        "target_month": str(target),
        "target_date": target.end_time.date().isoformat(),
        "as_of": artifact["metadata"]["as_of"],
        "last_complete_month": str(origin),
        "horizon_months": horizon,
        "last_close_usd": round(base, 2),
        "predicted_close_usd": round(float(point), 2),
        "predicted_change_pct": round(float(np.expm1(log_return) * 100), 2),
        "empirical_error_band_usd": {
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2),
            "historical_absolute_log_error_quantile": 0.8,
            "calibration_samples": artifact["metrics"][str(horizon)]["test_samples"],
            "note": "Faixa empírica de erros históricos; não garante cobertura futura de 80%.",
        },
        "model": f"StandardScaler + Ridge (retorno logarítmico, {'com' if halving_features else 'sem'} halving)",
        "halving_features": halving_features,
        "note": "Previsão experimental. A variação é relativa ao último mês fechado, não ao preço atual. Não é recomendação de investimento.",
    }
