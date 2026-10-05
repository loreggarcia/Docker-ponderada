"""Dados BTC/USD: coleta do histórico e preparação mensal para treinamento.

Coleta: download_history -> parse_source -> recover_missing_days -> CSV/source.json.
Treino: read_daily -> aggregate_monthly, sem acessar a rede.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_d.csv"
DAILY_COLUMNS = ["open", "high", "low", "close", "volume_btc"]


def _as_of_date(value: str) -> pd.Timestamp:
    """Treat a supplied date as the start of that day, in UTC."""
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("as_of deve ser uma data válida.")
    if date.tzinfo is not None:
        date = date.tz_convert("UTC").tz_localize(None)
    return date.normalize()


def _validate_daily(daily: pd.DataFrame, *, allow_gaps: bool = False) -> pd.DataFrame:
    missing = set(DAILY_COLUMNS).difference(daily.columns)
    if missing:
        raise ValueError(f"Colunas diárias ausentes: {sorted(missing)}")
    if daily.empty:
        raise ValueError("Nenhuma observação diária disponível.")
    if not isinstance(daily.index, pd.DatetimeIndex) or daily.index.hasnans:
        raise ValueError("Datas diárias inválidas.")
    if daily.index.tz is not None:
        raise ValueError("O índice deve conter datas UTC sem timezone.")
    if not daily.index.equals(daily.index.normalize()):
        raise ValueError("Cada candle deve começar à meia-noite UTC.")
    if daily.index.has_duplicates:
        raise ValueError("Datas diárias duplicadas.")
    result = daily.loc[:, DAILY_COLUMNS].sort_index().copy()
    result.index.name = "date"
    expected = pd.date_range(result.index[0], result.index[-1], freq="D")
    missing_dates = expected.difference(result.index)
    if len(missing_dates) and not allow_gaps:
        examples = ", ".join(missing_dates[:3].strftime("%Y-%m-%d"))
        raise ValueError(f"Dias ausentes no histórico: {examples}")
    try:
        result = result.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Preços e volumes devem ser numéricos.") from exc
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Histórico contém valores ausentes ou não finitos.")
    prices = result[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or (result["volume_btc"] < 0).any():
        raise ValueError("Preços devem ser positivos e volumes não negativos.")
    if (
        (result["high"] < prices.max(axis=1)).any()
        or (result["low"] > prices.min(axis=1)).any()
    ):
        raise ValueError("Candle inválido: low <= open/close <= high deve valer.")
    return result


def read_daily(path: str | Path) -> pd.DataFrame:
    """Leia o CSV local e rejeite duplicatas, lacunas e valores inválidos."""
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError("CSV deve conter a coluna date no formato YYYY-MM-DD.")
    try:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Datas devem usar o formato YYYY-MM-DD.") from exc
    return _validate_daily(frame.set_index("date"))


def aggregate_monthly(daily: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Agrupe apenas meses completos anteriores ao mês de referência.

    Meses nas bordas com cobertura parcial são excluídos. Lacunas internas são
    rejeitadas. A volatilidade é o desvio padrão populacional dos retornos
    logarítmicos diários observados em cada mês.
    """
    daily = _validate_daily(daily)
    cutoff = _as_of_date(as_of).replace(day=1)
    daily = daily.loc[daily.index < cutoff]
    if daily.empty:
        raise ValueError("Nenhum mês completo disponível antes da data de referência.")
    monthly = daily.resample("ME").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume_btc=("volume_btc", "sum"),
        mean_close=("close", "mean"),
        day_count=("close", "count"),
    )
    log_returns = np.log(daily["close"]).diff()
    monthly["volatility"] = log_returns.resample("ME").std(ddof=0)
    monthly = monthly.loc[monthly["day_count"] == monthly.index.days_in_month]
    if monthly.empty:
        raise ValueError("Nenhum mês possui todos os candles diários esperados.")
    monthly.index.name = "date"
    return monthly[
        ["open", "high", "low", "close", "volume_btc", "mean_close", "volatility", "day_count"]
    ]


def parse_source(content: bytes, as_of: str) -> pd.DataFrame:
    """Normalize o CSV externo; eventuais lacunas serão recuperadas na próxima etapa."""
    decoded = content.decode("utf-8-sig")
    if not decoded.strip():
        raise ValueError("A fonte retornou um CSV vazio.")
    skiprows = 1 if decoded.splitlines()[0].lower().startswith("http") else 0
    frame = pd.read_csv(io.StringIO(decoded), skiprows=skiprows)
    required = {"unix", "open", "high", "low", "close", "Volume BTC"}
    if not required.issubset(frame.columns):
        raise ValueError("A fonte não possui o esquema BTC/USD esperado.")
    timestamps = pd.to_numeric(frame["unix"], errors="raise")
    # A fonte usa timestamps Unix tanto em segundos quanto em milissegundos.
    seconds = timestamps.where(timestamps.abs() < 100_000_000_000, timestamps / 1000)
    frame.index = pd.to_datetime(seconds, unit="s", utc=True).dt.tz_localize(None)
    frame.index.name = "date"
    frame = frame.rename(columns={"Volume BTC": "volume_btc"})
    frame = frame.loc[
        (frame.index >= pd.Timestamp("2020-01-01"))
        & (frame.index < _as_of_date(as_of)),
        DAILY_COLUMNS,
    ]
    return _validate_daily(frame, allow_gaps=True)


def recover_missing_days(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Complete lacunas com a API Bitstamp e retorne dados + registro de cada recuperação."""
    expected = pd.date_range("2020-01-01", daily.index[-1], freq="D")
    missing = expected.difference(daily.index)
    if len(missing) > 31:
        raise ValueError("Mais de 31 dias ausentes: revise a integridade da fonte antes de prosseguir.")
    supplements = []
    for date in missing:
        start = int(date.tz_localize("UTC").timestamp())
        url = (
            "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
            f"?step=86400&limit=1&start={start}&end={start + 86399}"
            "&exclude_current_candle=true"
        )
        request = Request(url, headers={"User-Agent": "bitcoin-monthly-coursework/1.0"})
        with urlopen(request, timeout=30) as response:
            raw = response.read()
        payload = json.loads(raw)
        data = payload.get("data", {})
        candles = data.get("ohlc", [])
        if data.get("pair", data.get("market")) != "BTC/USD":
            raise ValueError("A Bitstamp respondeu com um par diferente de BTC/USD.")
        if len(candles) != 1 or int(candles[0]["timestamp"]) != start:
            raise ValueError(f"A Bitstamp não forneceu o candle UTC exato de {date.date()}.")
        candle = candles[0]
        supplements.append(
            {
                "source_url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_response": payload,
                "reason": "Daily candle absent from the CryptoDataDownload CSV.",
                "candle": {
                    "date": date.strftime("%Y-%m-%d"),
                    **{key: float(candle[key]) for key in ["open", "high", "low", "close"]},
                    "volume_btc": float(candle["volume"]),
                },
            }
        )
    if supplements:
        recovered = pd.DataFrame([item["candle"] for item in supplements])
        recovered["date"] = pd.to_datetime(recovered["date"], format="%Y-%m-%d")
        daily = pd.concat([daily, recovered.set_index("date")])
    return _validate_daily(daily), supplements


def download_history(output: str | Path, as_of: str) -> dict:
    """Baixe, normalize, recupere lacunas e salve o CSV com sua origem em source.json."""
    request = Request(SOURCE_URL, headers={"User-Agent": "bitcoin-monthly-coursework/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    daily = parse_source(content, as_of)
    daily, supplements = recover_missing_days(daily)

    output = Path(output)
    if daily.index[0] != pd.Timestamp("2020-01-01"):
        raise ValueError("A fonte não contém o início obrigatório de 2020-01-01.")
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output, date_format="%Y-%m-%d")
    metadata = {
        "provider": "CryptoDataDownload",
        "exchange": "Bitstamp",
        "pair": "BTC/USD",
        "source_url": SOURCE_URL,
        "documentation_url": "https://www.cryptodatadownload.com/data/bitstamp/",
        "frequency": "daily",
        "timezone": "UTC",
        "timestamp_column": "unix (seconds or milliseconds; source date strings ignored)",
        "columns": ["date", *DAILY_COLUMNS],
        "volume_unit": "BTC",
        "first_date": daily.index[0].strftime("%Y-%m-%d"),
        "last_date": daily.index[-1].strftime("%Y-%m-%d"),
        "rows": len(daily),
        "as_of": _as_of_date(as_of).strftime("%Y-%m-%d"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_source_sha256": hashlib.sha256(content).hexdigest(),
        "snapshot_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "supplemental_candles": supplements,
        "transformations": [
            "Keep candles starting on/after 2020-01-01 and strictly before as_of (UTC).",
            "Use Unix timestamps; sort ascending; retain OHLC and volume in BTC.",
            "Recover missing source candles from the official Bitstamp API with exact UTC timestamps; record every response.",
            "Reject missing days, duplicates, non-finite values and invalid OHLCV.",
            "No interpolation or synthetic observations.",
        ],
    }
    output.with_name("source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/btcusd_daily.csv")
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()
    metadata = download_history(args.output, args.as_of)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
