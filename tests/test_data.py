import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from bitcoin.data import (
    SOURCE_URL,
    aggregate_monthly,
    download_history,
    parse_source,
    read_daily,
    recover_missing_days,
)


SOURCE_WITH_GAP = (
    "unix,open,high,low,close,Volume BTC\n"
    "1577836800,99,104,94,100,11\n"
    "1578009600,102,106,95,103,12\n"
).encode()
MISSING_DAY_RESPONSE = (
    b'{"data":{"pair":"BTC/USD","ohlc":[{"timestamp":"1577923200",'
    b'"open":"100","high":"105","low":"95","close":"102","volume":"10"}]}}'
)


def daily_frame(start="2020-01-01", end="2020-04-04"):
    dates = pd.date_range(start, end, freq="D", name="date")
    close = np.arange(len(dates), dtype=float) + 100
    return pd.DataFrame(
        {"open": close - 1, "high": close + 2, "low": close - 2, "close": close, "volume_btc": 10.0},
        index=dates,
    )


def test_monthly_ohlcv_and_incomplete_month_exclusion():
    daily = daily_frame()
    monthly = aggregate_monthly(daily, as_of="2020-04-05")
    assert list(monthly.index.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-29", "2020-03-31"]
    january = daily.loc["2020-01"]
    assert monthly.iloc[0]["open"] == january.iloc[0]["open"]
    assert monthly.iloc[0]["close"] == january.iloc[-1]["close"]
    assert monthly.iloc[0]["high"] == january["high"].max()
    assert monthly.iloc[0]["low"] == january["low"].min()
    assert monthly.iloc[0]["volume_btc"] == 310
    assert monthly.iloc[1]["day_count"] == 29
    assert monthly.iloc[0]["mean_close"] == january["close"].mean()


def test_reference_date_excludes_current_month_even_if_source_contains_it():
    daily = daily_frame(end="2020-04-30")
    monthly = aggregate_monthly(daily, as_of="2020-03-31")
    assert monthly.index[-1] == pd.Timestamp("2020-02-29")


def test_partial_source_boundary_months_are_excluded():
    monthly = aggregate_monthly(daily_frame("2020-01-02", "2020-03-15"), as_of="2020-04-01")
    assert list(monthly.index) == [pd.Timestamp("2020-02-29")]


@pytest.mark.parametrize("problem", ["duplicate", "gap", "invalid_ohlc", "nonfinite", "negative_volume"])
def test_invalid_snapshot_is_rejected(tmp_path, problem):
    daily = daily_frame()
    if problem == "duplicate":
        daily = pd.concat([daily, daily.iloc[[0]]])
    elif problem == "gap":
        daily = daily.drop(daily.index[15])
    elif problem == "invalid_ohlc":
        daily.loc[daily.index[0], "high"] = 1
    elif problem == "nonfinite":
        daily.loc[daily.index[0], "close"] = float("inf")
    elif problem == "negative_volume":
        daily.loc[daily.index[0], "volume_btc"] = -1
    path = tmp_path / "daily.csv"
    daily.to_csv(path)
    with pytest.raises(ValueError):
        read_daily(path)


def test_source_uses_unix_and_excludes_unfinished_day():
    content = (
        "https://www.CryptoDataDownload.com\n"
        "unix,date,symbol,open,high,low,close,Volume BTC,Volume USD\n"
        "1577923200000,ambiguous date,BTC/USD,100,105,95,102,10,1000\n"
        "1577836800,wrong textual date,BTC/USD,99,104,94,100,11,1100\n"
    ).encode()
    daily = parse_source(content, as_of="2020-01-02")
    assert list(daily.index) == [pd.Timestamp("2020-01-01")]
    assert daily.iloc[0]["close"] == 100


def test_official_supplement_recovers_exact_missing_day(monkeypatch):
    seen_urls = []

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        return io.BytesIO(MISSING_DAY_RESPONSE)

    monkeypatch.setattr("bitcoin.data.urlopen", fake_urlopen)
    daily = parse_source(SOURCE_WITH_GAP, as_of="2020-01-04")
    recovered, supplements = recover_missing_days(daily)
    assert len(recovered) == 3
    assert recovered.loc["2020-01-02", "close"] == 102
    assert len(seen_urls) == 1
    assert "start=1577923200" in seen_urls[0]
    assert "exclude_current_candle=true" in seen_urls[0]
    assert supplements[0]["raw_response"]["data"]["pair"] == "BTC/USD"
    assert len(supplements[0]["response_sha256"]) == 64


@pytest.mark.parametrize(
    "response, message",
    [
        (b'{"data":{"pair":"BTC/USD","ohlc":[{"timestamp":"1577836800"}]}}', "exato"),
        (b'{"data":{"pair":"ETH/USD","ohlc":[]}}', "par diferente"),
    ],
)
def test_official_supplement_rejects_wrong_day_or_pair(monkeypatch, response, message):
    monkeypatch.setattr("bitcoin.data.urlopen", lambda *args, **kwargs: io.BytesIO(response))
    daily = parse_source(SOURCE_WITH_GAP, as_of="2020-01-04")
    with pytest.raises(ValueError, match=message):
        recover_missing_days(daily)


def test_download_saves_complete_history_with_verifiable_provenance(monkeypatch, tmp_path):
    seen_urls = []

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        if request.full_url == SOURCE_URL:
            return io.BytesIO(SOURCE_WITH_GAP)
        assert request.full_url.startswith("https://www.bitstamp.net/api/v2/ohlc/btcusd/")
        return io.BytesIO(MISSING_DAY_RESPONSE)

    monkeypatch.setattr("bitcoin.data.urlopen", fake_urlopen)
    output = tmp_path / "data" / "btcusd_daily.csv"
    download_history(output, as_of="2020-01-04")

    daily = read_daily(output)
    assert list(daily.index) == list(pd.date_range("2020-01-01", "2020-01-03"))
    assert list(daily["close"]) == [100, 102, 103]
    assert len(seen_urls) == 2

    source = json.loads(output.with_name("source.json").read_text())
    assert source["first_date"] == "2020-01-01"
    assert source["last_date"] == "2020-01-03"
    assert source["rows"] == 3
    assert source["as_of"] == "2020-01-04"
    assert source["raw_source_sha256"] == hashlib.sha256(SOURCE_WITH_GAP).hexdigest()
    assert source["snapshot_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    recovery = source["supplemental_candles"][0]
    assert recovery["candle"]["date"] == "2020-01-02"
    assert recovery["raw_response"] == json.loads(MISSING_DAY_RESPONSE)
    assert recovery["response_sha256"] == hashlib.sha256(MISSING_DAY_RESPONSE).hexdigest()
    assert recovery["source_url"] == seen_urls[1]
    assert pd.Timestamp(recovery["retrieved_at_utc"]).tzinfo is not None
