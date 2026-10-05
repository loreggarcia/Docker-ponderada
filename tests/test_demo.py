"""O cliente tolera a reinicialização da API, mas encerra erros permanentes."""

from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from scripts import demo


@pytest.fixture
def clock(monkeypatch):
    state = {"now": 0.0, "sleeps": []}

    def sleep(seconds):
        assert 0 < seconds <= 1
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(
        demo, "time", SimpleNamespace(monotonic=lambda: state["now"], sleep=sleep)
    )
    return state


def test_health_recovers_after_reset_and_service_unavailable(monkeypatch, clock):
    health = {"status": "ok", "model_loaded": True, "target_month": "2026-11"}
    responses = iter([
        ConnectionResetError("Connection reset by peer"),
        HTTPError("http://api/health", 503, "Service Unavailable", {}, BytesIO()),
        URLError("Connection refused"),
        TimeoutError("Request timed out"),
        health,
    ])
    requests = []

    def get_json(url, timeout=30):
        requests.append((url, timeout))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(demo, "get_json", get_json)

    assert demo.wait_for_api("http://api") == health
    assert len(requests) == 5
    assert all(url == "http://api/health" and 0 < timeout <= 3 for url, timeout in requests)
    assert clock["sleeps"] == [1, 1, 1, 1]


@pytest.mark.parametrize("status", [401, 404, 422, 500])
def test_non_transient_http_error_is_not_retried(monkeypatch, clock, status):
    error = HTTPError("http://api/health", status, "Request failed", {}, BytesIO())
    requests = []

    def get_json(url, timeout=30):
        requests.append(url)
        raise error

    monkeypatch.setattr(demo, "get_json", get_json)

    with pytest.raises(HTTPError) as caught:
        demo.wait_for_api("http://api")

    assert caught.value is error
    assert requests == ["http://api/health"]
    assert clock["sleeps"] == []


def test_wait_stops_at_deadline_including_request_time(monkeypatch, clock):
    requests = []

    def get_json(url, timeout=30):
        remaining = 30 - clock["now"]
        assert 0 < timeout <= min(3, remaining)
        requests.append((url, timeout))
        clock["now"] += timeout
        raise URLError("API still unavailable")

    monkeypatch.setattr(demo, "get_json", get_json)

    with pytest.raises(TimeoutError):
        demo.wait_for_api("http://api")

    assert clock["now"] == pytest.approx(30)
    assert requests
    assert requests[-1][1] < 3


def test_prediction_connection_error_exits_without_traceback(monkeypatch, capsys):
    requests = []
    monkeypatch.setattr(demo.sys, "argv", [
        "demo.py", "--url", "http://api/", "--target-month", "2026-11"
    ])
    monkeypatch.setattr(demo, "wait_for_api", lambda url: {"status": "ok"})

    def get_json(url, timeout=30):
        requests.append(url)
        raise ConnectionResetError("Connection reset by peer")

    monkeypatch.setattr(demo, "get_json", get_json)

    with pytest.raises(SystemExit) as caught:
        demo.main()

    output = capsys.readouterr()
    assert caught.value.code == 1
    assert requests == ["http://api/predict?target_month=2026-11"]
    assert output.out == ""
    assert "Não foi possível acessar a API" in output.err
    assert "Traceback" not in output.err


def test_get_json_passes_request_timeout(monkeypatch):
    requests = []

    def urlopen(url, timeout):
        requests.append((url, timeout))
        return BytesIO(b'{"status": "ok"}')

    monkeypatch.setattr(demo, "urlopen", urlopen)

    assert demo.get_json("http://api/health", timeout=1.5) == {"status": "ok"}
    assert requests == [("http://api/health", 1.5)]
