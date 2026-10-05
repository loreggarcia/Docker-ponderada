"""Consulta a API real e mostra a saúde do serviço e a previsão no terminal."""

import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


def get_json(url: str, timeout: float = 30) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def wait_for_api(base_url: str) -> dict:
    """Espera até 30 segundos pela API durante sua inicialização ou reinício."""
    deadline = time.monotonic() + 30
    last_error = "serviço indisponível"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"API não ficou pronta em 30 segundos. Último erro: {last_error}")
        try:
            return get_json(base_url + "/health", timeout=min(3, remaining))
        except HTTPError as exc:
            if exc.code != 503:
                raise
            last_error = "HTTP 503: modelo indisponível; confira os logs da API"
            exc.close()
        except (URLError, ConnectionError, TimeoutError) as exc:
            last_error = str(exc)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(1, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstração HTTP da previsão de Bitcoin.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Endereço da API")
    parser.add_argument("--target-month", help="Mês YYYY-MM, opcional")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    query = "?" + urlencode({"target_month": args.target_month}) if args.target_month else ""
    try:
        result = {
            "api_url": base_url,
            "health": wait_for_api(base_url),
            "prediction": get_json(base_url + "/predict" + query),
        }
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (URLError, ConnectionError, TimeoutError) as exc:
        print(f"Não foi possível acessar a API: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    document = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    print(document, end="")


if __name__ == "__main__":
    main()
