FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home app \
    && mkdir -p /app/artifacts /app/data \
    && chown -R app:app /app

COPY --chown=app:app bitcoin ./bitcoin

USER app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "bitcoin.api:app", "--host", "0.0.0.0", "--port", "8000"]
