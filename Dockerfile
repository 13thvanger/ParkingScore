FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --gid 10001 parking-score \
    && useradd --uid 10001 --gid parking-score --no-create-home parking-score \
    && mkdir -p /data/cache \
    && chown -R parking-score:parking-score /data

COPY pyproject.toml README.md ./
COPY parking_score ./parking_score
RUN pip install .

USER parking-score

CMD ["python", "-m", "parking_score", "run"]
