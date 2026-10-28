# The demo as one CPU container, with the release weights inside the image. Build from the repo root:
#   docker build --platform linux/amd64 -f space/Dockerfile -t redefiner-demo .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_OFFLINE=1 MODEL=/app/weights

RUN useradd --create-home app
COPY space/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md LICENSE /tmp/pkg/
COPY src /tmp/pkg/src
RUN pip install --no-deps /tmp/pkg && rm -rf /tmp/pkg

WORKDIR /app
COPY --chown=app:app weights /app/weights
COPY --chown=app:app space /app

USER app
EXPOSE 7860
CMD ["python", "app.py"]
