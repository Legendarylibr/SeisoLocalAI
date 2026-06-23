FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY seiso ./seiso
COPY seiso_cli ./seiso_cli
COPY third_party/adaptive-rl-quant ./third_party/adaptive-rl-quant
COPY deploy/model-router/config ./deploy/model-router/config

RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir \
      "httpx>=0.27" \
      "pydantic>=2.6" \
      "pydantic-settings>=2.2" \
      "pyyaml>=6.0" \
      "fastapi>=0.111" \
      "uvicorn[standard]>=0.29" \
      "psutil>=5.9" \
      "litellm>=1.55.0" && \
    pip install --no-cache-dir -e .

ENV SEISO_ROUTER_CONFIG_PATH=deploy/model-router/config/router.local.yaml
ENV PYTHONUNBUFFERED=1

EXPOSE 8780

CMD ["python", "-m", "uvicorn", "seiso.model_router.main:build_app", "--factory", "--host", "0.0.0.0", "--port", "8780"]
