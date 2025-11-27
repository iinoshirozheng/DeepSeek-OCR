FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG VLLM_VERSION=nightly
ENV VLLM_VERSION=${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv python3-pip python3-dev build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache-dir -r /app/requirements.txt

RUN if [ "$VLLM_VERSION" = "nightly" ]; then \
      uv pip install --system -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match; \
    else \
      uv pip install --system -U "vllm==${VLLM_VERSION}" --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match; \
    fi

COPY . /app

EXPOSE 8000

CMD ["python3", "app.py"]
