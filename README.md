# DeepSeek-OCR FastAPI + vLLM (CUDA)

Containerized FastAPI service exposing DeepSeek-OCR via vLLM. Uses `uv` for dependency/install speed and supports configurable vLLM versions (nightly by default).

## Prerequisites
- GPU with recent NVIDIA drivers (CUDA 12.x compatible)
- NVIDIA Container Toolkit configured for Docker/Podman
- `podman` or `docker` CLI

## Configuration
Environment variables (see `.env.example`):
- `MODEL_ID` (default `deepseek-ai/DeepSeek-OCR`)
- `MAX_TOKENS` (default `2048`)
- `VLLM_VERSION` (default `nightly`)

Create a `.env` from the example if needed:
```bash
cp .env.example .env
# adjust values if desired
```

## Build
Docker:
```bash
docker build -t deepseek-ocr --build-arg VLLM_VERSION=${VLLM_VERSION:-nightly} .
```

Podman (rootless):
```bash
podman build -t deepseek-ocr --build-arg VLLM_VERSION=${VLLM_VERSION:-nightly} .
```

## Run
Choose a host port (defaults to container port 8000). If you set `PORT` at runtime, match the mapping with `-p`.

Docker:
```bash
docker run --rm --gpus all \
  --env-file .env \
  -p 8000:8000 \
  deepseek-ocr
```

Podman:
```bash
podman run --rm \
  --env-file .env \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  -p 8000:8000 \
  deepseek-ocr
```

- To use a different port at runtime: add `-e PORT=9000` and map `-p 9000:9000`.
- Without `.env`, defaults apply (nightly vLLM, model `deepseek-ai/DeepSeek-OCR`).

## API
- `POST /ocr/full` (file)
- `POST /ocr/table` (file)
- `POST /ocr/custom` (file + `prompt` form field)
- `POST /explain` (file)
- `GET /health`

All OCR endpoints accept an image file upload (`multipart/form-data`).

## Notes
- CUDA wheels sourced from `https://download.pytorch.org/whl/cu124` in the Dockerfile.
- vLLM nightly required until stable releases fully support DeepSeek-OCR; override with `VLLM_VERSION` if pinning.
