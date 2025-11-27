#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="deepseek-ocr"
VLLM_VERSION_ENV=${VLLM_VERSION:-nightly}

usage() {
  cat <<USAGE
Usage: ./install.sh [--image-name NAME] [--vllm-version VERSION] [--runtime docker|podman]
Defaults: NAME=deepseek-ocr, VERSION=[1m$VLLM_VERSION_ENV[0m
USAGE
}

IMAGE_RUNTIME=${RUNTIME:-docker}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-name)
      IMAGE_NAME="$2"; shift 2;;
    --vllm-version)
      VLLM_VERSION_ENV="$2"; shift 2;;
    --runtime)
      IMAGE_RUNTIME="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if ! command -v "$IMAGE_RUNTIME" >/dev/null 2>&1; then
  echo "Runtime $IMAGE_RUNTIME not found" >&2
  exit 1
fi

$IMAGE_RUNTIME build -t "$IMAGE_NAME" --build-arg VLLM_VERSION="$VLLM_VERSION_ENV" .
echo "Built image $IMAGE_NAME (vLLM $VLLM_VERSION_ENV) with $IMAGE_RUNTIME"
