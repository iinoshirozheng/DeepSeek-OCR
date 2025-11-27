#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-deepseek-ocr}
RUNTIME=${RUNTIME:-docker}
PORT=${PORT:-8000}
ENV_FILE=${ENV_FILE:-.env}

usage() {
  cat <<USAGE
Usage: ./run.sh [--port PORT] [--image-name NAME] [--runtime docker|podman] [--env-file PATH]
Defaults: PORT=$PORT, NAME=$IMAGE_NAME, RUNTIME=$RUNTIME, ENV_FILE=$ENV_FILE
Example (defaults): ./run.sh --port $PORT --image-name $IMAGE_NAME --runtime $RUNTIME --env-file $ENV_FILE
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2;;
    --image-name)
      IMAGE_NAME="$2"; shift 2;;
    --runtime)
      RUNTIME="$2"; shift 2;;
    --env-file)
      ENV_FILE="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if ! command -v "$RUNTIME" >/dev/null 2>&1; then
  echo "Runtime $RUNTIME not found" >&2
  exit 1
fi

PORT_MAP="$PORT:$PORT"

common_args=(
  --rm
  --env-file "$ENV_FILE"
  -p "$PORT_MAP"
)

device_args=()
if [[ "$RUNTIME" == "docker" ]]; then
  device_args+=(--gpus all)
elif [[ "$RUNTIME" == "podman" ]]; then
  device_args+=(--device nvidia.com/gpu=all --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d)
fi

echo "Running: $RUNTIME run ${common_args[*]} ${device_args[*]} $IMAGE_NAME"
exec "$RUNTIME" run "${common_args[@]}" "${device_args[@]}" "$IMAGE_NAME"
