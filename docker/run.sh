#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_image="robotics-rl:latest"
fallback_image="nvcr.io/nvidia/pytorch:25.12-py3"
username="$(id -un 2>/dev/null || echo user)"

if [ -n "${IMAGE:-}" ]; then
  image="${IMAGE}"
elif docker image inspect "${default_image}" >/dev/null 2>&1; then
  image="${default_image}"
else
  image="${fallback_image}"
fi

workdir="${WORKDIR:-/workspace}"
mujoco_gl="${MUJOCO_GL:-egl}"
pyopengl_platform="${PYOPENGL_PLATFORM:-$mujoco_gl}"

exec docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MUJOCO_GL="${mujoco_gl}" \
  -e PYOPENGL_PLATFORM="${pyopengl_platform}" \
  -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  -e PYTHONUNBUFFERED=1 \
  -e USER="${USER:-$username}" \
  -e LOGNAME="${LOGNAME:-$username}" \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e TORCH_HOME=/tmp/.cache/torch \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib \
  -e WANDB_API_KEY \
  -v "${repo_root}":"${workdir}" \
  -w "${workdir}" \
  --user "$(id -u)":"$(id -g)" \
  "${image}" \
  "${@:-bash}"
