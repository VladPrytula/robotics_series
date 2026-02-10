#!/usr/bin/env bash
set -euo pipefail

# One-command developer entrypoint:
# - detects platform (Mac M4 vs DGX/NVIDIA)
# - launches the appropriate container
# - creates/updates a venv in the repo (./.venv)
# - installs Python deps from requirements.txt
# - drops you into an interactive shell with the venv activated

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
username="$(id -un 2>/dev/null || echo user)"

# Platform detection
os="$(uname -s)"
arch="$(uname -m)"
is_mac=false
is_nvidia=false

if [ "$os" = "Darwin" ]; then
    is_mac=true
elif [ "$os" = "Linux" ] && command -v nvidia-smi &>/dev/null; then
    is_nvidia=true
fi

# Image selection based on platform
if [ "$is_mac" = true ]; then
    default_image="robotics-rl:mac"
    fallback_image="python:3.11-slim"
    dockerfile="docker/Dockerfile.mac"
else
    default_image="robotics-rl:latest"
    fallback_image="nvcr.io/nvidia/pytorch:25.12-py3"
    dockerfile="docker/Dockerfile"
fi

if [ -n "${IMAGE:-}" ]; then
  image="${IMAGE}"
else
  image="${default_image}"
  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Docker image ${image} not found; building (required for MuJoCo rendering dependencies)..." >&2
    if ! docker build -t "${image}" -f "${repo_root}/${dockerfile}" "${repo_root}"; then
      echo "WARNING: build failed; falling back to ${fallback_image} (rendering may be unavailable)." >&2
      image="${fallback_image}"
    fi
  fi
fi

workdir="${WORKDIR:-/workspace}"

# Environment variables differ by platform
if [ "$is_mac" = true ]; then
    mujoco_gl="${MUJOCO_GL:-osmesa}"
else
    mujoco_gl="${MUJOCO_GL:-egl}"
fi
pyopengl_platform="${PYOPENGL_PLATFORM:-$mujoco_gl}"

cmd=( "$@" )
if [ ${#cmd[@]} -eq 0 ]; then
  cmd=( bash )
fi

# Build docker run arguments based on platform
docker_args=(
  --rm -it
  -e MUJOCO_GL="${mujoco_gl}"
  -e PYOPENGL_PLATFORM="${pyopengl_platform}"
  -e PYTHONUNBUFFERED=1
  -e USER="${USER:-$username}"
  -e LOGNAME="${LOGNAME:-$username}"
  -e HOME=/tmp
  -e XDG_CACHE_HOME=/tmp/.cache
  -e TORCH_HOME=/tmp/.cache/torch
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torch_inductor
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib
  -e WANDB_API_KEY
  -v "${repo_root}":"${workdir}"
  -w "${workdir}"
  --user "$(id -u)":"$(id -g)"
)

# NVIDIA-specific flags (only on Linux with GPU)
if [ "$is_nvidia" = true ]; then
    docker_args+=(
      --gpus all
      --ipc=host
      --ulimit memlock=-1
      --ulimit stack=67108864
      -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
    )
fi

# Inner script handles venv setup - differs slightly for Mac (no system torch to preserve)
if [ "$is_mac" = true ]; then
    inner_script='
    set -euo pipefail
    if [ ! -d .venv ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate
    req_hash="$(sha256sum requirements.txt | cut -d " " -f1)"
    stamp_file=".venv/.requirements.sha256"
    if [ ! -f "${stamp_file}" ] || [ "$(cat "${stamp_file}")" != "${req_hash}" ]; then
      python -m pip install -U pip
      # Install torch first for ARM64 native wheel, then other deps
      python -m pip install torch
      python -m pip install -r requirements.txt
      printf "%s" "${req_hash}" > "${stamp_file}"
    fi
    exec "$@"
  '
else
    inner_script='
    set -euo pipefail
    if [ ! -d .venv ]; then
      python -m venv --system-site-packages .venv
    fi
    if grep -q "^include-system-site-packages = false" .venv/pyvenv.cfg 2>/dev/null; then
      sed -i "s/^include-system-site-packages = false/include-system-site-packages = true/" .venv/pyvenv.cfg
    fi
    source .venv/bin/activate
    purelib="$(python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")"
    if [ -d "${purelib}/torch" ]; then
      echo "Removing venv-installed torch to use container PyTorch..." >&2
      rm -rf \
        "${purelib}/torch" \
        "${purelib}/torch-"*.dist-info \
        "${purelib}/torchgen" \
        "${purelib}/torchgen-"*.dist-info \
        "${purelib}/functorch" \
        "${purelib}/functorch-"*.dist-info
    fi
    req_hash="$(sha256sum requirements.txt | cut -d " " -f1)"
    stamp_file=".venv/.requirements.sha256"
    if [ ! -f "${stamp_file}" ] || [ "$(cat "${stamp_file}")" != "${req_hash}" ]; then
      python -m pip install -U pip
      python -m pip install -r requirements.txt
      printf "%s" "${req_hash}" > "${stamp_file}"
    fi
    exec "$@"
  '
fi

exec docker run "${docker_args[@]}" "${image}" bash -lc "${inner_script}" bash "${cmd[@]}"
