#!/usr/bin/env bash
set -euo pipefail

# One-command developer entrypoint:
# - launches the GPU container
# - creates/updates a venv in the repo (./.venv)
# - installs Python deps from requirements.txt
# - drops you into an interactive shell with the venv activated

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_image="robotics-rl:latest"
fallback_image="nvcr.io/nvidia/pytorch:25.12-py3"
username="$(id -un 2>/dev/null || echo user)"

if [ -n "${IMAGE:-}" ]; then
  image="${IMAGE}"
else
  image="${default_image}"
  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Docker image ${image} not found; building (required for MuJoCo EGL/OSMesa dependencies)..." >&2
    if ! docker build -t "${image}" -f "${repo_root}/docker/Dockerfile" "${repo_root}"; then
      echo "WARNING: build failed; falling back to ${fallback_image} (rendering may be unavailable)." >&2
      image="${fallback_image}"
    fi
  fi
fi

workdir="${WORKDIR:-/workspace}"
mujoco_gl="${MUJOCO_GL:-egl}"
pyopengl_platform="${PYOPENGL_PLATFORM:-$mujoco_gl}"

cmd=( "$@" )
if [ ${#cmd[@]} -eq 0 ]; then
  cmd=( bash )
fi

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
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torch_inductor \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib \
  -e WANDB_API_KEY \
  -v "${repo_root}":"${workdir}" \
  -w "${workdir}" \
  --user "$(id -u)":"$(id -g)" \
  "${image}" \
  bash -lc '
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
  ' bash "${cmd[@]}"
