#!/usr/bin/env bash
set -euo pipefail

# Record policy progression videos for Appendix E (both tutorial and book).
# Shows learning progression across training stages for state-based and pixel runs.
#
# Usage:  bash scripts/record_appendix_e_videos.sh
# Requires: Linux + NVIDIA GPU + Isaac Lab Docker image
#
# The script uses subprocess isolation (one env per process) because Isaac Lab's
# SimulationContext is a process-level singleton.

SCRIPT="scripts/appendix_e_isaac_manipulation.py"
ENV_ID="Isaac-Lift-Cube-Franka-v0"
CKPT_DIR="checkpoints"
VIDEO_DIR="videos"

mkdir -p "$VIDEO_DIR"

echo "================================================================"
echo "Appendix E: Recording policy progression videos"
echo "================================================================"

# --- State-based progression (3.5 MB checkpoints, MlpPolicy) ---
# These show the approach -> grasp -> lift -> track learning progression.
# From the 8M-step state-based training run (Feb 27).

echo ""
echo "--- State-based progression ---"

STATE_CKPTS=(
    "${CKPT_DIR}/appendix_e_sac_${ENV_ID}_seed0_3000064_steps.zip"   # ~3M: early grasping
    "${CKPT_DIR}/appendix_e_sac_${ENV_ID}_seed0_4999936_steps.zip"   # ~5M: lifting phase
    "${CKPT_DIR}/appendix_e_sac_${ENV_ID}_seed0_7999744_steps.zip"   # ~8M: converged
)
STATE_LABELS=(
    "3M_grasping"
    "5M_lifting"
    "8M_converged"
)

for i in "${!STATE_CKPTS[@]}"; do
    ckpt="${STATE_CKPTS[$i]}"
    label="${STATE_LABELS[$i]}"
    out="${VIDEO_DIR}/appendix_e_state_${label}.mp4"

    if [ ! -f "$ckpt" ]; then
        echo "  SKIP (not found): $ckpt"
        continue
    fi

    echo "  Recording state ${label}: $(basename "$ckpt")"
    bash docker/dev-isaac.sh python3 "$SCRIPT" record \
        --headless --env-id "$ENV_ID" \
        --ckpt "$ckpt" \
        --video-out "$out"
    echo "  -> $out"
done

# --- Pixel-based progression (34 MB checkpoints, MultiInputPolicy) ---
# These show the hockey-stick learning curve from TiledCamera training.
# From the 4M-step pixel training run (Feb 28).

echo ""
echo "--- Pixel-based progression ---"

PIXEL_CKPTS=(
    "${CKPT_DIR}/appendix_e_sac_${ENV_ID}_seed0_499968_steps.zip"    # 500K: pre-takeoff (random)
    "${CKPT_DIR}/appendix_e_sac_${ENV_ID}_seed0_1499904_steps.zip"   # 1.5M: post-takeoff (reaching)
    "${CKPT_DIR}/appendix_e_sac_${ENV_ID}_seed0_3999744_steps.zip"   # 4M: near-converged
)
PIXEL_LABELS=(
    "500K_random"
    "1500K_reaching"
    "4M_converged"
)

for i in "${!PIXEL_CKPTS[@]}"; do
    ckpt="${PIXEL_CKPTS[$i]}"
    label="${PIXEL_LABELS[$i]}"
    out="${VIDEO_DIR}/appendix_e_pixel_${label}.mp4"

    if [ ! -f "$ckpt" ]; then
        echo "  SKIP (not found): $ckpt"
        continue
    fi

    echo "  Recording pixel ${label}: $(basename "$ckpt")"
    bash docker/dev-isaac.sh python3 "$SCRIPT" record \
        --headless --pixel --env-id "$ENV_ID" \
        --ckpt "$ckpt" \
        --video-out "$out"
    echo "  -> $out"
done

echo ""
echo "================================================================"
echo "Done. Videos written to ${VIDEO_DIR}/appendix_e_*.mp4"
echo "================================================================"
ls -lh "${VIDEO_DIR}"/appendix_e_*.mp4 2>/dev/null || echo "(no videos found)"
