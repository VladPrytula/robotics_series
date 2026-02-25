# Ch09 Visual RL: Is 84x84 Enough? Stacking, Encoder, and Our Pipeline

Date: 2026-02-23
Owner: Ch09 (Pixel Push) — reference notes to guide runs and reviews

## TL;DR
- 84x84 pixels are generally sufficient for pixel-based control in simulation when:
  - Frames are stacked (3–4) to recover velocity and contact dynamics.
  - The encoder uses gentle downsampling (3x3 kernels, stride 2 in early layers) — not NatureCNN stride-4.
  - Strong per-sample augmentation (random shift) stabilizes learning.
- In our repo, these conditions are met by default in pixel mode: `frame_stack=4`, ManipulationCNN/SpatialSoftmax, and DrQ random shift.
- Our current blocker is optimization/gradient routing to the encoder under sparse rewards, not visual resolution.

## Why 84x84 Can Work (practical view)
- Pixel RL baselines that succeed broadly in DMControl/Atari use 64–84 px inputs, 3–4 stacked frames, and small-kernel convs. Larger images mainly help when using frozen/pretrained encoders (e.g., 224x224), not for end-to-end RL from scratch in simulation.
- The typical failure at 84x84 is architectural (aggressive stride) or optimization (weak gradients to CNN), not “too few pixels.”

## What Stacking Provides
- Velocity and contact cues come from temporal differences across frames; a single RGB frame creates a POMDP for manipulation tasks (e.g., Push) after contact.
- Stacking 4 frames converts this into a tractable partially-observed problem for the encoder without requiring explicit optical flow.

## How This Maps To Our Code
- Pixel wrapper and stacking:
  - `scripts/labs/pixel_wrapper.py` — `PixelObservationWrapper(..., frame_stack=4)` builds `(12, 84, 84)` pixel tensors in pixel mode.
  - Native render vs resize is controlled by `--native-render` in `scripts/ch09_pixel_push.py`.
- Encoder and spatial head:
  - `scripts/labs/manipulation_encoder.py` — `ManipulationCNN` (3x3, stride 2/1/1/2) preserves small-object spatial detail; `SpatialSoftmax` exposes per-channel (x, y) coordinates.
- Augmentation:
  - `scripts/labs/image_augmentation.py` — `RandomShiftAug(pad=cfg.drq_pad)` used by `HerDrQDictReplayBuffer` in pixel mode.
- Gradient routing (DrQ-v2 pattern):
  - `scripts/labs/drqv2_sac_policy.py` — DrQv2SACPolicy updates the shared encoder via critic TD loss; actor consumes detached features.
- Chapter runner & flags:
  - `scripts/ch09_pixel_push.py`
    - `--frame-stack` (default 4 in pixel mode)
    - `--critic-encoder` (enable DrQ-v2 gradient routing)
    - `--native-render` (render 84x84 natively)
    - `--full-state` (control run without pixels)

## Quick Checks Before/While Running
- Smoke output should show `frame_stack=4` and, in pixel mode, `pixels: shape=(12, 84, 84)`.
- If `--critic-encoder` is on, startup prints the DrQ-v2 mode and TB tag includes `criticEnc`.
- In `feature-stats`, SpatialSoftmax output should be bounded with LN+Tanh (means near 0, reasonable stds), and temperature should evolve over time.

## Targeted Experiments (no code changes)
1) Stacking ablation at 84x84 (partial observability probe)
- Expect fs=4 > fs=1 on early success and stability.

```
# fs=4 (baseline)
bash docker/dev.sh python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder --frame-stack 4 --total-steps 2000000

# fs=1 (ablation)
bash docker/dev.sh python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder --frame-stack 1 --total-steps 2000000
```

2) Native vs resized 84x84 (downsampling artifacts)
- Expect similar or slight win for native; if native wins clearly, resizing was smearing tiny features.

```
# Native render at target size
bash docker/dev.sh python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder --native-render --total-steps 2000000

# Standard resize path
bash docker/dev.sh python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder --total-steps 2000000
```

3) Minimal resolution sweep (only if (1) and (2) are inconclusive)
- Keep training identical; compare 84 vs 100(crop) vs 112. Stop early if 84 ties 100(crop).

```
# 100->crop to 84 (simulate CURL recipe)
# (Run by rendering 100x100 natively then center-cropping in aug if added later; not required now.)
```

## Failure Signatures vs Root Causes
- Critic loss monotonically decreasing, success oscillating at 2–14% → encoder not learning useful spatial features (likely gradient routing/optimization), not lack of pixels.
- Early spikes followed by instability with fs=1 → partial observability; stacking should fix.
- Native render wins big over resize → downsampling artifacts; keep native.

## When To Consider Higher Resolution
- If smallest objects occupy ~2–3 pixels after encoder layer 1 despite gentle downsampling, and (1) stacking, (2) native render, and (3) DrQ-v2 gradient routing are already in place with no improvement.
- Otherwise, prefer representation fixes (DrQ/CURL) or frozen encoders (R3M/VC-1) before paying the runtime cost of larger images.

## Run Hygiene
- Tags: `_config_tag` adds `criticEnc`, `fullState`, and `fsN`; verify unique run directories per config.
- Stop rules: at 1.0M steps, abort if success ≤ 10% and critic loss is flat/declining.
- Success signature: rising critic loss around 1.2–1.5M, reward toward −30/−20, sustained success > 15% then onward.

## References (context)
- DrQ/DrQ-v2: gentle downsampling + random shift, three consecutive frames common in setups.
- CURL: 100→84 crop; contrastive head improves pixel sample-efficiency.
- DreamerV3 / TD-MPC2: strong performance at 64×64, indicating resolution is not the primary limiter.
- DQN (historical): 84×84×4 frames established stacking as the default POMDP fix.

