# Pixel Push Fix: Implementation Record

**Date:** 2026-02-20
**Status:** Implemented, awaiting training runs

---

## Problem Statement

Pixel + HER + DrQ on FetchPush-v4 shows 2-8% success after 5M+ steps (8M
budget). State + HER solves the same task at 100% in 2M steps. Two root
causes identified:

### Root Cause 1: CNN features drown goal vectors (scale mismatch)

SB3's `CombinedExtractor` concatenates CNN(pixels) -> 256D features with
achieved_goal(3D) + desired_goal(3D) = 262D total. Goal vectors are 2.3%
of the input. Early in training, CNN features are random noise with
unbounded magnitude (post-ReLU). The MLP's gradients are dominated by
the 256 noise terms, drowning the 6 clean goal terms.

The critic cannot learn Q(s, a | g) ~ -distance(achieved, desired) because
the gradient dQ/d(goal) is tiny compared to dQ/d(CNN_noise). HER's value
wavefront never starts because the critic can't represent goal-distance
structure.

### Root Cause 2: Missing velocity information (single-frame observation)

The state observation for FetchPush includes 25 dimensions:
- End-effector position (3), velocity (3), gripper state (2)
- Object position (3), rotation (3), velocity (3), angular velocity (3)
- Gripper-object relative position (3), finger velocities (2)

A single 84x84 pixel frame contains ZERO velocity information. At least
11 of 25 state dimensions (all velocities) are physically unobservable
from a single photograph. For pushing, velocity is critical:
- Approach speed control (needs gripper velocity)
- Contact force modulation (needs relative velocity)
- Braking at target (needs object velocity)

---

## Changes Made

### 1. NormalizedCombinedExtractor (new class)

**File:** `scripts/labs/visual_encoder.py`
**Lines:** ~112-180 (after NatureCNN, before VisualGoalEncoder)

Replaces SB3's default `CombinedExtractor` with a version that applies
`LayerNorm -> Tanh` on CNN output before concatenation with goal vectors.
This is the DrQ-v2 trunk pattern (Yarats et al. 2021, arXiv:2107.09645).

Architecture per observation key:
```
image key  -> NatureCNN(cnn_output_dim) -> LayerNorm -> Tanh -> [-1, 1]
vector key -> Flatten (identity) -> natural scale
All outputs concatenated -> features
```

With cnn_output_dim=50 and goal_mode="both":
```
50 CNN features (bounded [-1, 1]) + 3 achieved_goal + 3 desired_goal = 56D
Goal vectors are 10.7% of input (vs 2.3% with default 256D CNN)
```

**Why 50?** DrQ-v2 uses feature_dim=50 as their default. SB3's default
of 256 was chosen for standalone CNN observations (no mixed modalities).
50 is sufficient for DMControl manipulation (demonstrated in the paper).

### 2. Frame stacking in PixelObservationWrapper

**File:** `scripts/labs/pixel_wrapper.py`
**Change:** Added `frame_stack` parameter to `PixelObservationWrapper`

When `frame_stack > 1`:
- Maintains a `collections.deque` of the last N rendered frames
- On `reset()`: fills deque with N copies of the first frame
- On `step()`: appends new frame, drops oldest
- Concatenates along channel dimension: (3, 84, 84) x4 -> (12, 84, 84)
- Only "pixels" key is stacked; "achieved_goal" and "desired_goal"
  remain instantaneous 3D vectors (NOT stacked)

**Why not VecFrameStack?** SB3's `VecFrameStack` stacks ALL dict keys,
which would incorrectly stack the 3D goal vectors into 12D. Our wrapper
selectively stacks only the pixel key.

**Reference:** Mnih et al. (2015) used 4 stacked frames for all Atari
games. DrQ and DrQ-v2 also use frame stacking.

### 3. CLI flags in ch10_visual_reach.py

**File:** `scripts/ch10_visual_reach.py`

New config fields:
- `norm: bool = False` — use NormalizedCombinedExtractor
- `cnn_dim: int = 50` — CNN output dimension (default 50 when --norm)
- `frame_stack: int = 1` — number of frames to stack

New CLI flags (added to train-push-pixel-her, train-push-pixel-her-drq,
push-all subcommands):
- `--norm` — enable NormalizedCombinedExtractor
- `--cnn-dim N` — CNN output dimension (auto-defaults to 50 with --norm)
- `--frame-stack N` — frame stacking (1=none, 4=standard)

Wiring:
- Both `cmd_train_push_pixel_her` and `cmd_train_push_pixel_her_drq`
  conditionally build `policy_kwargs` with the custom extractor
- Both pass `frame_stack` to the `PixelObservationWrapper` constructor
- Metadata JSON records `norm`, `cnn_dim`, and `frame_stack` fields

### 4. Resolution in _config_from_args

When `--norm` is set and `--cnn-dim` is not explicitly provided, defaults
to 50 (DrQ-v2's value). When `--norm` is not set, `cnn_dim` is not used
(SB3's default CombinedExtractor with cnn_output_dim=256 applies).

---

## Smoke Tests Passed

| Test | Command | Result |
|------|---------|--------|
| norm only | `--norm --cnn-dim 50 --learning-rate 1e-4` | PASS (2K steps, no crash) |
| norm + frame stack | `--norm --cnn-dim 50 --learning-rate 1e-4 --frame-stack 4` | PASS (2K steps, no crash) |

Note: frame_stack=4 with buffer_size=1M triggers memory warning (34GB).
Need to reduce buffer to ~300K-500K for frame-stacked runs.

---

## Planned Training Runs (after current baselines finish)

| Run | Flags | Buffer | Steps | Expected |
|-----|-------|--------|-------|----------|
| pixel+HER+DrQ+norm+fs4 | `--norm --frame-stack 4 --learning-rate 1e-4` | 300K | 8M | >50% |
| pixel+HER+norm+fs4 | `--norm --frame-stack 4 --learning-rate 1e-4` | 300K | 8M | >30% |

If both fail, ablation grid:
| Run | What it isolates |
|-----|-----------------|
| norm only (no fs) | Is normalization alone sufficient? |
| fs only (no norm) | Is velocity info alone sufficient? |
| norm+fs+DrQ, 16M steps | Is it just a budget issue? |

---

## Files Changed

| File | What changed |
|------|--------------|
| `scripts/labs/visual_encoder.py` | Added `NormalizedCombinedExtractor` class (~70 lines) |
| `scripts/labs/pixel_wrapper.py` | Added `frame_stack` param + deque + reset logic (~30 lines) |
| `scripts/ch10_visual_reach.py` | Config fields, CLI flags, policy_kwargs wiring, metadata |
| `tasks/pixel_push_fix_plan.md` | Original diagnosis and plan (created earlier) |
| `tasks/pixel_push_changes.md` | This document |

---

## How the Three Fixes Address the Two Root Causes

| Fix | Root Cause 1 (scale mismatch) | Root Cause 2 (missing velocity) |
|-----|-------------------------------|--------------------------------|
| LayerNorm + Tanh | **Direct fix.** Bounds CNN to [-1,1], matching goal scale | No effect |
| cnn_dim 256->50 | **Amplifies fix.** Goals go from 2.3% to 10.7% of input | No effect |
| lr 3e-4 -> 1e-4 | Stabilizes CNN training (less noisy updates) | No effect |
| Frame stack (4) | No effect | **Direct fix.** Velocity from frame differences |

Both root causes are independent. Fixing only one might not be sufficient.
The full fix applies all four interventions together.

---

## Book Integration

If the fix runs succeed, Ch9's narrative gains depth:

```
Act 1: Measure the pixel penalty (Reach: state vs pixel vs DrQ)
Act 2: The real test (Push fails from dense, HER solves from state)
Act 3: Naive pixel+HER fails -- WHY?
  -> Scale mismatch: CNN noise drowns goal vectors
  -> Missing velocity: single frame has no temporal information
Act 4: Resolution -- normalize features + stack frames + DrQ
  -> NormalizedCombinedExtractor (DrQ-v2 trunk pattern)
  -> Frame stacking (Mnih et al. 2015)
  -> Push from pixels works: [PENDING]% success
```

The failure-then-fix story is more pedagogically valuable than if naive
pixel+HER had worked on the first try. It teaches a real engineering
principle: when mixing learned and known features, normalize the learned
features to match the scale of the known ones.
