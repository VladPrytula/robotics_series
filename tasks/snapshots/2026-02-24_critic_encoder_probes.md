# Snapshot: Pixel Push -- Critic-Encoder + Hyperparameter Probes

**Date:** 2026-02-24
**Status:** Phase 1 complete, hyperparameter probe phase started

---

## 1. Where We Are in the Big Picture

### The problem

FetchPush-v4 from pixels is unsolved. State-based SAC+HER reaches 89-99%
success by 2M steps (Ch4). Every pixel-based configuration tried so far
oscillates at 2-14% success indefinitely -- no hockey-stick inflection.

### Timeline of attempts

| # | Date | Config | Steps | Success | Notes |
|---|------|--------|-------|---------|-------|
| 1 | Feb ~17 | NatureCNN(256) + HER, lr=3e-4 | 8M | 5-8% flat | Baseline |
| 2 | Feb ~17 | NatureCNN(256) + HER + DrQ, lr=3e-4 | 8M | 5-8% flat | + augmentation |
| 3 | Feb ~19 | NormExtractor(50) + HER + fs4, lr=1e-4 | 3M | 4-8% flat | + normalization + stacking |
| 4 | Feb ~19 | NormExtractor(50) + HER + DrQ + fs4, lr=1e-4 | 3M | 5-7% flat | + augmentation |
| 5 | Feb ~21 | ManipulationCNN + SS + proprio + DrQ + HER, lr=1e-4 | 2M | 2-14% osc | Research-backed arch |
| 6 | Feb ~22 | Proprio-only (16D, no pixels), lr=1e-4 | 2M | 0% | POMDP -- missing obj dynamics |
| 7 | Feb 23 | Full-state 25D + HER, lr=3e-4, n_envs=8, buf=1M | 2M | **89%** | Pipeline validated |
| 8 | Feb 23 | **Critic-encoder** + ManipCNN + SS + DrQ, lr=1e-4 | 1.4M+ | 6.7% mean | Run A (running) |
| 9 | Feb 24 | **Critic-encoder** + same, **lr=3e-4** | 5K+ | -- | Probe 1 (just launched) |

### Key discoveries along the way

1. **NatureCNN is wrong for manipulation** (Lesson 2): 8x8 stride-4 first
   layer destroys 3-5 pixel objects. Fixed with ManipulationCNN (3x3, stride 2).
   See `tasks/pixel_push_encoder_research.md` Section 2.

2. **Proprioception is essential** (Lesson 1): CNN should learn world-state,
   not self-state. 10D robot proprioception passed alongside pixels.
   See `tasks/lessons.md` Lesson 1.

3. **SB3 gradient routing is backwards from DrQ-v2** (Key Discovery):
   - SB3 `share_features_extractor=True`: encoder in **actor** optimizer only,
     critic uses `set_grad_enabled(False)`
   - DrQ-v2 paper: encoder in **critic** optimizer only, actor gets `obs.detach()`
   - Our custom `DrQv2SACPolicy` implements the DrQ-v2 pattern
   - Confirmed empirically with SB3 gradient probe (all 5 tests pass)
   - See `tasks/critic_encoder_status.md`, `scripts/labs/drqv2_sac_policy.py`

4. **Full-state control validates pipeline** (Run 7): 89% success at 2M steps
   with Ch9 script + `--full-state`, matching Ch4's known-good result.
   Hyperparams: lr=3e-4, buffer=1M, n_envs=8, gamma=0.95, ent_coef=0.05.

---

## 2. The Architecture Stack (Current)

### Observation pipeline

```
MuJoCo render (480x480) --> PIL resize (84x84) --> uint8
  --> PixelObservationWrapper: frame_stack=4 --> (12, 84, 84) uint8
  --> Proprioception: 10D float64 (grip_pos, gripper_state, grip_velp, gripper_vel)
  --> Goals: achieved_goal (3D), desired_goal (3D)
```

### Encoder: ManipulationExtractor (80D output)

```
ManipulationCNN (3x3 kernels):
  Conv(12, 32, 3, stride=2) --> ReLU     84x84 --> 42x42
  Conv(32, 32, 3, stride=1) --> ReLU     42x42 --> 42x42
  Conv(32, 32, 3, stride=1) --> ReLU     42x42 --> 42x42
  Conv(32, 32, 3, stride=2) --> ReLU     42x42 --> 21x21

SpatialSoftmax (learnable temperature):
  Per-channel expected (x, y) --> 64D (32 channels x 2 coords)

Concat: 64 spatial + 10 proprio + 3 achieved_goal + 3 desired_goal = 80D
```

Source: `scripts/labs/manipulation_encoder.py`

### Gradient routing: DrQv2SACPolicy

```
Shared encoder (in critic optimizer):
  critic_loss.backward() --> encoder params get gradients --> encoder learns
  actor_loss.backward() --> encoder features detached --> encoder untouched

Classes:
  CriticEncoderCritic  -- always enables grads through features_extractor
  CriticEncoderActor   -- calls features.detach() before policy MLP
  DrQv2SACPolicy       -- shared encoder in critic opt, excluded from actor opt
                          (identity-based param filtering for robustness)
```

Source: `scripts/labs/drqv2_sac_policy.py`

### Augmentation + Replay

- DrQ random shift (pad=4) applied on minibatch sample
- `HerDrQDictReplayBuffer`: HER relabeling + DrQ augmentation + uint8 pixel storage
- HER: future strategy, n_sampled_goal=4

Source: `scripts/labs/image_augmentation.py`

### Training: SAC + HER

- Algorithm: SAC (SB3 v2.7.1), HER with future goal relabeling
- Default hyperparams for pixel runs: lr=1e-4, gamma=0.95, ent_coef=0.05 (fixed),
  tau=0.005, batch_size=256, buffer_size=200K, n_envs=4, gradient_steps=1

---

## 3. Current Experiments

### Run A: Critic-Encoder Baseline (lr=1e-4) -- RUNNING

**Command:**
```bash
docker run --rm \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYTHONUNBUFFERED=1 \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e TORCH_HOME=/tmp/.cache/torch \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torch_inductor \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib -e USER=user -e LOGNAME=user \
  -v "$PWD:/workspace" -w /workspace --gpus all --ipc=host \
  robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python scripts/ch09_pixel_push.py train \
    --seed 0 --critic-encoder --total-steps 2000000'
```

**Hyperparams:** lr=1e-4, buffer=200K, n_envs=4, batch=256, gamma=0.95,
ent_coef=0.05, gradient_steps=1, HER n_sampled_goal=4

**Status at 1.41M / 2M steps (Feb 24, ~14.6h elapsed):**
- success_rate: 6.7% mean (flat across entire run, oscillates 2-13%)
- critic_loss: declined from 0.27 (200K) to 0.09 (1M+) -- converging to "everything fails"
- actor_loss: 1.9-3.1
- ep_rew_mean: -44 to -48
- fps: 25-26

**Mean success rate per 200K bucket (dead flat):**
```
  0-200K    6.7%
200-400K    6.8%
400-600K    5.8%
600-800K    6.7%
 800K-1M    6.6%
1.0-1.2M    6.8%
1.2-1.4M    6.9%
```

**Assessment:** Stop rule at 1M was technically triggered (success <= 10% AND
critic_loss declining). However, we're letting it run to 2M for comparison
with the LR probe. The flat mean with declining critic loss indicates the
encoder reached its representational quality early and stopped improving --
the critic is learning "all states are equally bad" rather than useful value
gradients.

**Monitor:** `tail -f /tmp/claude-1001/-home-vladp-src-robotics/tasks/bf636de.output`

### Probe 1: LR 3e-4 (highest-leverage change) -- JUST LAUNCHED

**Command:**
```bash
docker run --rm \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYTHONUNBUFFERED=1 \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e TORCH_HOME=/tmp/.cache/torch \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torch_inductor \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib -e USER=user -e LOGNAME=user \
  -v "$PWD:/workspace" -w /workspace --gpus all --ipc=host \
  robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python scripts/ch09_pixel_push.py train \
    --seed 0 --critic-encoder --learning-rate 3e-4 --total-steps 2000000'
```

**Hyperparams:** lr=**3e-4** (changed), buffer=200K, n_envs=4, batch=256,
gamma=0.95, ent_coef=0.05, gradient_steps=1, HER n_sampled_goal=4

**Rationale:** The full-state control that reached 89% used lr=3e-4. Run A uses
1e-4 (3x lower). With critic-encoder routing, 3x stronger critic gradients
should drive more aggressive encoder representation learning. This is the
single largest hyperparameter deviation from the working full-state config.

**Monitor:** `tail -f /tmp/claude-1001/-home-vladp-src-robotics/tasks/bb747bd.output`

### Run 0: Full-State Control -- COMPLETED

**Result:** 89% success at 2M steps
**Training time:** 7388s (~2h) at 271 fps
**Hyperparams:** lr=3e-4, buffer=1M, n_envs=8, batch=256, gamma=0.95, ent_coef=0.05
**Checkpoint:** `checkpoints/ch09_manip_noPix_fullState_FetchPush-v4_seed0.zip`
**Metadata:** `checkpoints/ch09_manip_noPix_fullState_FetchPush-v4_seed0.meta.json`

**Purpose:** Validates that the Ch9 script's SAC+HER wiring is correct independently
of pixel processing. Matches Ch4's known-good result.

---

## 4. Hyperparameter Probe Plan

### Why probes before code changes

Run A's flat 6.7% mean across 1.4M steps suggests the current hyperparams may
be too weak for pixel-based learning, not necessarily that the architecture or
gradient routing is wrong. Before changing code, we test hyperparameter changes
that existing CLI flags already support.

### Critical confound identified

Run A differs from the full-state control not just in pixel-vs-state but in
hyperparameters:

| Hyperparam | Full-state (89%) | Run A (6.7%) | Ratio |
|-----------|-----------------|-------------|-------|
| learning_rate | **3e-4** | 1e-4 | 3x lower |
| buffer_size | **1,000,000** | 200,000 | 5x smaller |
| n_envs | **8** | 4 | 2x fewer |

### Probe schedule (one variable at a time, stop at 2M)

| # | Change | Command delta | Status | Rationale |
|---|--------|--------------|--------|-----------|
| **Probe 1** | LR 3e-4 | `--learning-rate 3e-4` | **RUNNING** | Matches full-state; strongest encoder gradient signal |
| Probe 2 | gradient_steps 2 | `--gradient-steps 2` | PENDING | Doubles critic updates per env step |
| Probe 3 | HER-8 | `--her-n-sampled-goal 8` | PENDING | More positive targets in sparse regime |
| Probe 4 | n_envs 8 | `--n-envs 8` | PENDING | More decorrelated data |
| Probe 5 | gamma 0.98 | `--gamma 0.98` | PENDING (lower priority) | Longer effective horizon |
| Probe 6 | ent_coef 0.1 | `--ent-coef 0.1` | PENDING (optional) | More exploration pressure |

### Hardware constraints

| Resource | Total | Run A | Probe 1 | Available |
|----------|-------|-------|---------|-----------|
| GPU | 1x NVIDIA GB10 (unified mem) | ~1.2 GiB | ~1.2 GiB | Shared, not bottleneck |
| CPU | 20 cores | ~5 cores | ~5 cores | ~10 cores |
| RAM | 119 GiB | ~35 GiB | ~35 GiB | ~49 GiB |

**Constraint:** RAM limits us to 2 concurrent pixel experiments (~35 GiB each).
Probes 2-6 must wait for Run A or Probe 1 to finish.

### What to watch for

- **Probe 1 vs Run A early signal (by 100-200K):** Does critic_loss stay elevated
  longer with 3x LR? Does success rate distribute higher (more 8-12% readings,
  fewer 2-3% dips)?
- **Hockey-stick signature:** Critic loss RISING between 1-1.5M (policy discovers
  new states that surprise the critic), followed by success rate acceleration.
- **Failure signature:** Critic loss declining monotonically while success stays
  flat = encoder is learning "all states are equally bad."

---

## 5. Monitoring Milestones

### For Probe 1 (and future probes)

| Steps | Green light | Red flag |
|-------|------------|---------|
| 200K | success >= 1%, critic_loss NOT declining | critic_loss monotonically declining |
| 500K | success >= 10% sustained | flat at 5% (same as Run A) |
| 1.0M | success >= 15% with upward trend | <= 10% and critic_loss declining |
| 1.5M | Hockey-stick: success >= 30%, critic_loss rising | Still flat |
| 2.0M | success > 50% | Below 15% |

### Hockey-stick reference (full-state, from Ch4/Run 0)

From `tasks/hockey_stick_research.md`:
```
0-1.2M     3-10%  (flat)
1.28M      14%    (inflection begins)
1.36M      19%    (accelerating)
1.40M      32%    (positive feedback active)
1.52M      52%    (steep climb)
1.64M      77%
1.84M      90%    (near-saturation)
```

Pixel learning may need 1.5-2x more steps for the encoder warm-up phase,
pushing the potential inflection to 1.5-2.5M. This is why probes run to 2M.

---

## 6. If All Probes Fail

Ordered by cost (cheapest first):

1. **DrQ augmentation tightening** (no code): pad=8 instead of 4 -- stronger
   shift invariance, nearly free to test.

2. **Combined probe** (no code): LR 3e-4 + gradient_steps 2 + HER-8 together --
   if individual probes show partial improvement, combine them.

3. **CURL contrastive head** (code change): Self-supervised loss on encoder
   latents alongside TD loss. Reference: Srinivas et al. 2020 (arXiv:2004.04136).

4. **Frozen pretrained encoder** (code change): R3M or VC-1, bypass end-to-end
   learning entirely. Reference: Nair et al. 2022 (arXiv:2203.12601).

5. **Latent-goal HER** (narrative change): RIG/Skew-Fit style -- train VAE,
   do HER in latent space. Reference: Nair et al. 2018 (arXiv:1807.04742).

See `tasks/further_researc_and_plana.md` Phases 2-4 for details.

---

## 7. File Inventory

### Implementation files

| File | Status | Purpose |
|------|--------|---------|
| `scripts/ch09_pixel_push.py` | Modified | Main training script; `--critic-encoder`, `--full-state` flags |
| `scripts/labs/drqv2_sac_policy.py` | **NEW** | DrQv2SACPolicy + verification suite (`--verify`, `--probe`) |
| `scripts/labs/manipulation_encoder.py` | Unchanged | ManipulationCNN + SpatialSoftmax + ManipulationExtractor |
| `scripts/labs/pixel_wrapper.py` | Modified | PixelObservationWrapper (frame stack, proprio, goal_mode) |
| `scripts/labs/image_augmentation.py` | Unchanged | HerDrQDictReplayBuffer, RandomShiftAug |

### Research & planning documents

| File | Content |
|------|---------|
| `tasks/pixel_push_encoder_research.md` | Comprehensive encoder research (NatureCNN analysis, ManipCNN design, literature) |
| `tasks/further_researc_and_plana.md` | Multi-phase experiment plan (Phases 0-4) |
| `tasks/critic_encoder_status.md` | Implementation status, verification results, training commands |
| `tasks/hockey_stick_research.md` | Hockey-stick learning curve analysis (3 mechanisms) |
| `tasks/ch09_pixel_pixels_stack_notes.md` | 84x84 sufficiency, stacking, targeted experiments |
| `tasks/lessons.md` | 7 lessons learned (sensor separation, NatureCNN, gradient flow, etc.) |
| `tasks/snapshots/2026-02-24_critic_encoder_probes.md` | **THIS FILE** |

### Artifacts

| Artifact | Path |
|----------|------|
| Full-state checkpoint | `checkpoints/ch09_manip_noPix_fullState_FetchPush-v4_seed0.zip` |
| Full-state metadata | `checkpoints/ch09_manip_noPix_fullState_FetchPush-v4_seed0.meta.json` |
| TB logs (full-state) | `runs/ch09_manip_noPix_fullState/FetchPush-v4/seed0/` |
| TB logs (critic-enc Run A) | `runs/ch09_manip_criticEnc/FetchPush-v4/seed0_1/` |
| Run A live output | `/tmp/claude-1001/-home-vladp-src-robotics/tasks/bf636de.output` |
| Probe 1 live output | `/tmp/claude-1001/-home-vladp-src-robotics/tasks/bb747bd.output` |

---

## 8. Verification Checklist (All Pass)

These were run on 2026-02-23 before launching training:

| Test | Result | Command |
|------|--------|---------|
| SB3 gradient probe | PASS | `python scripts/labs/drqv2_sac_policy.py --probe` |
| Optimizer param membership | PASS (0/11 encoder in actor, 11/11 in critic) | `--verify` |
| Gradient flow (critic step) | PASS (encoder has grad, params change) | `--verify` |
| Gradient flow (actor step) | PASS (encoder NO grad, params unchanged) | `--verify` |
| Actor-through-critic path | PASS (critic fwd during actor doesn't corrupt) | `--verify` |
| Target critic Polyak | PASS (target encoder updated by soft update) | `--verify` |
| SB3 50-step training | PASS (DrQv2SACPolicy + ManipulationExtractor) | `--verify` |
| Save/load round-trip | PASS (predictions match, policy type preserved) | `--verify` |
| smoke-test --critic-encoder | PASS (10K steps, full pixel pipeline) | ch09 smoke-test |
| smoke-test --full-state | PASS (10K steps, raw 25D obs) | ch09 smoke-test |

---

## 9. Key Decisions and Reasoning

### Why critic-updates-encoder (not SB3's default sharing)

SB3's `share_features_extractor=True` routes encoder gradients through the
**actor** loss only. The actor loss is noisy policy gradient at low success rates
(~6% success = 94% of actor gradients are from failed trajectories). The critic
loss provides a denser, more stable TD signal. DrQ-v2 (Yarats et al. 2021)
established this as the correct pattern: encoder in critic optimizer, actor
gets detached features.

See: `tasks/further_researc_and_plana.md` decision D2,
`tasks/lessons.md` Lesson 7 (gradient flow before model complexity).

### Why LR 3e-4 as highest-priority probe

The original choice of lr=1e-4 was based on generic "pixel training needs lower
lr" advice. But with critic-encoder routing, the encoder's learning signal comes
from critic TD loss, which benefits from stronger gradients. The full-state
control used lr=3e-4 and succeeded. This is a 3x difference -- the single
largest hyperparameter gap between the working and failing configs.

### Why 2M steps (not 1M) for probes

The full-state hockey-stick started at ~1.28M steps. Pixel-based learning adds
an encoder warm-up phase (representation bootstrapping) that could shift the
inflection 1.5-2x later. A 1M cutoff risks killing a run that would have
succeeded at 1.5M. The flat trajectory of Run A (6.7% mean for 1.4M steps) is
concerning, but we want each probe to have the full window for comparison.

### Why identity-based optimizer param filtering

Initial implementation used name-based filtering (`"features_extractor" not in n`).
Switched to identity-based filtering (`id(p) not in encoder_param_ids`) for
robustness across SB3 versions where parameter naming conventions may vary.

---

## 10. How to Resume This Work

### Immediate next steps

1. **Monitor Probe 1 and Run A** -- both running to 2M steps
   ```bash
   # Quick status check
   grep "success_rate" /tmp/claude-1001/-home-vladp-src-robotics/tasks/bb747bd.output | tail -5
   grep "success_rate" /tmp/claude-1001/-home-vladp-src-robotics/tasks/bf636de.output | tail -5
   ```

2. **When a slot opens** (either finishes or is stopped), launch next probe:
   ```bash
   # Probe 2: gradient_steps 2
   docker run --rm \
     -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYTHONUNBUFFERED=1 \
     -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e TORCH_HOME=/tmp/.cache/torch \
     -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torch_inductor \
     -e MPLCONFIGDIR=/tmp/.cache/matplotlib -e USER=user -e LOGNAME=user \
     -v "$PWD:/workspace" -w /workspace --gpus all --ipc=host \
     robotics-rl:latest \
     bash -c 'source .venv/bin/activate && python scripts/ch09_pixel_push.py train \
       --seed 0 --critic-encoder --gradient-steps 2 --total-steps 2000000'

   # Probe 3: HER n_sampled_goal 8
   # Same docker pattern, add: --her-n-sampled-goal 8
   ```

3. **Compare trajectories** at 500K and 1M:
   - Extract mean success per 200K bucket (see Section 3 analysis)
   - Compare critic_loss trajectories
   - If Probe 1 (LR 3e-4) shows upward trend, run seeds 1 and 2

4. **If no probe shows improvement by 2M**, move to fallback plan (Section 6)

### Key commands reference

```bash
# Run verification suite
docker run --rm [env vars] robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python scripts/labs/drqv2_sac_policy.py --verify'

# Smoke test
docker run --rm [env vars] robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python scripts/ch09_pixel_push.py smoke-test --seed 0 --critic-encoder'

# Feature stats (check encoder output distribution)
docker run --rm [env vars] robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python scripts/ch09_pixel_push.py feature-stats --seed 0'

# Evaluate checkpoint
docker run --rm [env vars] robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python scripts/ch09_pixel_push.py eval \
    --ckpt checkpoints/ch09_manip_criticEnc_FetchPush-v4_seed0.zip --critic-encoder'

# TensorBoard
docker run --rm -p 6006:6006 [env vars] robotics-rl:latest \
  bash -c 'source .venv/bin/activate && tensorboard --logdir runs --bind_all'
```

### Session IDs (for Claude Code context recovery)

- Run A background task: `bf636de`
- Probe 1 background task: `bb747bd`
- Plan file: `/home/vladp/.claude/plans/cozy-riding-hopcroft.md`

---

## 11. Open Questions

1. **Is 2M steps sufficient for pixel Push?** The full-state hockey-stick starts
   at 1.28M. With encoder warm-up, the pixel inflection might need 3-5M. But
   Run A's dead-flat mean (no upward trend at all) suggests the issue is gradient
   magnitude, not patience.

2. **Should we combine probes?** If Probe 1 shows partial improvement (e.g.,
   mean rises from 6.7% to 12% but no hockey-stick), combining LR 3e-4 +
   gradient_steps 2 + HER-8 might push past the threshold. This sacrifices
   one-variable-at-a-time purity for practical progress.

3. **Buffer size confound:** Run A uses 200K buffer vs full-state's 1M. With
   200K and n_envs=4 at 50 steps/episode, the buffer cycles every ~1M steps.
   This means old data is overwritten before the hockey-stick phase begins.
   A larger buffer might help by preserving diverse early experience.

4. **Is this problem solvable end-to-end?** No published work demonstrates
   pixel + sparse HER + end-to-end CNN succeeding on manipulation. If all
   probes fail, this validates the research finding and motivates the narrative
   pivot to auxiliary representation learning (CURL) or pretrained encoders (R3M).
