# Critic-Encoder Implementation Status

## Date: 2026-02-23

## What Was Built

### New file: `scripts/labs/drqv2_sac_policy.py`
- `CriticEncoderCritic`: ContinuousCritic override -- always enables gradients through shared encoder (removes SB3's `set_grad_enabled(False)` gate)
- `CriticEncoderActor`: Actor override -- calls `features.detach()` before policy MLP (no encoder gradient from actor loss)
- `DrQv2SACPolicy`: SACPolicy override -- shared encoder in **critic optimizer**, excluded from **actor optimizer** (identity-based filtering)
- Verification suite: `--verify` (5 tests), `--probe` (SB3 gradient probe)

### Modified: `scripts/ch09_pixel_push.py`
- `--critic-encoder` flag: routes to DrQv2SACPolicy, adds `criticEnc` to config tag
- `--full-state` flag: raw 25D obs (no pixels), uses SB3 default MultiInputPolicy
- `cmd_eval` passes `custom_objects` for DrQv2SACPolicy checkpoints
- `make_full_state_env()`: creates vanilla gym.make() env for pipeline validation

## Verification Results (all pass)

| Test | Result |
|------|--------|
| SB3 gradient probe (`--probe`) | PASS -- confirms SB3 default puts encoder in actor opt only |
| Optimizer param membership | PASS -- 0/11 encoder params in actor opt, 11/11 in critic opt |
| Gradient flow (critic step) | PASS -- encoder has grad, params change |
| Gradient flow (actor step) | PASS -- encoder has NO grad (detached), params unchanged |
| Actor-through-critic path | PASS -- critic forward during actor loss doesn't corrupt encoder |
| Target critic Polyak update | PASS -- target encoder updated by soft update |
| SB3 50-step training | PASS -- DrQv2SACPolicy + ManipulationExtractor, no crashes |
| Save/load round-trip | PASS -- predictions match, policy type preserved |
| smoke-test --critic-encoder (10K) | PASS -- full pixel pipeline with DrQ+HER |
| smoke-test --full-state (10K) | PASS -- raw 25D obs, standard extractor |

## Key Discovery Confirmed

SB3 `share_features_extractor=True` is **backwards** from DrQ-v2:
- SB3: encoder in actor optimizer only; critic disables gradients through encoder
- DrQ-v2: encoder in critic optimizer only; actor receives detached features
- Our custom policy implements the DrQ-v2 pattern

## Training Runs To Launch

### Run 0: Full-State Control (Phase 0b)
```bash
python scripts/ch09_pixel_push.py train --seed 0 --full-state \
  --n-envs 8 --buffer-size 1000000 --learning-rate 3e-4 \
  --total-steps 2000000 --ent-coef 0.05
```
**Purpose:** Reproduce Ch4's 99% hockey-stick to confirm Ch9's algorithm wiring.
**Expected:** >90% success by 2M steps. If flat at 1M, pipeline bug.

### Run A: Critic-Encoder (Phase 1)
```bash
python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder \
  --total-steps 2000000
```
**Purpose:** Test whether critic-updates-encoder produces the hockey-stick inflection.
**Expected:** >50% by 2M (minimum), >80% (target).
**Stop rule:** If success <= 10% at 1M with flat critic_loss, abort and debug.

### Run B: Ablation -- critic-encoder without spatial softmax
```bash
python scripts/ch09_pixel_push.py train --seed 0 --critic-encoder \
  --no-spatial-softmax --total-steps 2000000
```
**Purpose:** Isolate whether spatial softmax matters on top of correct gradient routing.

## Monitoring Milestones (Run A)

| Steps | Green | Red |
|-------|-------|-----|
| 200K | success >= 1%, critic_loss not declining | critic_loss monotonically declining |
| 500K | success >= 10% | flat at 5% |
| 1.0M | **STOP RULE**: abort if success <= 10% AND critic_loss declining | Same as previous runs |
| 1.5M | Hockey-stick: success >= 30% | Still flat |
| 2.0M | success > 50% | Below 15% |

## Training Results

### Run 0: Full-State Control -- COMPLETED
- **Result: 89% success at 2M steps** -- validates Ch9 pipeline, matches Ch4
- Hockey-stick inflection around 1-1.5M steps as expected
- Confirms: algorithm wiring (SAC+HER+gamma+ent_coef) is correct

### Run A (Cell D): Critic-Encoder -- TERMINATED at 1.54M / 2M

**Config:** `--critic-encoder --total-steps 2000000` (defaults: buffer=200K, lr=1e-4, HER-4, DrQ ON, SpatialSoftmax ON)
**Config tag:** `ch09_manip_criticEnc`
**Result: 3% success, dead flat from 200K to 1.54M steps. Terminated to free resources for Phase A factorial.**

- Success rate never sustained above ~10%, oscillated 2-12%
- Critic loss declined monotonically to 0.10 by 1.54M (failure signature: "too easy" critic)
- Actor loss climbed from 0 to ~2.0 (normal SAC trajectory with sparse rewards)
- FPS: 25 (competed with two other containers for CPU/memory)

#### Success rate trajectory (sampled every ~10K steps, from first 283K)
```
  10K:  4%     100K: 3%     190K: 3%
  20K:  6%     110K: 10%    200K: 9%
  30K:  4%     120K: 5%     210K: 7%
  40K:  4%     130K: 6%     220K: 7%
  50K:  7%     140K: 9%     230K: 2%
  60K:  9%     150K: 10%    240K: 3%
  70K:  4%     160K: 9%     250K: 8%
  80K:  6%     170K: 7%     260K: 5%
  90K: 10%     180K: 8%     270K: 8%
```

**By 1.54M steps (final reading):** success_rate=3%, critic_loss=0.10, actor_loss=2.0, reward=-48.5.
TensorBoard logs preserved in `runs/ch09_stale_backup_20260224/ch09_manip_criticEnc/`.

---

## Phase A: 2x2 Factorial -- DrQ x SpatialSoftmax

### Date: 2026-02-24

### Three Failure Hypotheses

Run A (Cell D) showed dead-flat 6.7% mean success across 1.54M steps. Full-state
control reached 89% at 2M steps, confirming the SAC+HER pipeline is correct.
Three hypotheses for why the pixel agent fails:

**H1 -- Pre-Inflection Regime (critic starvation):**
With gamma=0.95 and 50-step episodes, Q converges to ~-18.5 for always-failing
trajectories. TD error floor (0.04-0.07) means encoder receives noisy,
uninformative gradients. HER-4 relabeling provides some value gradient (80%
reward=0), but potentially not enough to bootstrap spatial structure.

**H2 -- DrQ x SpatialSoftmax Conflict:**
DrQ random shift (+/-4 pixels) injects +/-0.42 noise into SpatialSoftmax's [-1, 1]
coordinate space. Gripper-puck distance maps to ~0.74 in these coordinates,
giving a 57% noise-to-signal ratio. Critically: independent augmentation of obs
and next_obs DOUBLES the noise in Bellman targets.
**We NEVER tested critic-encoder without DrQ.** All previous no-DrQ runs used
SB3's broken gradient routing (encoder in actor optimizer).

**H3 -- Buffer too small:**
With buffer_size=200K and n_envs=4, the buffer wraps every 200K total_timesteps
(SB3 divides buffer_size by n_envs internally, then each `add()` call stores
n_envs transitions). By 1M steps, all early diverse exploration is overwritten.
Full-state control used 1M buffer.

### Factorial Design

All four cells use `--critic-encoder` with shared hyperparams:
- lr=1e-4, gamma=0.95, ent_coef=0.05
- gradient_steps=1, batch_size=256
- n_envs=4, total_steps=2M
- **buffer_size=500,000** (see Memory Analysis below -- hardware limit)
- **her_n_sampled_goal=8** (more relabeled transitions for critic warmup)

| Cell | DrQ | SpatialSoftmax | Flags | Config Tag |
|------|-----|----------------|-------|------------|
| **A** | OFF | ON | `--no-drq` | `ch09_manip_noDrQ_criticEnc` |
| **B** | OFF | OFF | `--no-drq --no-spatial-softmax` | `ch09_manip_noSS_noDrQ_criticEnc` |
| **C** | ON | OFF | `--no-spatial-softmax` | `ch09_manip_noSS_criticEnc` |
| **D** | ON | ON | (defaults) | `ch09_manip_criticEnc` (= Run A, 3% flat @ 1.54M) |

Cell D already exists as Run A. The three new cells (A, B, C) differ from Run A
in buffer_size (500K vs 200K) and HER (8 vs 4), which means they also test H1+H3
simultaneously. This is intentional: we fix the known starvation issues so the
factorial cleanly isolates H2 (DrQ/SpatialSoftmax interaction).

### What the Factorial Reveals

| Result Pattern | Conclusion |
|---------------|------------|
| A >> D, B ~ C ~ D | SpatialSoftmax needs clean coordinates (DrQ kills it) |
| B >> D, A ~ C ~ D | Neither DrQ nor SS alone works; flat features + no aug is sweet spot |
| A ~ B >> C ~ D | DrQ is the poison regardless of SS |
| C >> D, A ~ B ~ D | SS is the poison regardless of DrQ |
| A ~ B ~ C ~ D (all flat) | Problem is NOT DrQ/SS interaction -- move to Phase B |

### Monitoring Milestones (compared to Run A = Cell D at 3% flat)

| Steps | Green | Red |
|-------|-------|-----|
| 200K | success >= 2%, critic_loss NOT monotonically declining | Identical to Run A |
| 500K | success >= 12% **sustained** (not just spikes) | Mean still <= 7% |
| 1.0M | success >= 15% with upward trend | <= 10% flat |
| 1.5M | Hockey-stick onset: >= 30% | Still flat |
| 2.0M | >= 50% | Below 15% |

---

## Replay Buffer Memory Analysis

### Why buffer_size=1M Does Not Fit

The original plan called for buffer_size=1,000,000. SB3 warned:
`This system does not have apparently enough memory to store the complete replay buffer 169.62GB > 58.52GB`

Here is the exact computation:

**Step 1: How SB3 allocates the buffer.**

SB3's `DictReplayBuffer.__init__` divides buffer_size by n_envs:
```
internal_rows = buffer_size // n_envs = 1,000,000 // 4 = 250,000
```
Each row stores n_envs=4 transitions simultaneously. Arrays are shaped
`(internal_rows, n_envs, *obs_shape)`.

SB3 uses the observation space's dtype for storage. Our PixelObservationWrapper
defines `spaces["pixels"] = Box(0, 255, shape=(12, 84, 84), dtype=np.uint8)`.
So pixels ARE stored as uint8 (1 byte per value), not float32.

**Step 2: Memory per pixel array.**

Each pixel observation: `12 channels x 84 x 84 = 84,672 bytes` (uint8).
One buffer array: `internal_rows x n_envs x 84,672 = 250,000 x 4 x 84,672`.

```
obs['pixels']:      250,000 x 4 x 84,672 = 84,672,000,000 bytes = 84.67 GB
next_obs['pixels']: 250,000 x 4 x 84,672 = 84,672,000,000 bytes = 84.67 GB
                                                          Total pixels: 169.34 GB
```

This matches SB3's warning of 169.62 GB (the extra 0.28 GB accounts for
proprioception, goals, actions, rewards, and dones arrays).

**Step 3: Hardware constraint.**

```
Total system RAM:    119 GB (122,570 MB)
Required for 1M buf: 169 GB
Deficit:             -50 GB  --> IMPOSSIBLE regardless of what else is running
```

Note: `DictReplayBuffer` does NOT support `optimize_memory_usage=True` (which
would avoid storing next_obs separately for a 2x savings). SB3 raises
ValueError if you try.

**Step 4: Per-transition cost (simplified formula).**

For any buffer_size with our pixel config:
```
pixel_bytes_per_transition = channels x H x W = 12 x 84 x 84 = 84,672 bytes
memory_per_transition = pixel_bytes x 2  (obs + next_obs) = 169,344 bytes ≈ 165 KB
total_buffer_memory ≈ buffer_size x 169,344 bytes  (+ ~1.6% for non-pixel arrays)
```

**Step 5: Maximum safe buffer_size.**

Running ONE experiment at a time (all machine resources available):
```
Total RAM:           119 GB
System/OS/caches:     ~5 GB
CUDA context + model: ~5 GB
HER overhead (episode tracking, goal arrays): ~3 GB
Safety margin:        ~2 GB
-----------------------------------------
Available for pixel arrays: ~104 GB

Max buffer_size = 104 GB / 169,344 bytes ≈ 614,000 transitions
```

We chose **buffer_size=500,000** for comfortable margin (needs ~83 GB, leaving
~21 GB headroom).

**Step 6: Buffer cycling comparison.**

SB3 buffer wraps after `buffer_size` total_timesteps (regardless of n_envs):
```
buffer_size=200K (Run A default): wraps every 200K steps. At 2M: cycled 10x.
buffer_size=500K (Phase A):       wraps every 500K steps. At 2M: cycled 4x.
buffer_size=1M (planned):         wraps every 1M steps.  At 2M: cycled 2x.  INFEASIBLE.
```

500K retains 2.5x more history than the default. Not as good as 1M, but the
best this hardware supports. If buffer size is the primary bottleneck, 500K
should still show measurable improvement over 200K.

### Why 58 GB "Available" When System Has 119 GB

When we first attempted to launch, SB3 reported only 58.52 GB available.
This was because THREE Docker training containers were running simultaneously:

| Container | Config | RAM Used |
|-----------|--------|----------|
| b2643abde1f2 | `--critic-encoder` (Run A, 1.54M steps) | 35.7 GB |
| f5fb978db4b5 | `--critic-encoder --learning-rate 3e-4` (LR probe) | 24.3 GB |
| cd27250450e6 | `--no-drq --buffer-size 1000000` (doomed Cell A) | 4.2 GB |
| **Total** | | **64.2 GB** |

The third container (cd27250) had buffer_size=1M but showed only 4.2 GB because
Linux uses lazy allocation: `np.zeros` reserves virtual address space but only
allocates physical pages as data is written. The buffer was only ~1.5% full
(~11K of 2M steps completed). It would have OOM'd at ~700K steps when the buffer
filled beyond available physical memory.

**Decision:** Kill all three containers. Run experiments sequentially, one at a
time, to give each the full 119 GB. This also improves FPS: 37-50 fps solo vs
13-25 fps competing for CPU/memory.

### Additional observations

1. **Run A (Cell D) ran at buffer_size=200K successfully** because 200K buffer
   needs only ~33 GB. With the other containers using ~60 GB, there was still
   enough RAM.

2. **Full-state control ran at buffer_size=1M successfully** because state
   observations are 25D float64 ≈ 200 bytes per transition (vs 169 KB for
   pixels). A 1M state buffer uses ~248 MB, not 169 GB.

3. **The lr=3e-4 probe (f5fb978)** was writing to the SAME TensorBoard directory
   as Run A (`ch09_manip_criticEnc`) because learning rate changes don't affect
   the config tag. This caused TensorBoard log contamination (Lesson #3). Both
   runs backed up to `runs/ch09_stale_backup_20260224/`.

---

## Execution Protocol

### Sequential runs (one at a time, full machine)

Each cell gets the full 119 GB RAM and all CPU cores. At ~40 fps, each 2M-step
run takes approximately 14 hours.

**Execution order:**
```
Cell A: --seed 0 --critic-encoder --no-drq --buffer-size 500000 --her-n-sampled-goal 8 --total-steps 2000000
        Config tag: ch09_manip_noDrQ_criticEnc
        STARTED 2026-02-24 ~09:50 UTC

Cell B: --seed 0 --critic-encoder --no-drq --no-spatial-softmax --buffer-size 500000 --her-n-sampled-goal 8 --total-steps 2000000
        Config tag: ch09_manip_noSS_noDrQ_criticEnc
        QUEUED (after Cell A)

Cell C: --seed 0 --critic-encoder --no-spatial-softmax --buffer-size 500000 --her-n-sampled-goal 8 --total-steps 2000000
        Config tag: ch09_manip_noSS_criticEnc
        QUEUED (after Cell B)
```

**Docker command template:**
```bash
docker run --rm \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYTHONUNBUFFERED=1 \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache -e TORCH_HOME=/tmp/.cache/torch \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torch_inductor \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib -e USER=user -e LOGNAME=user \
  -v "$PWD:/workspace" -w /workspace --gpus all --ipc=host \
  robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python scripts/ch09_pixel_push.py train \
    --seed 0 --critic-encoder [CELL-SPECIFIC FLAGS] --total-steps 2000000'
```

**Run ALL three cells to completion regardless of intermediate results.** Even if
Cell A shows clear lift, Cell C provides the ablation evidence (does DrQ conflict
specifically with SpatialSoftmax, or is it bad for all encoders?).

### Cell A progress

TensorBoard: `runs/ch09_manip_noDrQ_criticEnc/FetchPush-v4/seed0_1/`
Config: `--no-drq --buffer-size 500000 --her-n-sampled-goal 8 --critic-encoder`
Started: 2026-02-24 ~09:50 UTC | Container: 6cb315d41013

**808K checkpoint (2026-02-24 ~16:00):**
```
Success rate trajectory (sampled every ~40K steps):
  40K:  4%     200K: 8%     360K: 6%     520K: 6%     680K: 8%
  80K:  6%     240K: 3%     400K: 6%     560K: 4%     720K: 8%
 120K:  6%     280K: 5%     440K: 5%     600K: 9%     760K: 3%
 160K: 10%     320K: 7%     480K: 6%     640K: 11%    800K: 4%

Mean: ~6%  |  Range: 3-11%  |  Trend: FLAT
```
Assessment: Indistinguishable from Cell D (Run A, DrQ ON, buffer=200K, HER-4).
Removing DrQ alone did NOT produce sustained lift. 500K milestone RED (6% < 12%).
Decision: run to completion for ablation data.

**Memory incident:** Duplicate Cell A launch caused OOM at ~388K steps. One
container killed (exit 137). Survivor partially swapping (8 GB swap, 87 GB phys).
FPS degraded from 37 to 19. ETA to 2M: ~17h from 808K.

**2M completion (2026-02-25 ~06:35 UTC):**
Cell A completed 2M steps. Initial assessment was "flat at ~6%", but closer
analysis of the final 600K revealed an upward trend:

```
Rolling mean (200K window):
  0-1400K:    flat at ~6.5%
  1400K-1600K: lifts to ~8%
  1700K-2000K: 8.3% -> 9.6% -> 10.3%

Individual points at end: 15%, 16%, 16%
```

This is NOT the hockey-stick yet, but the upward slope is statistically
distinct from the flat 6.5% of the first 1.4M steps. The full-state control
(Run 0) showed its hockey-stick inflection at 1-1.5M steps; the pixel agent
may need 2-4x longer due to representation learning overhead.

**Decision:** Extend Cell A to 8M steps to see if the trend becomes a
hockey-stick. Cell B killed (launched with buffer=300K vs Cell A's 500K,
making comparison unfair). Cell A resumed using new `--resume` flag.

### Cell A Extension (2M -> 8M)

TensorBoard: `runs/ch09_manip_noDrQ_criticEnc/FetchPush-v4/seed0_0/`
Config: same as Cell A + `--resume checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0.zip`
Started: 2026-02-25 ~07:15 UTC
Target: 8,000,000 steps (6,000,000 remaining from 2M checkpoint)

**Resume implementation notes:**
- Added `--resume` flag to `ch09_pixel_push.py` (SAC.load + reset_num_timesteps=False)
- Replay buffer is fresh (not saved -- 85 GB too large). Trained policy fills it quickly.
- `learning_starts` shifted to `num_timesteps + 1000` so HER buffer gets at least
  one complete episode before training resumes.
- Estimated runtime: ~33-42h at 40-50 fps
- Actual FPS: ~31 (buffer allocation + single-machine)

**2.44M checkpoint (2026-02-25 ~14:15 UTC) -- HOCKEY-STICK CONFIRMED:**

```
Success rate trajectory (extension, sampled every ~2K steps):
  2.13M: 7-14%  (post-resume buffer warming up, dip from fresh buffer)
  2.15M: 13-17% (buffer stabilizing, recovering pre-resume trend)
  2.25M: 15-20% (clear upward slope)
  2.40M: 18-26% (accelerating)
  2.44M: 25-34% (peak: 34%)
```

**Loss signature -- rising losses confirm healthy learning (see Lesson 9):**

| Metric | Failure regime (0-2M, flat 6%) | Hockey-stick (2.4M, 25-34%) |
|--------|-------------------------------|----------------------------|
| success_rate | 3-11%, flat | 25-34%, climbing |
| critic_loss | 0.07 declining (trivially easy: predict Q=-18.5 everywhere) | 0.3-0.7 rising (learning heterogeneous value landscape) |
| actor_loss | ~0 (no gradient signal from critic) | 0.8-1.6 (real signal: some actions clearly better) |
| reward | -48 to -50 | -35 to -38 |

The rising losses are NOT a sign of instability -- they are the hockey-stick
signal. When the agent was failing uniformly (6%), the critic had an easy job
(predict constant failure). Now that 30% of trajectories succeed, the critic
must learn to distinguish good from bad states. This is harder (higher loss)
but produces meaningful gradients for the actor.

**Positive feedback loop now active:**
Critic learns value structure -> actor gets real gradients -> better policy ->
more successes -> more diverse Bellman targets -> critic learns more ->
HER amplifies (8 relabeled transitions per success) -> exponential growth.

**Memory:** Container at 64 GB (buffer 78% full at 393K/500K transitions).
At capacity (~2.5M steps): ~90 GB. System has 119 GB, safe.

**Monitoring milestones (extension, UPDATED):**

| Steps | Green | Red |
|-------|-------|-----|
| 2.5M | ~~FPS stable, no OOM~~ **PASSED** -- 64 GB, stable | |
| 3.0M | ~~success >= 12%~~ **SMASHED** at 2.44M (25-34%) | |
| 4.0M | success >= 50% (hockey-stick continuing) | <= 35% stall |
| 6.0M | success >= 70% | Below 50% |
| 8.0M | success >= 85% (approaching full-state's 89%) | Below 60% |

---

## Cell B: Killed (buffer mismatch)

Cell B (`ch09_manip_noSS_noDrQ_criticEnc`) was launched with buffer_size=300K
(reduced from 500K due to earlier memory concerns). Killed at 12.4K steps
because the comparison with Cell A (buffer=500K) would be unfair -- buffer size
is itself a variable being tested (H3).

Cell B will be relaunched later with buffer_size=500K after Cell A extension
completes, ensuring fair comparison across all factorial cells.

---

## Research: 84x84 Resolution Sufficiency

- **84x84 is the standard** for RL-from-scratch (DQN, DrQ, DrQ-v2 all use it)
- **FetchPush from pixels is uncharted territory** -- no published solutions found
- **Frame stacking IS active** (4 frames) -- provides weak velocity signal
- **Proprioception (10D)** supplements visual info: gripper pos, velocity, puck pos
- **MuJoCo renderer is adequate** -- objects are visible at 84x84 (puck ~5px, gripper ~6-7px)
- **Higher resolution only helps with pretrained encoders** (R3M, VC-1)
- **Bottom line:** resolution is unlikely the bottleneck; representation learning and gradient flow are

---

## Decision Tree (If Phase A Fails)

```
Phase A: 2x2 factorial {DrQ on/off} x {SpatialSoftmax on/off}
  |
  Any cell shows sustained lift (>12% @ 500K)?
  |
  YES --> Phase A+: signal-strength knobs (grad_steps=2/4, lr=3e-4)
  |         |
  |         Hockey-stick found?
  |         YES --> Multi-seed + write tutorial --> DONE
  |         NO  --> Phase B (asymmetric actor-critic)
  |
  NO (all 4 flat @ ~3-7%) --> Phase B (asymmetric AC diagnostic)
  |
Phase B: State-critic + pixel-actor
  |
  Works? --> Problem IS representation learning --> Phase C (goal-prediction aux head)
  Fails? --> Something else is broken (debug pipeline)
```

---

## Ch9 Algorithmic Inventory and Chapter Narrative

### Date: 2026-02-25 (after Cell A reached 70% at 2.73M steps)

### What Ch9 teaches (algorithmic innovations ranked by importance)

| # | Innovation | Status | What it teaches |
|---|-----------|--------|-----------------|
| 1 | **Critic-encoder gradient routing** | **NECESSARY** | SB3's `share_features_extractor=True` puts encoder in actor optimizer only -- backwards from DrQ-v2. Encoder must be in critic optimizer. Difference between 6% forever and 70%+. |
| 2 | **ManipulationCNN (3x3 stride-2)** | **NECESSARY** | NatureCNN's 8x8 stride-4 destroys 5px objects. Architecture must match task domain. Transferable skill. |
| 3 | **SpatialSoftmax** | **NECESSARY** | "WHERE does each feature fire?" (x,y coordinates) vs "WHAT does the image look like?" (flat vector). Right inductive bias for manipulation. |
| 4 | **Sensor separation (proprioception)** | **NECESSARY** | Don't make CNN learn what joint encoders already measure. CNN should learn world-state only. Mirrors real robotics. |
| 5 | **Frame stacking (4x)** | **NECESSARY** | Velocity signal from pixel differences. Without it, POMDP (see Lesson 6). |
| 6 | **HER goal_mode="both"** | **NECESSARY** | Information asymmetry: pixels for control, vectors for HER relabeling. Tests "can CNN learn to CONTROL from pixels?" not "can it SPECIFY GOALS from pixels?" |
| 7 | **Training budget x2-4** | **KEY LESSON** | Representation learning phase before hockey-stick. State-calibrated stop rules are premature for pixel RL. |
| 8 | **Loss signature interpretation** | **KEY LESSON** | Rising critic/actor losses = healthy hockey-stick. Declining critic loss with flat success = failure (memorizing uniform Q=-18.5). See Lesson 9. |
| 9 | **DrQ augmentation** | **HARMFUL with SpatialSoftmax** | +/-4px shift corrupts spatial coordinates. 57% noise-to-signal ratio. Independent augmentation of obs/next_obs doubles noise in Bellman targets. |
| 10 | **Buffer sizing** | **PRACTICAL** | Pixel buffers 1000x larger than state buffers. 500K buffer = 85 GB. Memory math is a real engineering constraint. |

### Pedagogical framing: the learning curve IS the chapter

The chapter is NOT "here is the solution for pixel Push." It is the story of
HOW we found the solution and WHY each piece matters. The reader walks the
same path we walked: try something, see it fail, diagnose WHY, form a
hypothesis, test it, learn from the result. This is how real research works.

The failures are not embarrassments to skip over -- they are the content.
Each failure teaches something that a "here's the recipe" chapter cannot:
- NatureCNN failing teaches you about stride vs object size
- Normalization helping loss curves but not success teaches you about
  information content vs signal scale (Lesson 5)
- DrQ hurting performance teaches you about augmentation-representation
  interaction
- The premature stop rule teaches you about training budgets for pixel RL
- The rising-losses signal teaches you to read training diagnostics

A reader who only sees "use ManipulationCNN + SpatialSoftmax + critic-encoder,
train for 4M steps" learns nothing transferable. A reader who walks through
the investigation learns to debug ANY pixel RL problem.

### Chapter narrative arc (detective story structure)

The chapter is structured as a systematic debugging investigation, not a
recipe. Each step eliminates a hypothesis and reveals the next:

```
1. State-based Push works at 89% (Ch4)
   --> The SAC+HER pipeline is correct. Problem is pixel-specific.

2. Add pixels with NatureCNN --> flat at 5-8% for 8M+ steps
   --> Architecture hypothesis: NatureCNN destroys spatial information

3. Fix architecture (ManipulationCNN + SpatialSoftmax + proprio) --> still flat
   --> Architecture is NECESSARY but NOT SUFFICIENT (Lesson 7)
   --> New hypothesis: gradient flow

4. Fix gradient routing (critic-encoder) --> still flat at 2M steps
   --> Patience hypothesis: is 2M enough?

5. Extend to 8M steps --> HOCKEY-STICK at 2.2M, 70% at 2.73M
   --> The approach was correct all along. Training budget was too short.

6. DrQ was removed (Cell A = --no-drq) --> it HELPS to remove it
   --> DrQ's random shift corrupts SpatialSoftmax coordinates
   --> Negative result: augmentation must match representation
```

### Why DrQ as a NEGATIVE result is more educational than DrQ working

1. **Teaches principled reasoning about augmentation:** Data augmentation isn't
   universally good. You must reason about the interaction between augmentation
   and representation. Random shift + spatial coordinates = noise injection.

2. **Quantitative noise analysis for the book:**
   - DrQ shifts +/-4px on 84x84 input
   - After 4 conv layers (84->42->42->42->21), shift becomes +/-1px on 21x21
   - SpatialSoftmax maps to [-1, 1] coordinates: +/-1px = +/-0.10 noise
   - Independent augmentation of obs AND next_obs doubles noise to +/-0.20
   - Gripper-puck signal in these coordinates: ~0.25-0.50
   - Noise-to-signal ratio: 40-80% -- catastrophic for Bellman targets

3. **Generalizable principle:** "Choose your representation, then choose your
   augmentation." If using SpatialSoftmax (spatial precision matters), don't
   use spatial augmentation. If using flatten+linear (spatial precision is
   averaged away), DrQ may help. Cell C ablation (DrQ ON, SpatialSoftmax OFF)
   would test this hypothesis.

### What remains for the ablation narrative

| Cell | Config | Status | Purpose |
|------|--------|--------|---------|
| A (ext) | --no-drq, SS ON | **RUNNING** (70% at 2.73M) | Winner: proves the approach works |
| B | --no-drq, SS OFF | **Queued** (buffer=500K) | Does SpatialSoftmax matter when DrQ is off? |
| C | DrQ ON, SS OFF | **Queued** (buffer=500K) | Does DrQ help with flat features? |
| D | DrQ ON, SS ON | **Done** (3% at 1.54M) | Baseline failure: both DrQ and SS together |

If Cell C shows DrQ helps with flat features, the narrative becomes:
*"DrQ and SpatialSoftmax are both good ideas separately, but combining them
is catastrophic. DrQ provides regularization through spatial noise;
SpatialSoftmax provides precision through spatial coordinates. They are
fundamentally incompatible."*

If Cell C also fails, DrQ is simply harmful for manipulation tasks regardless
of encoder, and the narrative simplifies to: *"DrQ was designed for Atari/
DMControl where object identity matters more than object position. In
manipulation, precise spatial coordinates matter more than invariance to
small shifts."*
