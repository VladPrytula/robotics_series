# Appendix E: Isaac Peg-In-Hole (GPU-Only)

**Goal:** Transfer the book's method to a contact-rich insertion task in Isaac Lab,
while keeping the same experiment contract: reproducible commands, metadata,
and JSON evaluation artifacts.

---

## Bridge: Why Appendix E Exists

Through Chapters 0-10, we developed a full workflow on Gymnasium-Robotics
Fetch tasks. That depth was intentional: one environment family lets us isolate
algorithmic effects.

This appendix adds **portability evidence**: the same methodological pattern
(define the contract -> build components -> run controlled experiments)
applied to Isaac Lab.

Two constraints shape scope:

1. **GPU-only runtime.** Isaac Lab requires Linux + NVIDIA GPU.
2. **State-first for insertion.** We prioritize state observations
   before adding pixel complexity.

---

## Prerequisites and Setup

### Hardware Requirements

Isaac Lab requires **Linux + NVIDIA GPU** (minimum ~12 GB VRAM, 20 GB+
recommended). Mac is not supported -- `docker/dev-isaac.sh` fails fast with
an explicit error message if it detects Darwin.

Why the hard requirement? Isaac Sim is built on NVIDIA Omniverse, which uses
CUDA for physics and Vulkan for rendering. There is no CPU fallback and no
Metal/MPS path. This is fundamentally different from our MuJoCo pipeline,
which runs on both platforms.

### Software Requirements

The host needs the same Docker + NVIDIA Container Toolkit stack as Chapter 0.
Verify with:

```bash
docker --version          # Docker installed
nvidia-smi                # NVIDIA driver + GPU visible
```

The NGC base image (`nvcr.io/nvidia/isaac-lab:2.3.2`) is **~15 GB**. The
first `docker pull` takes significant time on slow connections. Subsequent
builds are fast because Docker caches the base layer.

### Building the Isaac Image

```bash
bash docker/build.sh isaac
```

Or let `dev-isaac.sh` auto-build on first run -- it checks for the
`robotics-rl:isaac` image and builds from `docker/Dockerfile.isaac` if
missing. The Dockerfile is thin: it adds `imageio` + `Pillow` on top of the
NGC Isaac Lab 2.3.2 image. No venv, no MuJoCo, no Gymnasium-Robotics.

### How `dev-isaac.sh` Differs from `dev.sh`

Readers familiar with `dev.sh` from Chapters 0-10 will find `dev-isaac.sh`
similar in spirit but different in several important details:

| Aspect | MuJoCo (`dev.sh`) | Isaac (`dev-isaac.sh`) |
|--------|-------------------|----------------------|
| Platform | Mac + Linux | Linux + GPU only |
| Python env | Fresh `.venv` from `requirements.txt` | NGC embedded Python (no venv) |
| User | Non-root (host UID/GID) | Root (`OMNI_KIT_ALLOW_ROOT=1`) |
| Mount point | `/workspace` | `/workspace/project` |
| Network | Default bridge | `--network=host` |
| Caching | Per-container (ephemeral) | Named Docker volumes (persistent) |
| Boot time | ~5 seconds | ~15-30s (+ 30-90s shader compilation on first run) |

Why each difference exists:

- **Root user.** Omniverse Kit requires root for certain filesystem and device
  operations. The `OMNI_KIT_ALLOW_ROOT=1` env var acknowledges this
  explicitly.
- **`/workspace/project` mount.** Isaac Lab's internal code lives at
  `/workspace/isaaclab`. If we mounted our repo at `/workspace`, it would
  shadow Isaac Lab's own packages.
- **`--network=host`.** Isaac Sim uses network sockets for internal Omniverse
  services (LiveLink, Nucleus). Host networking avoids port-mapping
  complications.
- **Named volumes.** Kit shader caches, GL caches, and compute caches are
  stored in Docker named volumes (`isaac-cache-kit`, `isaac-cache-glcache`,
  etc.) so they survive across container restarts. Without these, every run
  triggers a full shader recompilation.
- **No venv.** The NGC image bundles Python, PyTorch, Isaac Lab, and all
  Omniverse dependencies in a carefully configured PYTHONPATH. Creating a
  separate venv would break these internal dependencies.

### First Launch Expectations

The first time you run `dev-isaac.sh`, expect:

1. **Verbose Omniverse/Kit/CARB logs.** Isaac Sim prints startup diagnostics
   from Kit, CARB (Carbonite runtime), and PhysX. This is normal. The output
   is much noisier than MuJoCo's clean startup.

2. **Shader compilation pause (30-90 seconds).** On the first launch with a
   given GPU, Isaac Sim compiles Vulkan shaders. The process appears to hang.
   Subsequent runs reuse the cached shaders (stored in named Docker volumes).

3. **GPU memory usage.** Isaac Sim uses 8-12 GB of GPU memory just to boot
   the simulation runtime, before any training begins. Monitor with
   `nvidia-smi` in a separate terminal. If your GPU is shared with other
   workloads, you may need to free memory first.

---

## Verification Protocol

Following Chapter 0's pattern, we verify the Isaac environment through a
4-test dependency chain. Each test assumes the previous tests pass:

```
Container Boots -> Tasks Register -> Env Steps -> Training Completes
```

### Test 1: Container Boots

**What we verify.** The Isaac container starts, Kit initializes, and PyTorch
sees the GPU.

**Why this matters.** Isaac Lab's runtime depends on Omniverse Kit, CARB, and
PhysX -- all of which require CUDA and Vulkan. If the container cannot boot,
nothing else works.

**What failure means.** Either:

1. NVIDIA drivers or Container Toolkit are not installed
2. The `robotics-rl:isaac` image was not built (run `bash docker/build.sh isaac`)
3. The GPU is unavailable or out of memory
4. Vulkan ICD (Installable Client Driver) is not mounted -- `dev-isaac.sh`
   handles this automatically, but custom `docker run` invocations may miss it

**The test.**

```bash
bash docker/dev-isaac.sh python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

Expected output: your GPU name (e.g., `NVIDIA A100-SXM4-80GB`). If this
prints a device name, the Isaac container and CUDA stack are functional.

### Test 2: Tasks Register

**What we verify.** Isaac Lab's task extensions import correctly and register
environment IDs with Gymnasium.

**Why this matters.** Isaac Lab environments are registered dynamically via
the `isaaclab_tasks` extension. If this import fails, `gym.make(...)` will
raise `EnvironmentNameNotFound`. Common causes: version mismatch between
Isaac Lab and the task extensions, or missing `--headless` flag on a headless
system (Isaac Sim tries to open a Vulkan display and segfaults).

**What failure means.** Either:

1. The `isaaclab_tasks` module is not installed in the NGC image
2. `--headless` was omitted on a headless machine (causes segfault or
   `VkResult` error)
3. Isaac Lab internal configuration issue

**The test.**

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py discover-envs --headless
```

Expected output: a list of registered `Isaac-*` environment IDs and a JSON
catalog at `results/appendix_e_isaac_env_catalog.json`. The command also
probes the observation space of the dense-first env (`Isaac-Reach-Franka-v0`).

### Test 3: Env Creates and Steps

**What we verify.** An Isaac env can be created, wrapped with
`Sb3VecEnvWrapper`, and stepped by SB3.

**Why this matters.** This is the integration test between Isaac Lab's batched
GPU simulation and SB3's numpy-based training loop. The wrapper must
correctly convert tensors to numpy, extract the `policy` observation key,
and clip infinite action bounds.

**What failure means.** Either:

1. Observation or action space mismatch between Isaac env and SB3
2. `Sb3VecEnvWrapper` import failure (missing `isaaclab_rl` package)
3. GPU memory exhaustion during env creation

**The test.**

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke \
    --headless --seed 0 --smoke-steps 100
```

With only 100 steps this completes in ~30 seconds (mostly Isaac Sim boot
time). It verifies that the env creates, wraps, and steps without error.

### Test 4: Training Loop Completes

**What we verify.** SAC trains for a non-trivial number of steps, and a
checkpoint + metadata JSON are saved.

**Why this matters.** Like Chapter 0's PPO smoke test, this validates the
full pipeline: environment interaction, replay buffer population, gradient
computation, checkpoint serialization. Many bugs only appear after
`learning_starts` transitions have been collected and the first gradient
update fires.

**What failure means.** Either:

1. SB3 configuration error (e.g., policy class mismatch)
2. CUDA out-of-memory during training
3. NaN in loss values (can happen with unclamped action bounds in custom
   wrappers -- `Sb3VecEnvWrapper` handles this)

**The test.**

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke \
    --headless --seed 0
```

This runs the default 10,000-step smoke test (~2-3 minutes). On completion,
check for:

- `checkpoints/appendix_e_sac_Isaac-Reach-Franka-v0_seed0.zip`
- `checkpoints/appendix_e_sac_Isaac-Reach-Franka-v0_seed0.meta.json`

### Summary

| Test | Command | Wall time | Produces |
|------|---------|-----------|----------|
| 1. Container boots | `dev-isaac.sh python3 -c "import torch; ..."` | ~15-30s | GPU name |
| 2. Tasks register | `dev-isaac.sh python3 ... discover-envs --headless` | ~30-60s | Env catalog JSON |
| 3. Env steps | `dev-isaac.sh python3 ... smoke --headless --smoke-steps 100` | ~30s | Console output |
| 4. Training completes | `dev-isaac.sh python3 ... smoke --headless` | ~2-3 min | Checkpoint + meta |

---

## Isaac Lab vs Gymnasium-Robotics Observations

Before diving into components, we need to address an important structural
difference between the two environment families.

**Gymnasium-Robotics (Fetch)** wraps every observation in a dict with three
keys:

```python
{"observation": (25,), "achieved_goal": (3,), "desired_goal": (3,)}
```

This structure enables HER: the `achieved_goal` / `desired_goal` split lets
the replay buffer swap goals after the fact and recompute rewards.

**Isaac Lab** uses a different convention. Most Isaac envs expose a single
`policy` key:

```python
{"policy": (1, 32)}    # (num_envs, obs_dim)
```

This is a flat observation vector -- there are no separate goal keys, no
`compute_reward` method, and no built-in mechanism for HER relabeling.

**Consequences for our pipeline:**

| Feature | Gymnasium-Robotics | Isaac Lab (typical) |
|---------|-------------------|---------------------|
| Observation keys | `observation`, `achieved_goal`, `desired_goal` | `policy` |
| HER compatible? | Yes | No (without wrapper) |
| SB3 policy class | `MultiInputPolicy` | `MlpPolicy` |
| Success metric | Goal distance or `is_success` | Return-based only |

Our pipeline uses Isaac Lab's official `Sb3VecEnvWrapper` (from `isaaclab_rl.sb3`),
the same wrapper Isaac Lab's own training scripts use. It extracts the `policy`
observation key automatically and clips infinite action bounds to `[-100, 100]`
for SB3 compatibility. The pipeline then checks whether the resulting observation
space is goal-conditioned (has `achieved_goal` / `desired_goal` keys) or flat,
and selects `MlpPolicy` or `MultiInputPolicy` accordingly.

The portability lesson here is honest: **porting the method does not mean
drop-in compatibility.** Adapting to a new simulator means understanding what
its observation API actually provides, not assuming it matches your previous
environment.

---

## Engineering Decisions

Two design decisions are worth surfacing before we move to the algorithm
components, because they affect how the code looks and how errors manifest.

**Why `Sb3VecEnvWrapper`.** We initially considered writing a custom Gymnasium
adapter to convert Isaac Lab's batched tensor API into the standard
`gym.Env` interface. Then we discovered that Isaac Lab ships its own SB3
integration: `isaaclab_rl.sb3.Sb3VecEnvWrapper`. This is the same wrapper
that Isaac Lab's own training scripts use. It subclasses SB3's `VecEnv`
directly, handles tensor-to-numpy conversion, and extracts the `policy`
observation key. The lesson: before writing adapter code, check whether the
framework already provides one. We would have saved ourselves a day of
debugging by checking earlier.

**Infinite action bounds.** Isaac Lab environments declare their action space
as `Box(-inf, inf)` because the environment's internal action manager handles
clipping and scaling. SB3's SAC assumes bounded actions for its tanh squashing
(the policy outputs `tanh(z)`, which maps to `[-1, 1]`, then rescales to
action bounds). `Sb3VecEnvWrapper` addresses this by clipping the declared
bounds to `[-100, 100]`. If you inspect the wrapped action space and see
`[-100, 100]` instead of the `[-1, 1]` you might expect from Gymnasium-Robotics,
this is why. The practical consequence: avoid writing custom wrappers that
pass infinite bounds through to SB3 -- you will get NaN in the actor loss.

---

## WHY: Insertion Is Different

Peg insertion differs from Reach/Push in one decisive way: **tolerance**.
A policy can be visually close but still fail due to orientation error,
contact geometry, or jamming.

That makes sparse rewards harsher:

- Most episodes fail with reward `-1`
- Success is binary and narrow
- Early learning depends on relabeling and replay quality

The practical consequence is the same as Chapter 5:
we need off-policy learning (SAC) and usually HER-style relabeling when the
environment exposes goal-conditioned structure. For envs that do not expose
goal structure, SAC alone with dense rewards is the starting point.

### Contact Requires Memory

Insertion is different from Reach or Push not just in tolerance but in
*information structure*. A Reach policy can decide its action from a single
observation -- current pose plus goal is sufficient. An insertion policy needs
to feel how forces evolve over multiple timesteps: is the peg catching on a
rim? Sliding into the chamfer? Jamming against the sidewall?

This temporal dependency is why NVIDIA's reference configuration for Factory
PegInsert uses LSTM (1024 units, 2 layers). The LSTM gives the policy implicit
memory of the recent force sequence, letting it distinguish "sliding in" from
"stuck against edge" even when the instantaneous observations look similar.

Our SAC pipeline uses feedforward MLPs -- no recurrent state. A single-frame
MLP processes each timestep independently, which is fine for Reach (where
the optimal action depends only on the current pose-to-goal vector) but
problematic for insertion (where the optimal action depends on the *history*
of contact forces).

The standard adaptation for adding temporal context to a memoryless policy is
**frame stacking**: concatenating the N most recent observations into a single
input vector. With 4 frames of 19D observations, the MLP sees 76D of input
that encodes position, velocity, and acceleration trends over a ~0.27-second
window (at the environment's 15 Hz control rate). This is coarser than an
LSTM's learned memory, but it gives the policy enough temporal signal to
distinguish contact modes without changing the algorithm.

---

## HOW: Build It Components

The Build It track is intentionally minimal and executable. It gives you the
core update math and relabeling mechanics independent of any one Isaac task ID.

### E.1 Dict Observation Encoder

```python
--8<-- "scripts/labs/isaac_sac_minimal.py:dict_flatten_encoder"
```

### E.2 Squashed Gaussian Actor

```python
--8<-- "scripts/labs/isaac_sac_minimal.py:squashed_gaussian_actor"
```

### E.3 Twin Q Critic

```python
--8<-- "scripts/labs/isaac_sac_minimal.py:twin_q_critic"
```

### E.4 SAC Losses

```python
--8<-- "scripts/labs/isaac_sac_minimal.py:sac_losses"
```

### E.5 SAC Update Step

```python
--8<-- "scripts/labs/isaac_sac_minimal.py:sac_update_step"
```

### E.6 Goal Transition Structures

```python
--8<-- "scripts/labs/isaac_goal_relabeler.py:goal_transition_structs"
```

### E.7 Goal Sampling

```python
--8<-- "scripts/labs/isaac_goal_relabeler.py:isaac_goal_sampling"
```

### E.8 Transition Relabeling

```python
--8<-- "scripts/labs/isaac_goal_relabeler.py:isaac_relabel_transition"
```

### E.9 Episode HER Processing

```python
--8<-- "scripts/labs/isaac_goal_relabeler.py:isaac_her_episode_processing"
```

!!! lab "Checkpoint"
    Validate Build It components before long training:

    ```bash
    # SAC math + update wiring -- tests both goal-conditioned and Isaac-style obs
    bash docker/dev-isaac.sh python3 scripts/labs/isaac_sac_minimal.py --verify

    # Goal relabeling invariants and data amplification
    bash docker/dev-isaac.sh python3 scripts/labs/isaac_goal_relabeler.py --verify
    ```

    The SAC verification now tests two observation conventions:

    1. **Goal-conditioned** (`{observation, achieved_goal, desired_goal}`) --
       the Gymnasium-Robotics layout used in Chapters 4-8.
    2. **Isaac Lab flat-dict** (`{policy: flat_vector}`) -- what most Isaac
       envs actually provide.

    The goal relabeler will print a note reminding you that HER requires
    goal-conditioned structure. This is by design: the relabeling mechanics
    are correct and simulator-agnostic, but they only apply when the env
    exposes the right observation keys.

---

## WHAT: Run It Pipeline

The production Appendix E pipeline is:

- `scripts/appendix_e_isaac_peg.py`

It provides the same chapter-style commands as the rest of the book:

- `discover-envs`
- `smoke`
- `train`
- `eval`
- `all`
- `compare`

### E.10 Discover Candidate Insertion Environments

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py discover-envs --headless
```

**Note:** `--headless` is required on DGX and other headless systems. Without
it, Isaac Sim attempts to open a Vulkan display and will segfault.

The command also **probes the observation space** of the dense-first env
(`Isaac-Reach-Franka-v0` by default). For the probed env, it reports:

- Observation type (Box or Dict) and shape
- Whether the env is goal-conditioned
- Action space shape

Only **one env is probed per process**. This is a deliberate constraint:
Isaac Lab's USD stage and PhysX scene are not fully clearable after
`env.close()`. Attempting to create a second env in the same process can
hang indefinitely because the first env's scene graph persists in memory.
We discovered this empirically -- the first probe completes in ~10 seconds,
but the second env load hangs without error. To probe a different env,
re-run with `--dense-env-id <other-env-id>`.

Output artifacts:

- `results/appendix_e_isaac_env_catalog.json` -- includes a `probed_envs`
  section with observation metadata

### E.11 Dense-First Smoke Check

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke --headless --seed 0
```

This is the "do we have a valid learning loop" gate before insertion runs.
With the default 10,000 steps, expect ~2-3 minutes of wall time (including
Isaac Sim boot).

### E.12 Train on Insertion Task

```bash
# Auto-select peg/insertion-like env from registry
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train --headless --seed 0

# Or select explicit env ID
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
  --headless --env-id Isaac-Factory-PegInsert-Direct-v0 --seed 0
```

Periodic checkpoints are saved every 100K steps by default (configurable via
`--checkpoint-freq`). This ensures you do not lose progress on long runs.
With 500K default steps, expect ~1-2 hours of wall time depending on GPU and
environment complexity.

### E.13 Evaluate a Checkpoint

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py eval \
  --headless \
  --ckpt checkpoints/appendix_e_sac_Isaac-Factory-PegInsert-Direct-v0_seed0.zip
```

### E.14 Compare Multiple Result Files

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py compare \
  --result results/appendix_e_sac_Isaac-Factory-PegInsert-Direct-v0_seed0_eval.json \
  --result results/appendix_e_sac_Isaac-Factory-PegInsert-Direct-v0_seed1_eval.json
```

### E.15 Full Pipeline (Smoke -> Train -> Eval)

The `all` subcommand orchestrates smoke, train, and eval by running each
phase as a **separate subprocess**. This is necessary because Isaac Lab's
`SimulationContext` is a process-level singleton -- only one Isaac env can
exist per Python process. Creating a second env after closing the first
raises `RuntimeError("Simulation context already exists")`.

Each subprocess gets its own fresh `SimulationApp`, avoiding the singleton
constraint at the cost of ~10 seconds of boot time per phase:

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py all \
  --headless --seed 0 --smoke-steps 5000 --total-steps 50000
```

If `--env-id` is not specified, the `all` command runs `discover-envs` as
a preliminary subprocess to auto-resolve a peg/insertion candidate from
the registry.

---

## What Can Go Wrong

We find it helpful to list the failure modes we have encountered, since Isaac
Lab's error messages are not always self-explanatory. We use the same
Symptom/Cause/Resolution format as Chapter 0's Appendix A.

### SimulationApp Singleton

**Symptom.** Script hangs after the first env closes, or raises
`RuntimeError("Simulation context already exists")` when creating a second
environment.

**Cause.** Isaac Lab's `SimulationContext` is a process-level singleton. After
`env.close()`, the PhysX scene and USD stage persist in memory. Creating a
second env in the same process triggers the error.

**Resolution.** One environment per process. The `all` subcommand handles this
by running each phase (smoke, train, eval) as a separate subprocess with its
own fresh `SimulationApp` (see E.15). If you write custom orchestration, use
`subprocess.call()` for each phase.

### Observation Space Mismatch

**Symptom.** `KeyError: 'achieved_goal'` or `KeyError: 'desired_goal'` when
creating a model or replay buffer.

**Cause.** Assuming Gymnasium-Robotics observation keys on an Isaac env that
uses `{'policy': ...}`. Most Isaac Lab envs are not goal-conditioned.

**Resolution.** Run `discover-envs` first and check the `probed_envs` section
of the catalog JSON. The pipeline detects this automatically and selects
`MlpPolicy` (flat obs) or `MultiInputPolicy` (dict obs) accordingly.

### Infinite Action Bounds

**Symptom.** NaN in actor loss or exploding Q-values when using a custom
Gymnasium wrapper around an Isaac env.

**Cause.** Isaac Lab environments declare `Box(-inf, inf)` action spaces. If
you pass these through to SB3 without clipping, the tanh squashing in SAC's
policy produces undefined gradients.

**Resolution.** Use `Sb3VecEnvWrapper`, which clips bounds to `[-100, 100]`
automatically. If you must write a custom wrapper, clip the action space
bounds explicitly before passing to SB3.

### Missing `--headless` Flag

**Symptom.** Segfault on startup, or `VkResult` / Vulkan error immediately
after Isaac Sim begins initializing.

**Cause.** Isaac Sim tries to open a Vulkan display for rendering. On a
headless system (DGX, CI), there is no display server.

**Resolution.** Always pass `--headless` when running on headless machines.
All example commands in this appendix include it. If you forget, the error
appears before any Python code runs, which can be confusing.

### Missing Peg/Insertion Environment

**Symptom.** `SystemExit: No peg/insertion-like env ID found automatically.`

**Cause.** Isaac Lab's available task IDs vary across releases. The
auto-selection regex may not match any registered env in your version.

**Resolution.** Run `discover-envs` to see what is actually registered, then
pass `--env-id` explicitly.

### GPU Memory Exhaustion

**Symptom.** `CUDA out of memory` error during env creation or training.

**Cause.** Isaac Sim uses 8-12 GB of GPU memory just to boot. If your GPU is
shared with other workloads, the remaining memory may be insufficient for
SB3's replay buffer and networks.

**Resolution.** Monitor with `nvidia-smi` before launching. Free other GPU
workloads, or use a GPU with more VRAM.

### First-Run Shader Compilation

**Symptom.** Isaac Sim appears to hang for 30-90 seconds after printing
initial startup logs.

**Cause.** First launch on a given GPU triggers Vulkan shader compilation.

**Resolution.** Wait. Subsequent runs reuse cached shaders (stored in named
Docker volumes). If you deleted the Docker volumes (`docker volume rm
isaac-cache-glcache`), the compilation will repeat.

### Low num_envs Throughput

**Symptom.** Training is much slower than expected (e.g., 3 fps on a powerful
GPU).

**Cause.** Using `num_envs=1` on a GPU-parallel physics engine. NVIDIA PhysX
is designed for batched execution -- `num_envs=1` wastes over 95% of compute
on kernel launch overhead.

**Resolution.** Use `--num-envs 64` or `--num-envs 128` (Factory's default).
`Sb3VecEnvWrapper` supports batched envs natively. Note that `learning_starts`
and `buffer_size` may need tuning to account for the higher data throughput
(SB3 divides these by `num_envs` internally).

---

## Deliverables

A complete Appendix E run should produce:

- `checkpoints/appendix_e_sac_<env>_seed<N>.zip`
- `checkpoints/appendix_e_sac_<env>_seed<N>.meta.json`
- `results/appendix_e_sac_<env>_seed<N>_eval.json`
- `results/appendix_e_sac_<env>_comparison.json`
- `results/appendix_e_isaac_env_catalog.json`

The `.meta.json` includes `obs_space_type`, `is_goal_conditioned`, and
`wrapper: "Sb3VecEnvWrapper"` fields, so you can verify after training
which observation convention was detected and which SB3 integration was used.

---

## Reproduce It

This section documents the full debugging journey, including the initial
failure and root cause analysis. We show the complete diagnostic process
rather than just the final working recipe, because the diagnostic skills
transfer to other environments.

### Act 1: First Attempt (Vanilla SAC, 500K Steps)

Our first run used default SAC hyperparameters with 64 parallel environments:

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --seed 0 \
    --env-id Isaac-Factory-PegInsert-Direct-v0 \
    --num-envs 64 --total-steps 500000
```

TensorBoard revealed a pattern we had seen before (Lesson 9):

| Metric | Start | End | Interpretation |
|--------|-------|-----|---------------|
| ep_rew_mean | 31.5 | 29.3 | Flat -- stuck at approach-tier reward |
| ep_len_mean | 149 | 149 | Always times out, never inserts |
| ent_coef | 0.89 | 0.10 | Policy becoming deterministic |
| actor_loss | -12.2 | -31.5 | Updating but no reward improvement |
| critic_loss | 3.7 | 11.9 | Oscillating -- no clear value structure |

**Diagnosis:** SAC is "confidently failing." Applying Lesson 9's three-phase
framework: the flat reward combined with an increasingly deterministic policy
(ent_coef 0.89 -> 0.10) and oscillating critic loss means the agent is stuck
in Phase 1 -- memorizing failure. The policy becomes more certain about a bad
strategy without ever discovering better trajectories.

### Act 2: Root Cause Analysis

**Why ~29 reward?** Factory PegInsert uses a multi-tier dense reward function
with baseline (approach), coarse (alignment), and fine (insertion) tiers. The
agent reaches the baseline tier -- it learns to move the arm toward the socket
-- but cannot progress to fine insertion. The ~29 reward corresponds to
maximizing the approach-tier bonus while getting zero from the finer tiers.

**Why can't it progress?** The 19D policy observation includes fingertip pose,
velocities, and previous actions -- a single snapshot. But fine insertion
requires sensing how forces change over multiple steps. Is the peg catching on
the rim? Sliding into the chamfer? Jamming? A memoryless MLP processes each
frame independently; it cannot reason about force trends.

**Reference configuration comparison.** NVIDIA's own training config for
Factory PegInsert uses RL-Games PPO with:

- LSTM (1024 units, 2 layers) for temporal reasoning
- Asymmetric actor-critic (policy sees 19D obs, critic sees 72D full state)
- MLP layers [512, 128, 64]
- gamma=0.995, 128 envs, ~3.27M total steps

The LSTM is the critical difference -- it gives the policy implicit memory
of the recent contact sequence.

**Connection to Chapter 9.** The same pattern appeared in pixel-based Push:
a representation bottleneck (there it was visual encoding, here it is temporal
encoding) causes a flat initial phase. The fix follows the same principle --
give the policy the information it needs.

### Act 3: Adaptation (Frame Stacking + Architecture + Budget)

We address the temporal gap with three changes:

1. **Frame stacking (4 frames).** With 19D observations, 4 frames produce 76D
   input, encoding ~0.27 seconds of position and velocity trends at 15 Hz.
   This approximates the LSTM's sequential memory without changing the
   algorithm.

2. **Larger network ([512, 256, 128]).** The richer 76D input needs more
   capacity than the default [256, 256]. This matches NVIDIA's [512, 128, 64]
   but wider, since our MLP lacks the LSTM's compression.

3. **Training budget and hyperparameters.** gamma=0.995 (longer effective
   horizon), lr=1e-4 (more stable for harder tasks), 128 envs (Factory
   default), 3M total steps (matching NVIDIA's budget).

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --seed 0 \
    --env-id Isaac-Factory-PegInsert-Direct-v0 \
    --num-envs 128 \
    --total-steps 3000000 \
    --checkpoint-freq 100000 \
    --frame-stack 4 \
    --net-arch "512,256,128" \
    --gamma 0.995 \
    --learning-rate 1e-4
```

| Parameter | Failed Run | Adapted Run | Why |
|-----------|-----------|-------------|-----|
| frame_stack | 1 (none) | **4** | Temporal context for contact reasoning |
| net_arch | [256, 256] | **[512, 256, 128]** | Larger capacity for richer input |
| gamma | 0.99 | **0.995** | Longer effective horizon |
| learning_rate | 3e-4 | **1e-4** | More stable for harder tasks |
| num_envs | 64 | **128** | Factory default, better GPU utilization |
| total_steps | 500K | **3M** | Matches NVIDIA's training budget |

**What to expect:**

- **Steps 0-500K:** Approach-tier reward (~29), similar to the first attempt.
- **Steps 500K-1.5M:** If frame stacking provides sufficient temporal context,
  reward should start climbing as the fine-insertion tier activates. Critic
  loss may rise -- this is the Phase 2 signal from Lesson 9.
- **Steps 1.5M-3M:** Convergence or plateau.

If reward is still flat at 1M steps, the frame stacking is insufficient and
the task genuinely requires recurrent architectures (LSTM/GRU via RL-Games
PPO).

!!! note "Results pending"
    Training results will be added here once the 3M-step run completes
    (~5.7 hours at 128 envs). Check for updated checkpoints and eval
    artifacts in the standard locations.

---

## Scope Boundary

This appendix focuses on **method transfer**, not exhaustive Isaac tuning.

Most Isaac Lab environments are **not** goal-conditioned in the
Gymnasium-Robotics sense. They expose flat observation vectors under a
`policy` key, with no separate `achieved_goal` / `desired_goal` structure.
As a result:

- **SAC without HER is the expected default** for most Isaac envs. With
  `--her auto`, the pipeline detects this and skips HER automatically.
- **If goal-conditioned envs are available**, HER activates automatically.
  The pipeline checks for `{observation, achieved_goal, desired_goal}` keys
  and a `compute_reward` method on the unwrapped env.
- **Adding goal structure to a non-goal-conditioned env** is possible by
  writing a thin Gymnasium wrapper that extracts achieved/desired goals from
  the observation vector and implements `compute_reward`. We do not attempt
  this here because it requires task-specific knowledge of which observation
  dimensions correspond to gripper pose vs. object pose.

We use `num_envs=64-128` for Factory tasks to achieve practical throughput.
GPU-parallel physics is essential for contact-rich environments: Factory
PegInsert at `num_envs=1` runs at ~3 fps; at `num_envs=64`, ~145 fps; at
`num_envs=128`, ~170 fps. SB3's `VecEnv` interface supports batched envs
natively through Isaac Lab's `Sb3VecEnvWrapper`. For readers interested in
native Isaac Lab training loops (which bypass SB3 entirely and scale to
thousands of parallel envs), we point to the `isaaclab_rl` extension and
the Isaac Lab documentation (Mittal et al., 2023).
