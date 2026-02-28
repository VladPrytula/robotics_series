# Appendix E: Isaac Lab Manipulation (GPU-Only)

**Goal:** Transfer the book's SAC methodology to a GPU-parallel manipulation
task in Isaac Lab, demonstrating that the same algorithm and diagnostic skills
work across simulators -- with dramatically different wall-clock performance.

---

## Bridge: Why Appendix E Exists

Through Chapters 0-10, we developed a full workflow on Gymnasium-Robotics
Fetch tasks. That depth was intentional: one environment family lets us isolate
algorithmic effects.

This appendix adds **portability evidence**: the same methodological pattern
(define the contract -> build components -> run controlled experiments)
applied to Isaac Lab. The primary demonstration task is
`Isaac-Lift-Cube-Franka-v0` -- a manager-based manipulation task that mirrors
our FetchPush work (approach, grasp, lift, track a target). The narrative
becomes: *the same SAC methodology, dramatically faster thanks to
GPU-parallel physics.*

Two constraints shape scope. First, Isaac Lab requires Linux + NVIDIA GPU,
so the entire appendix is GPU-only. Second, we practice honest
method-benchmark matching: we demonstrate SAC on tasks where it works
(Lift-Cube, a fully observable MDP) and explain where it does not
(PegInsert, a POMDP requiring recurrence).

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

Each difference has a concrete reason. The container runs as **root** because
Omniverse Kit requires root for certain filesystem and device operations (the
`OMNI_KIT_ALLOW_ROOT=1` env var acknowledges this explicitly). Our repo mounts
at **`/workspace/project`** rather than `/workspace` because Isaac Lab's
internal code lives at `/workspace/isaaclab` -- mounting at the parent would
shadow Isaac Lab's own packages. The container uses **`--network=host`**
because Isaac Sim relies on network sockets for internal Omniverse services
(LiveLink, Nucleus), so host networking avoids port-mapping complications.
**Named Docker volumes** (`isaac-cache-kit`, `isaac-cache-glcache`, etc.)
store Kit shader caches, GL caches, and compute caches so they survive across
container restarts; without these, every run triggers a full shader
recompilation. Finally, there is **no venv** because the NGC image bundles
Python, PyTorch, Isaac Lab, and all Omniverse dependencies in a carefully
configured PYTHONPATH, which means creating a separate venv would break these
internal dependencies.

### First Launch Expectations

The first time you run `dev-isaac.sh`, expect three things that might look
alarming but are normal. First, Isaac Sim prints **verbose Omniverse/Kit/CARB
logs** -- startup diagnostics from Kit, CARB (Carbonite runtime), and PhysX
that are much noisier than MuJoCo's clean startup. Second, on the first launch
with a given GPU, Isaac Sim compiles Vulkan **shaders**, which causes a
**30-90 second pause** during which the process appears to hang; subsequent
runs reuse the cached shaders (stored in named Docker volumes). Third, Isaac
Sim uses **8-12 GB of GPU memory** just to boot the simulation runtime, before
any training begins, so monitor with `nvidia-smi` in a separate terminal and
free other workloads if your GPU is shared.

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

**What failure means.** The most common causes are missing NVIDIA drivers or
Container Toolkit, an unbuilt `robotics-rl:isaac` image (run
`bash docker/build.sh isaac`), an unavailable or out-of-memory GPU, or an
unmounted Vulkan ICD (Installable Client Driver) -- `dev-isaac.sh` handles
ICD mounting automatically, but custom `docker run` invocations may miss it.

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

**What failure means.** The `isaaclab_tasks` module may not be installed in
the NGC image, or `--headless` was omitted on a headless machine (which causes
a segfault or `VkResult` error), or there is an Isaac Lab internal
configuration issue.

**The test.**

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py discover-envs --headless
```

Expected output: a list of registered `Isaac-*` environment IDs and a JSON
catalog at `results/appendix_e_isaac_env_catalog.json`. The command also
probes the observation space of the dense-first env (`Isaac-Reach-Franka-v0`
by default, or any env you specify with `--dense-env-id`).

### Test 3: Env Creates and Steps

**What we verify.** The Lift-Cube env can be created, wrapped with
`Sb3VecEnvWrapper`, and stepped by SB3.

**Why this matters.** This is the integration test between Isaac Lab's batched
GPU simulation and SB3's numpy-based training loop. The wrapper must
correctly convert tensors to numpy, extract the `policy` observation key,
and clip infinite action bounds.

**What failure means.** Typical causes include an observation or action space
mismatch between the Isaac env and SB3, an `Sb3VecEnvWrapper` import failure
(indicating a missing `isaaclab_rl` package), or GPU memory exhaustion during
env creation.

**The test.**

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke \
    --headless --seed 0 --dense-env-id Isaac-Lift-Cube-Franka-v0 --smoke-steps 100
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

**What failure means.** Possible causes are an SB3 configuration error (such
as a policy class mismatch), CUDA out-of-memory during training, or NaN in
loss values -- which can happen with unclamped action bounds in custom wrappers,
though `Sb3VecEnvWrapper` handles this automatically.

**The test.**

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke \
    --headless --seed 0 --dense-env-id Isaac-Lift-Cube-Franka-v0
```

This runs the default 10,000-step smoke test (~1-2 minutes including Isaac
boot). On completion, verify that
`checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip` and
`checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.meta.json`
both exist.

### Summary

| Test | Command | Wall time | Produces |
|------|---------|-----------|----------|
| 1. Container boots | `dev-isaac.sh python3 -c "import torch; ..."` | ~15-30s | GPU name |
| 2. Tasks register | `dev-isaac.sh python3 ... discover-envs --headless` | ~30-60s | Env catalog JSON |
| 3. Env steps | `dev-isaac.sh python3 ... smoke --headless --dense-env-id Isaac-Lift-Cube-Franka-v0 --smoke-steps 100` | ~30s | Console output |
| 4. Training completes | `dev-isaac.sh python3 ... smoke --headless --dense-env-id Isaac-Lift-Cube-Franka-v0` | ~1-2 min | Checkpoint + meta |

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

**Lift-Cube specifically** provides a 36D flat observation:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| `joint_pos` | 9 | Franka arm joint positions (7) + gripper (2) |
| `joint_vel` | 9 | Joint velocities |
| `object_position` | 3 | Cube position in world frame |
| `target_object_position` | 7 | Target pose (position + quaternion) |
| `actions` | 8 | Previous action (feedback for smoothness) |

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

## WHY: Lifting Is Different From Reaching

Lift-Cube requires a multi-phase control sequence that Reach does not:

1. **Approach** -- move the gripper toward the cube
2. **Grasp** -- close the gripper around the cube
3. **Lift** -- raise the cube off the table
4. **Track** -- move the held cube to the target pose

Each phase has a qualitatively different optimal action: approach requires
large arm movements, grasping requires precise gripper timing, lifting
requires coordinated upward motion, and tracking requires fine position
control. A Reach policy only needs phase 1.

### Staged Dense Reward

Isaac Lab's Lift-Cube environment uses a staged dense reward structure that
mirrors Chapter 5's curriculum pattern -- the reward design teaches the agent
in stages:

| Reward term | Weight | What it rewards |
|-------------|--------|----------------|
| `reaching_object` | 1.0 | Distance from gripper to cube |
| `lifting_object` | 15.0 | Height of cube above table |
| `object_goal_tracking` | 16.0 | Distance from cube to target pose |
| `object_goal_tracking_fine_grained` | 5.0 | Precise tracking bonus |
| `action_rate` | -0.0001 | Penalizes jerky actions |
| `joint_vel` | -0.0001 | Penalizes fast joint movement |

The weight structure creates a natural curriculum: the agent first maximizes
the reaching bonus (easy), then discovers that lifting unlocks a much larger
reward signal (15x reaching), and finally learns that tracking the target is
the largest reward component (16x+5x).

This is the same pedagogical pattern we saw in Chapter 5 (goal stratification
and curriculum learning) -- except here the curriculum is baked into the
reward function rather than requiring a separate wrapper.

### The Hidden Curriculum (and Why SAC Must Disable It)

Isaac Lab's Lift-Cube env has a second curriculum mechanism that is less
obvious but critically important: a **CurriculumManager** that scales the
`action_rate` and `joint_vel` penalty weights during training.

| Term | Initial weight | Final weight | Scaling |
|------|---------------|-------------|---------|
| `action_rate` | -0.0001 | -0.1 | 1000x |
| `joint_vel` | -0.0001 | -0.1 | 1000x |

The pedagogical idea is sound: start with tiny penalties so the agent can
explore freely -- thrash around, make big movements to discover how to reach
and grasp. Once it learns the basics, ramp up the penalties to encourage
smooth, controlled motion. *First learn WHAT to do, then learn to do it
SMOOTHLY.*

NVIDIA designed this curriculum for PPO. PPO is on-policy: it collects a
batch of experience, updates the policy, then **discards all old data**.
When the penalties increase, PPO's next batch is collected entirely under
the new reward scale. There is no conflict between old and new reward
magnitudes.

**SAC cannot tolerate this.** SAC stores hundreds of thousands of transitions
in a replay buffer. When the curriculum scales penalties by 1000x, those old
transitions have rewards of magnitude ~5, but new transitions have rewards of
magnitude ~5000. The Q-function trains on a mix of both and must somehow
reconcile two incompatible scales. It is like training a regression model
where half the labels are in meters and the other half are in kilometers,
with no indicator of which is which.

In our experiments, this caused a catastrophic reward collapse at ~4.6M
steps: the reward went from -4.14 (best ever, agent was lifting and tracking)
to -3,050 in a single 64K-step interval (see Reproduce It below for the
full diagnostic). The critic loss exploded 540x and the training never
recovered.

**Our fix:** The pipeline automatically disables the CurriculumManager for
SAC by setting all curriculum terms to `None` at env creation time. The
penalty signals still exist at their initial constant values (-0.0001) --
they just do not change magnitude mid-training. This preserves smooth
action incentives without poisoning the replay buffer.

**The general principle:** Any system that changes the reward function
mid-training is incompatible with off-policy replay buffers. The buffer
cannot distinguish "old reward scale" transitions from "new reward scale"
transitions. This is a specific instance of a broader assumption: replay
buffers assume the MDP is **stationary** -- that the transition dynamics
and reward function do not change over time. Curriculum learning that
modifies reward weights violates this stationarity assumption.

### Action Space: Joint-Level Control

The action space is 8-dimensional: 7 joint position targets for the Franka
arm + 1 gripper command. This is more complex than Fetch's 4D Cartesian
deltas (`dx, dy, dz, gripper`), but the same SAC algorithm handles both --
continuous actions are continuous actions, regardless of whether they represent
end-effector velocities or joint positions.

### Full Observability: MLP Is Sufficient

Unlike Factory PegInsert (which is a POMDP requiring recurrence -- see Scope
Boundary below), Lift-Cube is a fully observable MDP. The 36D observation
includes joint positions, joint velocities, object position, target pose,
and previous actions. A single observation frame contains everything the
agent needs to decide the optimal action -- no temporal reasoning required.

This is why SAC with a feedforward MLP works: the Markov property holds.
Compare to FetchPush in Chapter 4: similar structure, similar difficulty,
same methodology.

### Why SAC Should Work

The problem structure tells us SAC is the right algorithm. The action space is
continuous, which rules out DQN and points toward actor-critic methods. The
reward is dense, so off-policy learning works well without HER. The 36D state
vector is fully observable, which means an MLP suffices (no CNN, no LSTM). And
SAC's replay buffer reuses experience, providing superior sample efficiency
compared to PPO's on-policy waste.

NVIDIA's reference configuration for Lift-Cube uses PPO with 16.3M steps.
SAC's superior sample efficiency (off-policy reuse) should solve the task in
substantially fewer steps -- our budget is 2M, an 8x reduction.

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

The production Appendix E pipeline lives in
`scripts/appendix_e_isaac_peg.py` and provides the same chapter-style
subcommands as the rest of the book: `discover-envs`, `smoke`, `train`,
`eval`, `all`, `compare`, and `record` (video recording from checkpoints).

### E.10 Discover Available Isaac Environments

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py discover-envs --headless
```

**Note:** `--headless` is required on DGX and other headless systems. Without
it, Isaac Sim attempts to open a Vulkan display and will segfault.

The command also **probes the observation space** of the specified env
(`Isaac-Reach-Franka-v0` by default, or any env via `--dense-env-id`),
reporting the observation type (Box or Dict) and shape, whether the env is
goal-conditioned, and the action space shape.

Only **one env is probed per process**. This is a deliberate constraint:
Isaac Lab's USD stage and PhysX scene are not fully clearable after
`env.close()`. Attempting to create a second env in the same process can
hang indefinitely because the first env's scene graph persists in memory.
We discovered this empirically -- the first probe completes in ~10 seconds,
but the second env load hangs without error. To probe a different env,
re-run with `--dense-env-id <other-env-id>`.

The output artifact is `results/appendix_e_isaac_env_catalog.json`, which
includes a `probed_envs` section with observation metadata.

### E.11 Dense-First Smoke Check

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke \
    --headless --seed 0 --dense-env-id Isaac-Lift-Cube-Franka-v0
```

This is the "do we have a valid learning loop" gate before full training.
With the default 10,000 steps, expect ~1-2 minutes of wall time (including
Isaac Sim boot).

### E.12 Train on Lift-Cube

```bash
# Primary target: Lift-Cube with 256 parallel envs (8M steps, ~14 min)
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000

# Or any other Isaac env
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --env-id Isaac-Reach-Franka-v0 --seed 0
```

Periodic checkpoints are saved every 100K steps by default (configurable via
`--checkpoint-freq`). At 256 parallel envs on a modern GPU, expect ~9,000-
10,000 fps throughput -- 8M steps completes in approximately **14 minutes**.
Compare this to Chapter 9's pixel-based Push training: ~40 hours on MuJoCo.

### E.13 Evaluate a Checkpoint

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py eval \
    --headless \
    --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip
```

### E.14 Compare Multiple Result Files

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py compare \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --result results/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_eval.json \
    --result results/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed1_eval.json
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
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000
```

### E.16 Record Video From Checkpoint

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py record \
    --headless \
    --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip
```

The `record` subcommand creates the env with `render_mode="rgb_array"` and
`--enable_cameras`, loads the checkpoint, runs one deterministic episode, and
saves frames as MP4 via `imageio`. The env_id is inferred from the checkpoint
filename or `.meta.json` if available. Output defaults to
`videos/appendix_e_<env>_<ckpt_stem>.mp4`.

---

## Reproduce It

This section documents actual training results on Lift-Cube, plus the
debugging journey with PegInsert that led us to choose this benchmark.

### SAC on Lift-Cube (Primary Result)

Training command:

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99
```

Configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | SAC | Off-policy, continuous actions (Ch3) |
| Policy | MlpPolicy [256, 256] | 36D flat obs, fully observable MDP |
| num_envs | 256 | GPU parallelism for throughput |
| total_steps | 8,000,000 | Multi-phase task needs budget; NVIDIA uses 16.3M with PPO |
| learning_rate | 3e-4 | SB3 default, works for dense reward |
| gamma | 0.99 | Standard discount |
| HER | off (auto-detected) | Env is not goal-conditioned |
| frame_stack | 1 | MDP, no temporal reasoning needed |

**Training progression (2M steps, seed 0):**

| Timesteps | ep_rew_mean | Interpretation |
|-----------|------------|----------------|
| 64K | -29.7 | Reaching tier only |
| 512K | -26.1 | Reaching improving |
| 1.0M | -24.8 | Approaching grasping |
| 1.5M | -23.4 | Partial grasping |
| 2.0M | -21.8 | Grasping + some lifting |

Throughput: **9,363 fps** at 256 envs. Total wall time: **213.6 seconds**
(~3.6 minutes) for 2M steps.

**Evaluation (100 episodes, deterministic policy):**

| Metric | Value |
|--------|-------|
| return_mean | -4.73 +/- 4.53 |
| ep_len_mean | 250 (always runs to timeout) |
| Positive-return episodes | 41/100 (41%) |
| Mean return (positive eps) | +0.69 |
| Mean return (negative eps) | -8.50 |

The evaluation reveals a **bimodal distribution**: 41% of episodes achieve
positive returns (the agent reaches, grasps, lifts, and partially tracks the
target), while 59% fail to complete the grasping phase. This is consistent
with partial learning -- the agent has discovered the grasp-and-lift strategy
but has not generalized it across all goal configurations.

**Interpretation:** 2M steps showed steady improvement with no sign of
plateau. The reward was still climbing at the final timestep, suggesting the
agent would continue improving with more training. We extended to 8M steps.

### Act 2: The Curriculum Crash (and What We Learned)

Our first 8M run produced a **catastrophic reward collapse** at ~4.6M steps:

```
ts=4,624,128  rew=-4.14   <-- best ever
ts=4,688,128  rew=-3,050  <-- 740x worse in one interval
```

The critic loss exploded 540x (0.05 -> 26.2) and the entropy coefficient
loss flipped sign. The reward never recovered.

**Root cause:** Isaac Lab's CurriculumManager scales the `action_rate` and
`joint_vel` penalty weights from -0.0001 to -0.1 (a 1000x increase) when
the agent crosses a performance threshold. This is designed for PPO
(on-policy, no replay buffer). With SAC (off-policy), the replay buffer
contains transitions collected under the OLD penalty scale. When the
curriculum kicks in, new transitions have rewards 1000x larger in magnitude.
The Q-function, calibrated for rewards in [-30, 0], suddenly receives
Bellman targets computed from rewards in [-5000, 0].

**The fix:** Disable the CurriculumManager for off-policy training. Our
pipeline now sets all curriculum terms to `None` at env creation time,
keeping penalties at their initial constant values (-0.0001). The penalty
signals still exist -- they just don't change magnitude mid-training.

**The lesson (Lesson 14):** Replay buffers assume the MDP is stationary.
Curriculum learning that modifies reward weights violates this assumption.
The combination of curriculum + off-policy is a known pitfall but rarely
documented because most curriculum RL research uses on-policy methods.

### Act 3: Clean Run (8M Steps, Curriculum Disabled)

Training command (curriculum auto-disabled for SAC):

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99
```

**Training progression (8M steps, seed 0, curriculum disabled):**

| Timesteps | ep_rew_mean | Phase |
|-----------|------------|-------|
| 64K | -27.2 | Reaching only |
| 1M | -24.8 | Reaching improving |
| 2M | -21.8 | Approaching grasping |
| 3M | -15.4 | Grasping beginning |
| 4M | -6.4 | Lifting |
| 5M | -1.7 | Goal tracking |
| 6M | -1.5 | Tracking refinement |
| 7M | -1.5 | Near convergence |
| 8M | -1.4 | Converged |

Throughput: **9,377 fps** at 256 envs. Total wall time: **853 seconds**
(~14.2 minutes) for 8M steps.

**Evaluation (100 episodes, deterministic policy):**

| Metric | Value |
|--------|-------|
| return_mean | **+0.54 +/- 0.05** |
| ep_len_mean | 250 |
| Positive-return episodes | **100/100 (100%)** |
| Min return | +0.46 |
| Max return | +0.68 |

Every episode achieves positive return -- the agent consistently reaches,
grasps, lifts, and tracks the target across all goal configurations. The
tight standard deviation (0.05) indicates robust generalization.

**Compare with 2M checkpoint:** At 2M steps, only 41% of episodes achieved
positive returns, with a bimodal distribution (mean -4.73 +/- 4.53). The
extra 6M steps allowed SAC to generalize the grasp-lift-track sequence from
a subset of goal configurations to all of them.

### Wall-Clock Comparison

| Setup | Env | Steps | fps | Wall time |
|-------|-----|-------|-----|-----------|
| Ch9 pixel Push (MuJoCo) | FetchPush-v4 | 8M | 30-50 | ~40 hours |
| Ch4 state Push (MuJoCo) | FetchPush-v4 | 2M | 500-600 | ~1 hour |
| **Appendix E Lift-Cube (Isaac, 2M)** | Isaac-Lift-Cube-Franka-v0 | 2M | ~9,363 | **~3.6 min** |
| **Appendix E Lift-Cube (Isaac, 8M)** | Isaac-Lift-Cube-Franka-v0 | 8M | ~9,363 | **~14 min** |

The speedup is dramatic: Isaac Lab at 256 envs runs **~15x faster** than
MuJoCo state-based training and **~170x faster** than MuJoCo pixel-based
training. This is the core value proposition of GPU-parallel physics: what
takes hours on MuJoCo takes minutes on Isaac. Even the extended 8M run
completes in under 15 minutes.

### Video Progression: Seeing Learning Happen

Numbers tell us the agent improved; videos show us *how*. We recorded one
episode from three checkpoints spaced across the 8M training run to visualize
the progression from random behavior to competent manipulation.

| Stage | Checkpoint | Steps | Expected behavior |
|-------|-----------|-------|-------------------|
| Early | `*_199680_steps.zip` | ~200K | Random arm movements, occasionally approaching cube |
| Mid | `*_2999808_steps.zip` | ~3M | Reaching cube reliably, attempting grasp, partial lifts |
| Converged | `*_seed0.zip` | 8M (final) | Full reach-grasp-lift-track sequence, every episode |

Recording uses Isaac Lab's viewport camera via `env.render()`:

```bash
# Each runs as a separate subprocess (SimulationContext singleton)
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py record \
    --headless --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_199680_steps.zip

bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py record \
    --headless --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_2999808_steps.zip

bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py record \
    --headless --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip
```

Output: `videos/appendix_e_Isaac-Lift-Cube-Franka-v0_*.mp4`

The `record` subcommand creates the env with `render_mode="rgb_array"` and
`--enable_cameras` (for headless Vulkan rendering), warms up the renderer
(first few frames are often black during shader compilation), runs one
deterministic episode, and saves frames as MP4 at 30 fps via `imageio`.

**What to look for in the videos.** At the early checkpoint (200K), the arm
moves without purpose -- no reaching strategy, deep negative reward (~-27) --
because the agent is still in the random exploration phase, collecting
experience for the replay buffer. By the mid checkpoint (3M), the arm reaches
toward the cube and sometimes grasps it, though you may see partial lifts that
fail when the cube slips; reward is around -15, meaning the agent has learned
reaching and is discovering grasping. At the converged checkpoint (8M), the
motion is smooth and purposeful -- approach, close gripper, lift, track target
-- with positive reward (+0.54) and reliable manipulation across goal
configurations. This progression mirrors the training curve: the jumps between
phases (reaching -> grasping -> lifting -> tracking) are visible both in reward
plots and in behavior.

### PegInsert: What Came Before (Diagnostic Journey)

Before Lift-Cube, we attempted Factory PegInsert -- a contact-rich insertion
task that seemed like a natural target for manipulation RL. The debugging
process is instructive because it demonstrates how to diagnose
method-benchmark mismatches.

**First attempt (500K steps, 64 envs):**

| Metric | Start | End | Interpretation |
|--------|-------|-----|---------------|
| ep_rew_mean | 31.5 | 29.3 | Flat -- stuck at approach-tier reward |
| ep_len_mean | 149 | 149 | Always times out, never inserts |
| ent_coef | 0.89 | 0.10 | Policy becoming deterministic |

SAC is "confidently failing" -- the same Phase 1 pattern from Lesson 9.
The agent learns to approach the socket but cannot perform fine insertion.

**Root cause:** Factory PegInsert is a **POMDP**. The actor sees 19D partial
state (fingertip pose, velocities, previous actions) while the critic sees
72D full state (including socket geometry, contact forces, peg orientation).
Fine insertion requires temporal reasoning about contact sequences -- is the
peg catching, sliding, or jamming? A memoryless MLP processes each frame
independently and cannot distinguish these contact modes.

NVIDIA's reference config uses LSTM (1024 units, 2 layers) + asymmetric
actor-critic for exactly this reason. The LSTM gives the policy implicit
memory of the recent force sequence.

**What we tried:** Frame stacking (4 frames of 19D = 76D input), larger
network ([512, 256, 128]), longer budget (3M steps). Frame stacking
approximates temporal context but is fundamentally limited compared to
learned recurrence -- it provides a fixed window rather than adaptive memory.

**What we learned:** The diagnostic skill from Chapter 9 (recognizing the
three-phase loss pattern) transferred perfectly. The method did not fail
because SAC is wrong -- it failed because the problem structure (POMDP)
does not match the algorithm's assumptions (Markov observations). This
mismatch is exactly the kind of structural diagnosis the book teaches.

---

## Scope Boundary: PegInsert and the Limits of Transfer

Factory PegInsert is documented as a **scope boundary** -- a case where our
method does not work, and we can explain precisely why.

### The Mismatch

| Property | Lift-Cube | PegInsert |
|----------|-----------|-----------|
| Actor observation | 36D (full state) | 19D (partial state) |
| Critic observation | 36D (same) | 72D (full state) |
| Markov? | Yes | No (POMDP) |
| NVIDIA reference | PPO, MlpPolicy | PPO + LSTM, asymmetric |
| SAC+MLP result | Learns multi-phase control | Stuck at approach tier |

### Why SAC+MLP Cannot Solve PegInsert

The actor's 19D observation is a single-frame snapshot: fingertip position,
velocity, and previous action. But fine insertion requires sensing how forces
evolve over multiple timesteps. Consider three contact states:

1. **Peg sliding into chamfer** -- forces are low, decreasing. Correct action: continue downward.
2. **Peg catching on rim** -- lateral forces are rising. Correct action: retract and realign.
3. **Peg jammed against sidewall** -- forces are high, stable. Correct action: wiggle.

All three may produce identical single-frame observations (similar pose,
similar forces). Only the *sequence* of forces distinguishes them. A
memoryless MLP maps identical inputs to identical outputs -- it cannot
distinguish these three states.

### What Would Be Needed

To solve PegInsert with our framework, we would need one of three approaches.
The most direct is to **enrich the observation** so that it becomes Markov --
by including contact forces, peg orientation relative to socket, and enough
history that a single frame contains all decision-relevant information, which
effectively converts the POMDP into an MDP. Alternatively, we could **add
recurrence** (LSTM or GRU layers in the policy) to give it learned memory,
though this requires switching from SB3 (which does not support recurrent SAC)
to a framework like RL-Games or cleanrl. A coarser approximation is **frame
stacking**, which provides a fixed temporal window rather than adaptive memory;
our experiment with 4-frame stacking was inconclusive and may need more frames
or a longer training budget.

This is not a failure of SAC -- it is a mismatch between problem structure
and algorithm assumptions. Recognizing this mismatch is exactly the
diagnostic skill the book teaches (Lesson 13).

---

## Pixels on Isaac: Bridging to Chapter 9

Chapter 9 showed that SAC could learn from pixels instead of state vectors
on MuJoCo FetchPush. A natural question: can we do the same on Isaac Lab?

### Approach: Render-Based Pixel Wrapper

Rather than using Isaac Lab's native visuomotor environments (which have
`subtask_terms` reward issues and are a harder Stack-Cube task), we add pixel
observations to Lift-Cube using `env.render()` -- exactly the same pattern
Chapter 9 used with FetchPush.

**How it works.** We create Lift-Cube with `render_mode="rgb_array"` and
`--enable_cameras`, so that after `Sb3VecEnvWrapper` the env exposes
`Box(36,)` state observations and supports `render()` for viewport frames.
Our `IsaacPixelObsWrapper` then calls `render()` each step, resizes the
1280x720 viewport to 84x84 via PIL, transposes HWC to CHW, and creates a Dict
observation: `{"pixels": uint8(3, 84, 84), "state": float32(36,)}`. SB3's
`MultiInputPolicy` auto-detects the image key and routes it through NatureCNN
(CNN for pixels, Flatten for the state vector, concatenated before the
policy/value MLP).

This mirrors Chapter 9's `PixelObservationWrapper` but wraps a VecEnv
(Isaac Lab) rather than a Gymnasium env. The observation design follows the
same sensor separation principle (Lesson 1): the CNN processes world-state
from pixels, while the state vector provides proprioception directly.

### The Trade-Off: Parallelism vs Pixels

Isaac Lab's viewport camera is **global** -- it renders the scene from a
single viewport, not per-environment. This forces `num_envs=1` for pixel
mode, collapsing the GPU parallelism advantage that made state-based training
so fast.

| Mode | num_envs | Expected fps | Wall time (2M steps) |
|------|----------|-------------|---------------------|
| State-based | 256 | ~9,000 | ~3.6 min |
| Pixel-based | 1 | ~50-200 | hours |
| Ch9 MuJoCo pixel Push | 4 | ~30-50 | ~40 hours |

This is an honest trade-off: GPU physics helps state-based training
enormously (256 envs, 9,000+ fps), but pixel rendering bottlenecks back to
serial execution. The rendering pipeline (Vulkan viewport capture -> CPU
resize -> numpy conversion) adds ~10-20ms per step.

### Running Pixel Lift-Cube

```bash
# Smoke test: verify pixel wrapper + MultiInputPolicy initialization
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke \
    --headless --seed 0 --pixel \
    --dense-env-id Isaac-Lift-Cube-Franka-v0 --smoke-steps 2000

# Full training (2M steps, num_envs=1 forced by --pixel)
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --seed 0 --pixel \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --total-steps 2000000 \
    --buffer-size 200000 \
    --batch-size 256 \
    --learning-starts 5000
```

A few key choices deserve explanation. The **`--pixel`** flag injects
`--enable_cameras` automatically and forces `num_envs=1` (since the viewport
camera is global). We set **buffer_size=200K** because each 84x84x3 uint8
frame is ~21KB, so 200K transitions (obs + next_obs) consume ~8GB -- which
fits in RAM. The **batch_size** stays at 256, the same as state-based
training, and **learning_starts=5000** ensures we collect some pixel experience
before the first gradient update.

### What Isaac Lab Also Offers (Native Visuomotor)

Isaac Lab has five visuomotor Stack-Cube variants with per-environment
cameras (table-mounted + wrist-mounted, 200x200 RGB each). These are
architecturally richer than our render-based approach but require non-trivial
adapter work. Isaac Lab provides float32 [0, 1] images while SB3 expects
uint8 [0, 255] -- our `IsaacImageNormWrapper` handles this conversion, but the
visuomotor envs also use `subtask_terms` instead of standard reward terms.
Processing two 200x200 camera streams (table-mounted and wrist-mounted) doubles
CNN computation, and Stack-Cube is itself a harder task (stacking two cubes)
with a different reward structure than Lift-Cube.

We chose the render-based approach because it isolates the pixel observation
question from the task difficulty question: same task (Lift-Cube), same
reward, just different observations. This makes the comparison with
state-based Lift-Cube clean and interpretable.

Readers interested in the native visuomotor integration can consult Isaac
Lab's camera sensor documentation (`isaaclab.sensors.Camera`), NVIDIA's
RL-Games training scripts for visuomotor tasks, and the
`IsaacImageNormWrapper` in our pipeline (which handles the float32->uint8
conversion).

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

### GPU Memory Exhaustion

**Symptom.** `CUDA out of memory` error during env creation or training.

**Cause.** Isaac Sim uses 8-12 GB of GPU memory just to boot. If your GPU is
shared with other workloads, the remaining memory may be insufficient for
SB3's replay buffer and networks.

**Resolution.** Monitor with `nvidia-smi` before launching. Free other GPU
workloads, or use a GPU with more VRAM. With 256 parallel Lift-Cube envs,
expect ~12-15 GB total GPU memory usage.

### First-Run Shader Compilation

**Symptom.** Isaac Sim appears to hang for 30-90 seconds after printing
initial startup logs.

**Cause.** First launch on a given GPU triggers Vulkan shader compilation.

**Resolution.** Wait. Subsequent runs reuse cached shaders (stored in named
Docker volumes). If you deleted the Docker volumes (`docker volume rm
isaac-cache-glcache`), the compilation will repeat.

### Curriculum Reward Collapse (Off-Policy Only)

**Symptom.** Training reward improves steadily for millions of steps, then
catastrophically collapses (e.g., -4 to -3,050 in one interval). Critic loss
explodes 100-500x. Reward never recovers.

**Cause.** Isaac Lab's CurriculumManager scaled reward penalty weights
mid-training. The replay buffer contains transitions collected under the old
reward scale; new transitions use the new (much larger) scale. The Q-function
cannot reconcile the two, causing Bellman target mismatch.

**Resolution.** Our pipeline automatically disables the CurriculumManager for
off-policy algorithms (SAC/TD3). If you see this on a new Isaac Lab env,
check whether it has curriculum terms that modify reward weights:

```bash
# Look for CurriculumManager in the env startup logs
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke \
    --headless --dense-env-id <your-env> --smoke-steps 100 2>&1 | grep -i curriculum
```

If it reports active curriculum terms, the pipeline should disable them
automatically. If it does not (e.g., a custom env), override `env_cfg.curriculum`
before `gym.make()`.

**Note:** This only affects off-policy algorithms. PPO is immune because it
discards experience after each update -- there is no replay buffer to become
stale.

### Low num_envs Throughput

**Symptom.** Training is much slower than expected (e.g., 88 fps on a
powerful GPU).

**Cause.** Using `num_envs=1` on a GPU-parallel physics engine. NVIDIA PhysX
is designed for batched execution -- `num_envs=1` wastes over 95% of compute
on kernel launch overhead.

**Resolution.** Use `--num-envs 256` for Lift-Cube (or `--num-envs 128` for
heavier tasks like Factory). `Sb3VecEnvWrapper` supports batched envs
natively. At `num_envs=256`, Lift-Cube achieves ~9,000-10,000 fps vs ~88 fps
at `num_envs=1` -- a **~100x throughput improvement**.

---

## Deliverables

A complete Appendix E Lift-Cube run should produce:

- `checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip`
- `checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.meta.json`
- `results/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_eval.json`
- `results/appendix_e_isaac_env_catalog.json`
- `videos/appendix_e_Isaac-Lift-Cube-Franka-v0_*.mp4` (from `record` command)

The `.meta.json` includes `obs_space_type`, `is_goal_conditioned`, `pixel`,
and `wrapper: "Sb3VecEnvWrapper"` fields, so you can verify after training
which observation convention was detected and which SB3 integration was used.
