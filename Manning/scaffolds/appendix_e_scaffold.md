# Scaffold: Appendix E -- Isaac Lab Manipulation (GPU-Only)

## Classification

Type: Appendix (portability / simulator transfer). Closest protocol type: Algorithm + Engineering hybrid.
Source tutorial: tutorials/appendix_e_isaac_manipulation.md (~1,178 lines)
Book chapter output: Manning/chapters/appendix_e_isaac_manipulation.md

Lab files:
  - scripts/labs/isaac_sac_minimal.py (5 regions)
    - dict_flatten_encoder
    - squashed_gaussian_actor
    - twin_q_critic
    - sac_losses
    - sac_update_step
  - scripts/labs/isaac_goal_relabeler.py (4 regions)
    - goal_transition_structs
    - isaac_goal_sampling
    - isaac_relabel_transition
    - isaac_her_episode_processing

Production script: scripts/appendix_e_isaac_manipulation.py

Primary demonstration task: `Isaac-Lift-Cube-Franka-v0` (dense staged reward,
36D state, 8D joint-level actions, GPU-parallel physics). No peg-in-hole
or PegInsert references -- the appendix was reframed around Lift-Cube and
GPU-parallel scaling.

---

## Experiment Card

```
---------------------------------------------------------
EXPERIMENT CARD: Appendix E -- SAC on Isaac Lift-Cube
---------------------------------------------------------
Algorithm:    SAC (MlpPolicy [256, 256], auto-temperature)
Environment:  Isaac-Lift-Cube-Franka-v0
Fast path:    2,000,000 steps, seed 0, 256 envs
Time:         ~3.6 min (A100 GPU) -- includes Isaac Sim boot
Full run:     8,000,000 steps, seed 0, 256 envs, ~14 min

Run command (fast path):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 2000000

Run command (full):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99

Checkpoint track (skip training):
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip

Expected artifacts:
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.meta.json
  results/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_eval.json
  results/appendix_e_isaac_env_catalog.json

Success criteria (fast path, 2M steps):
  return_mean > -22  (approaching grasping phase)
  41% positive-return episodes (bimodal distribution)

Success criteria (full run, 8M steps):
  return_mean = +0.54 +/- 0.05
  100/100 positive-return episodes
  Throughput >= 9,000 fps at 256 envs

Pixel variant (4M steps, 64 envs, ~56 min):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 --pixel \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 64 --total-steps 4000000
  Success: return_mean ~ -1.07 at convergence
  Throughput: ~1,181 fps (TiledCamera, 64 envs)
---------------------------------------------------------
```

---

## Reproduce It Block

```
---------------------------------------------------------
REPRODUCE IT
---------------------------------------------------------
The results in this appendix come from these runs:

State-based (primary):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99

Evaluation:
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py eval \
    --headless \
    --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip

Pixel (TiledCamera):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 --pixel \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 64 --total-steps 4000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99

Hardware:     NVIDIA A100-SXM4-80GB (any modern GPU works; times will vary)
Time:         ~14 min (state, 8M steps) / ~56 min (pixel, 4M steps)
Seeds:        0 (single seed -- appendix scope)

Artifacts produced:
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.meta.json
  results/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_eval.json
  results/appendix_e_isaac_env_catalog.json
  videos/appendix_e_Isaac-Lift-Cube-Franka-v0_*.mp4

Results summary -- State-based (8M steps, seed 0):
  return_mean:  +0.54 +/- 0.05  (100 episodes, deterministic)
  positive_eps: 100/100 (100%)
  min_return:   +0.46
  max_return:   +0.68
  throughput:   9,377 fps (256 envs)
  wall_time:    853 seconds (~14 min)

Training progression (8M state-based):
  64K:  -27.2 (reaching only)
  1M:   -24.8 (reaching improving)
  2M:   -21.8 (approaching grasping)
  3M:   -15.4 (grasping beginning)
  4M:    -6.4 (lifting)
  5M:    -1.7 (goal tracking)
  6M:    -1.5 (tracking refinement)
  8M:    -1.4 (converged)

Results summary -- Pixel (4M steps, seed 0):
  return_mean:  ~-1.07 at convergence
  throughput:   ~1,181 fps (64 envs)
  wall_time:    ~56 min
  hockey-stick takeoff at ~1.0M steps

Wall-clock comparison:
  Ch9 pixel Push (MuJoCo):   ~40 hours (30-50 fps)
  Ch4 state Push (MuJoCo):   ~1 hour (500-600 fps)
  Isaac Lift-Cube (2M):      ~3.6 min (9,363 fps)
  Isaac Lift-Cube (8M):      ~14 min (9,377 fps)
  Isaac Lift-Cube pixel:     ~56 min (1,181 fps)

If your numbers differ by more than ~20%, check the
"What Can Go Wrong" section. Isaac throughput is sensitive
to GPU model, num_envs, and whether other workloads share
the GPU.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

---

## Build It Components

| # | Component | Equation / concept it implements | Lab file:region | Verify check |
|---|-----------|--------------------------------|-----------------|--------------|
| 1 | Dict observation encoder | Flatten selected dict keys into feature vector; supports both `{observation, achieved_goal, desired_goal}` and `{policy}` conventions | `scripts/labs/isaac_sac_minimal.py:dict_flatten_encoder` | "expect shape (batch, flat_dim); finite values; both Gym-Robotics and Isaac layouts" |
| 2 | Squashed Gaussian actor | $a = \tanh(u)$, $\log\pi(a|s) = \log\pi(u|s) - \sum\log(1-\tanh^2(u_i))$ | `scripts/labs/isaac_sac_minimal.py:squashed_gaussian_actor` | "actions in (-1,1); logp shape (batch,); all finite" |
| 3 | Twin Q critic | $Q_1(s,a), Q_2(s,a)$ -- clipped double Q to reduce overestimation | `scripts/labs/isaac_sac_minimal.py:twin_q_critic` | "two scalar Q-values per (obs, action) pair; finite" |
| 4 | SAC losses | Critic: $L = \text{MSE}(Q_i, r + \gamma(1-d)(\min Q_{\bar\theta} - \alpha\log\pi))$; Actor: $L = (\alpha\log\pi - \min Q)$; Temperature: $L = -\alpha(\log\pi + \bar{H})$ | `scripts/labs/isaac_sac_minimal.py:sac_losses` | "all losses finite after 25 updates; alpha > 0" |
| 5 | SAC update step | Wiring: critic update -> actor update -> temperature update -> Polyak target update ($\tau=0.005$) | `scripts/labs/isaac_sac_minimal.py:sac_update_step` | "returns dict with critic_loss, actor_loss, alpha_loss, alpha, entropy; all finite" |
| 6 | Goal transition structures | `GoalTransition` NamedTuple (obs, action, reward, next_obs, done, achieved_goal, desired_goal) + `GoalStrategy` enum | `scripts/labs/isaac_goal_relabeler.py:goal_transition_structs` | "GoalTransition fields match expected types; GoalStrategy has final/future/episode" |
| 7 | Goal sampling | Sample $k$ alternative goals from achieved goals in same episode using future/final/episode strategy | `scripts/labs/isaac_goal_relabeler.py:isaac_goal_sampling` | "returns exactly k goals; shapes match achieved_goal" |
| 8 | Transition relabeling | HER core: keep (obs, action, achieved_goal), replace desired_goal, recompute reward | `scripts/labs/isaac_goal_relabeler.py:isaac_relabel_transition` | "self-achieved relabel -> reward=0.0 (success); achieved_goal unchanged" |
| 9 | Episode HER processing | Combine original + relabeled transitions with her_ratio gating; compute success_fraction | `scripts/labs/isaac_goal_relabeler.py:isaac_her_episode_processing` | "HER transitions > original count; success_fraction increases" |

**Verify commands (CPU-only, no Isaac Sim needed):**

```bash
# SAC math -- tests both goal-conditioned and Isaac-style obs (~10-20s)
python scripts/labs/isaac_sac_minimal.py --verify

# Goal relabeling invariants and data amplification (~1-5s)
python scripts/labs/isaac_goal_relabeler.py --verify
```

**Ordering note:** Components 1-5 form the SAC core (foundation first: encoder
before actor/critic, networks before losses, losses before wiring step).
Components 6-9 form the HER relabeling track (structures before sampling,
sampling before relabeling, relabeling before episode processing). The two
tracks are independent and can be presented in either order; the tutorial
presents SAC first because Lift-Cube uses SAC without HER.

---

## Bridging Proof

- **Inputs (same data fed to both):** Synthetic batch with dict observations
  (both `{observation, achieved_goal, desired_goal}` and `{policy}` layouts)
- **From-scratch output:** `DictFlattenEncoder` -> `SquashedGaussianActor` ->
  `TwinQCritic` -> `sac_update_step` returns `{critic_loss, actor_loss,
  alpha_loss, alpha, entropy}` -- all finite after 25 updates
- **SB3 output:** `SAC(MlpPolicy, ...)` or `SAC(MultiInputPolicy, ...)` --
  the pipeline auto-detects observation convention and selects policy class
- **Match criteria:** Structural correspondence (not numerical identity --
  different initializations). The bridge maps:
  - `DictFlattenEncoder` -> `MultiInputPolicy` feature concat (goal-conditioned) or `MlpPolicy` feature extractor (Isaac flat-dict)
  - `SquashedGaussianActor` -> SAC actor network
  - `TwinQCritic` -> SAC critic / target critics
  - `temperature_loss` -> SAC `ent_coef` auto tuning
  - `sac_update_step` -> `model.learn()` gradient step internals
- **Lab mode:** `python scripts/labs/isaac_sac_minimal.py --bridge`

The bridge is a **structural mapping** rather than a numerical comparison
because the appendix reuses the same SAC components from Ch3-4. The primary
verification is that the from-scratch code handles both observation conventions
(Gymnasium-Robotics dict and Isaac Lab flat-dict) correctly, which `--verify`
confirms.

---

## What Can Go Wrong

| # | Symptom | Likely cause | Diagnostic |
|---|---------|-------------|------------|
| 1 | Script hangs after first env closes, or `RuntimeError("Simulation context already exists")` | Isaac Lab's `SimulationContext` is a process-level singleton; second env creation in same process fails | Use one env per process. The `all` subcommand handles this via subprocess isolation. For custom orchestration, use `subprocess.call()`. |
| 2 | `KeyError: 'achieved_goal'` or `KeyError: 'desired_goal'` when creating model | Assuming Gymnasium-Robotics observation keys on an Isaac env that uses `{'policy': ...}` | Run `discover-envs` first; check `probed_envs` in catalog JSON. Pipeline auto-detects and selects `MlpPolicy` vs `MultiInputPolicy`. |
| 3 | NaN in actor loss or exploding Q-values with custom wrapper | Isaac Lab declares `Box(-inf, inf)` action spaces; passing infinite bounds to SB3's tanh squashing produces undefined gradients | Use `Sb3VecEnvWrapper` (clips to `[-100, 100]` automatically). If custom wrapper, clip bounds explicitly. |
| 4 | Segfault on startup, or `VkResult` / Vulkan error | Missing `--headless` flag on headless system (DGX, CI) -- Isaac Sim tries to open Vulkan display | Always pass `--headless` on headless machines. Error appears before any Python code runs. |
| 5 | `CUDA out of memory` during env creation or training | Isaac Sim uses 8-12 GB GPU memory just to boot; shared GPU has insufficient remaining memory | Monitor with `nvidia-smi` before launching. Free other workloads. At 256 envs, expect ~12-15 GB total. |
| 6 | Isaac Sim appears to hang for 30-90 seconds after startup logs | First-launch Vulkan shader compilation on a new GPU | Wait. Subsequent runs reuse cached shaders (stored in named Docker volumes). Deleting volumes triggers recompilation. |
| 7 | Reward collapses catastrophically mid-training (e.g., -4 to -3,050 in one interval); critic loss explodes 100-500x; reward never recovers | CurriculumManager scaled reward penalty weights 1000x mid-training; replay buffer has transitions under old scale mixed with new scale, poisoning Q-function | Pipeline auto-disables CurriculumManager for SAC/TD3. For new envs, check for curriculum terms: `... smoke --headless --dense-env-id <env> --smoke-steps 100 2>&1 \| grep -i curriculum`. Only affects off-policy; PPO is immune. |
| 8 | Training much slower than expected (e.g., 88 fps on powerful GPU) | Using `num_envs=1` on GPU-parallel physics engine; PhysX designed for batched execution, `num_envs=1` wastes 95%+ compute on kernel launch overhead | Use `--num-envs 256` for Lift-Cube. At 256 envs: ~9,000 fps vs ~88 fps at 1 env (~100x difference). |

---

## Adaptation Notes

### Cut from tutorial

- Extended verification protocol prose (4 tests with full "What we verify /
  Why this matters / What failure means" blocks) -- compress to a quick-check
  table with 1-2 sentences per test
- Detailed `dev-isaac.sh` vs `dev.sh` comparison table -- move to a sidebar
  or footnote; book readers do not need the full rationale for each Docker
  difference
- Verbose Docker volume caching explanation -- one sentence suffices
- The `--enable_cameras` implementation details for `record` subcommand --
  keep the command, cut the internals
- Full listing of all Run It subcommand commands (E.10-E.16) -- consolidate
  into the Experiment Card and a "quick reference" command table

### Keep from tutorial

- **Bridge section** (Why Appendix E Exists) -- adapt to Manning chapter bridge format
- **WHY: Lifting Is Different From Reaching** -- the staged dense reward
  analysis and multi-phase control breakdown are excellent pedagogical content
- **The Hidden Curriculum (and Why SAC Must Disable It)** -- the curriculum
  crash narrative is the centerpiece debugging lesson; keep the full 3-act
  structure (initial run -> crash -> fix)
- **Honest Difficulty Comparison** -- essential for intellectual honesty;
  keep the Isaac-Lift-Cube vs FetchPickAndPlace comparison table and the
  "what this proves / does not prove" framing
- **Pixels on Isaac: Native TiledCamera Results** -- the hockey-stick curve,
  throughput comparison, and key lessons (`clip=(0,255)`, proprioception in
  Dict obs, curriculum must be disabled)
- **Wall-clock comparison table** -- the dramatic speedup numbers (15-170x)
  are the core value proposition
- **Observation space differences** (Gymnasium-Robotics vs Isaac Lab) -- this
  is the portability lesson
- **All 8 failure modes** from What Can Go Wrong -- battle-tested and specific
- **Training progression tables** (both state and pixel) with phase annotations
- **Build It components** (all 9 regions) -- already exist and verified

### Add for Manning

- **Opening Promise** (3-5 bullets) -- tutorial lacks this Manning format element
- **Chapter Bridge** connecting to Part 6 bonus material and Ch1-6 contract portability
- **Figure Plan** -- tutorial has no figures; the book needs learning curves,
  wall-clock comparison chart, and pixel hockey-stick curve at minimum
- **Exercises** (3-5, graduated difficulty) -- tutorial has none
- **Summary section** with forward-looking bridge (what this appendix does NOT
  cover: sparse reward on Isaac, POMDP contact tasks, multi-GPU scaling)
- **Concept Registry Additions** -- formalize the new terms introduced
- **Sidebar: "Curriculum + Off-Policy = Danger"** -- extract the general
  principle (replay buffers assume stationary MDP; curriculum violates this)
  into a reusable callout
- **Sidebar: "When NOT to Use HER"** -- Lift-Cube is dense + not
  goal-conditioned, so HER is inapplicable. Clarify the conditions under
  which HER applies (from Ch5) vs does not.

---

## Chapter Bridge

1. **Capability established:** Through Chapters 1-6, the reader built a
   complete workflow on Gymnasium-Robotics Fetch tasks: environment contracts,
   PPO, SAC, HER, and curriculum learning for multi-phase manipulation. Every
   experiment used MuJoCo physics on CPU.

2. **Gap:** The methodology is proven on one simulator family. Does it
   transfer? And can GPU-parallel physics make the same experiments run orders
   of magnitude faster?

3. **This appendix adds:** Portability evidence. The reader applies the same
   SAC methodology to Isaac Lab's Lift-Cube task -- a GPU-parallel manipulation
   environment with different observation conventions, action spaces, and
   engineering constraints. The same diagnostic skills (observation inspection,
   dense-first debugging, curriculum crash analysis) prove essential in the new
   simulator.

4. **Foreshadow:** This appendix demonstrates transfer on a dense-reward,
   fully observable task. Sparse-reward manipulation on Isaac Lab (requiring
   HER in a non-goal-conditioned env) and contact-rich POMDP tasks (requiring
   recurrent policies) remain open challenges beyond our MLP-based pipeline.

---

## Opening Promise

> **This appendix covers:**
> - Transferring the SAC methodology from MuJoCo Fetch to Isaac Lab's Lift-Cube task, demonstrating that the same algorithm and diagnostic skills work across simulators
> - Understanding how Isaac Lab's observation conventions (`{policy}` vs `{observation, achieved_goal, desired_goal}`) affect pipeline design -- and why porting a method does not mean drop-in compatibility
> - Diagnosing and fixing a catastrophic curriculum-induced reward collapse that reveals a fundamental incompatibility between off-policy replay buffers and non-stationary reward functions
> - Running pixel-based manipulation on Isaac Lab using the native TiledCamera sensor, achieving 24x faster throughput than MuJoCo pixel training
> - Comparing wall-clock performance: 14 minutes (Isaac, 8M steps) vs 40 hours (MuJoCo pixel, 8M steps) -- a 170x speedup from GPU-parallel physics

---

## Figure Plan

| # | Description | Type | Source command | Chapter location |
|---|------------|------|---------------|-----------------|
| 1 | State-based learning curve (8M steps): ep_rew_mean vs timesteps with phase annotations (reaching, grasping, lifting, tracking) | curve | Extract from TensorBoard logs or reconstruct from training progression table; matplotlib | After "Act 3: Clean Run" in Run It |
| 2 | Wall-clock comparison bar chart: Ch9 pixel MuJoCo (~40h) vs Ch4 state MuJoCo (~1h) vs Isaac state 8M (~14min) vs Isaac pixel 4M (~56min) -- log scale x-axis, labeled bars | comparison | matplotlib from wall-clock table data | After Wall-Clock Comparison section |
| 3 | Pixel learning curve (4M steps): hockey-stick shape with flat regime (0-800K) annotated, takeoff at ~1M, convergence at ~3M | curve | Extract from TensorBoard logs or reconstruct from pixel training progression table; matplotlib | After "Pixel Training Results" section |
| 4 | Curriculum crash diagnostic: overlaid reward curve showing clean run vs curriculum-enabled run, with crash point at ~4.6M annotated | comparison | Reconstruct from crash data (before: -4.14 at 4.6M, after: -3,050) vs clean run progression; matplotlib | After "Act 2: The Curriculum Crash" |

**Per-figure checklist (for Writer/Revisor):**
- [ ] Caption drafted (Figure E.N format with generation command)
- [ ] Alt-text written (for accessibility)
- [ ] Uses Wong (2011) colorblind-friendly palette
- [ ] Minimum 640x480, PNG format
- [ ] Referenced by number in the text

---

## Estimated Length

| Section | Words |
|---------|-------|
| Opening Promise + Chapter Bridge | 300 |
| WHY: Lifting Is Different (staged reward, hidden curriculum, action space, full observability, why SAC works) | 1,500 |
| HOW / Build It (9 components across 2 lab files, interleaved with math + verification checkpoints) | 2,000 |
| Bridge (from-scratch to SB3 structural mapping) | 400 |
| Run It (Experiment Card, commands, results tables, wall-clock comparison, honest difficulty comparison, pixel results) | 1,800 |
| What Can Go Wrong (8 failure modes) | 800 |
| Summary + Reproduce It + Exercises | 1,200 |
| **Total** | **~8,000** |

**Note:** At ~8,000 words, this exceeds the 6,000-word span threshold. The
Writer phase should split into 2 spans:
- **Span 1** (Opening Promise through Build It): ~4,200 words
- **Span 2** (Bridge through Exercises): ~3,800 words

Preferred split point: between last Build It section and Bridge section
(natural shift from implementation to verification/production).

---

## Concept Registry Additions

Terms this appendix introduces (to be added to the registry):

```
Appendix E: Isaac Lab, SimulationContext singleton, Sb3VecEnvWrapper,
     TiledCamera sensor, NGC base image, policy observation key,
     CurriculumManager (Isaac Lab), reward stationarity assumption,
     curriculum-replay incompatibility, staged dense reward,
     GPU-parallel physics throughput, headless rendering (Vulkan),
     shader compilation cache, named Docker volumes,
     observation convention portability
```

---

## Dependencies

- **Lab regions needed (for Lab Engineer):** All 9 regions already exist and
  are verified. No new lab code needed.
  - `scripts/labs/isaac_sac_minimal.py`: dict_flatten_encoder,
    squashed_gaussian_actor, twin_q_critic, sac_losses, sac_update_step
  - `scripts/labs/isaac_goal_relabeler.py`: goal_transition_structs,
    isaac_goal_sampling, isaac_relabel_transition, isaac_her_episode_processing

- **Pretrained checkpoints needed (for Reproduce It):**
  - `checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip`
  - Periodic checkpoints at 200K, 3M, 8M for video progression

- **Previous chapter concepts used:**
  - Ch1: reproducibility, proof-of-life verification pattern, container/image
  - Ch3: SAC (off-policy, replay buffer, maximum entropy objective, twin
    critics, automatic temperature tuning, squashed Gaussian policy)
  - Ch4: HER (goal relabeling, goal sampling strategies, off-policy
    requirement) -- referenced but NOT applied (Lift-Cube is not
    goal-conditioned)
  - Ch5: curriculum learning, goal stratification, dense-first debugging --
    the curriculum crash directly relates to Ch5's curriculum patterns
  - Ch6: multi-phase control (approach, grasp, lift, track) -- same structure
    as PickAndPlace capstone
  - Ch9: pixel observations, NatureCNN, hockey-stick learning curve, sample
    efficiency ratio -- pixel Isaac results extend Ch9's methodology

- **Infrastructure:**
  - `docker/dev-isaac.sh` (Isaac container entry point)
  - `docker/Dockerfile.isaac` (thin layer on NGC Isaac Lab 2.3.2)
  - `docker/build.sh isaac` (build command)
  - Linux + NVIDIA GPU required (no Mac support)
