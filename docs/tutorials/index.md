# Tutorials

A systematic course in goal-conditioned reinforcement learning for robotic manipulation.

## Quick Navigation

| Chapter | Topic | Key Deliverable | Run It | Build It |
|---------|-------|-----------------|--------|----------|
| [Chapter 0](ch00_containerized_dgx_proof_of_life.md) | Proof of Life | Working Docker environment, `ppo_smoke.zip` | `ch00_proof_of_life.py` | -- |
| [Chapter 1](ch01_fetch_env_anatomy.md) | Environment Anatomy | `reward-check` passes, baseline metrics | `ch01_fetch_env_anatomy.py` | -- |
| [Chapter 2](ch02_ppo_dense_reach.md) | PPO Baseline | >90% success on dense Reach, pipeline validated | `ch02_ppo_dense_reach.py` | `labs/ppo_from_scratch.py` |
| [Chapter 3](ch03_sac_dense_reach.md) | SAC + Replay Diagnostics | SAC matches PPO, Q-values stable | `ch03_sac_dense_reach.py` | `labs/sac_from_scratch.py` |
| [Chapter 4](ch04_her_sparse_reach_push.md) | HER for Sparse | HER vs no-HER separation on Reach/Push | `ch04_her_sparse_reach_push.py` | `labs/her_relabeler.py` |
| [Chapter 5](ch05_pick_and_place.md) | PickAndPlace | Stratified eval, stress testing, curriculum | `ch05_pick_and_place.py` | `labs/curriculum_wrapper.py` |
| [Chapter 6](ch06_action_interface.md) | Policies as Controllers | RL vs PD decomposition, engineering metrics | `ch06_action_interface.py` | `labs/action_interface.py` |
| [Chapter 7](ch07_robustness_curves.md) | Robustness Curves | Noise degradation analysis, robustness AUC | `ch07_robustness_curves.py` | `labs/robustness.py` |
| [Chapter 10](ch10_visual_reach.md) | Visual SAC | State vs pixel vs DrQ comparison, `--fast` mode | `ch10_visual_reach.py` | `labs/pixel_wrapper.py`, `labs/visual_encoder.py`, `labs/image_augmentation.py` |
| [Appendix E](appendix_e_isaac_manipulation.md) | Isaac Lab Manipulation | Isaac Lift-Cube training/eval, GPU-parallel scaling | `appendix_e_isaac_manipulation.py` | `labs/isaac_sac_minimal.py`, `labs/isaac_goal_relabeler.py` |

**Legend:**

- **Run It:** Production pipeline using Stable Baselines 3. Reproducible, optimized, tracked in TensorBoard.
- **Build It:** From-scratch implementations showing how equations map to code. Educational, not for production.

## The Learning Path

```
Chapter 0: Can I run code?
    |
Chapter 1: What does the robot see/do?
    |
Chapter 2: Can I train anything? (PPO baseline, on-policy)
    |
Chapter 3: Can I train off-policy? (SAC, replay buffer)
    |
Chapter 4: Can I handle sparse rewards? (HER)
    |
Chapter 5: Can I grasp and place? (PickAndPlace)
    |
Chapter 6: How well does it behave? (Policies as Controllers)
    |
Chapter 7: How fragile is it? (Robustness Curves)
    |
Chapter 10: Can it learn from pixels? (Visual SAC + DrQ)
    |
Appendix E: Does the method port to Isaac Lab? (GPU-only)
```

## Running Commands

All commands run through Docker via the `docker/dev.sh` wrapper:

```bash
# Pattern: bash docker/dev.sh <command>
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all
```

### Chapter Scripts

Each chapter has a self-contained orchestration script:

```bash
# Full pipeline (train + eval + compare)
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed 0
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all --seed 0
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py reach-all --seeds 0,1,2
bash docker/dev.sh python scripts/ch05_pick_and_place.py all --seeds 0,1,2
bash docker/dev.sh python scripts/ch06_action_interface.py all --seed 0 --include-push

# Partial execution
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py train --total-steps 100000
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py eval
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py compare
```

### Monitoring Training

```bash
# Start TensorBoard (http://localhost:6006)
bash docker/dev.sh tensorboard --logdir runs --bind_all
```

### Long Jobs with tmux

```bash
tmux new -s rl                    # Start session
# ... run training ...
# Ctrl-b d                        # Detach
tmux attach -t rl                 # Reattach later
```

## Chapter Summaries

### [Chapter 0: Proof of Life](ch00_containerized_dgx_proof_of_life.md)

**Goal:** Verify your experimental environment works.

**You will:**

- Build and enter a Docker container with GPU access
- Verify MuJoCo physics simulation runs
- Confirm headless rendering produces images
- Run a short PPO training loop

**Done when:** `ppo_smoke.zip` exists and loads successfully.

---

### [Chapter 1: Environment Anatomy](ch01_fetch_env_anatomy.md)

**Goal:** Understand exactly what the Fetch environments provide.

**You will:**

- Inspect observation space (dict with `observation`, `achieved_goal`, `desired_goal`)
- Understand action semantics (4D Cartesian velocity)
- Verify reward consistency with `compute_reward()`
- Collect random baseline metrics

**Done when:** `reward-check` passes and you can answer:

- What is the observation dimension for FetchReach? *(10)*
- What does action index 3 control? *(gripper)*
- Why can HER relabel goals? *(`compute_reward` accepts arbitrary goals)*

---

### [Chapter 2: PPO on Dense Reach](ch02_ppo_dense_reach.md)

**Goal:** Validate the training pipeline with the simplest method that should work.

**You will:**

- Train PPO on FetchReachDense-v4 (continuous reward signal)
- Understand PPO's clipped surrogate objective and why it exists
- Learn to read TensorBoard diagnostics for training health
- Establish baseline metrics for comparison

**Done when:** Success rate > 90% and you can answer:

- Why does PPO clip probability ratios instead of parameter updates?
- What does `clip_fraction = 0.15` mean in TensorBoard?
- Why use dense rewards for pipeline validation?

---

### [Chapter 3: SAC on Dense Reach](ch03_sac_dense_reach.md)

**Goal:** Validate off-policy learning before adding HER.

**You will:**

- Train SAC on FetchReachDense-v4 (same task as Chapter 2)
- Understand maximum-entropy RL and auto-tuned temperature
- Add replay buffer diagnostics (Q-values, entropy, goal distances)
- Benchmark throughput scaling with `n_envs`

**Done when:** SAC achieves >95% success and you can answer:

- Why does SAC add an entropy bonus to the objective?
- What does it mean if Q-values grow unbounded?
- Why must we validate SAC separately before adding HER?

---

### [Chapter 4: HER on Sparse Reach/Push](ch04_her_sparse_reach_push.md)

**Goal:** Demonstrate that HER is the difference-maker on sparse goals.

**You will:**

- Train SAC without HER on sparse environments (establish baseline difficulty)
- Train SAC + HER on the same tasks (see improvement)
- Run multi-seed experiments for statistical validity
- Discover why Reach is too easy and Push is the real test

**Actual Results (FetchReach-v4):**

| Method | Success Rate | Notes |
|--------|--------------|-------|
| SAC (no-HER) | 96.0% +/- 8.0% | Reach is easy enough for random exploration |
| SAC + HER | 100.0% +/- 0.0% | Perfect, but weak separation |

**Key insight:** FetchReach-v4 is too simple to demonstrate HER's value. The real test is FetchPush-v4, where object manipulation makes random success nearly impossible.

**Done when:** Clear separation on Push (>50 percentage points) and you can answer:

- Why does relabeling turn failures into successes?
- Why can only off-policy algorithms use HER?
- Why does Reach show weak separation but Push shows strong separation?

---

### [Chapter 5: PickAndPlace](ch05_pick_and_place.md)

**Goal:** Transfer SAC+HER from Push to the harder PickAndPlace task.

**You will:**

- Validate the pipeline with dense-first debugging (~5 min)
- Train SAC+HER on sparse PickAndPlace (5M steps)
- Evaluate with goal stratification (air vs table goals separately)
- Stress test with noise injection (observation + action noise)
- Build a curriculum learning wrapper (Build It lab)

**Done when:** Stratified evaluation completes and you can answer:

- Why is PickAndPlace harder than Push? *(multi-phase control: grasp, lift, place)*
- What does the air gap tell you? *(table goals may be solvable by pushing; air goals require grasping)*
- Why test with dense rewards first? *(catches pipeline bugs in 5 min instead of 5 hours)*

---

### [Chapter 6: Policies as Controllers](ch06_action_interface.md)

**Goal:** Treat learned policies like controllers and quantify stability.

**You will:**

- Evaluate SAC+HER checkpoints under action scaling and low-pass filtering
- Build a proportional controller (PD) as a classical baseline
- Compare RL vs PD with engineering metrics (smoothness, TTS, path length, energy)
- Decompose tasks into "planning" vs "control" problems

**Actual Results:**

| Environment | RL (SAC+HER) | PD (best Kp) | Gap | Task Type |
|-------------|:------------:|:------------:|:---:|-----------|
| FetchReach-v4 | 100% | 100% | 0% | Pure control |
| FetchPush-v4 | 100% | 5% | +95% | Requires planning |

**Done when:** You can compare RL vs baseline on controller metrics and answer:

- Why does scaling down hurt Push but not Reach? *(Push needs minimum contact force)*
- Why does heavy filtering cost 15% success on Push? *(delays approach-to-push transition)*
- What does a 95% gap between RL and PD tell you? *(RL learned multi-phase strategies PD cannot)*

---

### [Chapter 7: Robustness Curves](ch07_robustness_curves.md)

**Goal:** Quantify policy brittleness under observation and action noise.

**You will:**

- Inject calibrated noise into trained policies and measure degradation
- Compute robustness metrics (critical sigma, degradation slope, AUC)
- Compare robustness across algorithms and environments
- Train noise-augmented policies for improved robustness

**Done when:** Robustness sweeps complete and you can answer:

- What noise level causes 50% success drop? *(critical sigma)*
- Is the policy more sensitive to observation or action noise?
- Does noise-augmented training improve robustness without hurting clean performance?

---

### [Chapter 10: Pixels, No Cheating](ch10_visual_reach.md)

**Goal:** Quantify the cost of learning from pixels and recover part of the gap with DrQ.

**You will:**

- Train SAC on FetchReachDense with state observations (baseline)
- Train SAC from 84x84 pixel observations (no privileged state)
- Train SAC with DrQ random shift augmentation (Kostrikov et al., 2020)
- Measure the sample-efficiency gap across all three configurations
- Use `--fast` mode for 2-3x speedup (native rendering, SubprocVecEnv)

**Done when:** Three-way comparison table exists and you can answer:

- How many more samples does pixel SAC need vs state SAC?
- How does DrQ regularization help the Q-function?
- Why is rendering the bottleneck, not the GPU?

---

### [Appendix E: Isaac Lab Manipulation](appendix_e_isaac_manipulation.md)

**Goal:** Port the same experiment contract to Isaac Lab with GPU-parallel physics.

**You will:**

- Discover available Isaac env IDs and verify the Isaac container
- Train SAC on Lift-Cube (state-based: 256 envs, ~9K fps)
- Train SAC from pixels via native TiledCamera (64 envs, ~1.2K fps)
- Produce the same artifact pattern: checkpoint + metadata + eval JSON + comparison JSON
- Validate from-scratch Build It components for SAC update math and goal relabeling

**Done when:** You have a valid `appendix_e_sac_*` checkpoint and evaluation JSON, and can answer:

- How does Lift-Cube's dense reward compare to FetchPickAndPlace's sparse reward?
- Why must the CurriculumManager be disabled for off-policy SAC?
- What does the hockey-stick learning curve in pixel training tell you about visual representation learning?

---

## Prerequisites

Before starting, ensure you have:

- [ ] Access to a machine with NVIDIA GPU
- [ ] Docker installed with NVIDIA Container Toolkit
- [ ] Basic Python and command-line familiarity
- [ ] This repository cloned locally

## How to Use These Tutorials

1. **Work sequentially** - Each chapter builds on the previous
2. **Run every command** - Don't just read; execute
3. **Complete all deliverables** - They verify your understanding
4. **Understand before proceeding** - Speed is not the goal

---

*For the philosophical motivation behind this curriculum, see the [full README](https://github.com/VladPrytula/robotics_series/blob/main/tutorials/README.md).*
