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
Chapters 6-10: Advanced topics
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
