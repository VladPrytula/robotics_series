# Tutorials

A systematic course in goal-conditioned reinforcement learning for robotic manipulation.

## Quick Navigation

| Chapter | Topic | Key Deliverable |
|---------|-------|-----------------|
| [Chapter 0](ch00_containerized_dgx_proof_of_life.md) | Proof of Life | Working Docker environment, `ppo_smoke.zip` |
| [Chapter 1](ch01_fetch_env_anatomy.md) | Environment Anatomy | `reward-check` passes, baseline metrics |
| [Chapter 2](ch02_ppo_dense_reach.md) | PPO Baseline | >90% success on dense Reach, pipeline validated |
| [Chapter 3](ch03_sac_dense_reach.md) | SAC + Replay Diagnostics | SAC matches PPO, Q-values stable |
| Chapter 4 | HER for Sparse | *Coming soon* |

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
Chapters 5-10: Advanced topics
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
