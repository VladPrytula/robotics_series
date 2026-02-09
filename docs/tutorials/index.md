# Tutorials

A systematic course in goal-conditioned reinforcement learning for robotic manipulation.

## Quick Navigation

| Chapter | Topic | Key Deliverable |
|---------|-------|-----------------|
| [Chapter 0](ch00_containerized_dgx_proof_of_life.md) | Proof of Life | Working Docker environment, `ppo_smoke.zip` |
| [Chapter 1](ch01_fetch_env_anatomy.md) | Environment Anatomy | `reward-check` passes, baseline metrics |
| Chapter 2 | PPO Baseline | *Coming soon* |
| Chapter 3 | SAC on Dense | *Coming soon* |
| Chapter 4 | HER for Sparse | *Coming soon* |

## The Learning Path

```
Chapter 0: Can I run code?
    |
Chapter 1: What does the robot see/do?
    |
Chapter 2: Can I train anything? (PPO baseline)
    |
Chapter 3: Can I train better? (SAC)
    |
Chapter 4: Can I handle sparse rewards? (HER)
    |
Chapters 5-10: Advanced topics
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
