# Goal-Conditioned Robotic Manipulation

A Docker-first reinforcement learning laboratory for studying sparse-reward manipulation tasks.

## What Is This?

This repository teaches you to train neural network policies that control a simulated Fetch robot arm to manipulate objects to arbitrary goal positions. The approach uses **SAC + HER** (Soft Actor-Critic with Hindsight Experience Replay) to handle sparse binary rewards.

## Quick Start

```bash
# Clone and enter the repository
git clone https://github.com/VladPrytula/robotics_series.git
cd robotics_series

# Run proof-of-life (verifies GPU, MuJoCo, rendering, training)
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
```

## The Problem We Solve

You want a robot that can reach **any** position, push objects to **any** location, pick and place **any** object--with goals specified at runtime, not programming time.

This is hard because:

- **Sparse rewards**: Binary success/failure feedback (no partial credit)
- **Continuous control**: 4D velocity commands at 25Hz
- **Goal generalization**: Infinite possible goals, cannot train separately for each

## The Method

The algorithm choice follows from problem constraints:

| Constraint | Implication | Solution |
|------------|-------------|----------|
| Continuous actions | Need policy gradient | Actor-critic |
| Sparse rewards | Need sample reuse | Off-policy (replay buffer) |
| Goal conditioning | Need goal relabeling | HER |
| Exploration | Need stochastic policy | SAC (max entropy) |

**Result**: SAC + HER is the methodologically appropriate solution.

## Curriculum

| Week | Topic | Done When |
|------|-------|-----------|
| 0 | [Proof of Life](tutorials/ch00_containerized_dgx_proof_of_life.md) | `ppo_smoke.zip` exists |
| 1 | [Environment Anatomy](tutorials/ch01_fetch_env_anatomy.md) | Reward check passes |
| 2-10 | Training, HER, Robustness | See [syllabus](syllabus.md) |

## Math Rendering Test

Inline math: $\pi^* = \arg\max_\pi J(\pi)$

Block math:

$$
J(\pi) = \mathbb{E}_{g \sim p(g)} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t, s_{t+1}, g) \right]
$$

Goal-conditioned MDP: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{G}, P, R, \phi, \gamma)$
