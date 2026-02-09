# Goal-Conditioned Robotic Manipulation: A Research Platform

*A Docker-first reinforcement learning laboratory for studying sparse-reward manipulation tasks.*

This repository is structured for step-by-step study. A 10-week curriculum ([`syllabus.md`](syllabus.md)) provides executable commands and verification criteria. Tutorials ([`tutorials/`](tutorials/)) provide theoretical context. The goal is not to run scripts quickly, but to understand deeply.

---

## §1. Problem Formulation

**The Central Question.** Can an agent learn to manipulate objects to arbitrary goal configurations using only sparse binary feedback?

This is not merely an engineering challenge. It is a fundamental question in sequential decision-making under uncertainty. The problem exhibits three critical characteristics that make it non-trivial:

1. **High-dimensional continuous control.** The action space is $\mathcal{A} \subset \mathbb{R}^4$ (3D Cartesian delta + gripper command). Standard discrete action methods do not apply.

2. **Sparse binary rewards.** The reward function $R(s, a, s', g) = \mathbf{1}[\|g_{\text{achieved}}(s') - g\| < \epsilon]$ provides no gradient signal until the goal is achieved. Random exploration fails catastrophically.

3. **Goal conditioning.** The agent must generalize across a distribution of goals $p(g)$, not merely solve a single task. This requires learning a universal policy $\pi: \mathcal{S} \times \mathcal{G} \to \Delta(\mathcal{A})$.

**Definition (Goal-Conditioned MDP).** A goal-conditioned Markov Decision Process is a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{G}, P, R, \gamma)$ where:
- $\mathcal{S}$ is the state space
- $\mathcal{A}$ is the action space
- $\mathcal{G}$ is the goal space
- $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ is the transition kernel
- $R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{G} \to \mathbb{R}$ is the goal-conditioned reward
- $\gamma \in [0,1)$ is the discount factor

We seek $\pi^* = \arg\max_\pi \mathbb{E}_{g \sim p(g)} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}, g) \right]$.

---

## §2. Methodology

This repository implements **off-policy actor-critic methods with Hindsight Experience Replay** as the solution class. The approach combines three principles:

**Principle 1 (Off-Policy Learning).** Sample reuse is essential for efficiency. On-policy methods like PPO require fresh data for each gradient step; off-policy methods (SAC, TD3) can train on replay buffers containing trajectories from previous policies.

**Principle 2 (Maximum Entropy RL).** Exploration in continuous action spaces requires stochasticity. SAC optimizes $\pi^* = \arg\max_\pi \mathbb{E}_{\tau} \left[ \sum_{t} \gamma^t (R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t))) \right]$, where $\alpha$ trades off reward and entropy.

**Principle 3 (Goal Relabeling).** Failed trajectories contain information about achievable goals. HER relabels transitions $(s_t, a_t, s_{t+1}, g)$ with alternative goals $g'$ achieved during the episode, manufacturing dense learning signal from sparse feedback.

**Remark.** PPO is included for dense-reward baselines but cannot exploit HER (it requires off-policy learning). For sparse-reward tasks, SAC+HER or TD3+HER are required.

---

## §3. Repository Structure

```
robotics/
├── docker/              # Container definitions (Dockerfile, dev.sh, build.sh, run.sh)
├── scripts/             # Versioned experiment scripts (ch00_proof_of_life.py, ch01_env_anatomy.py)
├── tutorials/           # Textbook-style documentation (chNN_<topic>.md)
├── train.py             # Training CLI (PPO/SAC/TD3, HER, vectorized envs, TensorBoard)
├── eval.py              # Evaluation CLI (metrics: success rate, return, smoothness)
├── requirements.txt     # Python dependencies (pinned versions)
└── syllabus.md          # 10-week curriculum
```

**Generated Artifacts** (created as needed):
```
results/                 # Evaluation JSON reports
checkpoints/             # Model files (.zip + .meta.json)
videos/                  # Rendered episodes
runs/                    # TensorBoard logs
```

---

## §4. Prerequisites

**Hardware Requirements.**
- NVIDIA GPU with CUDA 12+ support (tested on DGX A100)
- Minimum 16GB GPU memory for parallel training (8 environments × batch size)

**Software Requirements.**
- Docker with NVIDIA Container Runtime
- Host OS: Linux (Ubuntu 22.04+ recommended)

**Domain Knowledge Prerequisites.**
- Reinforcement learning fundamentals (MDPs, policy gradients, Q-learning)
- Deep learning basics (neural networks, backpropagation, optimization)
- Python scientific computing (NumPy, PyTorch)
- Unix shell proficiency (Bash, docker commands)

**Remark.** This is not an introductory tutorial. Readers are expected to understand the difference between on-policy and off-policy algorithms, to recognize the temporal difference error $\delta_t = r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$, and to debug PyTorch tensor shape mismatches independently.

---

## §5. Quick Start

**Step 1: Verify Container Environment.**
```bash
bash docker/dev.sh
```
This creates `.venv`, installs dependencies, and drops you into an interactive shell. The environment is reproducible: same container, same packages, same results.

**Step 2: Run Proof-of-Life.**
```bash
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
```
This executes an end-to-end smoke test:
- GPU availability (CUDA device check)
- Gymnasium-Robotics Fetch environment instantiation
- MuJoCo rendering (saves `smoke_frame.png`)
- PPO training on FetchReachDense-v4 (1000 timesteps)

**Expected Output.** The script should complete without errors and produce `smoke_frame.png` (a rendered frame) and `ppo_smoke.zip` (a checkpoint).

**Step 3: Verify Reproducibility.**
```bash
bash docker/dev.sh python scripts/check_repro.py --algo ppo --env FetchReachDense-v4 --total-steps 50000
```
This trains the same configuration across multiple seeds and reports variance in final performance. High variance indicates training instability.

---

## §6. Training Examples

**Dense Reward (PPO Baseline).**
```bash
bash docker/dev.sh python train.py \
    --algo ppo \
    --env FetchReachDense-v4 \
    --seed 0 \
    --n-envs 8 \
    --total-steps 1000000
```

**Sparse Reward (SAC + HER).**
```bash
bash docker/dev.sh python train.py \
    --algo sac \
    --her \
    --env FetchReach-v4 \
    --seed 0 \
    --n-envs 8 \
    --total-steps 1000000
```

**Monitoring.** Training logs are written to `runs/` (TensorBoard format). View with:
```bash
bash docker/dev.sh tensorboard --logdir runs/
```

---

## §7. Evaluation Protocol

**Single Checkpoint Evaluation.**
```bash
bash docker/dev.sh python eval.py \
    --ckpt checkpoints/ppo_FetchReachDense-v4_s0.zip \
    --env FetchReachDense-v4 \
    --n-episodes 100 \
    --deterministic \
    --json-out results/eval_metrics.json
```

**Multi-Seed Evaluation.**
```bash
bash docker/dev.sh python eval.py \
    --ckpt checkpoints/ppo_FetchReachDense-v4_s0.zip \
    --env FetchReachDense-v4 \
    --n-episodes 10 \
    --seeds 0-9 \
    --deterministic \
    --json-out results/eval_multiseed.json
```

**Reported Metrics.**
- **Success Rate**: Fraction of episodes where $\|g_{\text{achieved}} - g_{\text{desired}}\| < \epsilon$
- **Mean Return**: Average cumulative discounted reward
- **Goal Distance**: Average final Euclidean distance to goal
- **Time to Success**: Mean timesteps until first success (for successful episodes)
- **Action Smoothness**: Variance of action deltas (lower = smoother)

**Remark.** A policy that achieves 95% success rate on a single seed but 60% on another seed has not solved the task reliably. Always report statistics over multiple seeds.

---

## §8. Gymnasium-Robotics Fetch Environments

This repository uses the Fetch robotic arm tasks from Gymnasium-Robotics:

| Environment | Task | Obs Dim | Goal Dim | Action Dim | Reward |
|-------------|------|---------|----------|------------|--------|
| `FetchReach-v4` | Move end-effector to target | 10 | 3 | 4 | Sparse |
| `FetchReachDense-v4` | Move end-effector to target | 10 | 3 | 4 | Dense |
| `FetchPush-v4` | Push block to target | 25 | 3 | 4 | Sparse |
| `FetchSlide-v4` | Slide puck to target | 25 | 3 | 4 | Sparse |
| `FetchPickAndPlace-v4` | Lift block to target | 25 | 3 | 4 | Sparse |

**Observation Structure.** Dict with keys:
- `observation`: Proprioceptive state (joint positions, velocities, gripper state)
- `achieved_goal`: Current 3D position of manipulated object (or end-effector for Reach)
- `desired_goal`: Target 3D position

**Action Space.** $\mathcal{A} = [-1, 1]^4$, interpreted as $(dx, dy, dz, \text{gripper})$ where the first three components are Cartesian position deltas and the fourth controls gripper closure.

---

## §9. Design Principles

**Docker-First.** Every command executes inside a container. No host-side package installation. No "it works on my machine" excuses. The `docker/dev.sh` script is the single entry point.

**Reproducibility.** All dependencies pinned in `requirements.txt`. All training runs seeded. All results reported with confidence intervals over multiple seeds.

**Quantification.** Claims require numbers. "The policy works" is inadmissible; "Success rate: 94.2% ± 2.1% (5 seeds, 100 episodes each)" is acceptable.

**Self-Contained Scripts.** Tutorials call scripts in `scripts/`; they do not contain inline Python or Bash snippets. Every experiment is a versioned, runnable artifact.

---

## §10. Common Failure Modes

**Problem: Training diverges after initial learning.**
**Diagnosis:** SAC entropy coefficient $\alpha$ too high. The policy remains stochastic when it should converge to deterministic goal-reaching.
**Solution:** Reduce `--sac-alpha` or enable automatic tuning.

**Problem: HER provides no benefit over standard replay.**
**Diagnosis:** Goal relabeling strategy mismatched to environment. `future` strategy requires episodes that make progress toward *some* region of goal space.
**Solution:** Verify goals are reachable. Check that `achieved_goal` updates during rollouts.

**Problem: Reproducibility check fails (high variance across seeds).**
**Diagnosis:** Neural network initialization, environment randomness, or optimizer state not seeded correctly.
**Solution:** Verify `torch.manual_seed(seed)` and `env.reset(seed=seed)` are called.

**Problem: Evaluation success rate << training success rate.**
**Diagnosis:** Overfitting to training goal distribution, or stochastic policy used at eval time.
**Solution:** Use `--deterministic` flag in `eval.py` and verify goal distribution matches training.

---

## §11. Curriculum: A Step-by-Step Study Plan

This repository is designed for systematic study, not casual browsing. The file [`syllabus.md`](syllabus.md) contains an executable 10-week curriculum with concrete commands, verification criteria, and "done when" checkpoints for each week.

**The Progression.**

| Week | Focus | Key Milestone |
|------|-------|---------------|
| 0 | DGX setup, containerization | Proof-of-life passes; reproducibility verified |
| 1 | Goal-conditioned environment anatomy | Understand dict observations, reward semantics |
| 2 | PPO on dense Reach | Stable baseline; evaluation protocol locked |
| 3 | SAC on dense Reach | Off-policy stack verified before adding HER |
| 4 | Sparse Reach + Push with HER | HER demonstrably outperforms no-HER baseline |
| 5 | PickAndPlace | Contacts/grasping; curriculum strategies |
| 6 | Action-interface engineering | Policies as controllers; smoothness metrics |
| 7 | Robustness | Noise injection, domain randomization, brittleness curves |
| 8 | Second suite (robosuite or Meta-World) | Avoid overfitting intuition to Fetch |
| 9 | Sweeps and ablations | Seed discipline; evidence-based conclusions |
| 10 | Capstone | Sparse PickAndPlace + robustness targets |

**The Structure of Each Week.**

Every week in `syllabus.md` follows the same format:
- **Goal:** What you will accomplish and why it matters
- **Steps:** Exact commands to run (copy-paste ready)
- **Done when:** Unambiguous criteria for completion

This structure serves two purposes. First, it prevents self-deception: you cannot convince yourself you understand something if you cannot produce the specified artifacts. Second, it builds skills incrementally: Week 4 assumes Week 3 is complete; skipping ahead creates gaps that surface later as mysterious failures.

**How to Use the Syllabus.**

Start at Week 0. Complete each step. Verify each "done when" criterion. Only then proceed to the next week. If something fails, diagnose it before moving on. The curriculum is designed to surface problems early, when they are easier to fix.

The tutorials in `tutorials/` provide theoretical context for each chapter. The syllabus provides the executable path. Use both together.

---

## §12. Contributing

**Standards.**
- All scripts must be executable via `bash docker/dev.sh python scripts/<name>.py`
- All new environments must include success rate benchmarks (5 seeds, 100 episodes)
- All documentation must follow WHY-HOW-WHAT structure
- All commits must pass the proof-of-life test (`scripts/ch00_proof_of_life.py all`)

**Pull Request Template.**
1. **Purpose:** What problem does this solve? (Problem formulation)
2. **Approach:** What method is implemented? (Derivation or reference)
3. **Verification:** What are the expected numerical outcomes?
4. **Artifacts:** Which new files are created? Where are they documented?

---

## §13. References

**Foundational Papers.**
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (Andrychowicz et al., 2017)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)

**Textbooks.**
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018)
- Bertsekas, *Reinforcement Learning and Optimal Control* (2019)

**Code Frameworks.**
- [Gymnasium-Robotics](https://robotics.farama.org/) (Fetch/Shadow Hand environments)
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) (PPO/SAC/TD3 implementations)

---

## §14. License

This repository is an educational research platform. Code is provided as-is for academic use. If you build upon this work, cite the foundational papers and acknowledge the pedagogical structure.

---

**A Final Remark.** This repository attempts to be more than a collection of scripts. It aspires to help readers understand goal-conditioned reinforcement learning from first principles—not just *how* to train a policy, but *why* policies learn and *when* they fail. Whether it succeeds is for the reader to judge. Corrections, criticisms, and improvements are welcome.
