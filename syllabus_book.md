# Robotics Reinforcement Learning in Action (Book Syllabus)

Working subtitle: Build reproducible goal-conditioned manipulation agents in MuJoCo (dense rewards, sparse rewards, pixels, and sim-to-real stress tests).

This book is intentionally practical: it teaches reinforcement learning (RL) for robots by making you produce artifacts you can trust (fixed-seed evaluation, JSON metrics, provenance metadata), not just screenshots of reward curves.

Core methods:
- Proximal Policy Optimization (PPO) for fast, dense-reward baselines.
- Soft Actor-Critic (SAC) for continuous control with replay buffers.
- Hindsight Experience Replay (HER) to make sparse, goal-conditioned tasks learnable.

Execution tracks (every chapter):
- Fast path: <= 500k environment steps (most readers can run).
- Full run: <= 3M environment steps (more stable results).
- Checkpoint track: evaluate provided pretrained checkpoints (for heavier chapters).

Core stack (kept stable to avoid "setup tax"):
- Gymnasium-Robotics Fetch tasks (goal-conditioned manipulation) in MuJoCo.
- Stable Baselines3 for PPO/SAC/TD3, plus HER replay buffers.
- Docker-first workflow with a single `train.py` / `eval.py` contract.

Repository contract (the "no vibes" rule):
- Training: `train.py` (SAC/PPO/TD3, optional HER for off-policy)
- Evaluation: `eval.py` (fixed-seed evaluation, JSON metrics, optional video)
- Artifacts:
  - Checkpoints: `checkpoints/*.zip` + `*.meta.json`
  - Metrics: `results/*.json`
  - Logs: `runs/` (TensorBoard)

All commands in this syllabus assume running inside the container wrapper:
```bash
bash docker/dev.sh <command...>
```

## Audience and Prerequisites

Target readers:
- ML engineers and software engineers who want to apply RL to robotics.
- Researchers who want a reproducible, engineering-first workflow.

Prerequisites:
- Comfortable with Python and the command line.
- Basic RL vocabulary (policy, value, replay buffer). We define terms as we use them.
- A GPU is recommended; CPU works for the early chapters but will be slow.

## Table of Contents (Draft, Manning-style)

Part 1 -- Start Running, Start Measuring
- Chapter 1: Proof of life -- Run a Fetch environment, render a frame, and train a policy long enough to see learning.
- Chapter 2: What the robot actually sees -- Inspect goal-conditioned observations, reward semantics, and success signals; write a metrics schema you will reuse.

Part 2 -- Baselines That Debug Your Pipeline
- Chapter 3: PPO as a lie detector -- Use dense rewards to validate your end-to-end training and evaluation pipeline.
- Chapter 4: Off-policy without mystery -- Train SAC on the same task and add replay diagnostics so later failures are debuggable.

Part 3 -- Sparse Goals, Real Progress
- Chapter 5: Learning from failure -- Show, with controlled experiments, why HER changes the outcome on sparse Reach and Push.
- Chapter 6: Capstone manipulation -- Train and evaluate sparse PickAndPlace with an honest report card and stress tests.

Part 4 -- Engineering-Grade Robotics RL
- Chapter 7: Policies as controllers -- Measure smoothness and stability; learn action scaling/filtering and controller-centric metrics.
- Chapter 8: Robustness curves -- Replace "it looks fine in videos" with degradation curves under noise and randomization.
- Chapter 9: Evidence-driven tuning -- Run minimal ablations and sweeps with seed discipline; write experiment cards you can reproduce.

Part 5 -- Pixels and the Reality Gap
- Chapter 10: Pixels, no cheating -- Learn Reach from rendered images only; build the wrapper and the convolutional encoder.
- Chapter 11: Visual robustness that matters -- Use augmentation to reduce brittleness and quantify the tradeoffs.
- Chapter 12 (optional advanced): Visual goals -- Specify goals as images ("make it look like this") and define a clean evaluation protocol.
- Chapter 13: A reality-gap playbook -- Domain randomization, sim-to-sim system identification, and a deployment-readiness checklist backed by tests.

Appendices / online bonus:
- Appendix A: Beyond Fetch -- adapt the same train/eval contract to one additional suite (robosuite or metaworld).
- Appendix B: Real robot (optional) -- calibration, safety, and a minimal interface that matches your simulator API.

## Implementation Map (Repo Alignment)

| Part | Book chapter | Repo script | Chapter title | Core deliverable | Status |
|------|---------|-------|------------------|---------|--------|
| 1 | 1 | `scripts/ch00_proof_of_life.py` | Proof of life | Fetch renders + PPO learns | Implemented |
| 1 | 2 | `scripts/ch01_env_anatomy.py` | What the robot actually sees | JSON schema + reward checks | Implemented |
| 2 | 3 | `scripts/ch02_ppo_dense_reach.py` | PPO as a lie detector | Stable baseline + eval protocol | Implemented |
| 2 | 4 | `scripts/ch03_sac_dense_reach.py` | Off-policy without mystery | Off-policy stack validated | Implemented |
| 3 | 5 | `scripts/ch04_her_sparse_reach_push.py` | Learning from failure (HER) | Clear HER vs no-HER separation | Implemented |
| 3 | 6 | `scripts/ch05_pick_and_place_capstone.py` | Capstone manipulation | Sparse PickAndPlace report card | Planned |
| 4 | 7 | `scripts/ch06_policies_as_controllers.py` | Policies as controllers | Smoothness/stability metrics | Planned |
| 4 | 8 | `scripts/ch07_robustness_curves.py` | Robustness curves | Degradation curves under noise | Planned |
| 4 | 9 | `scripts/ch08_ablations_and_sweeps.py` | Evidence-driven tuning | "What mattered?" via sweeps | Planned |
| 5 | 10 | `scripts/ch09_visual_reach.py` | Pixels, no cheating | Visual Reach (images only) | Planned |
| 5 | 11 | `scripts/ch10_visual_augmentation.py` | Visual robustness | Augmentation ablation report | Planned |
| 5 | 12 | `scripts/ch11_visual_goals.py` | Visual goals (optional) | Image-conditioned goals + eval | Planned |
| 5 | 13 | `scripts/ch12_reality_gap_playbook.py` | A reality-gap playbook | Domain rand + sim2sim sysid | Planned |

Appendices / online bonus (optional):
- A: "Beyond Fetch" adapter (robosuite or metaworld) using the same train/eval contract.
- B: Real robot deployment checklist + a minimal interface example.

---

## Part 1 -- Start Running, Start Measuring

### Chapter 1 -- Proof of Life (repo: `scripts/ch00_proof_of_life.py`)

You will validate the entire stack: MuJoCo loads, Fetch runs, headless rendering works, and a policy can learn enough to prove the loop is not broken.

Fast path (runs end-to-end):
```bash
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
```

Done when:
- `smoke_frame.png` exists and `ppo_smoke.zip` is created without crashes.

Artifacts:
- `smoke_frame.png`
- `ppo_smoke.zip`

### Chapter 2 -- What the Robot Actually Sees (repo: `scripts/ch01_env_anatomy.py`)

You will inspect goal-conditioned observations (what keys exist and what shapes they have), verify reward semantics, and lock a metrics schema you will use for the rest of the book.

Fast path:
```bash
bash docker/dev.sh python scripts/ch01_env_anatomy.py describe --env-id FetchReachDense-v4 --json-out results/ch01_env_describe.json
bash docker/dev.sh python scripts/ch01_env_anatomy.py reward-check --env-id FetchReachDense-v4 --n-steps 500
bash docker/dev.sh python scripts/ch01_env_anatomy.py random-episodes --env-id FetchReachDense-v4 --n-episodes 10 --json-out results/ch01_random_metrics.json
```

Done when:
- Reward semantics match `env.unwrapped.compute_reward(...)` (reward-check passes).

Artifacts:
- `results/ch01_env_describe.json`
- `results/ch01_random_metrics.json`

---

## Part 2 -- Baselines That Debug Your Pipeline

### Chapter 3 -- PPO as a Lie Detector (repo: `scripts/ch02_ppo_dense_reach.py`)

Dense rewards are your fastest debugging tool. You will build an end-to-end baseline with a fixed evaluation protocol and learn to spot common "nothing is learning" failure modes.

Fast path (<=500k):
```bash
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed 0 --total-steps 500000 --n-eval-episodes 50
```

Full run:
```bash
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed 0 --total-steps 1000000 --n-eval-episodes 100
```

Done when:
- You have an eval JSON with non-trivial success (> random rollouts) and stable training signals in TensorBoard.

Artifacts:
- `checkpoints/ppo_FetchReachDense-v4_seed0.zip`
- `checkpoints/ppo_FetchReachDense-v4_seed0.meta.json`
- `results/ch02_ppo_fetchreachdense-v4_seed0_eval.json`

### Chapter 4 -- Off-Policy Without Mystery (repo: `scripts/ch03_sac_dense_reach.py`)

You will train Soft Actor-Critic (SAC) on the same task and add replay-buffer diagnostics (entropy, rewards, Q-values) so later sparse-reward failures are debuggable.

Fast path (<=500k):
```bash
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all --seed 0 --total-steps 500000
```

Full run:
```bash
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all --seed 0 --total-steps 1000000
```

Done when:
- SAC trains stably and diagnostics are plausible (rewards/Q/entropy do not obviously diverge).

Artifacts:
- `checkpoints/sac_FetchReachDense-v4_seed0.zip`
- `checkpoints/sac_FetchReachDense-v4_seed0.meta.json`
- `results/ch03_sac_fetchreachdense-v4_seed0_eval.json`

---

## Part 3 -- Sparse Goals, Real Progress

### Chapter 5 -- Learning From Failure With HER (repo: `scripts/ch04_her_sparse_reach_push.py`)

You will run a controlled experiment that demonstrates why sparse rewards stall learning, and how Hindsight Experience Replay (HER) turns failures into training signal for goal-conditioned tasks.

Fast path (single seed, <=500k):
```bash
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchReach-v4 --seeds 0 --total-steps 500000
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4  --seeds 0 --ent-coef 0.1 --total-steps 500000
```

Full run (3-5 seeds, up to 2M):
```bash
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchReach-v4 --seeds 0-2 --total-steps 2000000
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4  --seeds 0-2 --ent-coef 0.1 --total-steps 2000000
```

Done when:
- The eval JSONs show a clear separation between HER and no-HER success rates.

Artifacts:
- `checkpoints/sac_FetchReach-v4_seed0.zip` and `checkpoints/sac_her_FetchReach-v4_seed0.zip` (plus meta JSONs)
- `results/ch04_sac_*` and `results/ch04_sac_her_*`

### Chapter 6 -- Capstone Manipulation: PickAndPlace With an Honest Report Card (Planned)

Goal:
- One "deliverable-grade" policy + an evaluation report that includes a stress test.

Fast path (<=500k, mostly validates harness on a harder task):
```bash
bash docker/dev.sh python train.py --algo sac --her --env FetchPickAndPlace-v4 --seed 0 --n-envs 8 --total-steps 500000 --out checkpoints/ch05_sac_her_pickplace_seed0
bash docker/dev.sh python eval.py  --ckpt checkpoints/ch05_sac_her_pickplace_seed0.zip --env FetchPickAndPlace-v4 --n-episodes 50 --seeds 0-49 --deterministic --json-out results/ch05_pickplace_fast_eval.json
```

Full run (<=3M):
```bash
bash docker/dev.sh python train.py --algo sac --her --env FetchPickAndPlace-v4 --seed 0 --n-envs 8 --total-steps 3000000 --out checkpoints/ch05_sac_her_pickplace_seed0_full
bash docker/dev.sh python eval.py  --ckpt checkpoints/ch05_sac_her_pickplace_seed0_full.zip --env FetchPickAndPlace-v4 --n-episodes 100 --seeds 0-99 --deterministic --json-out results/ch05_pickplace_full_eval.json
```

Checkpoint track:
- Provide a pretrained `checkpoints/ch05_*_full.zip` and run `eval.py` only.

Done when:
- Success rate improves meaningfully over training and holds up (degrades gracefully) under a defined stress split.

Planned implementation:
- `scripts/ch05_pick_and_place_capstone.py` with `train|eval|stress|all` subcommands, emitting a single report JSON.

---

## Part 4 -- Engineering-Grade Robotics RL

### Chapter 7 -- Policies as Controllers (Planned)

You will treat a learned policy like a controller: evaluate action smoothness, peak actions, and time-to-success; test action scaling and optional filtering; and learn to separate "planning" issues from "control" issues.

Focus:
- Action scaling and optional low-pass filtering.
- Controller-centric metrics: time-to-success, smoothness, peak action, path-length proxy.

Fast path:
- Run evaluation-only sweeps on an existing checkpoint (no retraining required).

Planned implementation:
- `scripts/ch06_policies_as_controllers.py` producing `results/ch06_controller_metrics.json`.

### Chapter 8 -- Robustness Curves (Planned)

You will quantify brittleness under controlled perturbations (noise and randomization) and produce degradation curves with confidence bands, replacing video-based evaluation with measurable robustness.

Focus:
- Controlled noise sweeps (obs noise, action noise).
- "Brittleness curves": success vs noise level with confidence bands across seeds.

Fast path:
- Evaluate a pretrained checkpoint under noise and plot/report JSON aggregates.

Planned implementation:
- `scripts/ch07_robustness_curves.py` producing `results/ch07_robustness.json`.

### Chapter 9 -- Evidence-Driven Tuning (Planned)

You will turn tuning into evidence: define a minimal ablation grid, enforce seed discipline, and generate "experiment cards" that make results reproducible and comparable.

Focus:
- Minimal ablation grid that answers "what mattered?" statistically.
- Seed discipline and standardized reporting.

Fast path:
- Small ablation on FetchReach/FetchPush at <=500k steps per config.

Planned implementation:
- `scripts/ch08_ablations_and_sweeps.py` + a YAML spec for grids (no new framework required).

---

## Part 5 -- Pixels and the Reality Gap

### Chapter 10 -- Pixels, No Cheating: Visual Reach (Planned)

You will remove privileged state observations and train from rendered images. The chapter focuses on making the pipeline work and measuring the sample-efficiency gap, not on chasing state-of-the-art visual RL.

Goal:
- Solve Reach using rendered images (no privileged coordinates).

Fast path:
- Train for <=500k steps to show learning trend; use checkpoint track for "high success."

Full run:
- Up to 3M steps for stable success, plus a fixed evaluation protocol.

Planned implementation:
- `scripts/ch09_visual_reach.py` (image wrapper + CNN encoder + train/eval).

### Chapter 11 -- Visual Robustness That Matters (Planned)

You will implement practical image augmentations (for example, random crop and color jitter), run an ablation, and measure robustness under visual perturbations.

Goal:
- Make visual policies robust to visual shifts without collecting more env experience.

Fast path:
- Evaluate augmentation vs no-augmentation on a small run; provide pretrained for the strong result.

Planned implementation:
- `scripts/ch10_visual_augmentation.py` (random crop, color jitter; ablation report JSON).

### Chapter 12 -- Visual Goals (Optional Advanced) (Planned)

You will explore goal specification as images ("make the scene look like this") and learn what changes in evaluation and debugging when the goal is high-dimensional.

Goal:
- "Make the scene look like this" image-goal conditioning; define a clean evaluation protocol.

Checkpoint track recommended:
- Image-goal pipelines are slower and higher variance; ship checkpoints + eval-only for readers.

Planned implementation:
- `scripts/ch11_visual_goals.py`.

### Chapter 13 -- A Reality-Gap Playbook (Planned)

You will produce a deployment-readiness report backed by stress tests: domain randomization, sim-to-sim system identification (treat one simulator config as "real"), and a checklist for what to verify before attempting hardware.

Goal:
- A pre-deployment checklist backed by stress tests, not demos.

Scope:
- Domain randomization (visual + dynamics).
- Sim2sim system identification (treat one simulator config as "real" and identify parameters).

Fast path:
- Evaluate robustness of an existing checkpoint under controlled perturbations.

Full run:
- Up to 3M steps training under randomization; checkpoint track for the full result.

Planned implementation:
- `scripts/ch12_reality_gap_playbook.py` producing `results/ch12_transfer_readiness.json`.

---

## Notes for a Manning Proposal

What makes this book different:
- It teaches robotics RL through measurement and reproducibility (JSON artifacts, fixed-seed eval), not screenshots of reward curves.
- It derives method choices from constraints (continuous control, sparse rewards, goal conditioning), then validates them empirically.
- It stays in one ecosystem long enough for the reader to build intuition, then extends to pixels and reality gap without swapping stacks every chapter.

Pretrained checkpoints:
- Used to keep the book runnable for readers while still presenting stable, meaningful results.
- Every checkpoint is paired with an evaluation protocol (`eval.py`) and a provenance file (`*.meta.json`) so results are auditable.
