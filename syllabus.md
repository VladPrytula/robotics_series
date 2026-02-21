# Spark DGX Robotics RL Syllabus (Executable, 10 weeks)

> **From zero reward signal to 100% success -- in 28 minutes of training.**

![Robot reaching goals](videos/fetch_reach_demo_grid.gif)

*No one programmed these movements. The robot discovered them through trial and error -- mostly error, at first.*

---

## What You Will Build

By Week 10, you will have trained a robotic arm to **pick up objects and place them at arbitrary goal positions** -- using only sparse binary feedback ("did you succeed?"). Along the way, you will:

| Week | Milestone | What You Learn |
|------|-----------|----------------|
| 0-1 | Environment works | Docker, MuJoCo, goal-conditioned observations |
| 2-3 | PPO and SAC baselines | On-policy vs off-policy, replay buffers, entropy |
| 4-5 | **HER unlocks sparse rewards** | Why failures teach success |
| 6-7 | Policies as controllers | Action smoothness, robustness, noise injection |
| 8-9 | **Pixels, no cheating** | Visual SAC, DrQ, HER+pixels synthesis on Push |
| 10 | Reality gap | Domain randomization, deployment readiness |

Each week includes runnable commands and "done when" criteria -- concrete checkpoints that help verify progress. We find this structure useful because RL can be frustrating: code runs without errors but the agent learns nothing. Having a target success rate makes it clear when something is working.

## Our Approach

We tried to understand *why* certain algorithms work for this problem, not just *how* to use them. The reasoning goes something like:

- Continuous actions → actor-critic methods (value-based methods like DQN struggle here)
- Sparse rewards → off-policy learning with replay buffers (on-policy methods discard too much data)
- Goal-conditioned sparse → HER (failed attempts contain useful information about *other* goals)
- Large goal space → entropy bonus (encourages exploration without manual schedules)

This leads to **SAC + HER** as the core method. We're not claiming it's the only way -- but we wanted to show how problem structure can guide algorithm choice, rather than just picking something that "usually works."

## Who This Is For

You should already understand MDPs, policy gradients, and basic PyTorch. You should have access to a GPU (DGX preferred, but any CUDA-capable machine works). You should be comfortable with Docker and command-line workflows.

**Time investment:** 10 weeks, 5-10 hours per week. The curriculum is dense. Rushing defeats the purpose.

---

## Quick Links

**Tutorials** (theory + context):
- [Chapter 0: Proof of Life](tutorials/ch00_containerized_dgx_proof_of_life.md)
- [Chapter 1: Environment Anatomy](tutorials/ch01_fetch_env_anatomy.md)
- [Chapter 2: PPO on Dense Reach](tutorials/ch02_ppo_dense_reach.md)
- [Chapter 3: SAC on Dense Reach](tutorials/ch03_sac_dense_reach.md)
- [Chapter 4: HER on Sparse Reach/Push](tutorials/ch04_her_sparse_reach_push.md)
- [Chapter 5: PickAndPlace](tutorials/ch05_pick_and_place.md)

**Scripts** (executable code):
- [`scripts/ch00_proof_of_life.py`](scripts/ch00_proof_of_life.py) -- verify your setup
- [`scripts/ch01_env_anatomy.py`](scripts/ch01_env_anatomy.py) -- inspect environments
- [`scripts/ch02_ppo_dense_reach.py`](scripts/ch02_ppo_dense_reach.py) -- PPO baseline
- [`scripts/ch03_sac_dense_reach.py`](scripts/ch03_sac_dense_reach.py) -- SAC with diagnostics
- [`scripts/ch04_her_sparse_reach_push.py`](scripts/ch04_her_sparse_reach_push.py) -- HER experiments
- [`scripts/ch05_pick_and_place.py`](scripts/ch05_pick_and_place.py) -- PickAndPlace with stratified eval

---

## Core references (keep open)
- Sutton & Barto (concepts)
- Spinning Up PPO/SAC (debugging + failure modes)
- Modern Robotics (kinematics intuition + baselines)

## Tooling stack (standardize early)
- Gymnasium-Robotics Fetch suite (goal-conditioned dict observations + success metrics) ([Fetch docs][fetch-docs])
- MuJoCo (maintained `mujoco` Python bindings; no `mujoco-py`) ([MuJoCo][mujoco])
- Stable-Baselines3 (PPO/SAC/TD3 + `HerReplayBuffer` for sparse goals; HER is off-policy only) ([SB3 HER][sb3-her])
- Tracking: W&B or TensorBoard

## Non-negotiables (Fetch "shape")
- **Observations** are dicts: `observation`, `desired_goal`, `achieved_goal`. This is what makes HER/relabeling natural. ([Fetch docs][fetch-docs])
- **Actions** are **4D**: small Cartesian deltas `dx, dy, dz` + gripper open/close. It is not torque control. ([Fetch docs][fetch-docs])

## Why SAC + HER? (Deriving the method from the problem)

The algorithm choice is not arbitrary. It follows from the problem constraints:

**1. Continuous actions → Actor-Critic.**
The action space is $\mathbb{R}^4$. Pure value-based methods (DQN) require $\arg\max_a Q(s,a)$, which is intractable for continuous $a$ without expensive discretization. We need a policy network that outputs actions directly, plus a critic to reduce variance. This means actor-critic.

**2. Sparse rewards → Off-Policy.**
In Gymnasium-Robotics, sparse step rewards are 0 when the success threshold is met and −1 otherwise (equivalently, a binary success indicator 1/0 up to a constant shift). Either way, informative signal is rare. On-policy methods (PPO) discard data after each update—wasteful when positive signal is scarce. Off-policy methods (SAC, TD3) store transitions in a replay buffer and reuse them. This is essential for sample efficiency with sparse rewards.

**3. Goal-conditioned + sparse → HER.**
A trajectory that fails to reach goal $g$ still demonstrates how to reach whatever state $s_T$ it ended in. Hindsight Experience Replay relabels transitions: replace $g$ with the achieved goal $g' = g_{\text{achieved}}(s_T)$, recompute rewards, and store both versions. Failed attempts become successful demonstrations for different goals. This manufactures dense signal from sparse feedback.

**4. Large goal space → Maximum Entropy (SAC).**
HER needs diverse experience to relabel. SAC's entropy bonus encourages exploration without explicit schedules:
$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_t \gamma^t (R_t + \alpha \mathcal{H}(\pi(\cdot|s_t)))\right]$$
The temperature $\alpha$ (often auto-tuned) balances exploration and exploitation.

**Summary:**
| Constraint | Requirement | Solution |
|------------|-------------|----------|
| Continuous actions | Direct policy output | Actor-critic |
| Sparse rewards | Sample reuse | Off-policy (replay buffer) |
| Goal-conditioned sparse | Learn from failures | HER |
| Large goal space | Exploration | Entropy bonus (SAC) |

**The method is SAC + HER.** This is derived, not chosen from a menu.

**Where does PPO fit?** PPO is on-policy and cannot use HER. It works for dense rewards (where every step provides signal) and serves as a sanity-check baseline in Week 2. But for the core sparse-reward tasks, SAC + HER is required.

## Reproducibility target (realistic on DGX)
You are not aiming for bitwise-identical curves. You are aiming for:
1) same code + same seed => evaluation metrics within a tolerance on the same machine class; and
2) across 3–5 seeds => mean±std is stable enough for ablations.

## Spark DGX checklist (do this once)
- Prefer running everything in a GPU-enabled container (`docker run --gpus all`) so you don’t install/pin Python packages on the DGX host.
- If the host does not provide `python`, do not fight it: run all Python via `bash docker/dev.sh python ...`.
- Run long jobs in `tmux`/`screen` or your scheduler (avoid “SSH session died” losses).
- Prefer a fast filesystem for checkpoints/replay/results if available.
- Headless rendering: try `MUJOCO_GL=egl` first; fall back to `osmesa`/`xvfb-run` if EGL isn’t available.

### Keep runs alive (tmux)
```bash
tmux new -s rl
# detach: Ctrl-b d
# reattach:
tmux attach -t rl
```

---

## Quickstart (30–60 min): DGX proof-of-life
Goal: “Fetch env loads, can render headless, PPO can train for 50k steps”.

### 0) Preflight (host)
```bash
cd /home/vladp/src/robotics
hostname
nvidia-smi | head
docker --version

# Minimal GPU-in-container check (pick any CUDA image you have access to):
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.12-py3 nvidia-smi
# Another local option (CUDA base image):
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
# Docker Hub example (substitute a tag you have/pulled):
docker run --rm --gpus all nvidia/cuda:<tag> nvidia-smi
```

### 1) One-command proof-of-life (recommended)
This runs everything from the host (no interactive shell required):
```bash
cd /home/vladp/src/robotics
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
```

**Done when**: `smoke_frame.png` exists and `ppo_smoke.zip` is created without crashes.

### 2) (Optional) Start an interactive Docker dev shell
Recommended (Python + CUDA already included):
```bash
cd /home/vladp/src/robotics
docker run --rm -it --gpus all --ipc=host \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/pytorch:25.12-py3 bash
```
If this creates root-owned files in your repo, use the wrapper below (runs as your user).

Convenience wrapper (same thing, but shorter to type):
```bash
cd /home/vladp/src/robotics
bash docker/run.sh
```

Auto-setup wrapper (recommended): creates `./.venv`, installs `requirements.txt`, and activates the venv.
```bash
cd /home/vladp/src/robotics
bash docker/dev.sh
```
Edit `requirements.txt` to add/remove Python deps; `docker/dev.sh` will reinstall when it changes.
If `robotics-rl:latest` is not present, `docker/dev.sh` builds it once (MuJoCo EGL/OSMesa system deps).

Optional: build a pinned “project image” (useful once things work and you want reproducibility):
```bash
cd /home/vladp/src/robotics
bash docker/build.sh robotics-rl:latest
IMAGE=robotics-rl:latest bash docker/run.sh
```

### 3) Fallback: host venv (only if Docker isn’t possible)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
```

Install packages (pin exact versions after this works)
```bash
python -m pip install "gymnasium>=0.29" "gymnasium-robotics>=1.2" "mujoco>=3" "stable-baselines3>=2.2" tensorboard imageio
```
Optional:
```bash
python -m pip install wandb
```

---

## Project contract (set up in Week 0; don’t churn it later)

### Folder skeleton (minimum)
Create these once and keep the rest of your work inside this structure:
```bash
mkdir -p configs envs wrappers callbacks scripts analysis results checkpoints videos
```

### Stable CLI contract
Your future self depends on this being stable.

`train.py`
- `--algo {ppo,sac,td3}`
- `--env {FetchReachDense-v4, FetchReach-v4, FetchPush-v4, FetchPickAndPlace-v4, ...}`
- `--seed`
- `--device {auto,cpu,cuda}`
- `--n-envs`
- `--total-steps`
- `--track {tb,none}`

`eval.py`
- `--ckpt path`
- `--env ...`
- `--n-episodes`
- `--seeds list_or_range` (or a base `--seed`)
- `--deterministic` (flag)
- `--json-out results/metrics.json`
- `--video` (flag)

### Metadata you always log
Commit hash, Python/pip lock, `mujoco`/`gymnasium-robotics`/SB3 versions, env id, seed(s), device, `n_envs`, wallclock, steps/sec, and evaluation protocol.
This repo’s default mechanism:
- `train.py` writes `<checkpoint>.meta.json` next to the `.zip`.
- `eval.py` writes a JSON report that includes versions + per-episode metrics.
- `scripts/print_versions.py` writes an environment snapshot for the machine/container.

### Metrics you track from Week 1 onward (don’t change later)
Per-episode:
- success (env-provided if present, plus your own goal-distance threshold)
- final goal distance
- return
- episode length
- time-to-success (first timestep that meets success)
- action smoothness proxy `Σ ||a_t - a_{t-1}||^2`
- max `|a|`

Aggregate:
- success-rate mean±CI (binomial CI is fine)
- return mean±std
- robustness curves (success vs noise/randomization)

---

## Week-by-week plan
Each week has: (1) goal, (2) steps you can run, (3) “done when”.

### Week 0 — DGX setup + workflow you can trust
Goal: end-to-end train/eval works; headless rendering is solved; you can reproduce an eval JSON within tolerance.

Steps
- [ ] Snapshot versions (inside container):
  - `bash docker/dev.sh python scripts/print_versions.py --json-out results/versions.json`
- [ ] Validate end-to-end (GPU + Fetch + render + PPO smoke):
  - `bash docker/dev.sh python scripts/ch00_proof_of_life.py all`
- [ ] Repro sanity (train twice, eval both, compare within tolerance):
  - `bash docker/dev.sh python scripts/check_repro.py --algo ppo --env FetchReachDense-v4 --total-steps 50000 --n-eval-episodes 50`
- [ ] Decide how you’ll run on Spark DGX (one of):
  - `tmux`-based workflow (simple, interactive)
  - scheduler job script (better for long runs; adapt to your cluster)

Done when
- Same commit runs on Spark DGX reliably.
- Save/load works and `eval.py` emits a stable JSON schema.

### Week 1 — Goal-conditioned RL literacy (dict obs, reward, success)
Goal: you understand exactly what SB3 “sees” and how the environment computes reward/success.

Steps
- [ ] Describe obs/action schema (JSON artifact):
  - `bash docker/dev.sh python scripts/ch01_env_anatomy.py describe --json-out results/ch01_env_describe.json`
- [ ] Reward semantics consistency check (`compute_reward` vs env reward):
  - `bash docker/dev.sh python scripts/ch01_env_anatomy.py reward-check --n-steps 500`
- [ ] Random-rollout metrics (baseline sanity + metric schema preview):
  - `bash docker/dev.sh python scripts/ch01_env_anatomy.py random-episodes --n-episodes 10 --json-out results/ch01_random_metrics.json`
- [ ] Decide normalization:
  - start with “none” until Week 2 baseline is stable; add normalization once you can detect regressions via `eval.py`.

Done when
- You can plot success-rate vs steps and distributions of final distance.
- Reward recomputation matches environment reward semantics.

### Week 2 — PPO on dense Reach (pipeline “truth serum”)
Goal: a stable baseline that proves your training loop is correct.

Steps
- [ ] Train PPO on a dense Reach env:
  - `bash docker/dev.sh python train.py --algo ppo --env FetchReachDense-v4 --seed 0 --n-envs 8 --total-steps 1000000 --out checkpoints/ppo_reachdense_v4_seed0`
- [ ] Lock an evaluation protocol:
  - separate eval env
  - fixed eval seeds
  - deterministic policy at eval
  - N=100 episodes (or N=50 for quick iteration)
  - `bash docker/dev.sh python eval.py --ckpt checkpoints/ppo_reachdense_v4_seed0.zip --env FetchReachDense-v4 --n-episodes 100 --seeds 0-99 --deterministic --json-out results/ppo_reachdense_v4_seed0_eval.json`
- [ ] Add an “early warning” dashboard:
  - value loss explosion, entropy collapse, KL drift, success flatline

Done when
- PPO reaches consistently high success on dense Reach.
- You can explain (practically) what PPO clipping is protecting you from.

### Week 3 — SAC on dense Reach + replay diagnostics
Goal: your off-policy stack is correct before you add HER.

Steps
- [ ] Train SAC on dense Reach using dict obs (`MultiInputPolicy`). ([SB3 examples][sb3-examples])
  - `bash docker/dev.sh python train.py --algo sac --env FetchReachDense-v4 --seed 0 --n-envs 8 --total-steps 1000000 --out checkpoints/sac_reachdense_v4_seed0`
  - `bash docker/dev.sh python eval.py --ckpt checkpoints/sac_reachdense_v4_seed0.zip --env FetchReachDense-v4 --n-episodes 100 --seeds 0-99 --deterministic --json-out results/sac_reachdense_v4_seed0_eval.json`
- [ ] Add replay diagnostics:
  - reward histogram
  - achieved vs desired goal distance distribution
  - Q-values + entropy coefficient trends
- [ ] Throughput scaling on DGX:
  - sweep `n_envs` upward and log steps/sec

Done when
- SAC trains stably (no divergence; returns/success improve).
- You can explain SAC’s entropy objective (maximum-entropy RL) at a practical level. ([SB3 SAC][sb3-sac])

### Week 4 — Sparse Reach + Push: introduce HER where it matters
Goal: demonstrate that HER is the difference-maker on sparse goals.

Steps
- [ ] Train SAC on sparse Reach (`FetchReach-*`) **without** HER (establish baseline difficulty).
- [ ] Train SAC + `HerReplayBuffer` on sparse Reach. ([SB3 HER][sb3-her])
  - no-HER:
    - `bash docker/dev.sh python train.py --algo sac --env FetchReach-v4 --seed 0 --n-envs 8 --total-steps 1000000 --out checkpoints/sac_reach_v4_seed0`
  - HER:
    - `bash docker/dev.sh python train.py --algo sac --her --her-goal-selection-strategy future --her-n-sampled-goal 4 --env FetchReach-v4 --seed 0 --n-envs 8 --total-steps 1000000 --out checkpoints/sac_her_reach_v4_seed0`
- [ ] Repeat on sparse Push (`FetchPush-*`):
  - no-HER baseline
  - HER with `goal_selection_strategy="future"` and `n_sampled_goal ∈ {2,4,8}`

Done when
- Clear separation in success-rate between HER and no-HER.
- You can report mean±CI across 3–5 seeds.

### Week 5 — PickAndPlace: dense debug → sparse+HER → curriculum
Goal: validate the pipeline on contacts/grasping without fooling yourself via accidental shaping.

Steps
- [ ] Dense validation first (if a dense PickAndPlace variant exists in your install).
  - If not: use temporary shaping only for debugging, and document it clearly.
- [ ] Sparse + HER:
  - start from stable Push hyperparams
  - increase training horizon (PickAndPlace takes longer)
- [ ] Curriculum (explicit schedule):
  - narrow goal range / easier initial object positions first
  - gradually expand to the full distribution
- [ ] Eval split:
  - “train-like” distribution
  - “stress” distribution (wider initial placements)

Done when
- Dense mode: high success + smooth behavior (no frantic oscillations).
- Sparse+HER: meaningful upward trend; stress test is worse but improving.

### Week 6 — Action-interface engineering (“policies as controllers”)
Goal: treat learned policies like controllers and quantify stability.

Steps
- [ ] Action scaling study: multiply actions by `{0.5, 1.0, 2.0}`; measure success + smoothness + time-to-success.
- [ ] Add an action wrapper:
  - clipping (already bounded) + optional low-pass filter
  - compare oscillations/contact stability
- [ ] Classical baseline (Reach + maybe Push):
  - simple “go-to-goal” PD-like rule in Cartesian delta space
- [ ] Define a controller-centric metric bundle:
  - success, time-to-success, smoothness, peak `|a|`, path-length proxy

Done when
- You can compare RL vs baseline on controller metrics (not just return).
- You can separate “planning” issues from “control” issues with evidence.

### Week 7 — Robustness: noise injection, brittleness curves, noise-augmented training
Goal: quantify brittleness, train more robust policies, verify the improvement.

Steps
- [ ] Controlled noise injection sweeps (obs + action noise)
- [ ] Degradation curves with confidence bands across seeds
- [ ] P-controller baseline: same sweeps for comparison
- [ ] Noise-augmented retraining: train SAC+HER under obs noise
- [ ] Compare clean vs robust degradation curves

Done when
- Degradation curves show clear brittleness patterns with CIs.
- Noise-augmented policy shows improved critical sigma and robustness AUC.
- Clean-vs-robust tradeoff is quantified with numbers.

### Week 8-9 — Pixels, no cheating: from Reach to Push (Ch9)
Goal: train visual SAC from raw pixels, measure the pixel penalty, then overcome it on Push with HER + DrQ.

Steps
- [ ] Train state SAC on FetchReachDense (baseline ceiling):
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py train-state --seed 0`
- [ ] Train pixel SAC on FetchReachDense (no augmentation):
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py train-pixel --seed 0`
- [ ] Train pixel+DrQ SAC on FetchReachDense:
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py train-pixel-drq --seed 0`
- [ ] Three-way Reach comparison (compute sample-efficiency ratio rho):
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py compare`
- [ ] Diagnose why Push from state (no HER) fails (~2% success):
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py train-push-state --seed 0`
- [ ] Train Push from state + HER (baseline: ~95%+):
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py train-push-her --seed 0`
- [ ] Train Push from pixels + HER + DrQ (the synthesis):
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py train-pixel-her-drq --seed 0 --env FetchPush-v4`
- [ ] Full Push comparison table:
  - `bash docker/dev.sh python scripts/ch10_visual_reach.py push-compare`

Done when
- Reach comparison shows clear pixel penalty (rho > 4x) with DrQ closing part of the gap.
- Push pixel+HER+DrQ achieves >80% success (target: >90%).
- You can explain why goal_mode="both" is not cheating (policy sees pixels, HER sees vectors).

### Week 10 — The reality gap: stress tests before hardware (Ch10)
Goal: domain randomization, visual robustness ablations, and a deployment-readiness checklist.

Steps
- [ ] Domain randomization sweep: what to randomize and how to measure it
- [ ] Visual robustness ablations: DrQ vs crop vs jitter (using Push from pixels model)
- [ ] Sim-to-sim system identification: controlled rehearsal with perturbed dynamics
- [ ] Write deployment-readiness checklist backed by quantitative tests

Deliverables
- Trained pixel Push model with robustness evaluation
- Augmentation ablation table with success rates across perturbation types
- Deployment-readiness checklist as a reusable protocol

Done when
- You can answer "will this transfer to a real robot?" with quantitative evidence.
- Augmentation ablation shows which augmentation strategy matters for each perturbation type.

---

[fetch-docs]: https://robotics.farama.org/envs/fetch/ "Fetch - Gymnasium-Robotics"
[mujoco]: https://mujoco.readthedocs.io/ "MuJoCo documentation"
[sb3-her]: https://stable-baselines3.readthedocs.io/en/master/modules/her.html "HER — Stable Baselines3"
[sb3-examples]: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html "Examples — Stable Baselines3"
[sb3-sac]: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html "SAC — Stable Baselines3"
