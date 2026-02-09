# Spark DGX Robotics RL Syllabus (Executable, 10 weeks)

This is a step-by-step plan you can execute on **Spark DGX** to learn goal-conditioned manipulation RL and build a reproducible training/evaluation workflow around Gymnasium-Robotics Fetch tasks.

If you prefer chapter-style “textbook” notes, start with:
- `tutorials/ch00_containerized_dgx_proof_of_life.md`
- `tutorials/ch01_fetch_env_anatomy.md`

## Core references (keep open)
- Sutton & Barto (concepts)
- Spinning Up PPO/SAC (debugging + failure modes)
- Modern Robotics (kinematics intuition + baselines)

## Tooling stack (standardize early)
- Gymnasium-Robotics Fetch suite (goal-conditioned dict observations + success metrics) ([Fetch docs][fetch-docs])
- MuJoCo (maintained `mujoco` Python bindings; no `mujoco-py`) ([MuJoCo][mujoco])
- Stable-Baselines3 (PPO/SAC/TD3 + `HerReplayBuffer` for sparse goals; HER is off-policy only) ([SB3 HER][sb3-her])
- Tracking: W&B or TensorBoard

## Non-negotiables (Fetch “shape”)
- **Observations** are dicts: `observation`, `desired_goal`, `achieved_goal`. This is what makes HER/relabeling natural. ([Fetch docs][fetch-docs])
- **Actions** are **4D**: small Cartesian deltas `dx, dy, dz` + gripper open/close. It is not torque control. ([Fetch docs][fetch-docs])

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

### Week 7 — Robustness: noise, randomization, brittleness curves
Goal: quantify brittleness and improvements (don’t rely on videos).

Steps
- [ ] Controlled noise injection sweeps:
  - observation noise `σ`
  - action execution noise `σ`
- [ ] Domain randomization (start light):
  - if physical params are hard to reach, begin with obs/action noise
- [ ] Plot degradation curves with confidence bands across seeds.

Done when
- At least one robustness technique measurably helps.
- You can quantify “how brittle” with plots.

### Week 8 — Second suite adapter (robosuite OR Meta-World)
Goal: avoid overfitting intuition to Fetch.

Steps (pick one suite)
- [ ] Implement an adapter to the same train/eval contract.
- [ ] Solve 1–2 simple tasks.
- [ ] If using robosuite, compare controller modes (this is where “torque vs OSC vs position” becomes real).

Done when
- You can train/eval in a second ecosystem with minimal harness changes.

### Week 9 — Engineering-grade RL: sweeps, ablations, seed discipline
Goal: turn tinkering into evidence.

Steps
- [ ] Seed discipline: 5 seeds per config; report mean±std + CI for success.
- [ ] Minimum ablation grid:
  - normalization method A vs B
  - HER `n_sampled_goal ∈ {2,4,8}`
  - SAC entropy: auto vs fixed target entropy
- [ ] Add a launcher:
  - YAML grid spec
  - Spark DGX job launcher script (scheduler or local parallel runner)
  - auto-collect results into a single report

Done when
- You can answer “what mattered?” with plots + statistics.
- A run is reproducible from scratch given commit+lock+config.

### Week 10 — Capstone: sparse PickAndPlace + robustness targets
Goal: one deliverable-grade model + eval protocol + write-up.

Capstone spec
- Environment: sparse PickAndPlace (`FetchPickAndPlace-*`) + HER
- Training: Spark DGX, vectorized envs, periodic checkpoints
- Evaluation:
  - standard test distribution
  - stress test (wider init positions + mild noise/randomization)
  - report success-rate, time-to-success, smoothness metrics

Deliverables
- model checkpoint(s) + normalization stats
- `eval.py` outputs one JSON with metrics + provenance (commit hash, versions, seeds, episode count)
- short experiment card:
  - observation handling + normalization
  - action filtering/scaling
  - hyperparams that mattered (from ablations)
  - what failed and how you debugged it

Stretch (only after capstone is stable)
- Vision (RGB) will spike sample complexity
- Demonstrations + BC warm-start can help manipulation
- CS285 robotics-adjacent assignments as structured extensions

---

[fetch-docs]: https://robotics.farama.org/envs/fetch/ "Fetch - Gymnasium-Robotics"
[mujoco]: https://mujoco.readthedocs.io/ "MuJoCo documentation"
[sb3-her]: https://stable-baselines3.readthedocs.io/en/master/modules/her.html "HER — Stable Baselines3"
[sb3-examples]: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html "Examples — Stable Baselines3"
[sb3-sac]: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html "SAC — Stable Baselines3"
