# AGENTS.md

This file provides guidance to coding agents (Codex CLI, Claude Code, etc.) when working with code in this repository.

**Note:** This file is intentionally NOT tracked in git (see `.gitignore`). It is local session guidance. Do not attempt to commit changes to this file.

**Source of truth:** `CLAUDE.md` is the approved canonical guidance. Keep this file aligned; if there is a mismatch, follow `CLAUDE.md` and then update this file.

## Project Overview

A Docker-first robotics RL research platform for goal-conditioned manipulation using Gymnasium-Robotics Fetch tasks. Built around Stable Baselines 3 (PPO/SAC/TD3) with HER support for sparse rewards.

**Methodological focus:** SAC + HER for sparse-reward goal-conditioned tasks. PPO is included only for dense-reward baselines/sanity checks.

## Build & Development Commands

### Platform Support

The project supports both DGX/NVIDIA and Mac (Apple Silicon). The scripts auto-detect the platform and configure appropriately.

| Platform | Image | Device | Rendering | Detection |
|----------|-------|--------|-----------|-----------|
| DGX / NVIDIA Linux | `robotics-rl:latest` | CUDA | EGL | `uname -s` = Linux + nvidia-smi available |
| Mac (Apple Silicon) | `robotics-rl:mac` | CPU (default) | OSMesa | `uname -s` = Darwin |
| Mac with `--device mps` | `robotics-rl:mac` | MPS (opt-in) | OSMesa | `uname -s` = Darwin |
| Linux (no GPU) | `robotics-rl:latest` | CPU | OSMesa | `uname -s` = Linux, no nvidia-smi |

**Note on MPS:** Apple's Metal Performance Shaders can be enabled with `--device mps` on Mac. This is experimental; CPU is the safer default.

All commands run from the host, executed inside containers:

```bash
# Interactive dev shell (auto-creates .venv, installs deps)
bash docker/dev.sh

# End-to-end smoke test (Fetch + render + PPO)
bash docker/dev.sh python scripts/ch00_proof_of_life.py all

# Reproducibility sanity check
bash docker/dev.sh python scripts/check_repro.py --algo ppo --env FetchReachDense-v4 --total-steps 50000

# Environment snapshot
bash docker/dev.sh python scripts/print_versions.py --json-out results/versions.json

# Build Docker image (MuJoCo EGL/OSMesa deps)
bash docker/build.sh robotics-rl:latest

# Build and serve documentation locally
python -m pip install -r requirements-docs.txt
mkdocs serve  # Opens at http://127.0.0.1:8000

# Build PDFs (requires pandoc + LaTeX)
python scripts/build_pdfs.py
```

### Training & Evaluation

```bash
# Train PPO on dense Reach (baseline only)
bash docker/dev.sh python train.py --algo ppo --env FetchReachDense-v4 --seed 0 --n-envs 8 --total-steps 1000000

# Train SAC + HER on sparse Reach (primary method)
bash docker/dev.sh python train.py --algo sac --her --env FetchReach-v4 --seed 0 --n-envs 8 --total-steps 1000000

# Evaluate checkpoint
bash docker/dev.sh python eval.py --ckpt checkpoints/...zip --env FetchReachDense-v4 --n-episodes 100 --seeds 0-99 --deterministic --json-out results/metrics.json
```

### Non-interactive Docker execution (scripts/CI)

`docker/dev.sh` uses `docker run -it` and will fail without a TTY. Use this pattern instead:

```bash
docker run --rm \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e PYTHONUNBUFFERED=1 \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e TORCH_HOME=/tmp/.cache/torch \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torch_inductor \
  -e MPLCONFIGDIR=/tmp/.cache/matplotlib \
  -e USER=user \
  -e LOGNAME=user \
  -v "$PWD:/workspace" \
  -w /workspace \
  --gpus all \
  --ipc=host \
  robotics-rl:latest \
  bash -c 'source .venv/bin/activate && python your_script.py'
```

Key environment variables needed:
- `HOME=/tmp` - PyTorch/matplotlib need a writable home directory
- `XDG_CACHE_HOME`, `TORCH_HOME`, `TORCHINDUCTOR_CACHE_DIR`, `MPLCONFIGDIR` - Cache directories
- `USER=user`, `LOGNAME=user` - Prevents `getpwuid` errors when running as non-root UID

## Architecture

- `train.py`: Training CLI supporting PPO/SAC/TD3, HER, vectorized envs, TensorBoard. Outputs `.zip` checkpoint + `.meta.json`.
- `eval.py`: Evaluation CLI with metrics. Outputs JSON reports.
- `docker/`: Container tooling. `dev.sh` is the primary entry point (handles venv, deps, GPU, rendering backend).
- `scripts/`: Versioned experiment scripts (`chNN_<topic>.py`). Tutorials only call these -- no inline runnable code blocks.
- `tutorials/`: Textbook-style chapters (`chNN_<topic>.md`) following WHY-HOW-WHAT structure.
- `docs/`: MkDocs site. `docs/tutorials/` contains symlinks to `../../tutorials/` for rendering; `docs/tutorials/index.md` is the only non-symlink file.
- `docs/pdfs/`: Generated PDFs (do not edit by hand; build from markdown).
- `syllabus.md`: 10-week executable curriculum with "done when" criteria.

## Key Conventions

**Gymnasium-Robotics Fetch environments:**
- Observations are dicts: `observation`, `desired_goal`, `achieved_goal`
- Actions are 4D Cartesian deltas: `dx, dy, dz, gripper`
- HER requires off-policy algorithms (SAC/TD3, not PPO) because goal relabeling requires replay buffers

**Docker-first workflow:**
- Avoid host-side installs or system modifications; use containers
- `MUJOCO_GL=egl` for headless rendering (fallback: osmesa)
- `docker/dev.sh` preserves host UID/GID to avoid root-owned files

**Coding style:**
- Python: 4-space indentation, `argparse` CLIs, `pathlib.Path`, type hints where practical; keep scripts self-contained and reproducible
- Bash: `set -euo pipefail`
- Artifacts: checkpoints + metadata JSON; evaluation metrics as JSON artifacts
- Chapter naming: `scripts/chNN_<topic>.py`, `tutorials/chNN_<topic>.md`

**Generated directories (create as needed):**
- `results/` - Evaluation JSON reports
- `checkpoints/` - Model files (`.zip` + `.meta.json`)
- `videos/` - Rendered episodes
- `runs/` - TensorBoard logs

## CLI Contracts

`train.py`: `--algo {ppo,sac,td3}`, `--env`, `--seed`, `--n-envs`, `--total-steps`, `--her`, `--device`, `--track`

`eval.py`: `--ckpt`, `--env`, `--n-episodes`, `--seeds`, `--deterministic`, `--json-out`, `--video`

## Running Commands and Monitoring

### Primary execution pattern

All Python commands run through Docker via `docker/dev.sh`:

```bash
# Pattern: bash docker/dev.sh <command>
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all
bash docker/dev.sh python train.py --algo sac --env FetchReach-v4 --total-steps 1000000
```

### Chapter scripts pattern

Each chapter has a self-contained script in `scripts/chNN_<topic>.py`:

```bash
# Chapter scripts handle train + eval + comparison in one command
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed 0
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all --seed 0

# Subcommands for partial execution
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py train --total-steps 100000
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py eval --ckpt checkpoints/...zip
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py compare
```

### Monitoring with TensorBoard

```bash
# Start TensorBoard (accessible at http://localhost:6006)
bash docker/dev.sh tensorboard --logdir runs --bind_all
```

### Long-running jobs with tmux

```bash
tmux new -s rl
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all --seed 0
# Detach: Ctrl-b d
tmux attach -t rl
```

### Artifact locations

| Artifact | Location | Format |
|----------|----------|--------|
| Checkpoints | `checkpoints/<algo>_<env>_seed<N>.zip` | SB3 model |
| Metadata | `checkpoints/<algo>_<env>_seed<N>.meta.json` | JSON |
| Eval reports | `results/chNN_<topic>_eval.json` | JSON |
| TensorBoard logs | `runs/<algo>/<env>/seed<N>/` | Event files |
| Videos | `videos/` | MP4/GIF |

### GPU utilization note

Low GPU utilization (~5-10%) during RL training is expected. The bottleneck is CPU-bound MuJoCo simulation, not GPU-bound neural network operations.

## Documentation Guidelines

**Single source of truth:**
- `tutorials/` is the canonical location for tutorial content
- `docs/tutorials/` contains symlinks to `../../tutorials/` for MkDocs
- Always edit files in `tutorials/`, never in `docs/tutorials/`
- `docs/tutorials/index.md` is the only non-symlinked file (MkDocs navigation)

**Content rules:**
- Tutorials follow WHY-HOW-WHAT structure (problem formulation -> methodology -> implementation)
- Define terms before use; avoid unexplained acronyms
- Keep commits imperative (e.g., `feat: add ch02 training harness`)

**Scope Boundary Convention:**
When a topic is relevant but out of scope, mark the boundary explicitly:

> "[Concept X] involves [brief reason it goes deep]. For our purposes, the property we need is [specific property]. Readers who want the full treatment may enjoy [Reference, Section N]."

**Code in tutorials:**
- Avoid large runnable code blocks. Use snippet-includes from `scripts/` or `scripts/labs/`.
- Prefer snippet-includes pulled from repo files (single source of truth). Use `pymdownx.snippets` syntax: `--8<-- "scripts/labs/ppo_from_scratch.py:region_name"`.
- Limit any single excerpt to 30-40 lines; focus on update logic, not boilerplate.
- Use collapsible `<details>` blocks (via `pymdownx.details`) for longer walkthroughs.
- Keep runnable, end-to-end code in `scripts/chNN_*.py` as the canonical "Run It" reference.
- When showing illustrative code that is not snippet-included, mark it with `!!! lab "Build It (illustrative)"` admonition.
- Each tutorial must link to the exact reference script and the commands used to produce results.

**Lab modules (`scripts/labs/`):**
- Contain minimal, readable "from-scratch" implementations for teaching (not production use).
- Each file exports small, labeled regions for snippet-includes using markers: `# --8<-- [start:region_name]` and `# --8<-- [end:region_name]`.
- Pedagogical: fewer features, explicit tensors, clear variable naming. Not optimized for performance.
- Verification: each lab includes lightweight sanity checks (finite values, expected trends) runnable in under 2 minutes.

**"Run It" vs "Build It" structure:**
- Run It: SB3 pipeline in `scripts/chNN_*.py` (reproducible, tracked)
- Build It: from-scratch code in `scripts/labs/` (educational, not production)

**Incremental Build pattern (Build It sections):**
- **Foundation first:** Before showing loss functions, show the data structures and networks they operate on. Every Build It section starts with "what are we building on?" (network architecture, replay buffer, data types).
- **One concept per step:** Each subsection introduces exactly one function or component, with a verification checkpoint before moving to the next.
- **Verification checkpoints:** After each snippet-include, add an illustrative code block (marked `!!! lab "Checkpoint"`) showing how to instantiate the component with concrete test inputs, what tensor shapes to expect, and what numerical values indicate correctness (e.g., "Q-values near 0 at init", "clip_fraction = 0.0 when policy unchanged").
- **Math-before-code:** Restate the relevant equation immediately before each snippet-include so the reader can trace math to implementation.
- **Expected output blocks:** Use fenced code blocks with `Expected output:` header showing what `--verify` prints, with brief annotations explaining why each value is correct.
- **Wiring step:** After showing individual components, include the orchestration function (e.g., `ppo_update`, `sac_update`) that wires them together. This was previously omitted.
- **Consistent numbering:** All Build It sections use `N.5.X` subsection numbering (e.g., 2.5.1, 2.5.2, etc.).

**ASCII-only text (tutorials/docs):**
- Use `--` instead of em-dash (—) or en-dash (–)
- Use straight quotes `"` and `'` instead of curly quotes (" " ' ')
- Use `...` instead of ellipsis (…)
- Use `->` instead of arrow (→)
- Avoid Unicode symbols; prefer ASCII equivalents

**Math in Markdown (GitHub-compatible):**
- Inline math: `$...$` (e.g., `$\pi^*$` renders as π*)
- Block math: `$$...$$` on separate lines
- For complex equations, use fenced code blocks with `math` language

## Pedagogy and Curriculum Voice

- Follow the Bourbaki principle: define terms before use; avoid unexplained acronyms.
- Keep WHY-HOW-WHAT structure in tutorials and chapters.
- Favor reproducibility, quantification (numbers, not adjectives), and derivations over "just call the library".
- For the prerequisite boundary, concept registry, and canonical references, follow `CLAUDE.md`.

## Core Principles

- Simplicity first: minimal, root-cause fixes.
- Minimal impact: touch only what's necessary.
- Reproducibility always: seeds, Docker, pinned deps, JSON artifacts.
- Quantify everything: success rates, returns, distances -- not "seems to work".
- Prefer understanding over black boxes: explain what you use.

## Workflow Expectations (agent behavior)

- Use an explicit plan for non-trivial work (3+ steps or architectural decisions).
- Use subagents for codebase exploration or parallel analysis when helpful.
- Verify before done: run scripts inside Docker (or use the non-interactive pattern), and check metrics move in expected directions.
- Prefer simplicity-first, minimal-impact changes; fix root causes and quantify results.

## Testing Guidelines

No pytest suite yet. Treat scripts as smoke tests:
- Minimum: `bash docker/dev.sh python scripts/ch00_proof_of_life.py all`
- Repro sanity: `bash docker/dev.sh python scripts/check_repro.py ...`

## Commit & Pull Request Guidelines

- Use clear, imperative commits (e.g., `feat: add ch04 HER experiment`).
- PRs include: purpose, exact commands run (platform/Docker), expected artifacts/paths, and doc updates (`tutorials/`, `syllabus.md`) when adding new scripts.
