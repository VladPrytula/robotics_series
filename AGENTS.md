# AGENTS.md

This file provides guidance to coding agents (Codex CLI, Claude Code, etc.) when working with code in this repository.

**Note:** This file is intentionally NOT tracked in git (see `.gitignore`). It is local session guidance. Do not attempt to commit changes to this file.

**Source of truth:**
- Tutorials/curriculum + code conventions: `CLAUDE.md`
- Manning book (chapters in `Manning/`): `Manning/CLAUDE.md` and `manning_proposal/`

Keep this file aligned. If there is a mismatch, follow the relevant canonical file(s) above and then update this file.

## Products and Boundaries

This repository is used for both the open-source tutorial series and a full Manning book draft.

**Tutorials (MkDocs site):**
- Canonical tutorial content lives in `tutorials/` (edit these files)
- `docs/tutorials/` contains symlinks to `../../tutorials/` for MkDocs rendering (do not edit symlinks)
- Tutorials may use MkDocs features (admonitions, snippet-includes)

**Manning book (pandoc build):**
- Canonical book content lives in `Manning/` (edit these files)
- Do not cross-contaminate: book chapters have different voice and markdown constraints than tutorials
- Follow `manning_proposal/chapter_production_protocol.md` and the role prompts in `manning_proposal/agents/`
- Book voice follows `manning_proposal/manning_writer_persona.md` (it overrides tutorial voice)

## Project Overview

A Docker-first robotics RL research platform for goal-conditioned manipulation using Gymnasium-Robotics Fetch tasks. Built around Stable Baselines 3 (PPO/SAC/TD3) with HER support for sparse rewards.

**Methodological focus:** SAC + HER for sparse-reward goal-conditioned tasks. PPO is included only for dense-reward baselines/sanity checks.

## Build & Development Commands

### Platform Support

The project supports both DGX/NVIDIA and Mac M4 (Apple Silicon). The scripts auto-detect the platform and configure appropriately.

| Platform | Image | Device | Rendering | Detection |
|----------|-------|--------|-----------|-----------|
| DGX / NVIDIA Linux | `robotics-rl:latest` | CUDA | EGL | `uname -s` = Linux + nvidia-smi available |
| Mac M4 (Apple Silicon) | `robotics-rl:mac` | CPU (default) | OSMesa | `uname -s` = Darwin |
| Mac M4 with `--device mps` | `robotics-rl:mac` | MPS (opt-in) | OSMesa | `uname -s` = Darwin |
| Linux (no GPU) | `robotics-rl:latest` | CPU | OSMesa | `uname -s` = Linux, no nvidia-smi |

**Note on MPS:** Apple's Metal Performance Shaders can be enabled with `--device mps` on Mac. This is experimental and may have edge cases with certain operations. CPU is the safer default.

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
pip install -r requirements-docs.txt
mkdocs serve  # Opens at http://127.0.0.1:8000

# GitHub CLI setup (for PR reviews, issues)
gh auth login              # Interactive browser auth
# Or with token: echo "$GH_TOKEN" | gh auth login --with-token

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
- `eval.py`: Evaluation CLI with metrics (success rate, return, goal distance, time-to-success, action smoothness). Outputs JSON reports.
- `docker/`: Container tooling. `dev.sh` is the primary entry point (handles venv, deps, GPU, rendering backend).
- `scripts/`: Versioned experiment scripts (`chNN_<topic>.py`). Tutorials only call these -- no inline runnable code blocks.
- `tutorials/`: Textbook-style chapters (`chNN_<topic>.md`) following WHY-HOW-WHAT structure.
- `docs/`: MkDocs site. `docs/tutorials/` contains symlinks to `../../tutorials/` for rendering; `docs/tutorials/index.md` is the only non-symlink file.
- `docs/pdfs/`: Generated PDFs (do not edit by hand; build from markdown).
- `Manning/`: Manning book draft (pandoc build via `scripts/build_book.py`). Separate product from tutorials; follow `Manning/CLAUDE.md`.
- `manning_proposal/`: Book proposal + chapter production protocol + role prompt files (`manning_proposal/agents/`).
- `syllabus.md`: 10-week executable curriculum with "done when" criteria.

## Key Conventions

**Gymnasium-Robotics Fetch environments:**
- Observations are dicts: `observation`, `desired_goal`, `achieved_goal`
- Actions are 4D Cartesian deltas: `dx, dy, dz, gripper`
- HER requires off-policy algorithms (SAC/TD3, not PPO) because goal relabeling requires replay buffers

**Docker-first workflow:**
- Never install packages on DGX host--always containerize
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

# Key metrics to watch:
# - rollout/success_rate: should increase over training
# - rollout/ep_rew_mean: should become less negative (for dense rewards)
# - train/value_loss: should decrease then stabilize
# - train/entropy_loss: should slowly decrease (policy becoming deterministic)
# - replay/q_min_mean (SAC): should stabilize, not explode
# - replay/ent_coef (SAC): should decrease from ~1.0 to ~0.1-0.5
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

Low GPU utilization (~5-10%) during RL training is **expected**. The bottleneck is CPU-bound MuJoCo simulation, not GPU-bound neural network operations. With small batch sizes (256) and simple MLPs, GPU operations complete in microseconds while the CPU runs physics. Typical throughput: ~600 fps. This is not a problem to solve--it's the nature of RL with physics simulators.

## Prerequisite and Concept Architecture

This section describes what the reader is expected to know coming in, what we
teach, and where to point them for topics we choose not to cover. These
boundaries help us honor the Bourbaki principle ("define before use") in a way
that is concrete rather than aspirational -- we can check our work against them.

### Prerequisite Boundary

We organize every concept into three categories. The goal is not to be
gatekeeping about prerequisites, but to be honest about what we assume so
readers know where they stand.

**Assumed known (the reader's responsibility):**
- Python (functions, classes, iterators, f-strings), NumPy (array ops, broadcasting)
- PyTorch basics (tensors, autograd, `nn.Module`, optimizers, `.backward()`)
- What a neural network does (maps inputs to outputs via learned parameters)
- Probability fundamentals (expectation, sampling, probability distributions)
- Linear algebra basics (vectors, matrices, dot products)
- What an MDP is at the level of Sutton & Barto (2018) Ch3.1-3.3 (states,
  actions, transitions, rewards, discount factor) -- we formalize
  goal-conditioned extensions in Ch1, but standard MDP literacy is assumed

If a concept is in this list, it can be used freely in any chapter without
re-definition. If the reader might appreciate a refresher, a parenthetical
with a reference is helpful: "the discount factor $\gamma$ (Sutton & Barto,
2018, Ch3.3)."

**Defined in this curriculum (our responsibility):**
- See the Concept Registry below for the exact chapter where each term is
  introduced
- The Bourbaki rule applies here: if Chapter N uses concept X, then X should
  appear in the registry at Chapter M where M < N
- If we need a concept that is not yet in the registry and not in the "assumed"
  list, we either (a) define it formally in the current chapter and add it to
  the registry, or (b) cite an external reference using the Scope Boundary
  Convention (see Documentation Guidelines)

**Out of scope (we point readers elsewhere):**

These are topics that matter but that we chose not to cover in depth. For each,
we name the concept and give a specific reference so the reader has somewhere
to go. We find this more respectful than saying "beyond the scope of this
tutorial" and leaving them stranded.

- The likelihood ratio trick derivation -- Sutton & Barto (2018) Ch13.2
- Importance sampling theory -- Sutton & Barto (2018) Ch5.5
- Neural network architecture design -- Goodfellow et al. (2016) Ch6-8
- Convex optimization fundamentals -- Boyd & Vandenberghe (2004) Ch1-2
- Information theory (entropy from first principles) -- Cover & Thomas (2006) Ch2
- Measure-theoretic probability -- not needed; we stay at the level of
  expectations and sampling throughout

### Concept Registry

Every formally introduced concept, listed by the chapter that owns its
definition. When writing Chapter N, any concept from Chapter M < N can be
used without re-definition. Concepts not in this registry and not in the
"assumed" list should be either defined or cited.

We find this registry useful as a living checklist: when a new chapter is
written, we add its concepts here, and the Bourbaki constraint stays checkable.

```
Ch0: reproducibility, well-posedness (three conditions), container, image,
     proof of life, rendering backend (EGL/OSMesa)

Ch1: goal-conditioned MDP (S, A, G, P, R, phi, gamma), goal-conditioned
     observation, goal-achievement mapping phi, compute_reward API,
     dense reward, sparse reward, success threshold epsilon,
     dictionary observation structure, critical invariant (reward
     recomputation), HER applicability theorem

Ch2: return G_t, discount factor gamma (formalized), time horizon T,
     expected discounted return J(theta), policy pi(a|s), advantage
     function A(s,a), Q-function Q(s,a), value function V(s),
     policy gradient theorem, probability ratio r_t(theta),
     PPO clipping (L^CLIP), actor-critic architecture,
     generalized advantage estimation (GAE), TD residual delta_t,
     lambda parameter, on-policy learning

Ch3: off-policy learning, replay buffer, maximum entropy objective,
     entropy H(pi), temperature parameter alpha, automatic temperature
     tuning, target entropy H_bar, Boltzmann policy, twin critic
     networks, clipped double Q-learning, target networks,
     soft update / Polyak averaging (tau), Bellman error,
     Bellman target y, squashed Gaussian policy, SAC

Ch4: hindsight experience replay (HER), goal relabeling, goal sampling
     strategies (future/final/episode), n_sampled_goal, data
     amplification, off-policy requirement for HER (three conditions),
     effective horizon T_eff, HER ratio, cumulative entropy bonus
     scaling

Ch5: dense-first debugging, multi-phase control, goal stratification,
     air goal / table goal classification, stress evaluation,
     noise injection (observation noise, action noise), degradation,
     curriculum learning, difficulty schedule (linear, success-gated),
     NoisyEvalWrapper, CurriculumGoalWrapper
```

### Canonical References

When a concept is outside our curriculum, or when we want to offer readers a
deeper path, we point to these specific sources. We standardize on these
editions and notations so that readers moving between our tutorials and these
references do not have to mentally translate.

| Topic | Reference | Specific Section |
|-------|-----------|------------------|
| MDP formalism | Sutton & Barto (2018) | Ch3.1-3.3 |
| Policy gradient theorem | Sutton & Barto (2018) | Ch13.1-13.2 |
| REINFORCE algorithm | Sutton & Barto (2018) | Ch13.3 |
| Importance sampling | Sutton & Barto (2018) | Ch5.5 |
| Actor-critic methods | Sutton & Barto (2018) | Ch13.5 |
| Function approximation | Sutton & Barto (2018) | Ch9-10 |
| PPO (original paper) | Schulman et al. (2017) | arXiv:1707.06347 |
| GAE (original paper) | Schulman et al. (2015) | arXiv:1506.02438 |
| SAC (original paper) | Haarnoja et al. (2018a) | arXiv:1801.01290 |
| SAC (applications) | Haarnoja et al. (2018b) | arXiv:1812.05905 |
| TD3 (original paper) | Fujimoto et al. (2018) | ICML 2018 |
| HER (original paper) | Andrychowicz et al. (2017) | arXiv:1707.01495 |
| Reproducibility in RL | Henderson et al. (2018) | AAAI 2018 |
| Fetch environments | Plappert et al. (2018) | arXiv:1802.09464 |
| Neural network basics | Goodfellow et al. (2016) | Ch6-8 |
| Information theory | Cover & Thomas (2006) | Ch2 (entropy) |
| RL algorithms survey | Spinning Up in Deep RL | spinningup.openai.com |

**Notation convention:** We follow Sutton & Barto (2018) notation throughout:
- Policy: $\pi(a|s)$ or $\pi_\theta(a|s)$ when parameterized
- State-value: $V^\pi(s)$ (or $v_\pi(s)$ in Sutton & Barto's lowercase)
- Action-value: $Q^\pi(s,a)$ (or $q_\pi(s,a)$)
- Return: $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$
- Advantage: $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$

When another paper uses different notation (e.g., Haarnoja's $J(\pi)$ vs.
Sutton's $J(\theta)$), we note the correspondence explicitly so the reader
is not left wondering whether these are the same quantity.

## Voice and Pedagogy (Tutorials)

This section applies to the open-source tutorials and docs site. For Manning
book chapters in `Manning/`, follow `manning_proposal/manning_writer_persona.md`.

This curriculum follows the Bourbaki tradition: rigorous, sequential, no undefined terms.

**The Bourbaki Principle: Define Before Use**
- We avoid using a term before defining it -- the Concept Registry (above)
  makes this checkable
- If Chapter N uses concept X, then X should be defined in Chapter M where
  M < N (or appear in the "assumed known" prerequisite list)
- When referencing future material, we say "we will define X in Chapter N"
  rather than using X as if known
- We hold off on acronyms (SAC, HER, PPO) until the chapter that introduces
  them; when an early mention is unavoidable, we provide an inline definition:
  "off-policy methods (algorithms that reuse past experience)"

**Definition Template (5-Step):**

When we introduce a new concept, we try to follow a sequence that builds
understanding gradually rather than dropping a formula and expecting it to
stick. Not every step is needed for every definition -- for lightweight
concepts a sentence or two is fine -- but for anything the reader will use
repeatedly, we find this pattern helpful:

1. **Motivating problem** (1-2 sentences): Why do we need this concept? What
   difficulty does it address? This grounds the definition in something the
   reader already feels.

2. **Intuitive description** (1-2 sentences): What does it mean in plain
   language? Analogies are welcome. This is the "what it feels like" before
   the "what it is."

3. **Formal definition**: Precise mathematical statement. We try to define
   every symbol before or within this block -- including domain, range, units,
   and typical values where helpful. We use the explicit "**Definition (X).**"
   format for key concepts.

4. **Grounding example**: Concrete numbers, a short code snippet, or a diagram.
   The reader should be able to verify the definition against this example by
   hand. We find this is often the part that actually creates understanding.

5. **Non-example or common misconception** (optional, but valuable for subtle
   concepts): What this does NOT mean. This is especially useful when a concept
   sounds like something the reader might already know but differs in ways that
   matter.

Example -- Advantage Function:

> 1. *Problem:* Policy gradient estimates have high variance. We need a way
>    to measure whether a specific action was better or worse than what we
>    would normally do, so we can reduce noise without introducing bias.
>
> 2. *Intuition:* The advantage answers: "how much better was this action
>    compared to my average behavior in this state?"
>
> 3. *Definition:* $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, where
>    $Q^\pi(s,a)$ is the expected return after taking action $a$ in state $s$
>    and then following $\pi$, and $V^\pi(s)$ is the expected return from
>    state $s$ under $\pi$. Note $\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0$
>    by construction.
>
> 4. *Example:* If $V^\pi(s) = -0.3$ and $Q^\pi(s, a_{\text{left}}) = -0.1$,
>    then $A^\pi(s, a_{\text{left}}) = +0.2$ -- moving left is better than
>    the policy's average from this state.
>
> 5. *Non-example:* A positive advantage does NOT mean the action leads to a
>    good outcome in absolute terms. It means the action is better than the
>    policy's average. An advantage of +0.2 with $V = -0.3$ still yields
>    $Q = -0.1$ -- a negative expected return.

**Symbol hygiene:**
- Before any equation, we define every symbol that appears in it
- We include domain (e.g., $\gamma \in [0,1)$), units (e.g., "meters"), and
  typical values (e.g., "$\gamma = 0.99$ in our experiments") where helpful
- When reusing a symbol from a previous chapter, a brief reminder helps:
  "the discount factor $\gamma$ (defined in Ch2)"
- When a paper's notation conflicts with ours, we note the correspondence:
  "Haarnoja et al. write $J(\pi)$; we use $J(\theta)$ following Sutton & Barto"

**The Hadamard Questions (ask before solving):**
1. Does a solution exist? (Can a neural network policy solve this task?)
2. Is it unique? (Are there multiple qualitatively different solutions?)
3. Does it depend continuously on the data? (Will different seeds give similar results?)

**WHY-HOW-WHAT Structure:**
- **WHY:** Problem formulation first. What mathematical object are we seeking? What makes this hard?
- **HOW:** Derive the method from constraints. We want the algorithm to feel inevitable, not handed down.
- **WHAT:** Concrete deliverables. Files that exist, tests that pass, metrics to compute.

**Chapter Bridge Rule:**

We find that readers get confused at chapter boundaries more than anywhere else.
A chapter that opens cold -- jumping into new formalism without connecting to
what came before -- can feel like a topic change rather than a continuation.

To help with this, we try to open every chapter (after Ch0) with a short bridge
that does four things:

1. **Name the capability established:** What can the reader now do, having
   completed the previous chapter?
2. **Identify the gap:** What can that capability NOT do? What question should
   the reader already be asking?
3. **State what this chapter adds:** How does this chapter close the gap?
4. **Foreshadow (optional):** If the current chapter's solution has its own
   limitations, briefly mention what comes next -- this helps the reader see
   the curriculum as a connected arc rather than isolated topics.

Example (Ch3 bridge):

> "In Chapter 2, we trained PPO to reach targets using dense rewards -- the
> robot received continuous feedback proportional to its distance from the
> goal. PPO achieved 100% success on FetchReachDense, but it required on-policy
> data: every training step discarded all previous experience. Can we learn
> more efficiently by reusing past transitions? This chapter introduces SAC, an
> off-policy algorithm that stores experience in a replay buffer and reuses it
> repeatedly. In Chapter 4, we will need this off-policy machinery for a harder
> problem: learning from sparse binary rewards."

Example (Ch4 bridge):

> "Chapter 3 showed that SAC learns FetchReachDense efficiently, converging
> faster than PPO by reusing experience. But dense rewards required us to
> hand-design a distance-based signal -- what if we only know whether the
> robot succeeded or failed? Sparse rewards ($R = 0$ or $R = -1$) are more
> natural but create a needle-in-a-haystack exploration problem. This chapter
> introduces Hindsight Experience Replay (HER), which transforms failures
> into learning signal by asking: 'what goal would this trajectory have
> achieved?'"

**Tone:**
- Humble but rigorous. Acknowledge field-wide problems (reproducibility crisis, sensitivity to hyperparameters)
- Explain failures as structural issues, not personal incompetence
- "The problem formulation was ill-posed" not "I made a mistake"
- Cite relevant literature when making claims (Henderson 2018 on reproducibility)

**Humble Writing Style:**

We explain and invite, not command and preach. The goal is collaboration with the reader, not lecturing.

| Instead of... | Write... |
|---------------|----------|
| "You must do this" | "We find this useful because..." |
| "The logic is..." | "The reasoning goes something like..." |
| "This is the only way" | "This works well for us; alternatives exist" |
| "Never do X" | "We avoid X because..." |
| "This is obvious" | (just explain it) |

**Voice guidelines:**
- First person plural ("we") invites collaboration; imperative mood ("do this") creates distance
- Explain *why* we do things instead of commanding readers to do them
- Acknowledge alternatives: "not the only way", "one approach is..."
- When something is hard, say so: "this took us several attempts" not "simply do X"
- Admit uncertainty: "we believe", "in our experience", "this seems to work"

**Examples:**

Bad: "You cannot fool yourself into thinking you understand something when the success rate is 12%."

Good: "We find concrete success metrics helpful -- it's easy to feel like things are working when the code runs without errors, but a 12% success rate tells you something is still wrong."

Bad: "This is not a recipe chosen from a menu; it is a consequence of the problem structure."

Good: "We're not claiming this is the only way -- but we wanted to show how problem structure can guide algorithm choice."

**The Four Commitments:**
1. **Reproducibility:** Multi-seed experiments, version-controlled everything, Docker containers
2. **Quantification:** Numbers, not adjectives. "94% success rate" not "works well"
3. **Derivation:** Show where algorithms come from, not just how to call them
4. **Understanding:** We try not to use tools we cannot explain -- no black boxes

## Documentation Guidelines

**Single source of truth:**
- `tutorials/` is the canonical location for tutorial content
- `docs/tutorials/` contains symlinks to `../../tutorials/` for MkDocs
- Always edit files in `tutorials/`, never in `docs/tutorials/`
- `docs/tutorials/index.md` is the only non-symlinked file (MkDocs navigation)
- `Manning/chapters/` is the canonical location for book chapters (do not edit generated `Manning/output/`)

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

**ASCII-only text:**
- Use `--` instead of em-dash (—) or en-dash (–)
- Use straight quotes `"` and `'` instead of curly quotes (" " ' ')
- Use `...` instead of ellipsis (…)
- Use `->` instead of arrow (→)
- Avoid Unicode symbols; prefer ASCII equivalents

**Math in Markdown (GitHub-compatible):**
- Inline math: `$...$` (e.g., `$\pi^*$` renders as π*)
- Block math: `$$...$$` on separate lines
- For complex equations, use fenced code blocks with `math` language

For Manning book builds (pandoc), avoid MathJax-only extensions and keep block
math (`$$...$$`) on its own line.

## Manning Book Production

The Manning book draft is produced from tutorials using an explicit protocol:
- Follow `manning_proposal/chapter_production_protocol.md` for every chapter
- Use the role prompts in `manning_proposal/agents/` (Scaffolder -> Lab Engineer + Writer -> Reviewer -> Publisher)
- Keep book chapters in `Manning/chapters/` and scaffolds in `Manning/scaffolds/`
- Book voice and structure follow `manning_proposal/manning_writer_persona.md` (it overrides tutorial-era voice guidance)
- Do not modify `tutorials/` as part of book writing; the scaffold is the handoff artifact
- Book markdown must be pandoc-friendly: avoid MkDocs-specific syntax (`!!!`, `--8<--`, `<details>`); use plain Markdown
- Validate/build via `python scripts/build_book.py` (Publisher phase runs pandoc/LaTeX; other phases are markdown-only)

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- This includes: new chapter scripts, tutorial rewrites, training pipeline changes, HER/algorithm config changes
- If an experiment or implementation goes sideways, stop and re-plan immediately
- Plan verification steps too (how to validate a training run before launching it)
- Write detailed specs upfront: environments, seeds, metrics, success criteria

### 2. Subagent Strategy

- Use subagents to keep the main context clean
- Offload secondary research, codebase exploration, and parallel analysis
- One task per subagent

### 3. Self-Improvement Loop

- After any correction from the user: record the pattern, root cause, and prevention rule
- If `tasks/lessons.md` exists, update it; otherwise keep a short "lessons" note in the repo-local workflow docs

### 4. Verification Before Done

- Never mark a task complete without proving it works
- For scripts: run them (or confirm they run) inside Docker via `docker/dev.sh` or the non-interactive pattern
- For tutorials: verify snippet-includes resolve, math renders, and linked scripts exist
- For training changes: check that metrics move in the expected direction (success_rate up, losses stabilizing)

### 5. Demand Elegance (Balanced)

- For non-trivial changes (new chapter scripts, algorithm implementations, lab modules): pause and ask "is there a more elegant way?"
- Skip this for simple, obvious fixes

### 6. Autonomous Bug Fixing

- When given a bug report (training divergence, Docker build failure, broken eval): fix it without hand-holding
- Use logs, errors, and metrics as the evidence trail

## Task Management

If `tasks/` exists, use it as lightweight session state:
1. Plan first: write a checklist to `tasks/todo.md`
2. Track progress: mark items complete as you go
3. Document results: add outcomes (metrics, artifacts) to `tasks/todo.md`
4. Capture lessons: update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Minimal code impact beats refactors.
- **No Laziness**: Find root causes. No temporary workarounds. If training fails, diagnose why.
- **Minimal Impact**: Touch only what's necessary. Avoid regressions in working pipelines.
- **Reproducibility Always**: Seeds, Docker, version-pinned deps, JSON artifacts.
- **Quantify Everything**: Numbers, not adjectives. Report success rates, returns, goal distances.

## Testing Guidelines

No pytest suite yet. Treat scripts as smoke tests:
- Minimum: `bash docker/dev.sh python scripts/ch00_proof_of_life.py all`
- Repro sanity: `bash docker/dev.sh python scripts/check_repro.py ...`
- Book build sanity (requires pandoc/LaTeX): `python scripts/build_book.py --validate-only --verbose`

## Commit & Pull Request Guidelines

- Use clear, imperative commits (e.g., `feat: add ch04 HER experiment`).
- PRs include: purpose, exact commands run (platform/Docker), expected artifacts/paths, and doc updates (`tutorials/`, `syllabus.md`, `Manning/`) when adding new scripts/chapters.
