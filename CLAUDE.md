# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note:** This file is intentionally NOT tracked in git. It is a local guidance file for Claude Code sessions. Do not attempt to commit changes to this file.

## Project Overview

A Docker-first robotics RL research platform for goal-conditioned manipulation using Gymnasium-Robotics Fetch tasks. Built around Stable Baselines 3 (PPO/SAC/TD3) with HER support for sparse rewards.

**Methodological focus:** SAC + HER for sparse-reward goal-conditioned tasks. This choice is derived from problem constraints (continuous actions -> actor-critic; sparse rewards -> off-policy; goal relabeling -> HER). PPO is included only for dense-reward baselines.

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

# End-to-end smoke test (GPU + Fetch + render + PPO)
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
```

## Training & Evaluation

```bash
# Train PPO on dense Reach (baseline only)
bash docker/dev.sh python train.py --algo ppo --env FetchReachDense-v4 --seed 0 --n-envs 8 --total-steps 1000000

# Train SAC + HER on sparse Reach (primary method)
bash docker/dev.sh python train.py --algo sac --her --env FetchReach-v4 --seed 0 --n-envs 8 --total-steps 1000000

# Evaluate checkpoint
bash docker/dev.sh python eval.py --ckpt checkpoints/...zip --env FetchReachDense-v4 --n-episodes 100 --seeds 0-99 --deterministic --json-out results/metrics.json
```

## Architecture

- **`train.py`**: Training CLI supporting PPO/SAC/TD3, HER, vectorized envs, TensorBoard. Outputs `.zip` checkpoint + `.meta.json`.
- **`eval.py`**: Evaluation CLI with metrics (success rate, return, goal distance, time-to-success, action smoothness). Outputs JSON reports.
- **`docker/`**: Container tooling. `dev.sh` is the primary entry point (handles venv, deps, GPU, rendering backend).
- **`scripts/`**: Versioned experiment scripts (`chNN_<topic>.py`). Tutorials only call these—no inline code blocks.
- **`tutorials/`**: Textbook-style chapters (`chNN_<topic>.md`) following WHY-HOW-WHAT structure.
- **`syllabus.md`**: 10-week executable curriculum with "done when" criteria.
- **`tasks/`**: Working documents -- plans, research notes, and lessons learned. NOT checked into git.
  - `lessons.md`: Patterns, root causes, and prevention rules discovered during development. Reviewed at session start.
  - `pixel_push_encoder_research.md`: Comprehensive research on CNN architectures for pixel-based manipulation (NatureCNN limitations, DrQ-v2 encoder, spatial softmax, proprioception, literature gap analysis).
  - `toc_revision_plan.md`: Book restructuring plan (13 -> 10 chapters).

## Key Conventions

**Gymnasium-Robotics Fetch environments:**
- Observations are dicts: `observation`, `desired_goal`, `achieved_goal`
- Actions are 4D Cartesian deltas: `dx, dy, dz, gripper`
- HER requires off-policy algorithms (SAC/TD3, not PPO) because goal relabeling requires replay buffers

**Docker-first workflow:**
- Never install packages on DGX host—always containerize
- `MUJOCO_GL=egl` for headless rendering (fallback: osmesa)
- `docker/dev.sh` preserves host UID/GID to avoid root-owned files

**Non-interactive Docker execution (for scripts/CI):**
When running Docker commands without a TTY (e.g., from scripts or Claude Code), the `docker/dev.sh` script fails due to `-it` flags. Use this pattern instead:

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

**Coding style:**
- Python 4-space indentation, `argparse` CLIs, `pathlib.Path`, type hints
- Bash scripts use `set -euo pipefail`
- Scripts are self-contained and reproducible
- All metrics/metadata as JSON artifacts
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

This wrapper:
- Starts a GPU-enabled container (`--gpus all`)
- Creates/activates a venv with dependencies from `requirements.txt`
- Sets rendering backend (`MUJOCO_GL=egl`)
- Preserves host UID/GID (avoids root-owned files)

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
# Start a persistent session
tmux new -s rl

# Inside tmux, run training
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all --seed 0

# Detach: Ctrl-b d
# Reattach later:
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

GPU utilization depends on the observation modality:

- **State-based RL (Ch1-8):** Low GPU utilization (~5-10%) is expected. Small MLPs (256x256) on 25D vectors complete forward+backward passes in microseconds. The bottleneck is CPU-bound MuJoCo simulation. Typical throughput: ~600 fps.
- **Pixel-based RL (Ch9+):** GPU utilization of 40-60% is normal. ManipulationCNN processes batches of 84x84x12 images (4-frame stack) through 4 conv layers for both obs and next_obs per training step. The CPU (MuJoCo physics for n_envs) and GPU (CNN forward/backward) form a balanced pipeline. Typical throughput: 30-50 fps with n_envs=4.

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

Ch6: action interface, action scaling, low-pass filter (EMA),
     smoothness (mean squared action difference), time-to-success (TTS),
     peak action, path length, action energy, controller metric bundle,
     proportional controller, planning-vs-control decomposition,
     ActionScalingWrapper, LowPassFilterWrapper, run_controller_eval

Ch7: degradation curve, critical sigma (sigma*), degradation slope,
     robustness AUC, brittleness fingerprint, noise injection
     (eval-time wrapper), observation noise model, action noise model,
     cross-seed aggregation, NoiseSweepResult, run_noise_sweep,
     aggregate_across_seeds, compute_degradation_summary,
     noise-augmented training, clean-vs-robust tradeoff,
     experiment card (formalized .meta.json pattern)

Ch9: pixel observation wrapper, goal mode (none/desired/both),
     render_and_resize, NatureCNN encoder (Mnih et al. 2015),
     sample-efficiency ratio (rho), DrQ (random shift augmentation),
     replicate padding, DrQ replay buffer, uint8 pixel storage,
     native resolution rendering, SubprocVecEnv (parallel envs),
     replay ratio / gradient steps, deceptively dense reward,
     HerDrQDictReplayBuffer, visual HER synthesis,
     information asymmetry (policy sees pixels, HER sees vectors),
     value wavefront (Bellman diffusion through goal space),
     hockey-stick learning curve (geometric phase transition + positive feedback),
     critical competence radius, effective horizon k* (Laidlaw et al. 2024)
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
| DrQ (data augmentation) | Kostrikov et al. (2020) | arXiv:2004.13649, Section 3.1 |
| NatureCNN | Mnih et al. (2015) | Nature 518(7540), 529-533 |
| Visual goal-conditioned RL | Nair et al. (2018) | NeurIPS 2018, arXiv:1807.04742 |
| Fetch environments | Plappert et al. (2018) | arXiv:1802.09464 |
| Neural network basics | Goodfellow et al. (2016) | Ch6-8 |
| Information theory | Cover & Thomas (2006) | Ch2 (entropy) |
| RL algorithms survey | Spinning Up in Deep RL | spinningup.openai.com |
| Effective horizon | Laidlaw et al. (2024) | ICLR 2024 Spotlight, arXiv:2312.08369 |
| Quasimetric Q-functions | Wang & Isola (2022) | ICLR 2022, arXiv:2206.15478 |
| Difficulty spectrum dynamics | Huang et al. (2025) | arXiv:2602.14872, Section 4 |
| Contrastive goal-conditioned RL | Eysenbach et al. (2022) | NeurIPS 2022, arXiv:2206.07568 |
| HER as implicit curriculum | Ren et al. (2019) | NeurIPS 2019, arXiv:1906.04279 |

**Notation convention:** We follow Sutton & Barto (2018) notation throughout:
- Policy: $\pi(a|s)$ or $\pi_\theta(a|s)$ when parameterized
- State-value: $V^\pi(s)$ (or $v_\pi(s)$ in Sutton & Barto's lowercase)
- Action-value: $Q^\pi(s,a)$ (or $q_\pi(s,a)$)
- Return: $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$
- Advantage: $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$

When another paper uses different notation (e.g., Haarnoja's $J(\pi)$ vs.
Sutton's $J(\theta)$), we note the correspondence explicitly so the reader
is not left wondering whether these are the same quantity.

## Voice and Pedagogy

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

**Tone and Prose Rhythm:**

The tone we aim for is humble but rigorous -- the voice of someone who has
struggled with the same problems the reader faces and can describe what they
learned without pretending the path was obvious. That means acknowledging
field-wide difficulties (the reproducibility crisis, sensitivity to
hyperparameters) as structural issues rather than personal failures, so that
"the problem formulation was ill-posed" replaces "I made a mistake," and
claims about algorithmic behavior are grounded in specific literature
(Henderson et al. 2018 on reproducibility, for instance) rather than
asserted from authority.

Prose rhythm matters as much as word choice. When guidance is written as a
sequence of short declarative bullets, the agent mirrors that rhythm in
tutorial prose -- producing text that reads like a checklist rather than an
argument. We prefer paragraphs that advance a single idea across several
sentences connected by logical tissue: "so that," "which means," "since,"
"provided that." Parentheticals woven into sentences (like this one) keep
qualifications close to the claims they modify, rather than splitting them
into separate bullet points. Lists remain useful for genuinely parallel
items -- environment names, file paths, CLI flags -- but not for prose
arguments that happen to have multiple parts.

A concrete example helps calibrate the difference:

> *Staccato (before):* Humble but rigorous. Acknowledge field-wide problems.
> Explain failures as structural issues. Cite literature.
>
> *Flowing (after):* The tone we aim for is humble but rigorous -- we
> acknowledge field-wide problems like the reproducibility crisis as
> structural issues rather than personal failures, and we ground claims
> in specific literature rather than asserting them from authority.

**Humble Writing Style and Voice:**

We write in first person plural ("we") because it frames the text as
collaboration rather than instruction -- "we find this useful because..."
instead of "you must do this," "the reasoning goes something like..." instead
of "the logic is...," "this works well for us; alternatives exist" instead of
"this is the only way." When we want to discourage a practice, "we avoid X
because..." is warmer than "never do X," and when something might seem
self-evident, we just explain it rather than writing "this is obvious" (which
helps no one and alienates anyone who did not find it obvious). The word
"simply" deserves special vigilance: if something were truly simple, it would
not need a sentence explaining it, so "simply configure the rendering backend"
becomes "configure the rendering backend -- here is what each option does."

The same principle extends to how we describe difficulty and uncertainty.
Saying "this took us several attempts" is more honest (and more useful) than
"simply do X," and phrases like "we believe," "in our experience," and "this
seems to work" signal that the field is empirical and our results are situated,
not universal. We acknowledge alternatives ("not the only way," "one approach
is...") because the reader will encounter different advice elsewhere and
deserves to know that reasonable people disagree.

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

**Content rules:**
- Tutorials follow WHY-HOW-WHAT structure (problem formulation -> methodology -> implementation)
- Keep commits imperative (e.g., `feat: add ch02 training harness`)

**Scope Boundary Convention:**

When a topic is relevant but outside what we cover, we find it more helpful to
mark the boundary explicitly than to hand-wave or over-explain. The pattern we
use:

> "[Concept X] involves [brief reason it goes deep]. For our purposes, the
> property we need is [specific property]. Readers who want the full treatment
> may enjoy [Reference, Section N]."

Good example:
> "Importance sampling corrects for the mismatch between the data-collection
> policy and the current policy. A full treatment involves careful analysis of
> variance and convergence (Sutton & Barto, 2018, Ch5.5). For our purposes,
> the key insight is simpler: SAC sidesteps importance sampling entirely by
> using a replay buffer with no explicit correction, which works in practice
> because the Q-function learns to approximate the current policy's value."

Patterns we try to avoid:
- **Hand-wave:** "This is beyond the scope of this tutorial" -- this gives the
  reader nowhere to go and can feel dismissive
- **Over-explain:** Three paragraphs on something we said is out of scope --
  if it needs that much space, it might actually be in scope (define it
  properly) or it might not (cite and move on)
- **Assume knowledge:** Using a concept without acknowledging it has not been
  defined in our curriculum -- this quietly violates the Bourbaki principle
- **Vague citation:** "See the literature" -- we try to always give a specific
  reference with a section number, so the reader can actually find it

**Code in tutorials:**
- Avoid large runnable code blocks. Use snippet-includes from `scripts/` or `scripts/labs/` to show small, annotated fragments that map derivations to code.
- Prefer snippet-includes pulled from repo files (single source of truth). Use `pymdownx.snippets` syntax: `--8<-- "scripts/labs/ppo_from_scratch.py:region_name"`.
- Limit any single excerpt to 30-40 lines; focus on update logic (actor loss, critic loss, advantage computation), not boilerplate.
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
- **Run It:** The production pipeline using SB3. Found in `scripts/chNN_*.py`. Reproducible, optimized, tracked.
- **Build It:** From-scratch implementations in `scripts/labs/`. Shows how equations map to code. Educational, not for production.

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
- For complex equations, use fenced code blocks with `math` language:
  ~~~
  ```math
  J(\pi) = \mathbb{E}[\sum_{t} \gamma^t R_t]
  ```
  ~~~
- GitHub renders LaTeX via MathJax; test rendering on GitHub after pushing

## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- This includes: new chapter scripts, tutorial rewrites, training pipeline changes, HER/algorithm config changes
- If an experiment or implementation goes sideways, STOP and re-plan immediately -- don't keep pushing on a broken approach
- Use plan mode for verification steps too, not just building (e.g., plan how to validate a training run before launching it)
- Write detailed specs upfront: which environments, seeds, metrics, success criteria

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload secondary research (e.g., "what SB3 hyperparameters does HER support?"), codebase exploration, and parallel analysis to subagents
- For complex problems (debugging training divergence, multi-file refactors), throw more compute at it via subagents
- One task per subagent for focused execution
- Use Explore subagents for codebase questions; use Plan subagents for architecture decisions

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern, root cause, and fix
- Write rules that prevent the same mistake (e.g., "always check reward type before choosing algorithm")
- Ruthlessly iterate on these lessons until mistake rate drops
- Review `tasks/lessons.md` at session start for relevant project context
- Current lessons (as of Ch9 pixel Push debugging):
  1. **Sensor Separation Principle**: Always pass proprioception alongside pixels -- CNN should learn world-state, not self-state
  2. **NatureCNN is wrong for manipulation**: 8x8 stride-4 destroys small objects; use 3x3 stride-2 (DrQ-v2 style)
  3. **TensorBoard log contamination**: Back up AND delete old TB dirs before relaunching with same run_id
  4. **Two kinds of Visual HER**: goal_mode="both" (our approach, vector goals) vs full image-goal HER (Nair 2018, VAE required)
  5. **Normalization fixes gradients, not features**: Clean bounded noise is still noise -- look at encoder architecture, not output normalization

### 4. Verification Before Done

- Never mark a task complete without proving it works
- For scripts: run them (or confirm they run) inside Docker via `docker/dev.sh` or the non-interactive pattern
- For tutorials: verify snippet-includes resolve, math renders, and linked scripts exist
- For training changes: check that metrics move in the expected direction (success_rate up, value_loss stabilizing)
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would this pass peer review in a reproducibility-focused RL lab?"

### 5. Demand Elegance (Balanced)

- For non-trivial changes (new chapter scripts, algorithm implementations, lab modules): pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes -- don't over-engineer a one-line config change
- Challenge your own work before presenting it, but respect the project's simplicity-first principle

### 6. Autonomous Bug Fixing

- When given a bug report (training divergence, Docker build failure, broken eval): just fix it. Don't ask for hand-holding
- Point at logs, errors, failing metrics -- then resolve them
- Zero context switching required from the user
- Go fix failing CI tests, broken Docker builds, or crashing scripts without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items (`- [ ]` / `- [x]`)
2. **Verify Plan**: Check in with the user before starting implementation on non-trivial work
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step (what changed, why, what metrics to expect)
5. **Document Results**: Add a review section to `tasks/todo.md` with outcomes (success rates, artifacts produced)
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections -- pattern, root cause, prevention rule

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code. A one-line hyperparameter fix is better than a refactored training loop.
- **No Laziness**: Find root causes. No temporary workarounds. Senior researcher standards -- if training fails, diagnose *why*, don't just retry with different seeds.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing regressions in working pipelines.
- **Reproducibility Always**: Every experiment must be reproducible. Seeds, Docker, version-pinned deps, JSON artifacts.
- **Quantify Everything**: Numbers, not adjectives. Report success rates, returns, goal distances -- not "it seems to work."
