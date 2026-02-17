# Manning Proposal Overview (Draft)

## Working Title and Positioning

Working title: Robotics Reinforcement Learning in Action

Working subtitle: Build reproducible goal-conditioned manipulation agents in MuJoCo (dense rewards, sparse rewards, pixels, and reality-gap stress tests).

Series fit: In Action (primary). The promise is practical competence: a reader who finishes the book can train, evaluate, and debug policies with an auditable experimental record (metrics JSON, provenance, checkpoints), not just produce videos.

## Target Audience

Primary reader:
- ML engineers and software engineers who want to apply reinforcement learning to robotics and continuous control.
- Researchers and practitioners who want a reproducible, engineering-first workflow for goal-conditioned manipulation.

Minimally qualified reader:
- Python proficiency, basic command-line comfort, and familiarity with neural networks.
- Basic RL vocabulary (policy, value function, replay buffer). The book defines concepts before use and does not assume robotics background beyond high-school physics intuition.

Constraints and expectations:
- A GPU is strongly recommended (especially for the later chapters). CPU-only readers can still run the early chapters and use provided checkpoints for heavier experiments.

## What the Reader Will Be Able to Do

After reading this book, the reader will be able to:
- Set up a containerized MuJoCo + Gymnasium-Robotics workflow that runs reproducibly across machines.
- Train PPO baselines for dense-reward tasks and use them as diagnostic instruments ("is my pipeline broken?").
- Train SAC for continuous control and interpret replay diagnostics (entropy, Q-values, reward distributions) when learning stalls.
- Use HER to make sparse, goal-conditioned tasks learnable and verify that relabeling is functioning correctly.
- Evaluate policies with fixed-seed protocols and produce a comparable metrics record (success, return, goal distance, time-to-success, action smoothness).
- Stress-test policies under controlled perturbations (noise and randomization) and quantify brittleness using degradation curves.
- Move from state-based policies to pixels within the same task family (Fetch) and measure the sample-efficiency tradeoffs.

## Why This Topic Is Important Now

Reinforcement learning for robotics is no longer gated by proprietary simulators and bespoke codebases:
- MuJoCo is widely available and actively maintained, and Gymnasium-Robotics provides goal-conditioned task interfaces with explicit success signals.
- Stable Baselines3 has made strong actor-critic baselines (PPO/SAC/TD3) and HER accessible, but many practitioners still lack reliable ways to diagnose failures and verify results.
- The robotics community has shifted from "one task, one demo" toward generalizable goal-conditioned behavior and principled evaluation (seeds, confidence intervals, stress testing).

The market gap is not "another RL introduction" but a practical, reproducible path from first run to a deliverable-grade manipulation policy with an honest evaluation protocol.

## Competition and How This Book Differs

Manning (directly adjacent titles):
- Deep Reinforcement Learning in Action (Zai, Brown) -- strong, example-rich deep RL coverage; not focused on robotics manipulation, goal-conditioned interfaces, or reproducible stress-tested evaluation.
- Grokking Deep Reinforcement Learning (Morales) -- approachable conceptual treatment; not centered on robotics task structure (continuous actions + sparse goals) and not organized around an experiment-contract workflow.
- Robotics for Programmers (Bihlmaier) -- robotics programming breadth; not reinforcement-learning-centric and does not cover HER-style goal relabeling workflows.
- Reinforcement Learning for Business (Aghazadeh) -- RL for optimization; different domain and evaluation constraints.

Other common references (ecosystem context):
- Sutton and Barto -- definitive foundations, but not an implementation-and-evaluation playbook for robotics.
- "Spinning Up" style resources -- helpful algorithms, but readers still need a reproducible robotics harness and stress-test methodology.

Differentiation in one line: this book is a robotics RL engineering playbook built around measurable outcomes and auditable artifacts, using a single coherent task family to minimize setup tax.

## Key Differentiators

1) One stable spine, many skills.
- The book stays in the Fetch manipulation suite long enough for the reader to develop intuition, then extends to pixels and reality-gap mitigation without switching stacks every chapter.

2) "No vibes" evaluation.
- Every chapter produces artifacts: checkpoints, metadata, and evaluation JSON. Videos are illustrations, not evidence.

3) Sparse rewards done methodically.
- The book does not handwave exploration. It shows, with controlled experiments, why sparse learning fails and why HER changes the data distribution enough to learn.

4) From scratch, then scale (Build It / Bridge / Run It).
- Build It: the reader implements core algorithm mechanics from scratch in PyTorch -- losses, updates, buffers, relabeling -- and verifies each component with concrete checks. This is the narrative spine of every algorithm chapter, not a sidebar.
- Bridge: a short section comparing the from-scratch implementation to SB3 on the same computation (same batch, same replay data), proving they agree. This connects understanding to production code.
- Run It: SB3-backed chapter scripts that scale the same math to multi-million-step robotics training, producing comparable artifacts (checkpoints, metadata, eval JSON) under a fixed train/eval contract.
- Verify It: lightweight sanity checks woven throughout Build It (finite values, shape checks, expected trends) and Run It (artifact inspection, success criteria), so readers can catch silent failures early.
- The reader builds the learning engine from scratch. SB3 is the scaling engine. The bridging proof shows they compute the same thing.

5) Five compute tiers, from 2 minutes to overnight.
- Build It `--verify` (< 2 min, CPU): sanity checks after each component.
- Build It `--demo` (~30 min, CPU): short from-scratch training that shows real learning curves.
- Run It fast path (<=500k steps, ~15 min GPU): SB3 single-seed validation during the chapter.
- Reproduce It (<=3M steps, multi-seed): full runs backing the chapter's results tables and pretrained checkpoints, documented as an end-of-chapter block.
- Checkpoint track (seconds): eval-only with pretrained models for readers without a GPU.

6) Debuggability as a first-class goal.
- The reader learns to diagnose failures (reward semantics, success definitions, replay buffer pathologies, entropy collapse) rather than only copying hyperparameters.

## Pedagogy and Voice (What the Reader Experiences)

The writing style aims for a mix of rigorous applied-math habits (define-before-use, explicit problem statements, measurable claims) and pragmatic modern RL engineering (run the experiment, inspect artifacts, iterate).

Concrete pedagogy choices:
- Define terms before use; no unexplained acronyms.
- Prefer "math-lite but honest" explanations: enough formal structure to prevent cargo-culting, without turning the book into a theory text.
- Treat evaluation as part of the method: fixed seeds, deterministic eval, comparable JSON reports, and stress tests.
- Use short runnable pipelines and small controlled experiments to build intuition and prevent handwavy claims.
- Offer three parallel tracks per chapter (Run It / Build It / Verify It) so readers can choose: reproduce results quickly, implement core pieces from scratch, or validate their understanding with checks.

## Supplementary Materials

The book is backed by a runnable repository that standardizes:
- `train.py` for training (PPO/SAC/TD3 with optional HER for off-policy).
- `eval.py` for evaluation (fixed-seed protocols and JSON output).
- Chapter scripts (`scripts/chNN_*.py`) that implement end-to-end experiments and produce consistent artifacts.
- Lab modules (`scripts/labs/*.py`) that implement core algorithms from scratch and include `--verify` sanity checks; tutorials and book chapters excerpt these as the single source of truth for "Build It" code.

These materials support self-study and also provide a consistent evaluation story for reviewers and instructors.
