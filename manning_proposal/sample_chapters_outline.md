# Sample Chapters to Submit (Outline Draft)

This file outlines two sample chapters aligned with the "In Action" structure. The final submitted samples should be written in full prose, but the outlines below are already organized around measurable objectives, artifacts, and reproducible commands.

## Chapter 1 -- Proof of Life: A Reproducible Robotics RL Loop

Promise: by the end of this chapter the reader has a working environment, a rendered frame, a trained checkpoint, and a repeatable "it runs" workflow they can reuse for every later chapter.

Learning objectives:
- Define the minimum working loop for robotics RL: environment -> policy -> training -> evaluation -> artifacts.
- Run a containerized Fetch task and understand what success looks like in outputs (not just in videos).
- Recognize and fix the most common early failures (missing env registration, headless rendering backends, checkpoint I/O).

Outline (numbered for a Manning proposal):
1. Proof of Life
1.1 The failure mode of robotics RL: silent non-learning
- Why a compilation error is easier than a flat success curve
- The difference between "runs" and "learns"
1.2 The task family we will use (defined before use)
- Fetch tasks in MuJoCo, goal-conditioned observations, explicit success signal
- What "goal-conditioned" means and why it matters for later chapters
1.3 The experiment contract (the "no vibes" rule)
- Training entrypoint, evaluation entrypoint, artifact locations
- What gets logged and why provenance matters
1.4 Run the proof-of-life pipeline
- Render a frame headlessly (what rendering backend is, why it fails)
- Train a short PPO run (why PPO is used here)
1.5 Interpret outputs and validate the loop
- What files you should see and what they mean
- Quick checks: success should improve on dense reach; if not, what to inspect
1.6 Summary
- What we now trust and what we still do not trust

Fast path (reader-run):
```bash
bash docker/dev.sh python scripts/ch00_proof_of_life.py all
```

Artifacts:
- `smoke_frame.png` (render validation)
- `ppo_smoke.zip` (checkpoint validation)

Suggested end-of-chapter exercises:
- Change the seed and rerun; confirm you still get artifacts and training does not crash.
- Force a rendering backend (EGL vs OSMesa) and document what changes on your machine.

---

## Chapter 5 -- Learning From Failure: HER on Sparse, Goal-Conditioned Tasks

Promise: by the end of this chapter the reader can reproduce a controlled experiment that shows HER vs no-HER on sparse Reach/Push, and can verify that relabeling is functioning correctly using artifacts rather than intuition.

Learning objectives:
- Define sparse reward, goal-conditioned reward computation, and why naive exploration fails.
- Explain HER as a data-distribution intervention: relabel transitions to manufacture positive learning signal.
- Run and evaluate HER vs no-HER experiments under a fixed protocol and interpret results statistically (across seeds).
- Verify HER correctness using diagnostics (reward recomputation, success metrics, and replay signal sanity checks).

Outline (numbered for a Manning proposal):
5. Learning From Failure With HER
5.1 The problem (made explicit)
- Sparse success rewards and why gradient signal disappears
- Continuous control and why we prefer SAC/TD3 over value-only methods
5.2 The environment requirements for HER (defined before use)
- Achieved goal vs desired goal; compute_reward for arbitrary goals
- What "goal relabeling" means operationally
5.3 Establish the baseline difficulty (SAC without HER)
- What we expect to see (often: flat success)
- How to tell "slow learning" from "broken pipeline"
5.4 The HER mechanism
- Relabeling with achieved goals ("future" strategy as default)
- The parameter that matters: `n_sampled_goal`
5.5 Implementation details that break easily
- Reward semantics: verify `compute_reward` matches env reward
- Evaluation: deterministic policy, fixed seeds, success metrics
5.6 Verification: does HER actually change the data?
- What to look for in artifacts: increased positive transitions, rising success curves
- The minimum acceptable evidence for a claim ("mean over seeds", not a single run)
5.7 Ablations (small, targeted)
- `n_sampled_goal` sweep (2/4/8)
- Entropy coefficient handling on sparse Push (practical default and failure modes)
5.8 Summary
- What HER solved, what it did not solve, and what changes in the capstone chapter

Fast path (reader-run, single seed, <=500k):
```bash
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchReach-v4 --seeds 0 --total-steps 500000
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4  --seeds 0 --ent-coef 0.1 --total-steps 500000
```

Full run (stable results, <=2M per run; still under the 3M cap):
```bash
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchReach-v4 --seeds 0-2 --total-steps 2000000
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4  --seeds 0-2 --ent-coef 0.1 --total-steps 2000000
```

Checkpoint track (reviewer/reader-friendly):
- Provide pretrained checkpoints for HER and no-HER runs.
- Run evaluation only with the same fixed seed protocol and compare JSON reports.

Artifacts:
- Checkpoints + provenance: `checkpoints/sac_*` and `checkpoints/sac_her_*` with `*.meta.json`
- Evaluation JSON: `results/ch04_*_eval.json` (HER vs no-HER comparisons)

Suggested end-of-chapter exercises:
- Change `n_sampled_goal` and show how it affects success in sparse Push.
- Replace the goal selection strategy ("future" vs "final") and explain the outcome you observe.
- Write a one-paragraph "experiment card" that allows someone else to reproduce your results from scratch.
