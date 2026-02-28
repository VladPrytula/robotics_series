# Proposed Table of Contents (In Action Style, Draft)

Part 1 -- Start Running, Start Measuring
1. Proof of life: a reproducible robotics RL loop
1.1 Why robotics RL fails silently
1.2 The task family: goal-conditioned Fetch manipulation
1.3 The experiment contract (training, evaluation, artifacts)
1.4 Running the proof-of-life pipeline
1.5 Reading results and debugging first failures
1.6 Summary

2. What the robot actually sees: observations, rewards, and success
2.1 The observation dictionary: observation, achieved_goal, desired_goal
2.2 Actions as Cartesian deltas and what that implies
2.3 Reward semantics and `compute_reward`
2.4 A metrics schema you will reuse (success, distance, smoothness)
2.5 Summary

Part 2 -- Baselines That Debug Your Pipeline
3. PPO as a lie detector (dense Reach)
3.1 Why dense rewards are diagnostic tools
3.2 PPO at a practical level (what it optimizes, what can go wrong)
3.3 A fixed evaluation protocol (seeds, determinism, JSON reports)
3.4 Debugging a flat curve (common causes and checks)
3.5 Summary

4. Off-policy without mystery (SAC on dense Reach)
4.1 Why off-policy methods matter for robotics
4.2 Replay buffers and what they change about learning
4.3 SAC intuition: entropy, critics, and target networks
4.4 Diagnostics: rewards, Q-values, entropy coefficient, goal distances
4.5 Summary

Part 3 -- Sparse Goals, Real Progress
5. Learning from failure with HER (sparse Reach and Push)
5.1 Sparse rewards and the exploration barrier
5.2 HER prerequisites: explicit goals and computable rewards
5.3 Baseline: SAC without HER (and why it stalls)
5.4 HER mechanism: relabeling and goal selection strategies
5.5 Verification: evidence that HER is working
5.6 Summary

6. Capstone manipulation: PickAndPlace with an honest report card
6.1 Why contact-rich manipulation is harder than Reach/Push
6.2 Dense-debugging vs sparse truth (and how not to fool yourself)
6.3 Curriculum and stress-test splits
6.4 Deliverable-grade evaluation and experiment cards
6.5 Summary

Part 4 -- Engineering-Grade Robotics RL
7. Policies as controllers: stability, smoothness, and action interfaces
7.1 Controller-centric metrics (beyond return)
7.2 Action scaling, clipping, and optional filtering
7.3 Time-to-success and "oscillation" as measurable quantities
7.4 Summary

8. Robustness curves: quantify brittleness
8.1 Why videos are not evidence
8.2 Observation noise, action noise, and controlled perturbations
8.3 Degradation curves with confidence bands across seeds
8.4 Experiment cards: formalizing the .meta.json pattern
8.5 Summary

Part 5 -- Pixels and the Reality Gap
9. Pixels, no cheating: from Reach to Push
9.1 What changes when you remove privileged state
9.2 Rendering wrappers and observation design (goal modes)
9.3 A practical CNN encoder for RL (NatureCNN)
9.4 Measuring the pixel penalty on Reach (state vs pixel vs DrQ)
9.5 Why Reach is too easy (and Push is "deceptively dense")
9.6 The bridge: HER + pixels with goal_mode="both"
9.7 Push from pixels: combining HER, DrQ, and visual observations
9.8 Making it fast: rendering as the bottleneck
9.9 Summary

10. The reality gap: stress tests before hardware
10.1 Domain randomization: what to randomize and how to measure it
10.2 Visual robustness: augmentation ablations (DrQ vs crop vs jitter)
10.3 Sim-to-sim system identification as a controlled rehearsal
10.4 A deployment-readiness checklist backed by tests
10.5 Summary

Part 6 -- Bonus Material (Online, Optional)
These are optional appendices. Goal: cover modern "frontier" ideas without
requiring lab-scale compute. Each appendix should have:
- a fast path (<= 500k env steps), and
- a checkpoint track (evaluate a provided model without training).

Appendix A -- Automated curriculum learning you can verify (PickAndPlace)
The point is not "a schedule" but a measurable shift in performance on hard
cases (stratified eval bins), under a fixed evaluation protocol.
A.1 The problem: manual curricula are tedious and brittle
A.2 Define a difficulty axis (table vs air goals, goal range, noise)
A.3 Two low-compute baselines: linear vs success-gated schedules
A.4 Optional: prioritized goal sampling (PLR-inspired, lightweight)
A.5 Deliverables: curriculum config in metadata + stratified results JSON
A.6 Summary

Appendix B -- Empowerment and unsupervised skill discovery (DIAYN-first)
HER is transformative, but it needs "meaningful interaction" to relabel. When
the agent never touches the object, we need objectives that create behavior
before the task reward is informative.
B.1 The limits of HER and random exploration
B.2 Skill discovery via mutual information (DIAYN: skills -> distinguishable states)
B.3 Low-compute hands-on: FetchReachDense (state-only) skill atlas (200k-500k steps)
B.4 Optional: zero-shot / low-shot downstream (sparse Reach fine-tune, 100k-200k)
B.5 Alternative (short): intrinsic curiosity (ICM) + coverage metrics
B.6 Summary

Appendix C -- Beyond MuJoCo: port the contract to PyBullet (PandaGym)
This appendix exists to fight simulator monotony and to prove a claim: the
experiment contract (train/eval/artifacts) is more portable than any single
engine. We keep it state-based to stay low-compute.
C.1 The invariant: goal-conditioned obs + recomputable rewards + success metrics
C.2 Setup: install a PyBullet goal-conditioned suite (e.g., PandaGym)
C.3 Dense baseline: Reach in PyBullet (prove the pipeline still learns)
C.4 Sparse + HER: Push/Pick in PyBullet (show the same HER separation)
C.5 Deliverables: same JSON reports, same .meta.json provenance, same plots
C.6 Summary

Appendix D -- World models and latent imagination (bonus-bonus)
This is intentionally scoped: a minimal demo plus a clear boundary. Full
Dreamer-style training loops are powerful but would require a second stack.
D.1 The rendering bottleneck (why pixels are FPS-limited)
D.2 Minimal demo: collect frames -> train a small VAE (latent compression)
D.3 Scope boundary: pointers to Dreamer/DayDreamer-style methods
D.4 Summary

Appendix E -- Isaac Lab Manipulation (GPU-only optional)
This appendix demonstrates portability: the same SAC methodology transfers to
Isaac Lab with GPU-parallel physics providing 15-170x wall-clock speedups. It is
optional because Isaac Lab requires Linux + NVIDIA GPU. Readers without that
hardware can still use the checkpoint track to run evaluation and inspect metrics.
E.1 Why this appendix exists: portability evidence and GPU-parallel scaling
E.2 Build It: SAC core for dict observations + goal relabeling
E.3 Run It: Lift-Cube training (state-based: 256 envs, ~9K fps, 14 min)
E.4 Honest difficulty comparison: Lift-Cube vs FetchPickAndPlace
E.5 Pixels on Isaac: TiledCamera results (64 envs, ~1.2K fps)
E.6 What can go wrong: curriculum crash, singleton, obs mismatch
E.7 Summary
