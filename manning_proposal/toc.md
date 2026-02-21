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
