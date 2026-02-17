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
8.4 Summary

9. Evidence-driven tuning: ablations and sweeps
9.1 What "reproducible" means in practice (tolerances, hardware classes)
9.2 Minimal ablations that answer "what mattered?"
9.3 Reporting: experiment cards and comparable JSON metrics
9.4 Summary

Part 5 -- Pixels and the Reality Gap
10. Pixels, no cheating: visual Reach in the same task family
10.1 What changes when you remove privileged state
10.2 Rendering wrappers and observation design
10.3 A practical CNN encoder for RL
10.4 Measuring the sample-efficiency gap
10.5 Summary

11. Visual robustness that matters: augmentation as a tool
11.1 Visual brittleness and why it appears
11.2 Practical augmentations (random crop, color jitter) and why they help
11.3 Ablations: what helps, what hurts
11.4 Summary

12. Visual goals (optional advanced): "make it look like this"
12.1 Goals as images: promise and pitfalls
12.2 Evaluation protocols for high-dimensional goals
12.3 Summary

13. A reality-gap playbook: stress tests before hardware
13.1 Domain randomization: what to randomize and how to measure it
13.2 Sim-to-sim system identification as a controlled rehearsal
13.3 A deployment-readiness checklist backed by tests
13.4 Summary
