# Reddit Post (r/MachineLearning)

**Title:** [P] Chapter 2: PPO as a diagnostic tool before tackling hard RL problems

---

*Building toward robots that pack boxes, fold clothes, assemble products--but first, we need infrastructure that doesn't lie to us.*

**TL;DR:** Before debugging your algorithm, debug your pipeline. New chapter in the open-source robotics RL curriculum uses PPO on dense Reach as a "pipeline truth serum." If this baseline fails, the problem is infrastructure, not algorithms. Achieved 100% success rate.

**The problem this solves:**

You implement a sophisticated RL algorithm. Train for 10 hours. Success rate: 0%. What went wrong?

The debugging surface is enormous: environment config, network architecture, hyperparameters, implementation bugs, insufficient training time, numerical issues. You're debugging in the dark.

**The solution:**

Establish a baseline where failure is informative:
- **PPO (Proximal Policy Optimization):** Well-understood, stable, works with default hyperparameters
- **Dense rewards:** Continuous feedback, no exploration problem
- **Reach task:** Simplest manipulation--no object dynamics, no contacts

If PPO on dense Reach fails, the problem is definitely infrastructure. If it succeeds (ours hit 100%), you've validated GPU, physics sim, training loop, and eval pipeline.

**What the tutorial covers:**

- Policy gradient theorem (intuition before math)
- PPO's clipped surrogate objective and why it prevents update instability
- Actor-critic architecture and why two networks (different objectives, stability)
- GAE for advantage estimation
- Diagnostic checklist for common failures

**Results:**

[Video: 2x2 grid of trained policy reaching targets]

500k steps, ~15 min on GPU. 100% success rate on FetchReachDense-v4.

**Links:**

- Tutorial: [ch02_ppo_dense_reach.md](https://github.com/VladPrytula/robotics_series/blob/main/tutorials/ch02_ppo_dense_reach.md)
- Full curriculum: [github.com/VladPrytula/robotics_series](https://github.com/VladPrytula/robotics_series)

**What's next:**

Chapter 3 introduces off-policy methods (replay buffers, target networks) on the same task. This validates the off-policy machinery before Chapter 4 adds HER for sparse rewards.

Feedback welcome--especially on explanations that don't land or missing content.
