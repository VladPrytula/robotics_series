# Your First Robot Policy -- And Why Starting Simple Isn't Lazy

*Week 2 of the robotics curriculum: using PPO as a diagnostic tool.*

---

## The Temptation to Skip Ahead

You want to train a robot. The end goal is impressive: pick up objects, place them precisely, maybe eventually fold laundry or pack boxes.

The temptation is to jump straight to the hard stuff. Sparse rewards. Complex algorithms. The techniques in the latest papers.

Don't.

Here's what happens when you do:

> You implement a sophisticated algorithm on a difficult task. Train for 10 hours. Success rate: 0%. What went wrong?

The debugging surface is enormous:
- Environment misconfiguration
- Wrong network architecture
- Bad hyperparameters
- Bug in your implementation
- The algorithm needing more time
- A subtle numerical issue

You're debugging in the dark. Every hypothesis takes hours to test. Most lead nowhere.

## The Diagnostic Mindset

There's a better approach: **establish a baseline where failure is informative.**

Chapter 2 of the curriculum uses Proximal Policy Optimization (PPO)--a widely-used RL algorithm--on the simplest task: moving a robot arm to a target position with dense rewards (continuous feedback based on distance to goal).

Why this combination?

**PPO is well-understood.** It's been tested on thousands of tasks. If PPO fails on your setup, the problem is almost certainly in your infrastructure, not in PPO.

**Dense rewards provide continuous signal.** Every action either improves or worsens the situation. No exploration problem. No credit assignment over long horizons.

**Reaching is the simplest manipulation task.** No object dynamics. No contact physics. No grasp planning. Just: see goal, move there.

If this fails, you know exactly where to look.

## What 500,000 Training Steps Produces

![Robot reaching goals](../../videos/fetch_reach_demo_grid.gif)

*Watch the distance counter. No inverse kinematics--the robot learned to reach any 3D position through gradient descent.*

Our test run achieved 100% success rate. When the baseline works, you've validated:

- GPU access and CUDA configuration
- Physics simulation (MuJoCo) and rendering
- The training loop and logging pipeline
- Checkpoint saving and model evaluation
- The observation/action interface

Now you can add complexity--harder tasks, sparse rewards, more sophisticated algorithms--with confidence that failures are algorithmic, not infrastructural.

## What PPO Actually Does

The tutorial goes deep on the algorithm. Here's the core idea:

**The problem:** You want to improve your policy, but changing it too much based on limited data causes instability. The new policy might be worse, and now you're collecting bad data, which makes the next update worse--a death spiral.

**PPO's solution:** Constrain how much the policy can change in one update. Specifically, clip the probability ratio between new and old policies:

```
ratio = new_policy(action) / old_policy(action)
clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
```

If an action's probability would increase by more than 20% (with epsilon=0.2), PPO caps the update. Same for decreases. Small, stable steps.

This is "proximal" in the optimization sense--staying close to the current solution.

## The Actor-Critic Architecture

PPO uses two neural networks:

**Actor:** Given the state and goal, output what action to take (as a probability distribution).

**Critic:** Given the state and goal, estimate how good this situation is (expected future reward).

Why two networks instead of one? Different objectives. The actor wants to find good actions. The critic wants accurate predictions. Training them together can cause conflicting gradients. Separating them (while possibly sharing early layers) keeps the optimization cleaner.

There's a deeper perspective here--the separation might reflect geometric structure in the problem, factoring through a shared representation space--but the practical benefits are clear even without the theory.

## Try It Yourself

The full tutorial walks through:

1. **WHY** -- The learning problem and policy gradients
2. **HOW** -- PPO's clipped objective and GAE
3. **WHAT** -- Running the experiment and interpreting results

Everything runs in Docker with GPU support:

```bash
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py train --seed 42
```

Chapter 2: [github.com/VladPrytula/robotics_series/.../ch02_ppo_dense_reach.md](https://github.com/VladPrytula/robotics_series/blob/main/tutorials/ch02_ppo_dense_reach.md)

---

*Next post: Off-policy learning and why it matters for sparse rewards.*
