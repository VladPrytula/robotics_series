# What I Learned From 3 Weeks of Watching a Robot Fail

*Notes on learning reinforcement learning by actually understanding what's happening.*

---

## The Problem With RL Tutorials

Open any reinforcement learning tutorial. You'll find something like this:

```python
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

Two lines. The robot learns. Magic.

Except when it doesn't. And in reinforcement learning, "doesn't work" is the default state. The field has a reproducibility problem that's well-documented: Henderson et al. (2018) showed that reported results often fail to replicate, even with the same code. Hyperparameters that work in one paper fail silently in another setting.

This isn't a skill issue--it's a structural one. RL algorithms are sensitive to details that tutorials gloss over: reward scaling, observation normalization, network initialization, the specific random seed. When something fails, the debugging surface is enormous.

The question isn't "why doesn't my code work?" That framing assumes the code is the problem. The deeper question is: "Under what conditions does this algorithm succeed, and have I established those conditions?"

## Asking Better Questions

There's a habit from applied math and physics that's surprisingly useful for ML: before trying to solve a problem, ask whether it's *well-posed*.

Three questions:

1. **Does a solution exist?**
2. **Is it unique?**
3. **Does it depend continuously on the input?**

These seem abstract--who cares about existence proofs when you just want a robot to pick up a block?

But they translate directly to reinforcement learning.

- **Existence:** Is there a neural network policy that can solve this task? (Not always obvious. Some tasks are impossible for certain architectures.)
- **Uniqueness:** Are there multiple qualitatively different solutions? (Often yes. Understanding this helps you interpret what your agent learned.)
- **Continuous dependence:** If I change the random seed slightly, do I get a similar policy? (Often no. This is why RL is notoriously unstable.)

Consider a common failure mode: you train a robot on a sparse-reward task, and it learns nothing. The instinct is to tune hyperparameters. But the right question is: "Is this problem well-posed? What are the necessary conditions for a solution to exist?"

With binary success/failure feedback, the probability of randomly reaching a goal in high-dimensional space is essentially zero. No gradient signal. No learning. The problem, as formulated, is ill-posed--no amount of hyperparameter tuning will fix it.

## A Different Approach

I'm trying to build a course that emphasizes understanding over recipes--**problem formulation before solution**.

The environment is real--a simulated Fetch robot arm, the same one used in research labs. The tasks are real--reaching, pushing, picking, placing. But the approach is different.

**Week 0** is not "run this code." It's "can you prove your experimental environment is reproducible?" We verify GPU access, physics simulation, rendering pipelines. We establish that the laboratory is functional before we do any experiments.

**Week 1** is not "train a policy." It's "what mathematical object are you seeking?" We dissect the environment interface: What exactly does the robot observe? What do actions mean? How is reward computed? We prove that the reward function can be evaluated for arbitrary goals--a necessary condition for the technique we'll use later.

Only in **Week 2** do we train anything. And even then, we start with dense rewards (continuous feedback) before attempting sparse rewards (binary success/failure). We establish baselines. We measure variance across seeds. We treat RL as experimental science, not alchemy.

## The Core Insight

**Hindsight Experience Replay (HER)** is not a trick or a hack. It's the mathematically principled solution to a well-posed problem.

The problem: sparse rewards provide no gradient signal when the goal is never reached.

The insight: a trajectory that fails to reach goal A still demonstrates how to reach wherever it ended up. Call that goal B.

The solution: relabel the trajectory. Store it twice--once as a failure for goal A, once as a success for goal B. Now you have gradient signal.

But HER only works if the environment provides three things:
1. Explicit separation of achieved and desired goals
2. A reward function that can be queried for arbitrary goals
3. A geometric success criterion

These aren't implementation details. They're mathematical preconditions. The Fetch environments satisfy them by design. Most environments don't. If you try to apply HER to an environment that doesn't expose these features, it will fail silently, and you'll spend three weeks wondering why.

## What This Curriculum Covers

**The Method:** SAC + HER (Soft Actor-Critic with Hindsight Experience Replay)

This isn't arbitrary. It follows from the problem structure:
- Continuous actions -> actor-critic (can't do argmax over continuous space)
- Sparse rewards -> off-policy (need to reuse failed trajectories)
- Goal conditioning -> HER (manufacture success from failure)
- Exploration -> maximum entropy (SAC's stochastic policies)

**The Tasks:**
- FetchReach: Move end-effector to target position
- FetchPush: Push object to target location
- FetchPickAndPlace: Pick up object, place at target

**The Infrastructure:**
- Docker containers for reproducibility
- Multi-seed experiments for statistical validity
- Version-controlled everything

**The Approach:**
- Try to derive algorithms from first principles
- Verify claims with experiments, not assumptions
- Aim for understanding, not just working code

## Who This Might Help

This curriculum assumes you know Python and basic probability. It doesn't assume you know RL--I try to derive everything from first principles (though I'm sure I've made mistakes along the way).

It might be useful if you've been frustrated by tutorials that work until they don't, or if you want to understand why algorithms work rather than just how to call them.

Fair warning: it's slow. The curriculum takes 10 weeks, and the chapters build on each other. If you need results fast, this isn't for you.

## Follow Along

The curriculum is open-source and in active development:

**GitHub:** [robotics_series](https://github.com/VladPrytula/robotics_series)
**Documentation:** [vladprytula.github.io/robotics_series](https://vladprytula.github.io/robotics_series/)

Chapters 0-1 are complete. Chapters 2-10 are in progress.

I'll be posting updates here every two weeks--what I've built, what I've learned, what I've gotten wrong. If you're working through the material, I want to hear from you. The goal is shared understanding, and that requires feedback.

---

*Next post: "Your First Robot Policy--And Why It Will Fail"*
