# Chapter 2: PPO on Dense Reach -- The Pipeline Truth Serum

## Abstract

This chapter establishes a baseline that validates the entire training infrastructure before introducing algorithmic complexity. We train Proximal Policy Optimization (PPO) on FetchReachDense-v4, a task with continuous reward signal where success is achievable by straightforward policy gradient methods.

The choice of PPO on dense rewards is not arbitrary--it is a *diagnostic*. If PPO fails to learn on dense Reach, the fault lies in our pipeline (environment configuration, hyperparameters, network architecture), not in algorithm sophistication. This is the "truth serum" principle: use the simplest method that should work to verify that your infrastructure is correct.

By the end of this chapter, you will have:
1. A trained PPO checkpoint achieving >90% success rate on FetchReachDense-v4
2. A locked evaluation protocol that you will use for all future experiments
3. An understanding of PPO's core mechanism (clipped surrogate objective) and why it exists
4. Early warning diagnostics for common training pathologies

---

## Part 0: The Practical Context

### 0.1 Where We Are

You have completed Chapters 0 and 1. You now have:
- A verified Docker environment with GPU access and MuJoCo rendering
- Understanding of the Fetch observation structure: `observation`, `achieved_goal`, `desired_goal`
- Understanding of the action space: 4D Cartesian velocity commands
- Scripts that verify reward computation and interface consistency

Your infrastructure is ready. Now we must verify that *learning* works.

### 0.2 The Diagnostic Mindset

Consider the following debugging scenario. You implement SAC + HER on sparse Push, train for one million steps, and observe no learning. What went wrong?

The possibilities are numerous:
- Environment misconfiguration (wrong observation normalization, action scaling)
- Network architecture issues (wrong input dimensions, initialization)
- Hyperparameter problems (learning rate, entropy coefficient, buffer size)
- Algorithm bugs (incorrect HER relabeling, target network updates)
- Task difficulty (sparse Push may simply require more steps or different hyperparameters)

You cannot diagnose the problem because you have too many variables.

**The solution**: establish a baseline where learning *must* work if your infrastructure is correct. FetchReachDense-v4 with PPO is that baseline:
- **Dense rewards** provide continuous gradient signal--no exploration bottleneck
- **Reach** is the easiest Fetch task--no object interaction, just move the end-effector
- **PPO** is a well-understood on-policy method with robust default hyperparameters

If this combination fails, the problem is in your code, not your algorithm choice. If it succeeds, you have confidence that your training loop, environment wrapper, and evaluation code are correct.

### 0.3 What We Will Build

| Artifact | Description |
|----------|-------------|
| `checkpoints/ppo_FetchReachDense-v4_seed0.zip` | Trained PPO checkpoint |
| `checkpoints/ppo_FetchReachDense-v4_seed0.meta.json` | Training metadata (versions, hyperparams) |
| `results/ch02_ppo_reachdense_eval.json` | Evaluation report (100 episodes, deterministic) |
| Diagnostic plots | Value loss, entropy, KL divergence, success rate curves |

---

## Part 1: WHY -- Problem Formulation

### 1.1 The Reinforcement Learning Objective

We seek a policy $\pi_\theta: \mathcal{S} \times \mathcal{G} \to \Delta(\mathcal{A})$ that maximizes expected cumulative reward:

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t, s_{t+1}, g) \right]
```

where $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ is a trajectory sampled by executing policy $\pi_\theta$.

**Remark (Gradient Estimation).** The challenge is computing $\nabla_\theta J(\theta)$. The expectation is over trajectories, which depend on $\theta$ through the policy. The policy gradient theorem provides the key identity:

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t, g) \cdot A^{\pi_\theta}(s_t, a_t, g) \right]
```

where $A^{\pi_\theta}(s, a, g) = Q^{\pi_\theta}(s, a, g) - V^{\pi_\theta}(s, g)$ is the *advantage function*--how much better action $a$ is compared to the average action under $\pi_\theta$.

### 1.2 The Problem with Vanilla Policy Gradient

The policy gradient estimator is unbiased but has high variance. In practice, we estimate it from a finite batch of trajectories and take gradient steps. Two problems arise:

**Problem 1: Step Size Sensitivity.** If the learning rate is too large, a single update can catastrophically change the policy. The new policy may visit entirely different states, making the advantage estimates (computed under the old policy) invalid. This leads to instability and collapse.

**Problem 2: Sample Inefficiency.** After each gradient update, we must discard the collected data and sample new trajectories under the updated policy. This is because the policy gradient requires on-policy data--samples from the *current* policy, not a previous one.

### 1.3 Why PPO? -- Constraining Policy Updates

PPO addresses Problem 1 by *limiting how much the policy can change* in a single update. The key insight: instead of constraining the step size in parameter space (which doesn't correspond to meaningful behavior change), constrain the change in *action probabilities*.

Define the probability ratio:

```math
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
```

This measures how much more (or less) likely action $a_t$ is under the new policy versus the old. If $r_t(\theta) = 1$, the policies agree on this action. If $r_t(\theta) = 2$, the new policy is twice as likely to take this action.

**The Clipped Surrogate Objective.** PPO optimizes:

```math
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
```

where $\epsilon$ is typically 0.2.

**Interpretation.** Consider what this objective does:

| Advantage $A_t$ | Desired change | Clipping effect |
|-----------------|----------------|-----------------|
| $A_t > 0$ (good action) | Increase $r_t$ (take action more often) | Clips at $1+\epsilon$ (don't increase too much) |
| $A_t < 0$ (bad action) | Decrease $r_t$ (take action less often) | Clips at $1-\epsilon$ (don't decrease too much) |

The clipping prevents the policy from changing too drastically, even if the advantage estimate suggests a large change would be beneficial. This is conservative--we sacrifice some potential improvement for stability.

**Why This Works.** The clipping creates a "trust region" around the old policy. Within this region, we optimize freely. Outside, we stop. This prevents the catastrophic updates that plague vanilla policy gradient, while still allowing meaningful learning.

### 1.4 Why Dense Rewards First?

The FetchReachDense-v4 environment provides reward:

```math
R(s, a, s', g) = -\| g_{\text{achieved}}(s') - g \|_2
```

This is the negative Euclidean distance from the achieved goal (end-effector position) to the desired goal. Every step provides gradient signal: "you got closer" or "you got farther."

Compare to sparse rewards ($R = 0$ if success, $-1$ otherwise), where the agent receives no useful signal until it accidentally reaches the goal. With random initialization, the probability of reaching a specific 3D goal by chance is essentially zero.

**Dense rewards decouple exploration from learning.** If PPO fails on dense Reach, the problem is not exploration--it's our implementation. This is why dense rewards serve as a diagnostic.

### 1.5 Why Not SAC? Why Not HER?

We will use SAC + HER in later chapters. But they are inappropriate here for diagnostic purposes:

**SAC** adds entropy regularization and off-policy learning. More moving parts, more ways to fail. If SAC doesn't learn, is it the entropy coefficient? The target network update rate? The replay buffer? The Q-function architecture?

**HER** requires sparse rewards to demonstrate its value. On dense rewards, HER provides no benefit--the original transitions already have useful reward signal.

**Principle: Isolate variables.** Validate the simplest method first, then add complexity incrementally.

---

## Part 2: HOW -- The PPO Algorithm

### 2.1 Algorithm Overview

PPO is an *on-policy actor-critic* method. "On-policy" means we train on data collected by the current policy. "Actor-critic" means we maintain two function approximators:

- **Actor** $\pi_\theta(a|s, g)$: the policy, outputs action probabilities
- **Critic** $V_\phi(s, g)$: the value function, estimates expected return from state $s$ with goal $g$

The training loop:
1. Collect trajectories using current policy $\pi_\theta$
2. Compute advantages $A_t$ using the critic $V_\phi$
3. Update both actor and critic for multiple epochs on the collected batch
4. Discard data, repeat

### 2.2 Advantage Estimation: Generalized Advantage Estimation (GAE)

The advantage $A_t = Q(s_t, a_t) - V(s_t)$ tells us how much better action $a_t$ was compared to average. We don't know $Q$ directly, but we can estimate it from returns.

**TD residual**: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

This is a one-step advantage estimate (high bias, low variance).

**Monte Carlo return**: $\hat{A}_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k} - V(s_t)$

This is a full-trajectory advantage estimate (low bias, high variance).

**GAE** interpolates between these extremes:

```math
\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}
```

The parameter $\lambda \in [0, 1]$ controls the bias-variance tradeoff:
- $\lambda = 0$: TD residual only (high bias, low variance)
- $\lambda = 1$: Monte Carlo (low bias, high variance)
- $\lambda = 0.95$: typical default, good balance

### 2.3 The Value Loss

The critic is trained to minimize the squared error between predicted and observed returns:

```math
L^{V}(\phi) = \mathbb{E}_t \left[ (V_\phi(s_t) - \hat{R}_t)^2 \right]
```

where $\hat{R}_t$ is the discounted return from timestep $t$.

**Remark (Value Clipping).** SB3's PPO implementation optionally clips value updates similar to policy updates. This can help stability but is not always beneficial.

### 2.4 Entropy Bonus

PPO adds an entropy bonus to encourage exploration:

```math
L^{\text{total}}(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_1 L^{V}(\phi) + c_2 \mathcal{H}(\pi_\theta)
```

where $\mathcal{H}(\pi_\theta) = -\mathbb{E}[\log \pi_\theta(a|s)]$ is the entropy of the policy.

High entropy means the policy is stochastic (explores). Low entropy means the policy is deterministic (exploits). The coefficient $c_2$ (typically 0.01) prevents premature convergence to deterministic policies.

### 2.5 Hyperparameters That Matter

| Parameter | SB3 Default | Effect |
|-----------|-------------|--------|
| `n_steps` | 2048 | Trajectory length before update. Longer = lower variance, slower iteration |
| `batch_size` | 64 | Minibatch size for gradient updates |
| `n_epochs` | 10 | Number of passes over the batch per update |
| `learning_rate` | 3e-4 | Step size. Too high = instability, too low = slow learning |
| `clip_range` | 0.2 | The $\epsilon$ in clipped surrogate. Smaller = more conservative |
| `gae_lambda` | 0.95 | Bias-variance tradeoff in advantage estimation |
| `ent_coef` | 0.0 | Entropy bonus coefficient |
| `vf_coef` | 0.5 | Value loss coefficient |

For FetchReachDense-v4, SB3 defaults work well. We use `n_steps=1024` with 8 parallel environments, giving 8192 transitions per update.

---

## Part 3: WHAT -- Implementation and Deliverables

### 3.1 Running the Experiment

Execute the Week 2 experiment script:

```bash
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed 0
```

This runs three phases:
1. **Train**: PPO on FetchReachDense-v4 for 1M steps
2. **Evaluate**: 100 episodes with deterministic policy
3. **Report**: Generate JSON metrics and diagnostic summary

For a quick sanity check (fewer steps):

```bash
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py train --total-steps 100000
```

### 3.2 Individual Commands

**Training only:**
```bash
bash docker/dev.sh python train.py \
    --algo ppo \
    --env FetchReachDense-v4 \
    --seed 0 \
    --n-envs 8 \
    --total-steps 1000000
```

**Evaluation only:**
```bash
bash docker/dev.sh python eval.py \
    --ckpt checkpoints/ppo_FetchReachDense-v4_seed0.zip \
    --env FetchReachDense-v4 \
    --n-episodes 100 \
    --seeds 0-99 \
    --deterministic \
    --json-out results/ppo_reachdense_eval.json
```

**TensorBoard visualization:**
```bash
bash docker/dev.sh tensorboard --logdir runs --bind_all
```

Then open `http://localhost:6006` in your browser.

### 3.3 Expected Results

After 1M timesteps (~20 minutes on DGX with 8 envs), you should observe:

| Metric | Expected Value | Concern If |
|--------|----------------|------------|
| Success Rate | > 95% | < 80% |
| Mean Return | > -5 | < -15 |
| Final Goal Distance | < 0.02m | > 0.05m |
| Training Stability | Monotonic improvement after 200k steps | Large oscillations |

**Remark (What "Success" Means).** The Fetch environments define success as `goal_distance < 0.05` (5cm threshold). On dense Reach, a well-trained policy achieves distances < 0.02m.

### 3.4 Diagnostic Signals

Monitor these quantities in TensorBoard:

**Healthy Training:**
- `rollout/ep_rew_mean`: Steadily increasing (less negative)
- `train/entropy_loss`: Slowly decreasing (policy becoming more deterministic)
- `train/value_loss`: Initially high, decreases, then stabilizes
- `train/approx_kl`: Small (< 0.02 typically)
- `train/clip_fraction`: Non-zero but not 1.0 (some updates being clipped)

**Pathological Signs:**

| Signal | Symptom | Likely Cause |
|--------|---------|--------------|
| `approx_kl` spikes | Policy changing too fast | Learning rate too high |
| `clip_fraction` near 1.0 | All updates being clipped | clip_range too small or LR too high |
| `entropy_loss` collapses to 0 | Premature determinism | Increase ent_coef |
| `value_loss` explodes | Critic diverging | Reduce LR, check reward scale |
| `ep_rew_mean` flatlines | No learning | Check env, check obs normalization |

### 3.5 Deliverables Checklist

After running `scripts/ch02_ppo_dense_reach.py all`, verify:

- [ ] `checkpoints/ppo_FetchReachDense-v4_seed0.zip` exists
- [ ] `checkpoints/ppo_FetchReachDense-v4_seed0.meta.json` exists and contains versions
- [ ] `results/ch02_ppo_reachdense_eval.json` exists
- [ ] Success rate in eval report > 90%
- [ ] TensorBoard logs in `runs/ppo/FetchReachDense-v4/seed0/`

### 3.6 Understanding the Clipping Mechanism

To solidify understanding, consider this thought experiment:

**Scenario**: At timestep $t$, the old policy has $\pi_{\theta_{\text{old}}}(a_t | s_t) = 0.3$. After one gradient step, suppose $\pi_\theta(a_t | s_t) = 0.6$. The advantage is $A_t = +2$ (good action).

**Question**: What is $r_t(\theta)$? Is the update clipped?

**Answer**:
- $r_t(\theta) = 0.6 / 0.3 = 2.0$
- With $\epsilon = 0.2$, the clip range is $[0.8, 1.2]$
- $r_t = 2.0 > 1.2$, so the objective becomes $\min(2.0 \cdot 2, 1.2 \cdot 2) = \min(4, 2.4) = 2.4$
- The gradient through $r_t \cdot A_t$ is clipped--we don't get credit for making the action *that* much more likely

This prevents the policy from overshooting, even when the advantage suggests a large change is good.

---

## Part 4: Verification Exercises

### Exercise 2.1: Reproduce the Baseline

Run the full experiment with seed 0. Record:
1. Final success rate
2. Final mean return
3. Wall-clock training time
4. Steps per second

Compare to the expected values in Section 3.3.

### Exercise 2.2: Multi-Seed Validation

Run with seeds 0, 1, 2, 3, 4:

```bash
for seed in 0 1 2 3 4; do
    bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed $seed
done
```

Compute mean and standard deviation of success rate across seeds. A well-posed result should have std < 5%.

### Exercise 2.3: Ablation -- Clip Range

Modify `clip_range` from 0.2 to 0.1 and 0.4. Observe:
- Does training stability change?
- Does final performance change?
- Does `clip_fraction` in TensorBoard change?

### Exercise 2.4: Explain the Clipping (Written)

In your own words, explain:
1. What problem does PPO's clipping solve?
2. Why is clipping in probability ratio space better than clipping in parameter space?
3. What would happen if $\epsilon = 0$ (no change allowed)?
4. What would happen if $\epsilon = 1$ (large changes allowed)?

---

## Part 5: Common Failures and Debugging

### 5.1 "Training runs but success rate stays at 0%"

**Check**: Is the environment correctly configured?

```python
import gymnasium as gym
import gymnasium_robotics  # registers envs

env = gym.make("FetchReachDense-v4")
obs, info = env.reset()
print("Observation keys:", obs.keys())
print("Observation shape:", obs["observation"].shape)
print("Goal shape:", obs["desired_goal"].shape)
```

Expected: `observation` shape (10,), `desired_goal` shape (3,).

**Check**: Is the action space correct?

```python
print("Action space:", env.action_space)
print("Action shape:", env.action_space.shape)
```

Expected: Box(-1, 1, (4,)).

### 5.2 "Value loss explodes"

**Likely cause**: Reward scale mismatch.

FetchReachDense-v4 rewards are negative distances, typically in range [-1, 0]. If you accidentally multiply rewards or use a different reward function, the scale may be wrong.

**Check**: Log rewards during rollout and verify they're in expected range.

### 5.3 "KL divergence spikes"

**Likely cause**: Learning rate too high.

**Fix**: Reduce `learning_rate` from 3e-4 to 1e-4 or 3e-5.

### 5.4 "Training is very slow"

**Check**: Is GPU being used?

```bash
bash docker/dev.sh nvidia-smi
```

In TensorBoard, check `time/fps`. Expected: >3000 steps/sec with 8 envs on a modern GPU.

---

## Conclusion

This chapter established the baseline that validates our training infrastructure. PPO on FetchReachDense-v4 is the "truth serum"--if it works, our pipeline is correct; if it fails, we debug before adding complexity.

**Key Takeaways:**

1. **Diagnostic first**: Use the simplest method that should work to verify infrastructure
2. **PPO's mechanism**: Clipped surrogate objective prevents catastrophic policy updates
3. **Dense rewards**: Provide continuous gradient signal, decoupling exploration from learning
4. **Monitoring**: TensorBoard diagnostics reveal training health before final metrics

**What's Next:**

In Chapter 3, we introduce SAC on dense Reach. This adds:
- Off-policy learning (replay buffer)
- Maximum entropy objective (automatic exploration)
- Target networks (stability for Q-learning)

SAC is more complex but more sample-efficient. On dense Reach, both should succeed--this validates SAC before we add HER for sparse rewards.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438.

3. Stable Baselines3 PPO Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

4. Spinning Up PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
