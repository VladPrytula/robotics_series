# Scaffold: Chapter 3 -- PPO on Dense Reach

## Classification
Type: Algorithm
Source tutorial: tutorials/ch02_ppo_dense_reach.md
Book chapter output: Manning/chapters/ch03_ppo_dense_reach.md
Lab file: scripts/labs/ppo_from_scratch.py (existing; all regions present)
Production script: scripts/ch02_ppo_dense_reach.py (existing; the "Run It" reference)

---

## Experiment Card

```
---------------------------------------------------------
EXPERIMENT CARD: PPO on FetchReachDense-v4
---------------------------------------------------------
Algorithm:    PPO (clipped surrogate, on-policy)
Environment:  FetchReachDense-v4
Fast path:    500,000 steps, seed 0
Time:         ~5 min (GPU) / ~30 min (CPU)

Run command (fast path):
  bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all \
    --seed 0 --total-steps 500000

Checkpoint track (skip training):
  checkpoints/ppo_FetchReachDense-v4_seed0.zip

Expected artifacts:
  checkpoints/ppo_FetchReachDense-v4_seed0.zip
  checkpoints/ppo_FetchReachDense-v4_seed0.meta.json
  results/ch02_ppo_fetchreachdense-v4_seed0_eval.json
  runs/ppo/FetchReachDense-v4/seed0/    (TensorBoard logs)

Success criteria (fast path):
  success_rate >= 0.90
  mean_return > -10.0
  final_distance_mean < 0.02

Full multi-seed results: see REPRODUCE IT at end of chapter.
---------------------------------------------------------
```

---

## Reproduce It Block

```
---------------------------------------------------------
REPRODUCE IT
---------------------------------------------------------
The results and pretrained checkpoints in this chapter
come from these runs:

  for seed in 0 1 2; do
    bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all \
      --seed $seed --total-steps 1000000
  done

Hardware:     Any modern GPU (tested on NVIDIA GB10; CPU works but ~6x slower)
Time:         ~8 min per seed (GPU), ~45 min per seed (CPU)
Seeds:        0, 1, 2

Artifacts produced:
  checkpoints/ppo_FetchReachDense-v4_seed{0,1,2}.zip
  checkpoints/ppo_FetchReachDense-v4_seed{0,1,2}.meta.json
  results/ch02_ppo_fetchreachdense-v4_seed{0,1,2}_eval.json
  runs/ppo/FetchReachDense-v4/seed{0,1,2}/

Results summary (what we got):
  success_rate: 1.00 +/- 0.00  (3 seeds x 100 episodes)
  return_mean:  -0.40 +/- 0.05
  final_distance_mean: 0.005 +/- 0.001

If your numbers differ by more than ~5%, check the
"What Can Go Wrong" section above.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

---

## Build It Components

This is a full Algorithm chapter. The reader derives PPO, implements it
component by component from scratch, verifies each piece, then wires them
together. All lab regions already exist in `scripts/labs/ppo_from_scratch.py`.

| # | Component | Equation / concept | Lab file:region | Verify check |
|---|-----------|-------------------|-----------------|--------------|
| 1 | Actor-Critic Network | Actor $\pi_\theta(a \mid x)$ as Gaussian with learned mean/std; Critic $V_\phi(x)$ as scalar output; shared backbone with separate heads | `labs/ppo_from_scratch.py:actor_critic_network` | Instantiate with obs_dim=16, act_dim=4; forward pass produces dist.mean shape (1,4), value shape (1,); total params ~5,577; all outputs finite |
| 2 | GAE Computation | $\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$ where $\delta_t = r_t + \gamma(1-d_t)V(x_{t+1}) - V(x_t)$ | `labs/ppo_from_scratch.py:gae_computation` | T=10, reward=1.0 at last step, done=1.0 at last step: advantages[-1] > 0 (got unexpected reward); all advantages and returns finite; returns = advantages + values |
| 3 | PPO Clipped Loss | $L^{\text{CLIP}} = \mathbb{E}_t[\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]$ where $\rho_t = \pi_\theta(a_t \mid x_t) / \pi_{\theta_\text{old}}(a_t \mid x_t)$ | `labs/ppo_from_scratch.py:ppo_loss` | Same model, same params -> clip_fraction = 0.000, ratio_mean = 1.000, approx_kl ~ 0.000; loss is finite |
| 4 | Value Loss | $L_\text{value} = \frac{1}{2}\mathbb{E}[(V_\phi(x_t) - \hat{G}_t)^2]$ where $\hat{G}_t = \hat{A}_t + V_\text{rollout}(x_t)$ | `labs/ppo_from_scratch.py:value_loss` | Near-zero value predictions vs random returns -> value_loss ~ 0.5, explained_variance ~ 0.0; loss is finite |
| 5 | PPO Update (wiring) | $\mathcal{L} = -L^{\text{CLIP}} + c_1 L_\text{value} - c_2 \mathcal{H}[\pi]$ with $c_1=0.5$, $c_2=0.0$; gradient clipping at max_grad_norm=0.5 | `labs/ppo_from_scratch.py:ppo_update` | 10 updates on mock batch: value_loss decreases from initial; approx_kl < 0.05; total_loss and grad_norm are finite |
| 6 | Training Loop (rollout + batch assembly) | Collect n_steps transitions, compute GAE, assemble minibatches, run n_epochs of updates | `labs/ppo_from_scratch.py:ppo_training_loop` | `--demo` mode: CartPole-v1 solves (avg return >= 195) within 50k steps (~30 sec on CPU); value_loss shows decreasing trend |

**Ordering rationale:** Foundation first -- network architecture (1) before the
losses that operate on it (2-4). GAE (2) before PPO loss (3) because advantages
are an input to the clipped objective. Value loss (4) is independent of PPO loss
but shown after to maintain the "actor then critic" narrative flow. Wiring (5)
assembles components 2-4. Training loop (6) uses the wiring step to show
end-to-end learning.

---

## Bridging Proof

The bridge connects the from-scratch implementation to SB3, proving they
compute the same quantities. The existing lab supports this via `--compare-sb3`.

- **Inputs (same data fed to both):**
  Random rewards, values, and dones of horizon=64, generated with seed=0.
  Same gamma=0.99, gae_lambda=0.95 for both implementations.

- **From-scratch output (lab code):**
  `compute_gae()` from `ppo_from_scratch.py` produces advantages and returns
  tensors of shape (64,).

- **SB3 output:**
  `RolloutBuffer.compute_returns_and_advantage()` from stable-baselines3
  produces advantages and returns arrays of shape (64,).

- **Match criteria:**
  - `max_abs_advantage_diff <= 1e-6` (exact float match within atol)
  - `max_abs_returns_diff <= 1e-6` (exact float match within atol)
  - Both implementations handle episode boundaries (done masks) identically

- **Lab mode:** `--compare-sb3`
  ```bash
  bash docker/dev.sh python scripts/labs/ppo_from_scratch.py --compare-sb3
  ```
  Expected output:
  ```
  PPO From Scratch -- SB3 Comparison
  Max abs advantage diff: ~0
  Max abs returns diff:   ~0
  [PASS] Our GAE matches SB3 RolloutBuffer
  ```

- **Narrative bridge (for the writer):**
  After showing the match, explain what SB3 adds beyond our from-scratch code:
  vectorized environments (n_envs parallel rollouts), learning rate scheduling,
  multi-epoch minibatch shuffling, and MultiInputPolicy for dict observations.
  Map SB3 TensorBoard metrics to our code: `train/value_loss` is our
  `compute_value_loss`, `train/clip_fraction` is from our `compute_ppo_loss`,
  `train/approx_kl` is the same KL approximation.

---

## What Can Go Wrong

| Symptom | Likely cause | Diagnostic |
|---------|-------------|------------|
| `ep_rew_mean` flatlines near -20 for entire training | Environment misconfigured (wrong obs/action shapes) or policy network not receiving goal info | `print({k: v.shape for k, v in obs.items()})` -- should show observation (10,), achieved_goal (3,), desired_goal (3,); verify MultiInputPolicy is used, not MlpPolicy |
| Success rate stays at 0% after 200k steps | Wrong environment ID (using sparse FetchReach-v4 instead of FetchReachDense-v4) | Check env_id string; print reward from a random step (dense should be in [-1, 0], sparse is exactly 0 or -1) |
| `value_loss` explodes (>100) early in training | Reward scale mismatch or returns not computed correctly | Check reward range with random policy (should be in [-1, 0] for FetchReachDense); verify GAE returns are in [-50, 0] range |
| `approx_kl` consistently > 0.05 | Learning rate too high; policy changing too fast per update | Reduce learning_rate from 3e-4 to 1e-4; or reduce n_epochs from 10 to 5 |
| `clip_fraction` near 1.0 every update | Update steps too aggressive; all actions being clipped | Reduce learning_rate; reduce clip_range from 0.2 to 0.1; check that advantages are normalized |
| `clip_fraction` always 0.0 | Policy not learning (learning_rate too low or stuck at initialization) | Increase learning_rate; verify optimizer is attached to model parameters; check grad_norm is nonzero |
| `entropy_loss` immediately goes to 0 | Policy collapsed to deterministic (log_std went to -inf) | Add entropy coefficient (ent_coef=0.01); check initial log_std parameter value |
| Training very slow (< 300 fps on GPU) | Not using GPU, or n_envs too low | Check `nvidia-smi` shows python process; increase n_envs from 1 to 8; note: 500-1300 fps is typical even with GPU (CPU-bound MuJoCo) |
| `--compare-sb3` shows mismatch > 1e-6 | Episode boundary handling differs (done mask applied differently) | Check that dones mask is applied to both next_value and last_gae in the GAE loop; verify episode_starts alignment with SB3 convention |
| Build It `--verify` fails on "Value loss should decrease" | Random seed produced an adversarial batch | Re-run; if persistent, check that optimizer.step() is being called and model parameters are changing |

---

## Adaptation Notes

### Cut from tutorial

- **Part 0 entirety (Sections 0.1-0.3, ~600 words):** "Setting the Stage" is
  informal preamble. The task hierarchy table and "diagnostic mindset" are
  covered more concisely in the Chapter Bridge and WHY section. Cut the
  "Option A vs Option B" framing (too elementary for Manning audience).

- **Part 4 "Understanding What You Built" (Sections 4.1-4.3, ~500 words):**
  Section 4.1 (what the policy learned) repeats Ch2 observation structure.
  Section 4.2 (clipping in action) is better integrated into the Build It
  checkpoint for PPO loss (component 3). Section 4.3 (pipeline validation)
  is the chapter's conclusion, not a standalone section. Fold 4.3 into the
  Summary.

- **Exercise 2.3 "Explain the Clipping (Written)":** Written-answer exercises
  don't fit Manning's hands-on format. Replace with a code-based clipping
  ablation.

- **The animated GIF reference and demo grid image:** These are web-specific.
  Replace with a description of what the trained policy does, or a static
  figure placeholder.

- **References section:** Move to inline citations within the text. Manning
  chapters don't typically have a standalone references section.

- **"If you want to run first and read later" jump-ahead link:** Web-specific
  navigation. Drop for linear book format.

- **Optional aside: geometric perspective on actor-critic (the collapsible
  details block):** Too tangential. Cut entirely.

### Keep from tutorial

- **Section 1.1-1.6 (WHY, ~1800 words):** The full derivation from reward ->
  return -> policy gradient -> advantage -> instability -> PPO clipping. This
  is the chapter's mathematical spine. Keep the progression and the concrete
  numerical example (Section 1.5). Trim slightly for Manning voice.

- **Section 2.1-2.3 (HOW, ~1200 words):** Actor-critic architecture rationale
  (three reasons for separate heads), training loop pseudocode, hyperparameter
  table. Keep all of this; it sets up Build It.

- **Section 2.5 (Build It, ~2200 words):** All six components with
  snippet-includes and verification checkpoints. This is the narrative spine.
  Keep the math-before-code pattern and the checkpoint blocks. Tighten the
  prose around each listing.

- **Section 2.5.7 (SB3 comparison):** Keep as the Bridge section; this is
  the bridging proof.

- **Section 3 (Run It, ~1200 words):** Experiment card, commands, TensorBoard
  interpretation table, milestone table, artifact paths. Keep and adapt to
  Manning format.

- **Section 6 (Common Failures, ~400 words):** Keep and expand into the
  "What Can Go Wrong" section with the full table above.

- **GAE equation summary block (Section 2.2):** The compact equation summary
  is a useful reference. Keep as a sidebar or callout.

### Add for Manning

- **Chapter Bridge (from Ch2):** Ch2 established the environment anatomy --
  observations, actions, rewards, goals. Gap: we understand the environment
  but have not trained a policy. Ch3 adds PPO training, from-scratch
  implementation, and pipeline validation. Foreshadow: PPO is on-policy
  and uses dense rewards; Ch4 introduces SAC (off-policy) for better sample
  efficiency, preparing for sparse rewards in Ch5.

- **Opening Promise:** 3-5 bullet "This chapter covers" block.

- **Explicit "on-policy" definition and discussion:** The tutorial mentions
  on-policy in passing (Section 2.2 Step 5). For Manning, define on-policy
  learning formally as a concept registry entry: data must come from the
  current policy; after each update, old data is discarded. Contrast with
  off-policy (preview of Ch4). This is critical because the PPO->SAC
  transition in Ch4 depends on the reader understanding this limitation.

- **TensorBoard metric-to-code mapping table:** After the Bridge section,
  add an explicit table mapping SB3 TensorBoard log keys to the from-scratch
  code: `train/value_loss` -> `compute_value_loss`, `train/clip_fraction` ->
  `compute_ppo_loss` info dict, `train/approx_kl` -> same, `train/entropy_loss`
  -> `dist.entropy()`. This helps readers connect monitoring to understanding.

- **Exercises 3-5 (see Exercises section below):** Graduated exercises specific
  to Manning format.

---

## Chapter Bridge

1. **Capability established:** Chapter 2 dissected the Fetch environment --
   observation dictionaries, action semantics, dense and sparse reward
   computation, goal relabeling, and the random-policy baseline (0% success,
   mean return ~ -20). You understand what the agent sees and what the numbers
   mean.

2. **Gap:** Understanding the environment is necessary but not sufficient. A
   random policy achieves 0% success. You need an algorithm that converts
   observations into intelligent actions -- one that improves through
   experience. But which algorithm, and how do you verify it is working?

3. **This chapter adds:** PPO (Proximal Policy Optimization), an on-policy
   algorithm that learns by clipping likelihood ratios to prevent destructive
   updates. You will derive the PPO objective, implement it from scratch
   (actor-critic network, GAE, clipped loss, value loss), verify each
   component, bridge to SB3, and train a policy that reaches 100% success
   on FetchReachDense-v4. This validates your entire training pipeline.

4. **Foreshadow:** PPO works here because dense rewards provide continuous
   gradient signal. But PPO is on-policy: it discards all data after each
   update, wasting expensive simulation time. Chapter 4 introduces SAC, an
   off-policy algorithm that stores and reuses experience in a replay buffer.
   That off-policy machinery is required for Chapter 5 (HER), where we tackle
   sparse rewards.

---

## Opening Promise

> **This chapter covers:**
>
> - Deriving the PPO clipped surrogate objective from the policy gradient
>   theorem -- why constraining the likelihood ratio prevents the catastrophic
>   updates that plague vanilla policy gradient
> - Implementing PPO from scratch: actor-critic network, Generalized Advantage
>   Estimation (GAE), clipped policy loss, value loss, and the full update loop
> - Verifying each component with concrete checks (tensor shapes, expected
>   values, learning curves) before assembling the complete algorithm
> - Bridging from-scratch code to Stable Baselines 3 (SB3): confirming that
>   both implementations compute the same GAE advantages, and mapping SB3
>   TensorBoard metrics to the code you wrote
> - Training PPO on FetchReachDense-v4 to 100% success rate, establishing
>   the pipeline baseline that every future chapter builds on

---

## Figure Plan

| # | Description | Type | Source command | Chapter location |
|---|------------|------|---------------|-----------------|
| 1 | FetchReachDense-v4 annotated screenshot: arm reaching toward target, labeled goal and gripper positions | screenshot | `python scripts/capture_proposal_figures.py env-setup --envs FetchReach-v4` | After opening bridge / Section 3.1 WHY, establishing the visual context |
| 2 | PPO clipping diagram: ratio vs objective plot showing clipped region for A>0 and A<0 | diagram | matplotlib in Build It text (illustrative) | After Section 3.5 (PPO clipped loss), illustrating the clipping mechanism |
| 3 | Learning curve (from-scratch demo): episode return over training steps on CartPole | curve | `python scripts/labs/ppo_from_scratch.py --demo` output | After Section 3.8 (Build It training loop), showing the from-scratch implementation learns |

---

## Estimated Length

| Section | Words |
|---------|-------|
| Opening promise + chapter bridge | 450 |
| 3.1 WHY: The learning problem (return, policy gradient, instability, PPO clipping, dense rewards, concrete example) | 1,800 |
| 3.2 HOW: Actor-critic architecture, training loop, hyperparameters | 1,000 |
| 3.3 Build It: Actor-Critic Network (component 1) | 500 |
| 3.4 Build It: GAE computation (component 2) | 600 |
| 3.5 Build It: PPO clipped loss (component 3) | 600 |
| 3.6 Build It: Value loss (component 4) | 400 |
| 3.7 Build It: PPO update wiring (component 5) | 500 |
| 3.8 Build It: Training loop demo (component 6) | 400 |
| 3.9 Bridge: From-scratch vs SB3 comparison | 500 |
| 3.10 Run It: Experiment card, commands, TensorBoard, milestones, artifacts | 1,200 |
| 3.11 What Can Go Wrong | 800 |
| 3.12 Summary + bridge to Ch4 | 500 |
| Reproduce It block | 300 |
| Exercises (5 exercises) | 700 |
| **Total** | **~9,250** |

(Target range: 6,000-10,000 words. Code listings counted separately by
Manning but included in overall page count estimate.)

---

## Concept Registry Additions

Terms this chapter introduces (to be added to the Concept Registry under Ch2,
which maps to Manning Ch3):

- **return $G_t$**: $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$, the discounted sum of future rewards from timestep $t$
- **discount factor $\gamma$** (formalized): $\gamma \in [0,1)$, controls how much future rewards are valued; $\gamma=0.99$ in our experiments
- **time horizon $T$**: maximum number of timesteps per episode; $T=50$ for FetchReach
- **expected discounted return $J(\theta)$**: $J(\theta) = \mathbb{E}[\sum_{t=0}^{T-1} \gamma^t r_t]$, the objective function
- **policy $\pi(a \mid s)$**: probability distribution over actions conditioned on state; parameterized by $\theta$
- **advantage function $A(s,a)$**: $A(s,a) = Q(s,a) - V(s)$; how much better action $a$ is vs. average
- **Q-function $Q(s,a)$**: expected return after taking action $a$ in state $s$ and following $\pi$
- **value function $V(s)$**: expected return from state $s$ under policy $\pi$
- **policy gradient theorem**: $\nabla_\theta J(\theta) = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid x_t) A(x_t, a_t)]$
- **probability ratio $\rho_t(\theta)$**: $\rho_t = \pi_\theta(a_t \mid x_t) / \pi_{\theta_\text{old}}(a_t \mid x_t)$; measures how action likelihood changed
- **PPO clipping ($L^{\text{CLIP}}$)**: $\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)$; constrains policy updates
- **actor-critic architecture**: two networks (or heads) -- actor for policy, critic for value estimation
- **Generalized Advantage Estimation (GAE)**: $\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}$; bias-variance tradeoff in advantage estimation
- **TD residual $\delta_t$**: $\delta_t = r_t + \gamma(1-d_t)V(x_{t+1}) - V(x_t)$; was this transition better than expected?
- **lambda parameter**: $\lambda \in [0,1]$ in GAE; $\lambda=0$ is one-step TD, $\lambda=1$ is Monte Carlo
- **on-policy learning**: data must come from the current policy; old data is discarded after each update

These match the Concept Registry entries for "Ch2" in the root CLAUDE.md.

---

## Dependencies

- **Lab regions needed (for Lab Engineer):**
  All regions already exist in `scripts/labs/ppo_from_scratch.py`:
  - `actor_critic_network` (lines 67-108) -- ActorCritic class
  - `gae_computation` (lines 128-187) -- compute_gae function
  - `ppo_loss` (lines 194-265) -- compute_ppo_loss function
  - `value_loss` (lines 268-327) -- compute_value_loss + explained_variance
  - `ppo_update` (lines 330-408) -- ppo_update function
  - `ppo_training_loop` (lines 537-637) -- collect_rollout + transitions_to_batch
  - Modes: `--verify` (runs all checks, < 2 min CPU), `--demo` (CartPole training,
    ~30 sec CPU), `--compare-sb3` (GAE comparison with SB3 RolloutBuffer)
  - SB3 comparison helper: `scripts/labs/sb3_compare.py` (compare_ppo_gae_to_sb3)
  - **No new lab code needed.** Lab Engineer should verify existing regions
    work correctly and consider adding a `--bridge` alias if it doesn't exist
    (currently the mode is called `--compare-sb3`).

- **Pretrained checkpoints needed (for Reproduce It):**
  - `checkpoints/ppo_FetchReachDense-v4_seed0.zip`
  - `checkpoints/ppo_FetchReachDense-v4_seed1.zip`
  - `checkpoints/ppo_FetchReachDense-v4_seed2.zip`
  - Generated by: `bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed {0,1,2}`

- **Previous chapter concepts used (from Manning Ch2 / tutorial Ch1):**
  - goal-conditioned MDP, goal-conditioned observation, dense reward, sparse reward
  - dictionary observation structure (observation, achieved_goal, desired_goal)
  - compute_reward API, critical invariant (reward recomputation)
  - success threshold epsilon = 0.05
  - The three diagnostic questions (from Manning Ch1)
  - Docker dev.sh workflow (from Manning Ch1)
  - Random-policy baseline: 0% success, mean return ~ -20 (from Manning Ch2)

- **Production script used:** `scripts/ch02_ppo_dense_reach.py` (existing; no changes needed)

---

## Exercises

**1. (Verify) Reproduce the single-seed baseline.**

Run the fast path command and verify your results match:

```bash
bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all \
  --seed 0 --total-steps 500000
```

Check the eval JSON: success_rate should be >= 0.90, mean_return > -10.
Record your training time and steps/second for comparison with Ch4 (SAC).

**2. (Tweak) GAE lambda ablation.**

In the lab verification, modify gae_lambda and observe the effect:
- `gae_lambda = 0.0` (one-step TD, high bias)
- `gae_lambda = 0.5` (midpoint)
- `gae_lambda = 1.0` (Monte Carlo, high variance)

Question: How do the advantage magnitudes change? Why does lambda=0 produce
smaller magnitude advantages? Expected: lambda=0 advantages are dominated by
single TD residuals; lambda=1 advantages accumulate over the full trajectory,
producing larger magnitudes and more variance.

**3. (Tweak) Clip range ablation.**

Train PPO with different clip_range values (0.1, 0.2, 0.4) for 500k steps each:

```bash
# Requires modifying the --ppo-clip-range flag in train.py (if supported)
# or editing the script temporarily
```

Compare: (a) final success rate, (b) training stability (watch approx_kl in
TensorBoard), (c) clip_fraction. Expected: 0.1 is more conservative (slower
but stable); 0.4 is more aggressive (may be faster but risk instability).

**4. (Extend) Add training time tracking.**

Modify the eval report to include wall-clock training time and compute
steps/second. Compare across GPU and CPU. Expected: GPU is ~2-5x faster
(the bottleneck is CPU-bound MuJoCo, not GPU-bound neural network ops).

**5. (Challenge) Train the from-scratch implementation on FetchReachDense.**

Extend the `--demo` mode in `ppo_from_scratch.py` to work with
`FetchReachDense-v4` instead of CartPole. You will need to handle dictionary
observations (concatenate observation + desired_goal as the network input) and
continuous actions. Does it learn? How does it compare to SB3 in sample
efficiency?

Expected: It should learn but more slowly than SB3 (SB3 uses vectorized envs,
optimized rollout storage, and proper obs preprocessing). Success rate may
reach 50-80% in 500k steps with a single environment.
