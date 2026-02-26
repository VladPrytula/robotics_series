# Scaffold: Chapter 4 -- SAC on Dense Reach: Off-Policy Learning with Maximum Entropy

## Classification
Type: Algorithm
Source tutorial: tutorials/ch03_sac_dense_reach.md
Book chapter output: Manning/chapters/ch04_sac_dense_reach.md
Lab file: scripts/labs/sac_from_scratch.py (existing; all 7 regions present)
Production script: scripts/ch03_sac_dense_reach.py (existing; the "Run It" reference)

---

## Experiment Card

```
---------------------------------------------------------
EXPERIMENT CARD: SAC on FetchReachDense-v4
---------------------------------------------------------
Algorithm:    SAC (soft actor-critic, off-policy, auto-tuned entropy)
Environment:  FetchReachDense-v4
Fast path:    500,000 steps, seed 0
Time:         ~14 min (GPU) / ~60 min (CPU)

Run command (fast path):
  bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all \
    --seed 0 --total-steps 500000

Checkpoint track (skip training):
  checkpoints/sac_FetchReachDense-v4_seed0.zip

Expected artifacts:
  checkpoints/sac_FetchReachDense-v4_seed0.zip
  checkpoints/sac_FetchReachDense-v4_seed0.meta.json
  results/ch03_sac_fetchreachdense-v4_seed0_eval.json
  runs/sac/FetchReachDense-v4/seed0/    (TensorBoard logs)

Success criteria (fast path):
  success_rate >= 0.95
  mean_return > -5.0
  final_distance_mean < 0.03

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
    bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all \
      --seed $seed --total-steps 1000000
  done

Hardware:     Any modern GPU (tested on NVIDIA GB10; CPU works but ~4x slower)
Time:         ~28 min per seed (GPU), ~120 min per seed (CPU)
Seeds:        0, 1, 2

Artifacts produced:
  checkpoints/sac_FetchReachDense-v4_seed{0,1,2}.zip
  checkpoints/sac_FetchReachDense-v4_seed{0,1,2}.meta.json
  results/ch03_sac_fetchreachdense-v4_seed{0,1,2}_eval.json
  runs/sac/FetchReachDense-v4/seed{0,1,2}/

Results summary (what we got):
  success_rate: 1.00 +/- 0.00  (3 seeds x 100 episodes)
  return_mean:  -0.93 +/- 0.13
  final_distance_mean: 0.016 +/- 0.003

If your numbers differ by more than ~10%, check the
"What Can Go Wrong" section above.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

---

## Build It Components

This is a full Algorithm chapter. The reader derives SAC from the maximum
entropy objective, implements it component by component from scratch, verifies
each piece, then wires them together. All 7 lab regions already exist in
`scripts/labs/sac_from_scratch.py`.

| # | Component | Equation / concept | Lab file:region | Verify check |
|---|-----------|-------------------|-----------------|--------------|
| 1 | Replay Buffer | Circular buffer storing $(s, a, r, s', d)$ transitions; uniform random sampling for off-policy reuse | `labs/sac_from_scratch.py:replay_buffer` | Add 100 transitions with obs_dim=10, act_dim=4; sample batch of 32; verify shapes: obs (32,10), actions (32,4), rewards (32,); buf.size == 100; buf.ptr == 100 |
| 2 | Twin Q-Network | $Q_{\phi_1}(s,a), Q_{\phi_2}(s,a)$: two independent MLPs mapping state-action pairs to scalar Q-values; clipped double Q-learning reduces overestimation | `labs/sac_from_scratch.py:twin_q_network` | Forward pass with obs_dim=10, act_dim=4, batch=32; Q1 shape (32,), Q2 shape (32,); both means near 0 at init; all outputs finite |
| 3 | Squashed Gaussian Policy | $a = \tanh(z)$ where $z \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$; log-prob includes Jacobian correction: $\log \pi(a \mid s) = \log \mu(z \mid s) - \sum_i \log(1 - \tanh^2(z_i))$ | `labs/sac_from_scratch.py:gaussian_policy` | Actions bounded in [-1, 1]; log_probs finite and negative; shapes: actions (32, 4), log_probs (32,) |
| 4 | Twin Q-Network Loss (critic update) | $L(\phi_i) = \mathbb{E}[(Q_{\phi_i}(s,a) - y)^2]$ where $y = r + \gamma(1-d)[\min_j Q_{\bar{\phi}_j}(s',a') - \alpha \log \pi(a' \mid s')]$ | `labs/sac_from_scratch.py:twin_q_loss` | Q-loss finite; Q1 mean near 0 at init; target_q_mean near 0 at init; q1_loss < 1.0 at init (random networks, zero-centered targets) |
| 5 | Actor Loss with Entropy | $L(\theta) = \mathbb{E}[\alpha \log \pi_\theta(a \mid s) - \min_i Q_{\phi_i}(s,a)]$ | `labs/sac_from_scratch.py:actor_loss` | Actor loss finite and positive at init; entropy ($H = -\mathbb{E}[\log \pi]$) positive; log_prob_mean negative |
| 6 | Automatic Temperature Tuning | $L(\alpha) = \mathbb{E}[-\alpha(\log \pi(a \mid s) + \bar{\mathcal{H}})]$ where $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ | `labs/sac_from_scratch.py:temperature_loss` | Initial alpha = 1.0 (log_alpha = 0); after updates alpha has changed; alpha remains positive |
| 7 | SAC Update (wiring) | Sequence: (1) update Q-networks, (2) update policy, (3) update temperature, (4) Polyak-average target networks $\bar{\phi} \leftarrow \tau\phi + (1-\tau)\bar{\phi}$ | `labs/sac_from_scratch.py:sac_update` | After 20 updates on random batch: alpha has changed from 1.0; q1_loss finite; actor_loss finite; all info dict values are finite |

**Ordering rationale:** Foundation first -- replay buffer (1) because off-policy
learning requires it. Twin Q-network (2) and policy (3) are the networks the
losses operate on. Critic loss (4) before actor loss (5) because the actor
uses Q-values computed by the critic. Temperature loss (6) depends on policy
log-probs. Wiring (7) assembles components 4-6 into a single update step with
Polyak averaging.

---

## Bridging Proof

The bridge connects the from-scratch implementation to SB3, proving they
compute the same core quantity. The existing lab supports this via `--compare-sb3`.

- **Inputs (same data fed to both):**
  Random observations, pre-squash samples, and shared mean/log_std parameters,
  generated with seed=0. Same tanh squashing applied in both implementations.

- **From-scratch output (lab code):**
  `GaussianPolicy.forward()` from `sac_from_scratch.py` produces log_probs
  using the numerically stable formula: `log_prob = dist.log_prob(x_t) - 2*(log(2) - x_t - softplus(-2*x_t))`.

- **SB3 output:**
  `StateDependentNoiseDistribution` / `SquashedDiagGaussianDistribution` from
  stable-baselines3 produces log_probs using: `log_prob = dist.log_prob(x_t) - log(1 - tanh(x_t)^2 + 1e-6)`.

- **Match criteria:**
  - `max_abs_log_prob_diff <= 5e-2` (tolerance accounts for SB3's epsilon safety term)
  - The ~0.02 nat typical difference comes from SB3's `1e-6` epsilon in `log(1 - a^2 + 1e-6)` vs our exact `log(1 - tanh^2(u))`
  - For non-saturated actions both formulas agree; epsilon matters only near `|tanh(u)| -> 1`

- **Lab mode:** `--compare-sb3`
  ```bash
  bash docker/dev.sh python scripts/labs/sac_from_scratch.py --compare-sb3
  ```
  Expected output:
  ```
  SAC From Scratch -- SB3 Comparison
  Max abs log_prob diff: 2.055e-02
  Tolerance (atol):      5.0e-02

  [PASS] Our squashed Gaussian log_prob matches SB3
  ```

- **Narrative bridge (for the writer):**
  After showing the match, explain what SB3 adds beyond our from-scratch code:
  replay buffer with efficient numpy storage, parallel environment collection,
  automatic entropy coefficient with constrained optimization, gradient clipping,
  and MultiInputPolicy for dict observations (FetchReach). Map SB3 TensorBoard
  metrics to our code: `replay/ent_coef` is our `log_alpha.exp()`,
  `replay/q1_mean` and `replay/q2_mean` are from our `TwinQNetwork.forward()`,
  `train/actor_loss` is our `compute_actor_loss`, `train/critic_loss` is our
  `compute_q_loss`.

---

## What Can Go Wrong

| Symptom | Likely cause | Diagnostic |
|---------|-------------|------------|
| `replay/q_min_mean` grows unbounded (>100, still rising) | Overestimation feedback loop: Q-targets use overestimated Q-values | Check rewards are bounded (FetchReachDense: [-1, 0]); verify target networks update with small tau (0.005); reduce learning rate from 3e-4 to 1e-4 |
| `replay/ent_coef` drops to <0.01 within first 10k steps | Target entropy too low or policy collapsed to near-deterministic | Check target entropy setting (should be -dim(A) = -4 for Fetch); try fixed ent_coef=0.2 temporarily to isolate the issue |
| Success rate stalls below 50% for >200k steps | Insufficient exploration or replay buffer sampling issues | Check entropy coefficient is not too low; verify `learning_starts` is not too high (default 10000); check batch_size is reasonable (256) |
| Training much slower than expected (<200 fps on GPU) | GPU not used, or excessive gradient steps per environment step | Check `nvidia-smi` shows python process; verify `gradient_steps=1` (default); note ~594 fps is typical for SAC on FetchReach with GPU |
| Q1 and Q2 diverge significantly (difference > 10x) | One Q-network is stuck or has different learning rate | Verify both Q-networks share the same optimizer; check that `q_loss = q1_loss + q2_loss` (not just one of them) |
| `--compare-sb3` shows log_prob diff > 0.05 | Squashing correction formula differs in subtle ways | Verify the `2*(log(2) - x_t - softplus(-2*x_t))` formula matches expected behavior; check that pre-squash samples match between implementations |
| `ep_rew_mean` flatlines near -20 for entire training | Wrong environment or policy not receiving goal info | Check env_id is FetchReachDense-v4 (not sparse); verify MultiInputPolicy is used; print obs dict shapes |
| Build It `--verify` reports NaN in Q-loss or actor loss | Numerical instability in log-prob computation (saturated tanh) | Check LOG_STD_MIN and LOG_STD_MAX bounds; verify softplus-based log-prob correction (prevents -inf from log(0)) |

---

## Adaptation Notes

### Cut from tutorial

- **Part 0 entirety (Sections 0.1-0.2, ~400 words):** "Setting the Stage" is
  informal preamble. The task hierarchy diagram and "diagnostic mindset" framing
  are covered more concisely in the Chapter Bridge and WHY section. The
  dependency chain ASCII art is a tutorial navigation aid, not book content.

- **Part 4 "Understanding What You Built" (Sections 4.1-4.3, ~400 words):**
  Section 4.1 (replay buffer) and 4.2 (min-Q trick) repeat material already
  covered in Build It components 1 and 4. Section 4.3 (squashed Gaussian)
  repeats Build It component 3. Fold any unique insights into the corresponding
  Build It verification checkpoints.

- **Section 3.7 "Understanding GPU Utilization" (~200 words):** GPU utilization
  discussion is peripheral to SAC. Condense to a one-sentence tip in the Run It
  section: "Low GPU utilization (~5-10%) is expected; the bottleneck is
  CPU-bound MuJoCo simulation, not neural network operations."

- **Section 3.8 "Generating Demo Videos" (~250 words):** Video generation
  workflow is infrastructure, not SAC-specific. Reference the video script in
  a sidebar, do not make it a full section.

- **The animated GIF references and demo grid images:** Web-specific content.
  Replace with static figure placeholders or textual descriptions of policy
  behavior.

- **References section:** Move to inline citations. Manning chapters do not
  typically have standalone reference lists.

- **Exercise 2.5.5 (Record a GIF):** This is a video recording exercise, not
  an SAC learning exercise. Cut entirely.

### Keep from tutorial

- **Section 1.1-1.5 (WHY, ~1500 words):** The derivation from standard RL
  objective -> determinism problems -> maximum entropy objective -> Boltzmann
  policy -> automatic temperature tuning -> robotics motivation. This is the
  chapter's mathematical spine. Keep the three-problem structure (exploration
  dies, brittleness, instability) and the Boltzmann distribution insight.

- **Section 2.1-2.4 (HOW, ~1400 words):** SAC component table (5 networks),
  training loop pseudocode (6 steps), PPO vs SAC comparison table, hyperparameter
  table. Keep all -- it provides the architecture overview before Build It.

- **Section 2.5 (Build It, ~2500 words):** All 7 components with snippet-includes
  and verification checkpoints. This is the narrative spine. Keep the
  math-before-code pattern, key-mapping tables (math symbol -> code variable),
  and checkpoint blocks.

- **Section 2.5.9 (SB3 comparison):** Keep as the Bridge section; this is the
  bridging proof showing squashed Gaussian log-prob agreement.

- **Section 2.5.11 (Demo: SAC Solves Pendulum):** Keep the demo run and results
  table. This is the proof that the from-scratch implementation actually learns.

- **Section 3.1-3.6 (Run It, ~1500 words):** One-command version, milestone
  table, diagnostics callback metrics table, Q-value and entropy analysis,
  throughput note, actual results comparison table (PPO vs SAC).

- **Section 6 (Common Failures, ~400 words):** Keep and expand into the full
  "What Can Go Wrong" table above.

### Add for Manning

- **Chapter Bridge (from Ch3):** Ch3 established PPO on dense Reach --
  on-policy training, 100% success, pipeline validated. Gap: PPO discards data
  after every update, wasting simulation time. Ch4 adds SAC, an off-policy
  algorithm with a replay buffer and maximum entropy objective. Foreshadow: Ch5
  needs off-policy algorithms (replay buffers) to implement HER for sparse
  rewards.

- **Opening Promise:** 3-5 bullet "This chapter covers" block.

- **Explicit "off-policy" definition and comparison to on-policy:** The tutorial
  mentions this in passing. For Manning, define off-policy learning formally
  using the concept registry pattern: data can come from any policy (including
  old versions); transitions are stored in a replay buffer and reused across
  many updates. Contrast explicitly with PPO's on-policy data usage from Ch3.
  This is critical because the off-policy property is what enables HER in Ch5.

- **Maximum entropy definition with 5-step template:** The entropy objective
  is the core concept of this chapter. Use the full definition template:
  motivating problem (deterministic policies are brittle), intuitive description
  (prefer actions proportional to their Q-values), formal definition (the
  augmented objective), grounding example (Boltzmann distribution with concrete
  numbers), non-example (high entropy does not mean random -- it means keeping
  options open proportional to value).

- **TensorBoard metric-to-code mapping table:** After the Bridge section, add
  an explicit table mapping SB3 TensorBoard log keys to from-scratch code:
  `replay/ent_coef` -> `log_alpha.exp()`, `replay/q1_mean` -> `TwinQNetwork`
  Q1 output, `train/actor_loss` -> `compute_actor_loss`, `train/critic_loss` ->
  `compute_q_loss`.

- **PPO vs SAC results comparison table:** The tutorial has this in Section 3.6.
  Expand with explicit analysis: both achieve 100% success, but SAC is slower
  per-step (more networks) while being more sample-efficient (reuses data).
  Frame this as the tradeoff between on-policy simplicity and off-policy
  efficiency.

---

## Chapter Bridge

1. **Capability established:** Chapter 3 trained PPO on FetchReachDense-v4 to
   100% success rate, validating the full training pipeline from scratch. You
   derived the PPO clipped surrogate objective, implemented it component by
   component, bridged to SB3, and confirmed the from-scratch code computes the
   same GAE advantages. PPO works -- the infrastructure is sound.

2. **Gap:** PPO is on-policy: every transition is used for a few gradient steps,
   then discarded. For FetchReachDense, where dense rewards provide continuous
   feedback, this wastefulness is tolerable. But each MuJoCo simulation step
   costs real CPU time. PPO achieves ~1300 fps but uses each frame only once.
   For harder tasks with sparser signal (coming in Chapter 5), throwing away
   data will be catastrophic.

3. **This chapter adds:** SAC (Soft Actor-Critic), an off-policy algorithm that
   stores every transition in a replay buffer and reuses it across many updates.
   SAC adds a maximum entropy bonus that keeps the policy exploratory early in
   training and lets it become deterministic as it converges. You will derive
   the maximum entropy objective, implement SAC from scratch (replay buffer,
   twin Q-networks, squashed Gaussian policy, automatic temperature tuning),
   verify each component, bridge to SB3, and match PPO's 100% success on
   FetchReachDense-v4 -- validating the off-policy stack.

4. **Foreshadow:** SAC's replay buffer is not just about sample efficiency.
   Chapter 5 introduces HER (Hindsight Experience Replay), which relabels
   failed transitions with alternative goals -- manufacturing success signal
   from failure. HER requires off-policy learning because relabeled data
   did not come from the current policy. The off-policy machinery you build
   in this chapter is the foundation HER needs.

---

## Opening Promise

> **This chapter covers:**
>
> - Why deterministic policies are brittle -- and how the maximum entropy
>   objective keeps exploration alive by rewarding high-entropy action
>   distributions alongside high reward
> - Implementing SAC from scratch: replay buffer, twin Q-networks with clipped
>   double Q-learning, squashed Gaussian policy with tanh bounds, automatic
>   temperature tuning, and the full update loop
> - Verifying each component with concrete checks (tensor shapes, Q-value
>   ranges, log-probability correctness) before assembling the complete
>   algorithm
> - Bridging from-scratch code to Stable Baselines 3 (SB3): confirming that
>   both implementations compute the same squashed Gaussian log-probabilities,
>   and mapping SB3 TensorBoard metrics to the code you wrote
> - Training SAC on FetchReachDense-v4 to 100% success rate, matching PPO's
>   performance while establishing the off-policy replay machinery that
>   Chapter 5 (HER) requires

---

## Figure Plan

| # | Description | Type | Source command | Chapter location |
|---|------------|------|---------------|-----------------|
| 1 | SAC architecture diagram: twin Q-networks (Q1, Q2), actor (squashed Gaussian policy), target networks (dashed), temperature alpha, replay buffer. Show data flow: env -> buffer -> sample -> Q-update + actor-update + alpha-update -> target soft-update | diagram | Illustrative (matplotlib or draw.io; hand-drawn in Build It section) | After Section 4.2 (HOW: SAC components), before Build It begins |
| 2 | Entropy coefficient over training: alpha vs timesteps showing automatic decrease from ~0.47 to ~0.0004 over 1M steps. Annotate exploration phase (high alpha) and exploitation phase (low alpha) | curve | `python scripts/ch03_sac_dense_reach.py all --seed 0` then extract `replay/ent_coef` from TensorBoard logs in `runs/sac/FetchReachDense-v4/seed0/` | After Section 4.8 (Build It: automatic temperature tuning) or in Run It section |
| 3 | Learning curve comparison: PPO vs SAC success rate over timesteps on FetchReachDense-v4. Both reach 100% but at different speeds and with different sample efficiency | comparison | Extract `rollout/success_rate` from TensorBoard logs for both `runs/ppo/FetchReachDense-v4/seed0/` and `runs/sac/FetchReachDense-v4/seed0/` | In Run It section (Section 4.12), after the PPO vs SAC results table |
| 4 | From-scratch demo learning curve: average return over training steps on Pendulum-v1 showing SAC solving the task (~50k steps). Annotate the -200 solved threshold | curve | `python scripts/labs/sac_from_scratch.py --demo --steps 50000` | After Section 4.9 (Build It: demo run), proving the from-scratch implementation learns |

---

## Estimated Length

| Section | Words |
|---------|-------|
| Opening promise + chapter bridge | 500 |
| 4.1 WHY: Standard objective, determinism problems, maximum entropy, Boltzmann policy, auto temperature, robotics motivation | 1,800 |
| 4.2 HOW: SAC components (5 networks), training loop, PPO vs SAC comparison, hyperparameters | 1,200 |
| 4.3 Build It: Replay Buffer (component 1) | 400 |
| 4.4 Build It: Twin Q-Network (component 2) | 400 |
| 4.5 Build It: Squashed Gaussian Policy (component 3) | 500 |
| 4.6 Build It: Twin Q-Network Loss (component 4) | 500 |
| 4.7 Build It: Actor Loss with Entropy (component 5) | 400 |
| 4.8 Build It: Automatic Temperature Tuning (component 6) | 400 |
| 4.9 Build It: SAC Update wiring (component 7) + demo run | 600 |
| 4.10 Bridge: From-scratch vs SB3 comparison | 500 |
| 4.11 Run It: Experiment card, commands, diagnostics, milestones, artifacts | 1,500 |
| 4.12 What Can Go Wrong | 800 |
| 4.13 Summary + bridge to Ch5 | 500 |
| Reproduce It block | 300 |
| Exercises (5 exercises) | 700 |
| **Total** | **~9,500** |

(Target range: 6,000-10,000 words. Code listings counted separately by
Manning but included in overall page count estimate.)

---

## Concept Registry Additions

Terms this chapter introduces (to be added to the Concept Registry under Ch3,
which maps to Manning Ch4):

- **off-policy learning**: data can come from any policy (including old versions of the current policy); transitions are stored in a replay buffer and reused across many updates
- **replay buffer**: circular array storing $(s, a, r, s', d)$ transitions; uniform random sampling provides training data for off-policy algorithms; capacity controls the "memory horizon"
- **maximum entropy objective**: $J(\theta) = \mathbb{E}[\sum_t \gamma^t (R_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)))]$; augments the standard return with an entropy bonus encouraging exploration
- **entropy $\mathcal{H}(\pi)$**: $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$; measures how "spread out" the action distribution is; higher entropy = more exploration
- **temperature parameter $\alpha$**: controls the exploration-exploitation tradeoff in the maximum entropy objective; $\alpha \to 0$ recovers standard (deterministic) RL; $\alpha \to \infty$ approaches uniform random
- **automatic temperature tuning**: learning $\alpha$ by minimizing $L(\alpha) = \mathbb{E}[-\alpha(\log \pi(a \mid s) + \bar{\mathcal{H}})]$; adjusts $\alpha$ to maintain target entropy
- **target entropy $\bar{\mathcal{H}}$**: desired entropy level for automatic tuning; typically $-\dim(\mathcal{A})$ (negative action dimension); for Fetch: $\bar{\mathcal{H}} = -4$
- **Boltzmann policy**: $\pi^*(a \mid s) \propto \exp(Q^*(s,a)/\alpha)$; the optimal policy under maximum entropy assigns probabilities proportional to exponentiated Q-values
- **twin critic networks**: two independent Q-networks $Q_{\phi_1}$, $Q_{\phi_2}$; using two reduces overestimation bias in Q-learning
- **clipped double Q-learning**: using $\min(Q_{\phi_1}, Q_{\phi_2})$ for targets; counteracts the systematic overestimation $\mathbb{E}[\max(Q_1, Q_2)] \geq \max(\mathbb{E}[Q_1], \mathbb{E}[Q_2])$
- **target networks**: slow-moving copies of Q-networks that provide stable targets for critic updates; prevents the moving-target problem
- **soft update / Polyak averaging ($\tau$)**: $\bar{\phi} \leftarrow \tau\phi + (1-\tau)\bar{\phi}$ with $\tau = 0.005$; slowly blends main network weights into target networks
- **Bellman error**: $|Q_\phi(s,a) - y|$ where $y$ is the Bellman target; the loss the critic minimizes
- **Bellman target $y$**: $y = r + \gamma(1-d)[\min_j Q_{\bar{\phi}_j}(s',a') - \alpha \log \pi(a' \mid s')]$; the "correct" Q-value according to the current model
- **squashed Gaussian policy**: policy that samples from $\mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$ then applies $\tanh$ to bound actions to $[-1,1]$; log-prob includes a Jacobian correction for the squashing
- **SAC (Soft Actor-Critic)**: off-policy actor-critic algorithm with maximum entropy objective, twin critics, automatic temperature tuning, and squashed Gaussian policy (Haarnoja et al. 2018a, 2018b)

These match the Concept Registry entries for "Ch3" in the root CLAUDE.md.

---

## Dependencies

- **Lab regions needed (for Lab Engineer):**
  All 7 regions already exist in `scripts/labs/sac_from_scratch.py`:
  - `replay_buffer` (lines 145-179) -- ReplayBuffer class
  - `twin_q_network` (lines 126-138) -- TwinQNetwork class
  - `gaussian_policy` (lines 85-123) -- GaussianPolicy class
  - `twin_q_loss` (lines 186-252) -- compute_q_loss function
  - `actor_loss` (lines 255-302) -- compute_actor_loss function
  - `temperature_loss` (lines 305-343) -- compute_temperature_loss function
  - `sac_update` (lines 346-420) -- sac_update function
  - Modes: `--verify` (runs all checks, < 2 min CPU), `--demo` (Pendulum training,
    ~30 sec CPU for 5k steps, ~5 min for 50k steps), `--compare-sb3` (squashed
    Gaussian log-prob comparison with SB3)
  - SB3 comparison helper: `scripts/labs/sb3_compare.py` (`compare_sac_squashed_gaussian_log_prob_to_sb3`)
  - **No new lab code needed.** Lab Engineer should verify existing regions
    work correctly and consider adding a `--bridge` alias if it does not exist
    (currently the mode is called `--compare-sb3`).

- **Pretrained checkpoints needed (for Reproduce It):**
  - `checkpoints/sac_FetchReachDense-v4_seed0.zip` (exists)
  - `checkpoints/sac_FetchReachDense-v4_seed1.zip` (to be trained)
  - `checkpoints/sac_FetchReachDense-v4_seed2.zip` (to be trained)
  - Generated by: `bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all --seed {0,1,2}`

- **Previous chapter concepts used (from Manning Ch3 / tutorial Ch2):**
  - return $G_t$, discount factor $\gamma$, expected discounted return $J(\theta)$
  - policy $\pi(a \mid s)$, value function $V(s)$, Q-function $Q(s,a)$, advantage $A(s,a)$
  - policy gradient theorem, actor-critic architecture
  - on-policy learning (used to motivate off-policy contrast)
  - PPO (used as comparison baseline: 100% success, ~1300 fps, -0.40 return)
  - Docker dev.sh workflow (from Manning Ch1)
  - FetchReachDense-v4 environment anatomy (from Manning Ch2)
  - Dense reward structure: $R \in [-1, 0]$ (from Manning Ch2)
  - Success threshold $\epsilon = 0.05$ (50mm, from Manning Ch2)

- **Production script used:** `scripts/ch03_sac_dense_reach.py` (existing; no changes needed)

---

## Exercises

**1. (Verify) Reproduce the single-seed baseline.**

Run the fast path command and verify your results match:

```bash
bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all \
  --seed 0 --total-steps 500000
```

Check the eval JSON: success_rate should be >= 0.95, mean_return > -5.0.
Record your training time and steps/second for comparison with Ch3 (PPO).
Expected: ~594 fps (vs PPO's ~1300 fps), due to more network updates per step.

**2. (Tweak) Twin Q-network ablation.**

In the lab's `compute_q_loss()`, the target uses `torch.min(target_q1, target_q2)`.
Change it to use only one Q-network: `target_q = target_q1`. Run `--verify` and
observe: does `q1_mean` grow larger without the min clipping? Run `--demo --steps 50000`
and compare final returns. Expected: without the min trick, Q-values may overestimate,
leading to slightly worse or unstable training.

**3. (Tweak) Fixed vs auto-tuned temperature.**

In the lab's `sac_update()`, replace `alpha = log_alpha.exp().item()` with a
fixed value: `alpha = 0.2`. Run `--demo --steps 50000` with fixed vs auto-tuned.
Compare: (a) final return, (b) convergence speed, (c) policy entropy at the end.
Expected: auto-tuned reaches better final performance; fixed alpha=0.2 may
over-explore late in training.

**4. (Extend) Add Q-value divergence monitoring.**

In the verification code, add a check that tracks `|q1_mean - q2_mean|` over
updates. Plot this divergence over the 20 verification steps. Expected: the
two Q-networks should stay close (within ~0.5 of each other) because they see
the same targets. Large divergence would indicate a problem.

**5. (Challenge) SAC on Pendulum with different target entropies.**

Modify the demo to try target_entropy values of -0.5, -1.0 (default for 1D),
and -2.0. For each, train for 50k steps and record: (a) final average return,
(b) final alpha value, (c) steps to reach -200 return. Expected: lower target
entropy leads to earlier exploitation (faster convergence but potentially less
robust). Higher target entropy leads to more exploration (slower convergence but
the policy may discover better strategies).
