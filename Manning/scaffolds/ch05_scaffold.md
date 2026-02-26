# Scaffold: Chapter 5 -- HER on Sparse Reach/Push: Learning from Failure

## Classification
Type: Algorithm
Source tutorial: tutorials/ch04_her_sparse_reach_push.md
Book chapter output: Manning/chapters/ch05_her_sparse_reach_push.md
Lab file: scripts/labs/her_relabeler.py (existing; 4 snippet regions: data_structures, goal_sampling, relabel_transition, her_buffer_insert)
Production script: scripts/ch04_her_sparse_reach_push.py (existing; the "Run It" reference)

---

## Experiment Card

```
---------------------------------------------------------
EXPERIMENT CARD: SAC + HER on FetchPush-v4 (sparse)
---------------------------------------------------------
Algorithm:    SAC + HER (future strategy, n_sampled_goal=4, ent_coef=0.05, gamma=0.95)
Environments: FetchReach-v4 (sparse, validation), FetchPush-v4 (sparse, primary)
Fast path:    FetchPush-v4, 500,000 steps, seed 0
Time:         ~14 min (GPU) / ~60 min (CPU)

Run command (fast path):
  bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py train \
    --env FetchPush-v4 --her --seed 0 --total-steps 500000 --ent-coef 0.05

Checkpoint track (skip training):
  checkpoints/sac_her_FetchPush-v4_seed0.zip

Expected artifacts:
  checkpoints/sac_her_FetchPush-v4_seed0.zip
  checkpoints/sac_her_FetchPush-v4_seed0.meta.json
  results/ch04_sac_her_fetchpush-v4_seed0_eval.json
  runs/sac_her/FetchPush-v4/seed0/    (TensorBoard logs)

Success criteria (fast path):
  success_rate >= 0.90
  mean_return > -20.0
  final_distance_mean < 0.05

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

  # Sparse Reach: HER vs no-HER (3 seeds, 1M steps each)
  bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all \
    --env FetchReach-v4 --seeds 0,1,2 --total-steps 1000000

  # Sparse Push: HER vs no-HER (3 seeds, 2M steps each)
  bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all \
    --env FetchPush-v4 --seeds 0,1,2 --total-steps 2000000 --ent-coef 0.05

Hardware:     Any modern GPU (tested on NVIDIA GB10; CPU works but ~4x slower)
Time:         ~28 min per seed for Reach (GPU), ~56 min per seed for Push (GPU)
Seeds:        0, 1, 2

Artifacts produced:
  checkpoints/sac_her_FetchReach-v4_seed{0,1,2}.zip
  checkpoints/sac_her_FetchReach-v4_seed{0,1,2}.meta.json
  checkpoints/sac_FetchReach-v4_seed{0,1,2}.zip          (no-HER baselines)
  checkpoints/sac_her_FetchPush-v4_seed{0,1,2}.zip
  checkpoints/sac_her_FetchPush-v4_seed{0,1,2}.meta.json
  checkpoints/sac_FetchPush-v4_seed{0,1,2}.zip            (no-HER baselines)
  results/ch04_fetchreach-v4_comparison.json
  results/ch04_fetchpush-v4_comparison.json

Results summary (what we got):

  FetchReach-v4 (sparse):
                   HER          no-HER
    success_rate:  1.00 +/- 0.00   0.96 +/- 0.07   (3 seeds x 100 episodes)
    return_mean:  -1.68 +/- 0.02  -2.92 +/- 2.04
    final_dist:    17.0mm +/- 6mm  19.5mm +/- 8mm

  FetchPush-v4 (sparse):
                   HER          no-HER
    success_rate:  0.99 +/- 0.01   0.05 +/- 0.00   (3 seeds x 100 episodes)
    return_mean: -13.20 +/- 1.48  -47.50 +/- 0.00
    final_dist:    25.7mm +/- 1mm  184.5mm +/- 0mm

If your numbers differ by more than ~10%, check the
"What Can Go Wrong" section above.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

---

## Build It Components

This is a full Algorithm chapter. The reader learns HER by implementing its
core operations from scratch: goal sampling, transition relabeling, reward
recomputation, and full episode processing.

All 4 snippet regions already exist in `scripts/labs/her_relabeler.py`.
The data structures region establishes the foundation types before the three
operational components.

| # | Component | Equation / concept | Lab file:region | Verify check |
|---|-----------|-------------------|-----------------|--------------|
| 1 | Data Structures: Transition and Episode | Goal-conditioned transition tuple $(s, a, r, s', d, g_a, g_d)$; Episode as a list of transitions; GoalStrategy enum (final/future/episode) | `labs/her_relabeler.py:data_structures` | Construct a Transition with obs_dim=10, act_dim=4, goal_dim=3; verify all 7 fields are present; verify GoalStrategy.FUTURE.value == "future" |
| 2 | Goal Sampling | HER goal sampling strategies: FINAL uses last achieved goal; FUTURE samples from timesteps > current (most common); EPISODE samples from anywhere. Andrychowicz et al. (2017) Section 3.3 | `labs/her_relabeler.py:goal_sampling` | Create 20-step synthetic episode; sample 4 goals with FUTURE strategy at idx=5; verify 4 goals returned, each with shape (3,); verify FINAL always returns same goal k times; verify FUTURE goals come from idx > 5 |
| 3 | Transition Relabeling + Reward Recomputation | Core HER operation: replace $g_d$ with $g_a^{t'}$ (achieved goal from another timestep), then recompute $r = R(g_a, g_d^{\text{new}})$. Sparse reward: $r = 0$ if $\|g_a - g_d\| < \epsilon$, else $r = -1$. Critical invariant: reward must be recomputed, not assumed | `labs/her_relabeler.py:relabel_transition` | Relabel a failed transition (r=-1) with its own achieved_goal; verify relabeled reward = 0.0 (success); verify achieved_goal unchanged; verify desired_goal replaced; verify relabeling with distant goal gives r=-1 |
| 4 | HER Episode Processing (wiring) | Full pipeline: for each transition, add original + k relabeled copies. Data amplification: 50-step episode with k=4, her_ratio=0.8 produces ~210 transitions. Success fraction should jump from ~0% to ~60-80% | `labs/her_relabeler.py:her_buffer_insert` | Process 50-step synthetic episode with k=4, her_ratio=0.8; verify output has more transitions than input; verify success fraction > original; verify data amplification ratio is approximately 1 + k * her_ratio |

**Ordering rationale:** Foundation first -- data structures (1) define the
types that all subsequent components operate on. Goal sampling (2) determines
*which* goals to relabel with. Transition relabeling (3) performs the
substitution and reward recomputation on a single transition. Episode
processing (4) wires 2 and 3 together to process an entire episode,
demonstrating the data amplification effect.

---

## Bridging Proof

The bridge connects the from-scratch relabeler to SB3's `HerReplayBuffer`,
proving they compute the same fundamental invariant: rewards in the buffer
match what `compute_reward(achieved_goal, desired_goal)` returns.

- **Inputs (same data fed to both):**
  A synthetic 64-step episode of goal-conditioned transitions with fixed seed=0,
  obs_dim=10, act_dim=4, goal_dim=3. Goals are set far apart (10+ units) to
  guarantee failures under the original desired goal. Both implementations use
  n_sampled_goal=4, future strategy.

- **From-scratch output (lab code):**
  `process_episode_with_her()` from `her_relabeler.py` produces relabeled
  transitions; `compute_success_fraction()` reports the fraction of success
  transitions (reward >= 0). The success fraction should be > 0 (HER created
  successes from failures).

- **SB3 output:**
  `HerReplayBuffer` from stable-baselines3 stores the same transitions, then
  `buf.sample(512)` returns a batch. For each sample, recompute reward via
  `env.compute_reward(achieved_goal, desired_goal)` and compare to the
  sampled reward.

- **Match criteria:**
  - `max_abs_reward_diff == 0.0` (exact match -- sparse rewards are binary, no floating-point tolerance needed)
  - `success_fraction > 0.0` (HER actually relabeled some transitions as successes)
  - Both implementations produce relabeled successes where none existed originally

- **Lab mode:** `--compare-sb3`
  ```bash
  bash docker/dev.sh python scripts/labs/her_relabeler.py --compare-sb3
  ```
  Expected output:
  ```
  HER Relabeler -- SB3 Comparison
  Max abs reward diff: 0.000e+00
  Success fraction:    0.832
  n_sampled_goal:      4

  [PASS] SB3 HER relabeling is reward-consistent (compute_reward invariant)
  ```

- **Narrative bridge (for the writer):**
  After showing the match, explain what SB3 adds beyond the from-scratch code:
  vectorized episode storage across parallel envs, automatic episode boundary
  detection, integration with SAC's training loop (no manual episode collection),
  online relabeling at sample time (not at insertion time -- SB3 relabels lazily
  when sampling batches, which is more memory-efficient). Map the SB3 constructor
  parameters to our concepts: `n_sampled_goal=4` is our `k`, `goal_selection_strategy="future"`
  is our `GoalStrategy.FUTURE`, `online_sampling=True` means SB3 relabels
  at sample time (our code relabels at insertion time for clarity).

---

## What Can Go Wrong

| Symptom | Likely cause | Diagnostic |
|---------|-------------|------------|
| Push success_rate stalls at ~5% for entire training (with HER enabled) | Entropy coefficient too high -- SAC over-explores and never exploits successful relabeled transitions | Check `ent_coef` value; use fixed `ent_coef=0.05` for Push (not `auto`); verify `--her` flag is actually passed |
| Push success_rate stalls at ~5% even with `ent_coef=0.05` | Wrong gamma; too-long effective horizon dilutes sparse signal | Verify `gamma=0.95` (not 0.99); check `total_steps >= 2000000` for Push; verify n_envs=8 |
| Reach success_rate is 0% for first 100k steps then jumps to 100% | Normal behavior -- SAC+HER on sparse Reach takes longer to start learning than dense Reach; the jump is expected once enough relabeled successes accumulate | No fix needed; this is the "hockey-stick" learning curve pattern |
| `--compare-sb3` fails with reward mismatch | SB3 version incompatibility in `HerReplayBuffer` sampling | Check `stable-baselines3 >= 2.0`; verify the GoalEnvStub's `compute_reward` handles batched inputs correctly |
| `--compare-sb3` shows `success_fraction == 0.0` | Goals not far enough apart in stub env; or n_sampled_goal=0 | Verify the stub places desired_goal 10+ units from achieved_goal; check n_sampled_goal=4 |
| Training much slower than expected (<300 fps on GPU with HER) | HER sampling overhead; too many gradient steps per env step | Check `gradient_steps=1`; note HER adds ~20% overhead vs plain SAC due to goal relabeling; ~500 fps is typical for SAC+HER on Fetch with GPU |
| `ep_rew_mean` stays at -50 throughout training (no improvement) | HER not enabled; or wrong environment (dense instead of sparse) | Verify `--her` flag; check env_id is FetchPush-v4 (not FetchPushDense-v4); print SB3 model to verify HerReplayBuffer is used |
| NaN in Q-values during Push training | Sparse reward + high gamma creates large Bellman targets | Reduce gamma from 0.99 to 0.95; check ent_coef is not 0.0 (needs some exploration) |
| Reach HER barely outperforms no-HER baseline | Expected -- FetchReach-v4 is simple enough that random exploration sometimes succeeds; HER benefit is marginal on Reach. Push is the convincing comparison | Run Push experiment to see the dramatic 5% -> 99% improvement |

---

## Adaptation Notes

### Cut from tutorial

- **Section 0 "Setting the Stage" (~300 words):** Informal preamble with
  dependency chain and week overview. Covered by the Chapter Bridge and
  Opening Promise.

- **Section 1.1 "Sparse Rewards: The Natural Formulation" (first ~200 words
  of motivation):** Overlaps with Ch2 (Environment Anatomy) where dense vs
  sparse rewards were formally defined. Keep only the FetchPush-specific
  motivation (why Push is harder than Reach with sparse rewards).

- **Section 3.3 entropy debugging diary and all `auto-floor`/`schedule`/
  `adaptive` entropy modes (~800 words):** The production script supports 4
  entropy modes but the book should present only the winning configuration
  (fixed ent_coef=0.05 for Push). Mention auto-tuning as a caveat in "What
  Can Go Wrong" rather than a full section.

- **Section 3.4 ablation study on n_sampled_goal (~400 words):** Interesting
  but secondary. Move to an exercise ("Tweak: try n_sampled_goal=2 and 8").

- **The animated GIFs and demo grid images:** Web-specific content. Replace
  with static figure placeholders or textual descriptions.

- **References section:** Move to inline citations throughout the chapter.

- **Appendix-level troubleshooting for entropy modes:** Cut the schedule/
  adaptive/auto-floor troubleshooting; keep only the core failure modes in
  "What Can Go Wrong."

### Keep from tutorial

- **Section 1.1-1.3 (WHY: sparse rewards, needle-in-a-haystack, the insight,
  ~1200 words):** The failure-first pedagogy: show SAC-without-HER flatline
  (Push: 5% success), then motivate HER as the solution. The "what if we just
  changed the question?" insight is the chapter's emotional core.

- **Section 1.4 "The Mathematical Framework" (~600 words):** HER formal
  definition, goal sampling strategies, data amplification math, off-policy
  requirement theorem. This is essential -- the reader needs to understand
  WHY HER requires off-policy algorithms (connects back to Ch4's SAC).

- **Section 2.1-2.4 (Build It, ~1200 words):** All 4 components with
  snippet-includes and verification checkpoints. Data structures, goal
  sampling, transition relabeling, episode processing. Math-before-code
  pattern.

- **Section 2.5 (SB3 comparison):** Keep as the Bridge section. The
  `compute_reward` invariant proof is the core bridge.

- **Section 3.1-3.2 (Run It, ~1000 words):** Two-environment experiment
  (Reach + Push), the dramatic comparison table, training configuration,
  milestone table for Push.

- **Section 4 (Failure modes, ~400 words):** Keep and expand into the full
  "What Can Go Wrong" table.

- **The FetchPush environment introduction (~200 words):** Brief description
  of 7-DoF + object dynamics, annotated figure reference.

### Add for Manning

- **Chapter Bridge (from Ch4):** Ch4 established SAC on dense Reach --
  off-policy learning with replay buffer validated. Gap: dense rewards
  required hand-designed distance signal. Sparse rewards are more natural but
  create an exploration crisis. Ch5 adds HER, which relabels failures with
  achieved goals. Foreshadow: Ch6 applies everything to PickAndPlace.

- **Opening Promise:** 3-5 bullet "This chapter covers" block.

- **Failure-first opening:** Before introducing HER, show the SAC-without-HER
  results on FetchPush-v4: 5% success rate, flat learning curve, 184.5mm
  final distance. Make the reader feel the problem viscerally before
  presenting the solution.

- **Explicit "off-policy requirement for HER" definition with 5-step
  template:** HER requires off-policy learning because relabeled transitions
  were not generated by the current policy. This connects directly to Ch4's
  SAC and explains why PPO+HER is impossible. Use the formal three-condition
  statement from the tutorial.

- **Data amplification visualization and calculation:** Walk through the
  concrete math: a 50-step episode with k=4 and her_ratio=0.8 generates
  ~210 transitions, with ~60-80% being successes vs ~0% originally. This is
  HER's key quantitative insight.

- **Two-environment comparison table:** FetchReach-v4 (marginal HER benefit)
  vs FetchPush-v4 (transformative HER benefit). The contrast reveals when
  HER matters: tasks where random exploration almost never succeeds.

- **TensorBoard metric-to-concept mapping:** Map SB3 HER metrics to
  from-scratch concepts: `rollout/success_rate` is the metric HER directly
  improves, `replay/n_relabeled` (if logged) corresponds to our k parameter,
  `replay/ent_coef` (if auto) shows entropy evolution.

- **Push environment description:** Brief introduction to FetchPush-v4
  (7-DoF arm + puck on table, 25-dimensional observation, 4D actions, object
  dynamics), with annotated figure. This is the reader's first multi-object
  manipulation task.

---

## Chapter Bridge

1. **Capability established:** Chapter 4 derived SAC from the maximum entropy
   objective, implemented it from scratch (replay buffer, twin Q-networks,
   squashed Gaussian policy, automatic temperature tuning), bridged to SB3,
   and achieved 100% success on FetchReachDense-v4. The off-policy replay
   buffer machinery is validated and the reader understands why reusing past
   experience is efficient.

2. **Gap:** Dense rewards required us to hand-design a distance-based signal
   -- the robot received continuous feedback proportional to how far it was
   from the goal. But what if we only know whether the robot succeeded or
   failed? Sparse rewards ($r = 0$ or $r = -1$) are more natural -- they
   do not require knowing the right distance metric -- but they create a
   needle-in-a-haystack exploration problem. On FetchPush-v4 with sparse
   rewards, SAC alone achieves only 5% success: the puck almost never lands
   on the goal by chance, so the agent has no signal to learn from.

3. **This chapter adds:** Hindsight Experience Replay (HER), which transforms
   failures into learning signal by asking: "what goal would this trajectory
   have achieved?" You will implement goal sampling, transition relabeling,
   and reward recomputation from scratch, verify each component, bridge to
   SB3's HerReplayBuffer, and watch success on FetchPush-v4 jump from 5%
   to 99%. HER uses the off-policy replay buffer from Chapter 4 -- relabeled
   transitions were not generated by the current policy, so only off-policy
   algorithms can use them.

4. **Foreshadow:** HER solves the exploration problem for goal-conditioned
   tasks with known goal spaces. Chapter 6 applies the full SAC+HER stack to
   the hardest Fetch task -- PickAndPlace -- where the robot must lift an
   object off the table, requiring curriculum strategies and stress-testing
   to achieve reliability.

---

## Opening Promise

> **This chapter covers:**
>
> - Why sparse rewards create a needle-in-a-haystack problem -- and why SAC
>   alone achieves only 5% success on FetchPush-v4 when rewards are binary
> - The HER insight: relabeling failed trajectories with goals that were
>   actually achieved, turning every failure into a learning opportunity
> - Implementing HER from scratch: goal sampling strategies (future, final,
>   episode), transition relabeling with reward recomputation, and the full
>   episode processing pipeline that amplifies data from 50 transitions to ~210
> - Bridging from-scratch code to SB3's HerReplayBuffer: verifying the
>   compute_reward invariant holds across both implementations
> - Training SAC+HER on two sparse-reward tasks: FetchReach-v4 (marginal
>   improvement, 96% -> 100%) and FetchPush-v4 (transformative improvement,
>   5% -> 99% success rate across 3 seeds)

---

## Figure Plan

| # | Description | Type | Source command | Chapter location |
|---|------------|------|---------------|-----------------|
| 1 | Sparse reward flatline: SAC without HER on FetchPush-v4, success_rate stuck at ~5% over 2M steps. Three seeds overlaid showing consistent failure. Annotate the 5% baseline and contrast with the 99% HER result (shown as dashed target line) | curve | Extract `rollout/success_rate` from TensorBoard logs `runs/sac/FetchPush-v4/seed{0,1,2}/` (no-HER runs) | After Section 5.1 (WHY: the exploration crisis), before HER introduction |
| 2 | HER relabeling diagram: a 5-step trajectory shown as a horizontal sequence of states. Original desired goal (red X) is far away; achieved goals (blue dots) are along the path. Arrows show how HER substitutes the desired goal with an achieved goal from a future timestep, turning a failure (r=-1) into a success (r=0). Label the three strategies (future, final, episode) | diagram | Illustrative (matplotlib or draw.io) | After Section 5.2 (HOW: the HER algorithm), before Build It begins |
| 3 | Data amplification visualization: bar chart showing original episode (50 transitions, ~0% success) vs HER-processed episode (~210 transitions, ~70% success). Two bars side by side for transition count and success fraction. Annotate the amplification factor (k=4, her_ratio=0.8) | comparison | `python scripts/labs/her_relabeler.py --demo` output + matplotlib | After Build It component 4 (episode processing), showing the amplification effect |
| 4 | HER vs no-HER learning curves on both environments: 2x1 subplot. Left: FetchReach-v4 success_rate (HER and no-HER both reach ~100%, marginal difference). Right: FetchPush-v4 success_rate (no-HER stuck at 5%, HER reaches 99%). Three seeds per condition, show mean +/- std shading | comparison | Extract `rollout/success_rate` from TensorBoard logs for all conditions: `runs/sac{,_her}/Fetch{Reach,Push}-v4/seed{0,1,2}/` | In Run It section (Section 5.10), after the results comparison table |
| 5 | FetchPush-v4 environment annotated screenshot: top-down or perspective view showing the 7-DoF arm, the puck (object to push), the target goal position (red marker), and the table surface. Annotate key elements: gripper, puck, goal, workspace bounds | screenshot | `python scripts/capture_proposal_figures.py env-setup` (fetch_push_setup.png already exists in figures/) | After Section 5.1 or at the start of the Push experiment discussion |

---

## Estimated Length

| Section | Words |
|---------|-------|
| Opening promise + chapter bridge | 500 |
| 5.1 WHY: Sparse rewards, the exploration crisis, failure-first (SAC-no-HER flatline), needle-in-a-haystack framing | 1,800 |
| 5.2 HOW: The HER insight, formal definition, goal sampling strategies, data amplification math, off-policy requirement | 1,500 |
| 5.3 Build It: Data structures (component 1) | 300 |
| 5.4 Build It: Goal sampling (component 2) | 500 |
| 5.5 Build It: Transition relabeling + reward recomputation (component 3) | 500 |
| 5.6 Build It: Episode processing / wiring (component 4) + demo | 600 |
| 5.7 Bridge: From-scratch vs SB3 comparison | 500 |
| 5.8 Run It: Two-environment experiment, Experiment Card, commands, milestone table, results table | 1,800 |
| 5.9 What Can Go Wrong | 800 |
| 5.10 Summary + bridge to Ch6 | 500 |
| Reproduce It block | 400 |
| Exercises (5 exercises) | 700 |
| **Total** | **~10,400** |

(Target range: 6,000-10,000 words. Slightly over target due to two-environment
coverage. Code listings counted separately by Manning but included in overall
page count estimate. Consider trimming WHY or Run It if the writer finds the
total exceeds 11,000 words.)

**Writer spans:** 2 spans recommended.
- Span 1: Opening + Bridge + 5.1 WHY + 5.2 HOW + 5.3-5.6 Build It (~5,700 words)
- Span 2: 5.7 Bridge + 5.8 Run It + 5.9 What Can Go Wrong + 5.10 Summary + Reproduce It + Exercises (~4,700 words)

Preferred split point: Between Build It (Section 5.6) and Bridge (Section 5.7)
-- natural shift from implementation to verification.

---

## Concept Registry Additions

Terms this chapter introduces (to be added to the Concept Registry under Ch4,
which maps to Manning Ch5):

- **hindsight experience replay (HER)**: technique that relabels failed
  episodes by substituting the original desired goal with a goal that was
  actually achieved, turning failures into successes for learning
  (Andrychowicz et al. 2017)
- **goal relabeling**: the core HER operation -- replacing the desired goal
  $g_d$ in a transition with an alternative goal $g'$ (typically an achieved
  goal from another timestep) and recomputing the reward
- **goal sampling strategies (future/final/episode)**: methods for choosing
  which achieved goal to use as the relabeled goal. FUTURE (sample from
  timesteps after the current one) is most common and effective; FINAL uses
  the episode's last achieved goal; EPISODE samples uniformly from the full
  episode
- **n_sampled_goal**: the number of alternative goals sampled per transition
  during HER relabeling; typically k=4 (Andrychowicz et al. 2017)
- **data amplification**: HER's quantitative effect -- a T-step episode
  produces T original + T * k * her_ratio relabeled transitions, dramatically
  increasing the fraction of "success" examples in the buffer
- **off-policy requirement for HER (three conditions)**: HER requires
  off-policy learning because: (1) relabeled transitions were not generated
  by the current policy, (2) the reward is retroactively changed, (3) the
  goal is modified after collection. This is why HER works with SAC/TD3 but
  not PPO
- **effective horizon $T_{\text{eff}}$**: the number of timesteps within which
  a randomly-exploring agent has reasonable probability of reaching the goal;
  for FetchReach ~2-5 steps, for FetchPush ~20-50 steps
- **HER ratio**: the probability that a transition receives HER relabeling
  (typically 0.8, meaning 80% of transitions get k relabeled copies); controls
  the balance between real and relabeled experience
- **cumulative entropy bonus scaling**: the interaction between SAC's entropy
  bonus and HER's sparse rewards; with gamma=0.99 the cumulative entropy bonus
  can dominate the sparse reward signal, requiring lower gamma (0.95) or fixed
  low ent_coef (0.05)

These match the Concept Registry entries for "Ch4" in the root CLAUDE.md.

---

## Dependencies

- **Lab regions needed (for Lab Engineer):**
  All 4 snippet regions already exist in `scripts/labs/her_relabeler.py`:
  - `data_structures` (lines 68-98) -- Transition, Episode, GoalStrategy
  - `goal_sampling` (lines 101-169) -- sample_her_goals function
  - `relabel_transition` (lines 176-237) -- relabel_transition + sparse_reward
  - `her_buffer_insert` (lines 244-304) -- process_episode_with_her + compute_success_fraction
  - Modes: `--verify` (runs all checks, < 1 min CPU), `--demo` (shows
    relabeling on synthetic data, < 10 sec CPU), `--compare-sb3` (reward
    invariant comparison with SB3 HerReplayBuffer)
  - SB3 comparison helper: `scripts/labs/sb3_compare.py`
    (`compare_her_relabeling_to_sb3`)
  - **No new lab code needed.** Lab Engineer should verify existing regions
    work correctly and that `--verify`, `--demo`, and `--compare-sb3` all
    pass. Consider adding a `--bridge` alias if it does not exist (currently
    the mode is called `--compare-sb3`).

- **Pretrained checkpoints needed (for Reproduce It):**
  All checkpoints already exist:
  - `checkpoints/sac_her_FetchReach-v4_seed{0,1,2}.zip` (exist)
  - `checkpoints/sac_FetchReach-v4_seed{0,1,2}.zip` (exist, no-HER baselines)
  - `checkpoints/sac_her_FetchPush-v4_seed{0,1,2}.zip` (exist)
  - `checkpoints/sac_FetchPush-v4_seed{0,1,2}.zip` (exist, no-HER baselines)
  - Generated by: `bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env <env> --seeds 0,1,2`

- **Eval JSON files (for results tables):**
  - `results/ch04_fetchreach-v4_comparison.json` (exists)
  - `results/ch04_fetchpush-v4_comparison.json` (exists)

- **Previous chapter concepts used (from Manning Ch4 / tutorial Ch3):**
  - off-policy learning, replay buffer (Ch4 -- the foundation HER builds on)
  - SAC, maximum entropy objective, temperature parameter alpha (Ch4)
  - squashed Gaussian policy, twin critic networks, Polyak averaging (Ch4)
  - dense reward, sparse reward, success threshold epsilon (Ch2)
  - goal-conditioned MDP, dictionary observation structure, compute_reward API (Ch2)
  - critical invariant (reward recomputation) (Ch2 -- extended here to HER relabeling)
  - Docker dev.sh workflow (Ch1)

- **Figures already available:**
  - `figures/fetch_push_setup.png` (exists from `capture_proposal_figures.py env-setup`)
  - TensorBoard logs for all conditions (exist in `runs/`)

- **Production script used:** `scripts/ch04_her_sparse_reach_push.py`
  (existing; no changes needed)

---

## Exercises

**1. (Verify) Reproduce the single-seed Push baseline.**

Run the fast path command and verify your results match:

```bash
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py train \
  --env FetchPush-v4 --her --seed 0 --total-steps 500000 --ent-coef 0.05
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py eval \
  --ckpt checkpoints/sac_her_FetchPush-v4_seed0.zip
```

Check the eval JSON: success_rate should be >= 0.90, mean_return > -20.0.
Compare training time to Ch4's SAC on FetchReachDense (expect ~20% overhead
from HER relabeling).

**2. (Tweak) Change n_sampled_goal.**

The default is k=4. Try k=2 and k=8:

```bash
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py train \
  --env FetchPush-v4 --her --seed 0 --total-steps 2000000 --ent-coef 0.05 \
  --n-sampled-goal 2
```

Compare success_rate at convergence. Expected: k=2 may be slightly slower to
converge; k=8 may be slightly faster but with diminishing returns. The original
paper found k=4 to be a good balance.

**3. (Tweak) Replace FUTURE strategy with FINAL.**

In the lab's `process_episode_with_her()`, change the default strategy from
`GoalStrategy.FUTURE` to `GoalStrategy.FINAL`. Run `--demo` and compare
the success fraction. Expected: FINAL produces slightly lower success
fraction because all relabeled goals are the same (less diversity), but it
still works. FUTURE provides more diverse relabeled goals.

**4. (Extend) Visualize the data amplification effect.**

Write a short script that processes 10 synthetic episodes with `process_episode_with_her()`
and plots: (a) number of original vs relabeled transitions, (b) success
fraction before and after HER, (c) distribution of distances between achieved
and relabeled goals. Use matplotlib bar charts and histograms.

**5. (Challenge) SAC without HER on sparse Reach vs Push.**

Run the no-HER baseline on both environments:

```bash
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py train \
  --env FetchReach-v4 --seed 0 --total-steps 1000000
bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py train \
  --env FetchPush-v4 --seed 0 --total-steps 2000000 --ent-coef 0.05
```

Compare the two: Reach achieves ~96% without HER, Push achieves ~5%. Explain
why in terms of the effective horizon $T_{\text{eff}}$: for Reach, random
exploration reaches the goal within ~2-5 steps; for Push, the puck must
be contacted AND pushed to the target, requiring ~20-50 coordinated steps.
HER's benefit scales with the difficulty of the exploration problem.
