# Lessons Learned

Patterns, root causes, and prevention rules discovered during development.
Updated after each correction or significant debugging session.

---

## Lesson 1: Sensor Separation Principle

**Context:** Pixel+HER on FetchPush failed at 5-8% for 8M+ steps across
multiple configurations (baseline, normalization fix, frame stacking).

**Pattern:** When a CNN receives only pixel inputs for a task where the robot's
own state is directly measurable, the CNN wastes capacity learning self-state
(gripper position, joint angles) that cheaper sensors already provide.

**Root cause:** The CNN must solve two problems simultaneously:
1. "Where is my arm?" (answerable from joint encoders)
2. "Where is the object?" (only answerable from camera)

Problem 1 is unnecessary -- proprioceptive data provides it with millimeter
precision at microsecond latency.

**Rule:** Always pass robot proprioception (end-effector position, velocity,
gripper state) alongside pixel observations. The CNN should only learn about
the WORLD, not about the robot itself. This mirrors how real robotic systems
operate: joint encoders + cameras, never cameras alone.

**Generalization:** Use the cheapest, most precise sensor for each piece of
information. Don't make learned representations solve problems that direct
measurement already solves.

---

## Lesson 2: NatureCNN is Wrong for Manipulation

**Context:** Same pixel+HER failure investigation.

**Pattern:** NatureCNN (Mnih et al. 2015) was designed for Atari where
game sprites are 10-30 pixels and decisions are coarse. Its 8x8 stride-4
first layer reduces 84x84 to 20x20 in one step, destroying objects that
are only 3-5 pixels wide (gripper, puck in Fetch).

**Root cause:** Aggressive spatial downsampling in the first conv layer.
After layer 1, a 4-pixel object becomes ~1 pixel -- the spatial relationship
between gripper and puck is destroyed.

**Rule:** For manipulation tasks, use 3x3 kernels with stride 2 (DrQ-v2
style), not 8x8 with stride 4 (NatureCNN). Modern visual RL (DrQ-v2,
TD-MPC2, DreamerV3) all use gentler downsampling. Consider spatial softmax
instead of flattening for tasks where precise spatial coordinates matter.

---

## Lesson 3: TensorBoard Log Contamination

**Context:** Fix runs were writing to the same TensorBoard directory as
baseline runs, causing mixed data in EventAccumulator analysis.

**Pattern:** SB3's tensorboard_log creates directories with `_1` suffix.
If the same `run_id` is used for different experiment configs, new event
files are appended to existing directories.

**Root cause:** Both baseline and fix runs used identical run_id format:
`sac_push_pixel_her/{env_id}/seed{seed}`.

**Rule:** Before launching new experiment runs, either:
1. Back up AND delete old TensorBoard directories for the same run_id
2. Use a different run_id that encodes the experiment config
3. Check that the target directory contains only the expected event files

We lost ~2.5 hours of training data to this contamination.

---

## Lesson 4: Two Kinds of Visual HER

**Context:** Designing the pixel+HER architecture for Ch9.

**Pattern:** "Visual HER" can mean two very different things:
1. **Our approach (goal_mode="both"):** Policy sees pixels, HER operates on
   3D goal vectors. Relabeling is trivial (swap 3D vectors). The pixel
   observations are irrelevant to the relabeling process.
2. **Full visual HER (Nair et al. 2018, RIG):** Goals ARE images. HER must
   relabel goal images. Requires a VAE or contrastive encoder to define
   distance in image space.

**Rule:** When writing the tutorial, make this distinction explicit. Our
approach is an intentional middle ground: honest pixel learning for the policy,
efficient vector relabeling for HER. The information asymmetry is a feature,
not a limitation. Frame it as: "We are testing whether a CNN can learn to
CONTROL from pixels, not whether it can learn to SPECIFY GOALS from pixels."

---

## Lesson 5: Normalization Fixes Gradients, Not Features

**Context:** NormalizedCombinedExtractor (LayerNorm + Tanh on CNN output)
fixed the scale mismatch between CNN features and goal vectors but did not
improve success rate.

**Pattern:** The normalization made critic loss lower and more stable from
the start (0.11 vs 0.75 at 200K steps) and increased actor loss (stronger
gradient signals). But by 2M steps, fix and baseline converged to identical
success rates (~5%).

**Root cause:** The CNN features were well-scaled but uninformative. Clean,
bounded noise is still noise. The normalization addressed the symptom (scale
mismatch) but not the cause (CNN can't extract spatial information).

**Rule:** When a representation fix improves training diagnostics (loss
curves) but not task performance (success rate), the problem is deeper than
the representation's scale or normalization -- it's the information content
of the features themselves. Look at the encoder architecture, not just its
output normalization.

---

## Lesson 6: Proprio-Only is a POMDP, Not a Fair Baseline

**Context:** Ablation Run 1 (2026-02-22) tested `--no-pixels` mode with
`ProprioGoalWrapper`, which strips FetchPush-v4's 25D observation to 16D:
proprioception (10D: grip_pos, gripper_state, grip_velp, gripper_vel) +
achieved_goal (3D) + desired_goal (3D). Ran for 2M steps at 571-613 fps.

**Pattern:** 0% success at 1.67M steps. Reward flat at -48.6. No
hockey-stick inflection. Critic loss declining monotonically (same "too
easy" pattern as pixel runs).

**Root cause:** The observation is missing object dynamics -- `object_velp`
(3D), `object_velr` (3D), `object_rel_pos` (3D), `object_rot` (3D). The
agent knows WHERE the object is (via `achieved_goal` = `object_pos`) and
WHERE the target is (via `desired_goal`), but not HOW the object is moving
after contact. Without velocity, the agent cannot predict push outcomes.

Compare to Ch4's state+HER @ 99%: Ch4 used the full 25D observation (31D
total with goals) including all object dynamics. That's an MDP with
complete state. Our 16D is a POMDP.

**Rule:** A vector-only sanity check must use the SAME observation as the
known-good reference. Changing the observation at the same time changes
two variables (modality AND information content), making the result
uninterpretable as a "pipeline sanity check." The correct control for
"is the Ch9 script wired correctly?" is running Ch4's script
(`scripts/ch04_her_sparse_reach_push.py`) or passing the full 25D
observation through `--no-pixels`.

**Generalization:** When designing ablations, change one variable at a
time. If Run 1 was meant to test "does the algorithm work?", the
observation should match the known-good reference exactly.

---

## Lesson 7: ManipulationCNN + SpatialSoftmax is Necessary but Not Sufficient

**Context:** Implemented the full research-backed architecture from
`tasks/pixel_push_encoder_research.md` Section 7: ManipulationCNN (3x3
kernels, stride 2/1/1/2) + SpatialSoftmax + proprioception (10D) + DrQ +
HER. Trained for 2M+ steps (of 8M planned).

**Pattern:** Success rate oscillated 2-14% through 2M steps. Critic loss
declined monotonically from 0.29 to 0.07. No hockey-stick inflection. Same
failure signature as NatureCNN runs, despite verified-correct feature
statistics (pixel features bounded [-0.40, 0.79], no saturation, 80D
features = 64 spatial + 10 proprio + 3 ag + 3 dg).

**Root cause (hypothesis, not yet confirmed):** Fixing the encoder
architecture (Lesson 2) and adding proprioception (Lesson 1) are necessary
conditions but may not be sufficient. The fundamental challenge of
pixel-based goal-conditioned manipulation with end-to-end training remains:
simultaneous representation learning + policy learning from sparse rewards.
No published work demonstrates this exact combination working (research
doc Section 5: "nobody does pixel HER on manipulation").

**What to investigate next:**
1. `--share-encoder` (DrQ-v2 pattern): critic gradients flow through CNN,
   providing encoder with learning signal from Q-values rather than only
   policy gradients.
2. Full-state control: confirm Ch9 script wiring is correct by running with
   the full 25D observation (see Lesson 6).
3. If encoder sharing fails, consider separated representation learning
   (VAE pre-training per RIG, or frozen pre-trained encoder per R3M).

**Rule:** When changing encoder architecture doesn't produce the
hockey-stick inflection, the problem may be optimization (how gradients
reach the encoder) rather than architecture (what the encoder can
represent). Check gradient flow next before adding model complexity.

---

## Lesson 8: Pixel Replay Buffer Memory -- Know Your Capacity

**Context:** Phase A factorial (Ch9 pixel Push debugging) planned buffer_size=1M
to prevent early exploration data from being overwritten. SB3 warned:
`169.62GB > 58.52GB`. The 1M buffer doesn't fit on a 119 GB machine.

**Pattern:** SB3's DictReplayBuffer stores pixel observations at the observation
space dtype (uint8 for our PixelObservationWrapper). But even uint8 pixels are
large: `12 channels × 84 × 84 = 84,672 bytes` per image. With obs + next_obs,
that's 169,344 bytes per transition. A 1M buffer = 169 GB -- exceeds total RAM.

Three compounding factors made memory worse than expected:
1. SB3 divides buffer_size by n_envs internally (`internal_rows = buffer_size // n_envs`),
   but each row stores n_envs transitions. Total memory = buffer_size × 169 KB regardless.
2. DictReplayBuffer does NOT support `optimize_memory_usage` (which avoids storing
   next_obs separately). No 2× savings available.
3. Multiple concurrent Docker training runs can silently eat RAM. Three runs consumed
   64 GB, leaving only 58 GB "available" on a 119 GB machine.

**Rule:** Before launching pixel-based training, compute buffer memory:
```
pixel_bytes = channels × H × W  (uint8: 1 byte/pixel)
per_transition = pixel_bytes × 2  (obs + next_obs)
total_buffer = buffer_size × per_transition
```
For 84×84, 4-frame-stack: `12 × 84 × 84 × 2 = 169 KB/transition`.
Max safe buffer_size ≈ `(available_RAM - 15 GB headroom) / 169 KB`.

**Applied fix:** Run experiments sequentially (one at a time) on the full machine.
With 119 GB total, max safe buffer_size ≈ 600K. Used 500K (2.5× default) for margin.

---

## Lesson 9: Rising Losses Are the Hockey-Stick Signal

**Context:** Cell A extension (Ch9 pixel Push, --no-drq, SpatialSoftmax ON,
critic-encoder, buffer=500K, HER-8). After 2M steps flat at 6%, the agent
began climbing at ~2.15M. By 2.44M steps, success rate reached 25-34%.
Simultaneously, critic_loss rose from 0.07 to 0.3-0.7 and actor_loss rose
from ~0 to 0.8-1.6.

**Pattern:** In SAC+HER with sparse rewards, the hockey-stick inflection
manifests as RISING losses, not declining ones. This is counterintuitive --
in supervised learning, training progress means loss goes DOWN. In RL with
sparse rewards, the opposite happens at the inflection point.

**Root cause -- two regimes of critic loss:**

*Failure regime (flat at 6%, critic_loss declining to 0.07):*
The critic's job is trivially easy. With the agent almost always failing,
Q converges to a uniform ~-18.5 everywhere (sum of gamma^t * (-1) for 50
steps with gamma=0.95). TD error is tiny because the Q-landscape is
uniformly wrong -- predicting failure everywhere and being "right" because
the agent always fails. The critic is memorizing a constant, not learning
structure. A declining critic_loss here is a RED flag, not a green one.

*Hockey-stick regime (climbing through 30%, critic_loss rising to 0.3+):*
Now some trajectories succeed (reward=0 on success steps) and others fail
(reward=-1). The Q-landscape becomes HETEROGENEOUS: Q(s,a) near 0 for
good state-action pairs, near -18.5 for bad ones. The Bellman targets have
real variance. The critic must distinguish between states where the puck
is reachable and states where it is not. This is HARDER -- hence higher
loss. Rising critic_loss means the critic is learning meaningful value
structure.

**Actor loss follows the same logic:**
SAC's actor loss is approximately alpha * log(pi) - Q(s, a_sampled).
When actor_loss was ~0, the critic gave no useful gradient -- all actions
looked equally bad. At actor_loss ~1.0, the critic now says "this action
leads to Q=-2 but that action leads to Q=-15" -- real signal. The actor
is being pulled toward high-Q actions with meaningful force.

**The positive feedback loop (geometric phase transition):**
1. Critic learns value structure -> actor gets real gradients -> better policy
2. Better policy -> more successes -> more diverse Bellman targets -> critic
   learns more fine-grained structure
3. HER amplifies: each success relabels N future transitions with reward=0,
   seeding new value wavefronts
4. This self-amplifying loop is why the curve goes exponential

**Rule for monitoring pixel-based goal-conditioned RL:**

| Metric | Failure Regime | Hockey-Stick Regime |
|--------|---------------|---------------------|
| success_rate | Flat 3-7% | Rising 10%+ |
| critic_loss | Declining to < 0.1 (too easy) | Rising to 0.3+ (learning structure) |
| actor_loss | Near 0 (no signal) | Rising to 1+ (real gradient) |
| reward | Flat at -48 to -50 | Rising toward -40, -30... |

**If critic_loss is declining AND success_rate is flat, the critic is NOT
learning useful representations -- it has converged to predicting uniform
failure. This is the most common failure mode in pixel-based sparse-reward
RL.** The fix is not to reduce loss further (that makes it worse) but to
provide the critic with more informative gradients: larger buffer (more
diverse transitions), more HER relabeling (more reward=0 transitions),
or auxiliary losses (goal prediction) to force encoder to extract spatial
structure.

---

## Lesson 10: Pixel RL Needs 2-4x the Training Budget of State RL

**Context:** Cell A extension confirmed the hockey-stick for pixel Push.
Full-state control hit its inflection at ~1-1.5M steps. Pixel agent hit
its inflection at ~2.2M steps. Ratio: ~1.5-2x.

**Pattern:** Pixel-based RL agents require a "representation learning phase"
before the policy can meaningfully improve. During this phase, the encoder
is building spatial representations from raw pixels. The agent appears
stuck (flat success rate, declining critic loss), but the encoder IS
learning -- just not yet well enough to support policy improvement.

**Quantitative evidence from our experiments:**

| Agent | Hockey-stick onset | 40%+ success | Ratio vs state |
|-------|-------------------|--------------|----------------|
| Full-state (25D, Ch9 Run 0) | ~1.2M | ~1.8M | 1.0x |
| Pixel (84x84, Cell A ext.) | ~2.2M | ~2.55M | ~1.8x |

**Literature confirmation:** Yarats et al. (2022, DrQ-v2) report that
pixel-based agents on DMControl tasks typically need 2-3x the frames of
state-based agents to reach equivalent performance. Our 2.2M/1.2M = 1.8x
is right in that range.

**Why the overhead exists:**
1. The state agent starts with perfect features (25D = exact MDP state).
   The pixel agent must LEARN to extract spatial coordinates from 84x84
   images before the value function becomes meaningful.
2. The encoder warm-up phase (~0-1.4M steps in our case) looks like failure
   but is actually prerequisite work. The flat 6.5% success rate with
   declining critic loss is the encoder slowly building spatial structure
   (see Lesson 9).
3. Once the encoder is good enough, the hockey-stick follows the SAME
   exponential trajectory as state-based RL -- the positive feedback loop
   (Lesson 9) takes over.

**Rule:** When setting training budgets and stop rules for pixel-based RL:
- Multiply the state-based budget by 2-4x for the same task
- Do NOT apply state-calibrated stop rules (e.g., "abort if <10% at 1M")
  to pixel agents -- the representation learning phase makes early steps
  uninformative
- Monitor encoder quality (feature statistics, critic loss regime) rather
  than success rate alone during the first 50% of training
- The original Run A (terminated at 1.54M) and Cell D stop rule ("abort if
  success <=10% at 1M") were both premature -- they killed the agent during
  its representation learning phase, right before the payoff

**Book framing:** "The 2M-step budget that works for state-based training
is insufficient for pixels. The stop rule calibrated for state-based RL
becomes a self-fulfilling prophecy for pixel RL: you terminate the run
precisely because the encoder hasn't yet learned, which ensures you never
see it learn."

---

## Reference Notes

- See `tasks/ch09_pixel_pixels_stack_notes.md` for a concise mapping of research to our code (84x84 sufficiency with stacking, encoder/aug choices, flags to check, and targeted experiments).

