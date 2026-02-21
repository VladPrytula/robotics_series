# Draft: Ch9 Section 9.8 -- The Representation Tax and What Comes Next

> **Note to ourselves:** This sidebar goes after section 9.7 (Push from pixels
> results). Numbers marked [PENDING] will be filled in from our pixel training
> runs once they complete.

---

## 9.8 The Representation Tax

### Why pixels cost more than state

Our experiments tell a consistent story:

| Config | Task | Steps to converge | Success |
|--------|------|-------------------:|--------:|
| State SAC | Reach | 500K | 100% |
| Pixel SAC | Reach | 2M | 98% |
| Pixel SAC + DrQ | Reach | 2M | 100% |
| State SAC + HER | Push (sparse) | 2M | 100% |
| Pixel SAC + HER | Push (sparse) | [PENDING: 8M] | [PENDING]% |
| Pixel SAC + HER + DrQ | Push (sparse) | [PENDING: 8M] | [PENDING]% |

The pattern: **going from state to pixels multiplies the sample budget by
roughly 4x on Reach.** We expect a similar or larger multiplier on Push --
the CNN must now learn to distinguish object position, gripper position, and
goal position from 84x84 images, while simultaneously learning a pushing
policy through the HER hockey-stick curve.

[PENDING: Update multiplier with actual Push pixel numbers.]

This is not a failure of our pipeline. It reflects a fundamental tension that
the field has identified clearly: **representation learning and RL policy
learning compete for gradient signal** (Laskin et al. 2020; Stooke et al.
2021). The sparse HER reward provides no gradient at all during the flat
phase, so the CNN must learn useful features from the augmentation signal
alone (in the DrQ case) or from the trickle of relabeled near-successes.

### Three strategies the field uses to close this gap

The research community has developed three broad approaches to reduce the
representation tax. We summarize them here not because our pipeline needs
them -- our goal is to show the honest cost of pixels -- but because the
reader who wants to push further should know where the field is headed.

**Strategy 1: Bring a pre-trained encoder.**

Rather than learning visual features from RL rewards, use an encoder
pre-trained on large-scale video data. The policy then operates on compact,
meaningful features from the first step.

- **R3M** (Nair et al. 2022, CoRL; arXiv:2203.12601) trains a ResNet-50 on
  4,500 hours of Ego4D human manipulation video using time-contrastive
  learning and video-language alignment. Frozen R3M features improve
  manipulation task success by ~20% over training from scratch.

- **VIP** (Ma et al. 2023, ICLR Spotlight; arXiv:2210.00030) goes further:
  it trains a visual encoder whose embedding distance serves as a dense
  reward signal for goal-conditioned RL. This is particularly relevant to
  our setting -- VIP embeddings could replace both the CNN encoder and the
  sparse reward, potentially eliminating the need for HER entirely.

The honest caveat: these encoders were trained on real-world egocentric
video (kitchens, workshops, outdoor scenes). MuJoCo's rendered frames look
nothing like Ego4D. The domain gap is real, and we have not evaluated whether
frozen R3M or VIP features transfer to Gymnasium-Robotics environments.
We find it more useful to name this limitation than to pretend it does not
exist.

**Strategy 2: Add an auxiliary learning signal.**

Keep learning the CNN from scratch, but give it additional gradients beyond
the RL reward. Self-supervised objectives provide dense signal from the
first step, bridging the gap until the RL reward kicks in.

- **CURL** (Laskin et al. 2020; arXiv:2004.04136) adds a contrastive loss
  on augmented observation pairs, achieving 1.9x sample efficiency on
  DMControl. DrQ (Kostrikov et al. 2020; arXiv:2004.13649) -- which we
  already use -- showed that the contrastive loss is not strictly necessary;
  the augmentation itself provides sufficient regularization. But CURL's
  insight remains: explicit representation objectives help.

- **DrM** (Xu et al. 2024, ICLR; arXiv:2310.19668) diagnoses a specific
  pathology: during early visual RL training, a large fraction of neurons in
  the policy network become "dormant" -- they produce near-zero activations
  and receive near-zero gradients. DrM monitors the dormant ratio and
  periodically perturbs weights to reactivate dead neurons. This directly
  addresses the flat phase we observe in our HER hockey-stick curve: the
  value wavefront is stalling in part because the representation is
  functionally collapsed.

- **TACO** (Zheng et al. 2023, NeurIPS; arXiv:2306.13229) adds a temporal
  action-driven contrastive loss that jointly learns state and action
  representations. The key idea: good representations should predict not
  just what the agent sees, but what it did. This yielded ~40% improvement
  at 1M steps on DMControl manipulation tasks.

**Strategy 3: Learn a world model.**

Instead of learning a policy directly from pixel interactions, learn a
dynamics model in latent space and train the policy on imagined trajectories.
A single real transition can generate many imagined ones, dramatically
improving sample efficiency.

- **DreamerV3** (Hafner et al. 2023; arXiv:2301.04104) learns a world model
  from pixels and trains entirely in imagination. It solves 150+ tasks with
  a single set of hyperparameters, including Minecraft diamond collection
  from sparse reward. Sample efficiency is typically 5-20x better than
  model-free methods.

- **TD-MPC2** (Hansen et al. 2023; arXiv:2310.16828) learns a decoder-free
  latent world model and performs trajectory optimization (model-predictive
  control) in latent space. It scales to 80+ tasks including Meta-World
  manipulation from pixels.

The honest caveat: neither DreamerV3 nor TD-MPC2 natively supports HER-style
goal relabeling. HER operates on real transitions in a replay buffer; world
models generate imagined trajectories. Combining them -- imagined HER -- is
an open problem with limited published work. Adopting either approach would
mean replacing our SB3 pipeline entirely, not augmenting it.

### A fourth direction: replacing HER altogether

A line of recent work asks whether hindsight relabeling is the right
abstraction for visual goal-conditioned RL in the first place.

- **Contrastive RL** (Eysenbach et al. 2024, ICLR Spotlight;
  arXiv:2306.03346) learns goal-reaching policies through self-supervised
  contrastive objectives, without explicit reward functions or goal
  relabeling. The agent learns a representation where temporal proximity in
  trajectories implies proximity in goal space. This was demonstrated on
  real-world image-based manipulation.

- **GCHR** (Lei et al. 2025; arXiv:2508.06108) stays closer to HER but adds
  a hindsight regularization term that extracts more learning signal from
  each relabeled transition. The authors report substantially improved
  sample reuse on Fetch manipulation tasks.

These approaches suggest that the HER hockey-stick we observed in section 9.6
is not the only path. The flat phase exists because HER's relabeled signal
must propagate through Bellman backups before the test-goal policy improves.
Methods that learn goal-conditioned representations directly -- without the
Bellman bottleneck -- may avoid the hockey-stick entirely.

### What our experiments add

The contribution of this chapter is not a new algorithm. It is an honest
measurement of costs that the literature often glosses over:

1. **The representation tax is real and quantifiable.** Going from state to
   pixels costs [PENDING: Nx] in sample budget on Reach, [PENDING: Nx] on
   Push. We report both multipliers with specific step counts and success
   rates.

2. **DrQ helps quality, not necessarily speed.** On Reach, DrQ closes the
   success gap (98% -> 100%) without reducing convergence time. On Push,
   [PENDING: does DrQ converge faster, reach higher success, or both?].
   If augmentation primarily improves final performance rather than
   convergence speed, that is a finding worth reporting.

3. **The hockey-stick persists in pixel space.** The phase-transition
   structure of HER learning (section 9.6) does not disappear when we add
   pixels -- [PENDING: confirm whether the pixel runs show a similar
   hockey-stick pattern, and at what step count the inflection occurs].

We find it more valuable to present these honest measurements than to chase
a higher number by importing pre-trained representations. The reader who
wants to close the gap further now has three well-cited directions to pursue.

---

### References (new for this section)

- Chi, C. et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via
  Action Diffusion." arXiv:2303.04137.
- Eysenbach, B. et al. (2024). "Contrastive RL: Self-Supervised Goal-Conditioned
  Learning from Observation." ICLR 2024 Spotlight. arXiv:2306.03346.
- Hafner, D. et al. (2023). "Mastering Diverse Domains through World Models."
  arXiv:2301.04104.
- Hansen, N. et al. (2023). "TD-MPC2: Scalable, Robust World Models for
  Continuous Control." arXiv:2310.16828.
- Laskin, M. et al. (2020). "CURL: Contrastive Unsupervised Representations
  for Reinforcement Learning." arXiv:2004.04136.
- Lei, H. et al. (2025). "Goal-Conditioned Hindsight Regularization."
  arXiv:2508.06108.
- Ma, Y. et al. (2023). "VIP: Towards Universal Visual Reward and Representation
  via Value-Implicit Pre-Training." ICLR 2023 Spotlight. arXiv:2210.00030.
- Nair, S. et al. (2022). "R3M: A Universal Visual Representation for Robot
  Manipulation." CoRL 2022. arXiv:2203.12601.
- Stooke, A. et al. (2021). "Decoupling Representation Learning from
  Reinforcement Learning." arXiv:2009.08319.
- Xu, G. et al. (2024). "DrM: Mastering Visual Reinforcement Learning through
  Dormant Ratio Minimization." ICLR 2024. arXiv:2310.19668.
- Zheng, R. et al. (2023). "TACO: Temporal Latent Action-Driven Contrastive
  Loss for Visual Reinforcement Learning." NeurIPS 2023. arXiv:2306.13229.
