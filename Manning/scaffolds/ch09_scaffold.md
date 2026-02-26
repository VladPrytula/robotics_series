# Scaffold: Chapter 9 -- Pixels, No Cheating: From State Vectors to Camera Images

## Classification
Type: Pixels
Source tutorial: tutorials/ch09_pixel_push.md
Book chapter output: Manning/chapters/ch09_pixel_push.md
Lab files:
  - scripts/labs/pixel_wrapper.py (3 regions: render_and_resize, pixel_obs_wrapper, pixel_replay_buffer)
  - scripts/labs/manipulation_encoder.py (3 regions: spatial_softmax, manipulation_cnn, manipulation_extractor)
  - scripts/labs/image_augmentation.py (3 regions: random_shift_aug, drq_replay_buffer, her_drq_replay_buffer)
  - scripts/labs/visual_encoder.py (5 regions: nature_cnn, normalized_combined_extractor, visual_goal_encoder, visual_gaussian_policy, visual_twin_q_network)
  - scripts/labs/drqv2_sac_policy.py (3 regions: critic_encoder_critic, critic_encoder_actor, drqv2_sac_policy)
Production script: scripts/ch09_pixel_push.py (existing; the "Run It" reference)

---

## Experiment Card

```
---------------------------------------------------------
EXPERIMENT CARD: SAC + HER + Pixel Pipeline on FetchPush-v4
---------------------------------------------------------
Algorithm:    SAC + HER (critic-encoder gradient routing,
              ManipulationCNN + SpatialSoftmax, no DrQ)
Environment:  FetchPush-v4 (pixel observations, sparse reward)
Fast path:    5,000,000 steps, seed 0
Time:         ~40 hours (GPU); not feasible on CPU for full run
              (Build It --verify: < 2 min CPU; --demo: ~10 min CPU)

Run command (fast path):
  bash docker/dev.sh python scripts/ch09_pixel_push.py train \
    --seed 0 --critic-encoder --no-drq --buffer-size 500000 \
    --her-n-sampled-goal 8 --total-steps 5000000 \
    --checkpoint-freq 500000

Checkpoint track (skip training):
  checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0.zip

Expected artifacts:
  checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0.zip
  checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0.meta.json
  checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0_{500K,1M,...,5M}_steps.zip
  results/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0_eval.json
  runs/ch09_manip_noDrQ_criticEnc/FetchPush-v4/seed0/

Success criteria (fast path):
  success_rate >= 0.90 (at 5M steps)
  hockey-stick inflection visible in TensorBoard by ~2.5M steps
  critic_loss non-monotonic trajectory (decline -> rise -> decline)

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
    bash docker/dev.sh python scripts/ch09_pixel_push.py train \
      --seed $seed --critic-encoder --no-drq --buffer-size 500000 \
      --her-n-sampled-goal 8 --total-steps 5000000 \
      --checkpoint-freq 500000
  done

Hardware:     NVIDIA GPU with >= 60 GB system RAM
              (tested on DGX; any modern GPU works, times will vary)
Time:         ~40 hours per seed at ~30 fps
Seeds:        0, 1, 2

Artifacts produced:
  checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed{0,1,2}.zip
  checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed{0,1,2}.meta.json
  results/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed{0,1,2}_eval.json
  runs/ch09_manip_noDrQ_criticEnc/FetchPush-v4/seed{0,1,2}/

Results summary (what we got):
  success_rate: 0.95 +/- [PENDING]  (3 seeds x 100 episodes)
  hockey_stick_onset: ~2.2M steps
  90%_success: ~3.5M steps
  training_time: ~40h per seed

Comparison runs (failure baselines for the investigation):
  # Step 0: NatureCNN baseline
  bash docker/dev.sh python scripts/ch09_pixel_push.py train \
    --seed 0 --total-steps 2000000

  # Step 4: DrQ ablation (shows augmentation-representation conflict)
  bash docker/dev.sh python scripts/ch09_pixel_push.py train \
    --seed 0 --critic-encoder --buffer-size 500000 \
    --her-n-sampled-goal 8 --total-steps 2000000

If your numbers differ by more than ~10% (hockey-stick not visible
by 3M, or final success below 80%), check the "What Can Go Wrong"
section above.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

---

## Build It Components

This is a Pixels chapter. The reader builds every visual component that SB3
uses, then runs the production pipeline with those components integrated. The
chapter has 11 components across 5 lab files -- substantially more than an
Algorithm chapter because the visual pipeline has more moving parts.

**Ordering rationale:** Foundation first. Start with the pixel observation
wrapper (how images become tensors), then the encoder architectures (NatureCNN
first as the "wrong" encoder the reader will see fail, then ManipulationCNN as
the fix), then the spatial feature head (SpatialSoftmax), then the SB3-
compatible feature extractor, then data augmentation (DrQ), then gradient
routing (critic-encoder). The gradient routing comes last because it is the
most subtle and the reader needs to understand the full pipeline before
appreciating why gradients need to flow differently.

| # | Component | Equation / concept | Lab file:region | Verify check |
|---|-----------|-------------------|-----------------|--------------|
| 1 | render_and_resize | MuJoCo camera -> CHW uint8: reshape (H,W,3) -> transpose -> stack 4 frames -> (12, 84, 84) uint8 tensor | `labs/pixel_wrapper.py:render_and_resize` | Output shape (3, 84, 84) uint8; values in [0, 255]; dtype is np.uint8 not float32 |
| 2 | PixelObservationWrapper | goal_mode="both": pixel obs + vector goals; frame stacking (4 frames -> 12 channels); proprioception passthrough (10D robot state); observation space dict with correct shapes and dtypes | `labs/pixel_wrapper.py:pixel_obs_wrapper` | obs dict has keys "pixels" (12, 84, 84), "proprioception" (10,), "achieved_goal" (3,), "desired_goal" (3,); pixels dtype uint8; step produces valid obs |
| 3 | Pixel replay buffer | uint8 storage for pixel observations (not float32); float32 conversion at sample time; memory: 12 x 84 x 84 x 1 byte = 84,672 bytes per pixel obs | `labs/pixel_wrapper.py:pixel_replay_buffer` | Stored dtype is uint8; sampled dtype is float32; sampled values in [0, 1] range (normalized); memory per transition ~169 KB (obs + next_obs) |
| 4 | NatureCNN (the "wrong" encoder) | Conv2d(12, 32, 8, stride=4) -> Conv2d(32, 64, 4, stride=2) -> Conv2d(64, 64, 3, stride=1) -> Flatten -> Linear(3136, 512); spatial progression: 84->20->9->7 | `labs/visual_encoder.py:nature_cnn` | Output shape (B, 512); 5-pixel object -> ~1 pixel after layer 1; total params ~1.7M; forward produces finite values |
| 5 | ManipulationCNN (the "right" encoder) | 4 x Conv2d(32, 3x3, stride=2/1/1/2, pad=1); spatial progression: 84->42->42->42->21; preserves small objects | `labs/manipulation_encoder.py:manipulation_cnn` | Output shape (B, 32, 21, 21); 5-pixel object -> ~3 pixels after layer 1; total params ~28K; forward produces finite values |
| 6 | SpatialSoftmax | For each of C channels: apply spatial softmax across HxW -> expected (x,y) coordinate; output: 2C values in [-1, 1] representing "where" each feature fires, not "what" the image looks like | `labs/manipulation_encoder.py:spatial_softmax` | Input (B, 32, 21, 21) -> output (B, 64); values in [-1, 1]; sum of softmax weights per channel = 1.0; gradient flows through (non-zero grad) |
| 7 | ManipulationExtractor (SB3 compatible) | CNN -> SpatialSoftmax -> concat with proprio (10D) + ag (3D) + dg (3D) -> 80D feature vector; routes dict obs keys to appropriate sub-encoders | `labs/manipulation_encoder.py:manipulation_extractor` | features_dim = 80 (64 spatial + 10 proprio + 3 ag + 3 dg); forward with dict obs produces (B, 80) tensor; all outputs finite; SB3 policy_class accepts this extractor |
| 8 | DrQ random shift augmentation | Pad image by 4 pixels (replicate padding), then random crop back to original size; shift range: +/-4 pixels; independent augmentation of obs and next_obs | `labs/image_augmentation.py:random_shift_aug` | Input (B, 12, 84, 84) -> output (B, 12, 84, 84); output shape matches input; output != input (shift applied); replicate padding at edges |
| 9 | DrQ replay buffer | Augment at sample time (not store time); sample batch -> apply random_shift to obs and next_obs independently -> return augmented batch | `labs/image_augmentation.py:drq_replay_buffer` | Samples have same shapes as non-DrQ buffer; two calls with same indices return different augmented images (random shift varies); uint8 storage preserved |
| 10 | CriticEncoderCritic | Override ContinuousCritic.forward: always enable gradients through shared encoder (remove SB3's set_grad_enabled(False) gate during critic training) | `labs/drqv2_sac_policy.py:critic_encoder_critic` | After critic.backward(): encoder params have .grad != None; encoder param values change after optimizer step |
| 11 | CriticEncoderActor | Override Actor.forward: call features.detach() before policy MLP; encoder receives NO gradient from actor loss | `labs/drqv2_sac_policy.py:critic_encoder_actor` | After actor.backward(): encoder params have .grad == None (or zero); encoder param values do NOT change after actor optimizer step |
| 12 | DrQv2SACPolicy (wiring) | SACPolicy override: shared encoder in critic optimizer (identity-based param filtering); encoder excluded from actor optimizer; Polyak soft update includes target encoder | `labs/drqv2_sac_policy.py:drqv2_sac_policy` | 0/N encoder params in actor optimizer, N/N in critic optimizer; target critic has separate encoder copy; save/load round-trip preserves policy type and predictions; SB3 50-step training smoke test passes |

---

## Bridging Proof

The bridging proof for a Pixels chapter is different from an Algorithm chapter.
There is no single "loss comparison" because the chapter's Build It components
are infrastructure (wrappers, encoders, routing), not learning algorithms. The
bridge demonstrates that the from-scratch components produce the same
observations, features, and gradient behavior as the SB3 pipeline that uses them.

- **Inputs (same data fed to both):**
  Same FetchPush-v4 reset state, same pixel observation, same random batch
  from replay buffer.

- **From-scratch output (lab code):**
  1. `pixel_wrapper.py --verify`: Pixel wrapper produces correct dict obs
     with correct shapes, dtypes, and value ranges.
  2. `manipulation_encoder.py --verify`: ManipulationCNN + SpatialSoftmax
     produces 64D spatial features in [-1, 1]; ManipulationExtractor produces
     80D combined features.
  3. `drqv2_sac_policy.py --verify`: Gradient routing verified -- encoder
     gets grad from critic, not from actor.

- **SB3 output:**
  1. `drqv2_sac_policy.py --probe`: SB3's default SAC with
     `share_features_extractor=True` puts encoder in actor optimizer only
     (the broken default). Our override puts it in critic optimizer.
  2. After 50 SB3 training steps with DrQv2SACPolicy: no crashes, encoder
     params updated by critic loss, encoder params NOT updated by actor loss.

- **Match criteria:**
  - Wrapper obs shapes and dtypes match SB3's expected dict observation space
  - ManipulationExtractor features_dim matches SB3 policy's expected input dim
  - Gradient routing: 0 encoder params in actor optimizer, all in critic optimizer
  - Save/load round-trip: predictions match within 1e-6 tolerance
  - SB3 training with DrQv2SACPolicy produces non-NaN losses and finite Q-values

- **Lab mode:** `--verify` across all three lab files + `--probe` for gradient
  routing comparison with SB3 default.
  ```bash
  bash docker/dev.sh python scripts/labs/pixel_wrapper.py --verify
  bash docker/dev.sh python scripts/labs/manipulation_encoder.py --verify
  bash docker/dev.sh python scripts/labs/drqv2_sac_policy.py --verify
  bash docker/dev.sh python scripts/labs/drqv2_sac_policy.py --probe
  ```

- **Narrative bridge (for the writer):**
  After showing the verify/probe outputs, explain what SB3 does with these
  components: the ManipulationExtractor becomes the `features_extractor` in
  SB3's policy; the DrQv2SACPolicy overrides SB3's default gradient routing;
  the pixel wrapper produces observations that SB3's `DictReplayBuffer` stores.
  Map SB3 TensorBoard metrics to Build It components: `train/critic_loss` is
  the TD error through the CriticEncoderCritic's encoder; `train/actor_loss`
  uses features detached by CriticEncoderActor; `rollout/success_rate` is the
  end-to-end pipeline using all components together.

---

## What Can Go Wrong

| Symptom | Likely cause | Diagnostic |
|---------|-------------|------------|
| Flat at 5% after 3M+ steps | Missing `--critic-encoder` flag; encoder not receiving critic gradients | Run `drqv2_sac_policy.py --verify` to confirm gradient routing; check train command includes `--critic-encoder` |
| OOM during training | Pixel replay buffer exceeds available RAM; 500K buffer needs ~40 GB | Reduce `--buffer-size` to 300K (~24 GB) or 200K (~16 GB); run one training container at a time; check `free -h` before launch |
| `RuntimeError: Unable to sample before end of first episode` on resume | `learning_starts` not shifted past checkpoint steps | Ensure ch09_pixel_push.py sets `learning_starts = num_timesteps + 1000` on resume |
| Flat at 5% WITH `--critic-encoder` after 2M steps | Normal pre-hockey-stick phase; representation learning not yet complete | Extend to 4-5M steps; check TensorBoard for slow upward trend at 1.5-2M; monitor critic_loss (should be declining toward 0.04-0.07 -- this is Phase 1, normal) |
| `critic_loss` declining monotonically for 3M+ steps, success still flat | Critic memorizing uniform Q ~ -18.5; not learning value structure | Check `--critic-encoder` is set; verify buffer_size >= 300K; increase `--her-n-sampled-goal` to 8; if all correct, may need longer training (5M+) |
| Full-state control (`--full-state`) at 0% success | Missing object dynamics in observation | Ensure `--full-state` is set (uses full 25D obs), not a custom wrapper that strips object velocity |
| DrQ + SpatialSoftmax stuck at 3% | Augmentation-representation conflict: +/-4px shift corrupts SpatialSoftmax coordinates (40-80% noise-to-signal ratio) | Use `--no-drq` with SpatialSoftmax; DrQ is only compatible with flatten-based feature extraction |
| Very slow FPS (< 15 fps) | Multiple Docker containers competing for CPU/RAM; or rendering at 480x480 with PIL resize | Run one container at a time; add `--native-render` for direct 84x84 MuJoCo rendering; expect 25-35 fps solo |
| TensorBoard shows mixed/confusing curves | Log contamination from multiple runs with same config tag | Back up AND delete old TensorBoard directories (`runs/ch09_*`); use unique config tags for each experiment |
| `--compare-sb3` or `--bridge` shows encoder in actor optimizer | Using SB3's default SACPolicy instead of DrQv2SACPolicy | Confirm `--critic-encoder` flag routes to `DrQv2SACPolicy`; run `--probe` mode to compare default vs override |
| Hockey-stick at 2.2M but convergence stalls at 50-60% | Buffer too small; early diverse exploration data overwritten before convergence | Increase `--buffer-size` to 500K (max safe on 120 GB system); check `--her-n-sampled-goal 8` |

---

## Adaptation Notes

### Cut from tutorial

- **Long Docker command blocks (6 occurrences, ~60 lines total):** The tutorial
  repeats the full `docker run --rm -e MUJOCO_GL=egl ...` invocation for every
  experiment step. Replace with the `bash docker/dev.sh` shorthand established
  in Ch1. Show the full Docker command once (in a sidebar or tip) for readers
  who need it, then use the short form.

- **Section 9.16 "Full-State Control" as a separate section (~20 lines):** This
  is a pipeline validation step, not a chapter-level concept. Fold into the
  "What Can Go Wrong" section as a diagnostic command, or mention it briefly in
  the opening Bridge when establishing that Ch5's pipeline is validated.

- **Section 9.14 "Resume from Checkpoint" (~25 lines):** Checkpoint resume is
  an engineering detail. Compress to a tip box (3-4 lines) within Run It, or
  move to What Can Go Wrong under "Training interrupted."

- **References section (standalone, 8 entries):** Move to inline citations
  within the text per Manning convention. No standalone references section.

- **The `--8<--` snippet-include markers in HOW sections (Sections 9.5-9.6):**
  In the tutorial, code appears mid-narrative in the HOW debugging journey. For
  the book, Build It code listings should be in the Build It sections (9.2-9.5);
  the investigation narrative (9.7 Run It) should reference back to Build It:
  "the encoder we built in Section 9.3."

### Keep from tutorial

- **Section 9.1 "Why Push from Pixels Is Harder" (~300 words):** The four
  compounding challenges (tiny objects, sparse rewards, two learning problems,
  contact dynamics). This is the WHY section's core content. Keep and tighten.

- **Section 9.2 "Visual HER: Two Different Things" (~200 words):** The
  goal_mode="both" vs full visual HER distinction. Critical concept for
  understanding the architecture. Keep as a callout or Definition box.

- **Section 9.3 "The Observation Structure" (~200 words):** The dict obs
  diagram and proprioception rationale. Keep as a figure/diagram + brief text.

- **The 5-step investigation arc (Steps 0-4, Sections 9.4-9.8, ~3000 words):**
  This IS the chapter. The failure-first progression is the pedagogical spine.
  Keep the structure but recast as Run It experiments that reference Build It
  components.

- **The three-phase loss signature (Section 9.7, ~600 words):** Novel content
  not found elsewhere. The Phase 1/2/3 interpretation table is essential for
  readers monitoring their own training. Keep as a major section with a figure.

- **The DrQ + SpatialSoftmax noise analysis (Section 9.8, ~400 words):** The
  quantitative noise-to-signal ratio calculation. Keep -- this is the
  "receipts" for the negative result claim.

- **What Can Go Wrong table (8 entries):** Keep and expand to 10+ entries per
  the scaffold table above.

- **All five exercises:** Keep and adapt numbering for Manning format. These are
  well-graduated and cover verify, tweak, explore, and challenge levels.

### Add for Manning

- **Chapter Bridge (from Ch8 / previous chapter):** Chapter 8 established
  robustness testing -- noise injection, degradation curves, brittleness
  fingerprints. Gap: all that work assumed the agent observes state vectors
  directly. Real robots see pixels. This chapter adds the visual pipeline:
  pixel wrappers, CNN encoders, gradient routing, and the patience to let
  representation learning complete.

- **Opening Promise:** 3-5 bullet "This chapter covers" block. Draft below.

- **Build It sections with math-before-code pattern (Sections 9.2-9.5):** The
  tutorial puts code in the HOW narrative. For Manning, create dedicated Build
  It sections with the standard pattern: equation/concept -> code listing ->
  verification checkpoint. 11 components across 4 Build It sections.

- **Bridge section (9.6):** Connect Build It to SB3. Show that all from-scratch
  components are what SB3 uses internally. Map TensorBoard metrics to Build It
  components.

- **Figure: NatureCNN vs ManipulationCNN spatial progression:** Side-by-side
  diagram showing how a 5-pixel puck is processed through each encoder's
  layers. This makes the "stride-4 destroys small objects" argument visual.

- **Figure: Three-phase loss signature:** Learning curve with critic_loss,
  actor_loss, and success_rate on the same time axis. Annotate the three
  phases directly on the figure with color regions.

- **Figure: The 5-step investigation summary:** A comparison chart or table
  figure showing all 5 steps side-by-side with success rates, highlighting
  which component was added at each step and whether it helped.

- **Sidebar: Pixel replay buffer memory math:** The memory calculation from
  `tasks/critic_encoder_status.md` (169 KB per transition, 500K buffer = ~40
  GB). This is practical engineering knowledge that readers need. 4-5 lines.

- **Sidebar: The hockey-stick explained (3 mechanisms):** Condensed from
  `tasks/hockey_stick_research.md`. Value propagation bottleneck, geometric
  phase transition, positive feedback loop. ~300-400 words + citation of
  Laidlaw et al. (2024) for effective horizon.

---

## Chapter Bridge

1. **Capability established:** Through Chapters 3-8, you built SAC from scratch,
   added HER for sparse rewards, solved Push at 89% from state vectors, tested
   robustness under noise, and learned to tune hyperparameters systematically.
   You have a working, validated, robust manipulation pipeline -- but it
   assumes the agent directly observes state vectors (joint positions, object
   coordinates, goal positions).

2. **Gap:** Real robots do not observe state vectors. They see pixels from
   cameras. Can the pipeline you built survive the switch from 25D vectors to
   84x84 images? The answer is: not without significant changes to the encoder,
   the gradient routing, and the training budget.

3. **This chapter adds:** A visual observation pipeline (pixel wrappers, CNN
   encoders, spatial feature extraction), the gradient routing that makes the
   encoder actually learn (critic-encoder pattern from DrQ-v2), and the
   diagnostic skills to read pixel RL training curves. You will discover these
   through a 5-step investigation: try the obvious thing, watch it fail,
   diagnose why, fix one component at a time, and arrive at a working solution.

4. **Foreshadow:** This chapter achieves 95% Push from pixels, matching state-
   based performance at the cost of 2-4x more training steps. The
   representation learning phase is an unavoidable overhead when learning from
   scratch. Chapter 10 will explore how pre-trained encoders and world models
   can reduce this tax -- and whether HER itself is the right abstraction for
   visual goal-conditioned RL.

---

## Opening Promise

> **This chapter covers:**
>
> - Building a complete visual observation pipeline from scratch -- pixel
>   wrappers, CNN encoders, spatial feature extraction, and memory-efficient
>   replay buffers -- that turns 84x84 camera images into features a policy
>   can act on
> - Understanding why NatureCNN fails on manipulation tasks (stride-4 destroys
>   5-pixel objects) and implementing ManipulationCNN + SpatialSoftmax as the
>   fix (the "right eyes")
> - Discovering why architecture alone is not enough: SB3's default gradient
>   routing starves the encoder of learning signal, and the DrQ-v2 pattern
>   (encoder in the critic's optimizer) is the 15 lines that make the
>   difference between 5% and 95%
> - Reading the three-phase loss signature in pixel RL -- declining losses with
>   flat success means the critic is memorizing failure, not learning structure;
>   rising losses during the hockey-stick means real value learning has begun
> - Running a 5-step progressive investigation that arrives at a pixel Push
>   agent achieving 95%+ success, with each failure teaching a transferable
>   debugging principle

---

## Figure Plan

| # | Description | Type | Source command | Chapter location |
|---|------------|------|---------------|-----------------|
| 1 | FetchPush-v4 pixel observation: raw 84x84 camera image showing gripper, puck, and goal target. Annotated with object sizes in pixels (puck ~5px, gripper ~4px) and obs dict structure (pixels, proprioception, goals). | screenshot + annotation | `python scripts/capture_proposal_figures.py env-setup --envs FetchPush-v4` + manual annotation or custom script rendering 84x84 frame | After opening bridge / Section 9.1 WHY |
| 2 | NatureCNN vs ManipulationCNN spatial progression: two side-by-side columns showing how a 5-pixel puck survives (or does not survive) each layer. NatureCNN: 84->20->9->7 (puck -> 1px). ManipulationCNN: 84->42->42->42->21 (puck -> 3px). | diagram | matplotlib diagram generated in Build It section or by custom figure script | After Section 9.3.1/9.3.2 (encoder architecture Build It) |
| 3 | Three-phase loss signature: time-series plot with three subplots sharing the x-axis (training steps). Top: success_rate showing flat->hockey-stick->saturation. Middle: critic_loss showing decline->rise->decline. Bottom: actor_loss showing ~0->rise->negative. Three color regions annotating Phase 1 (failure), Phase 2 (hockey-stick), Phase 3 (convergence). | curve (multi-panel) | TensorBoard export from `runs/ch09_manip_noDrQ_criticEnc/FetchPush-v4/seed0/` processed with matplotlib | After Section 9.8 (Reading the training curve) |
| 4 | Five-step investigation summary: comparison bar chart or table figure showing success rate at 2M steps (or final) for each step. Step 0 (NatureCNN): 5%. Step 1 (ManipCNN+SS): 5%. Step 2 (+ critic-encoder): 95% at 4.4M. Step 4 (+ DrQ): 3%. Clear visual showing that Step 2 is the breakthrough. | comparison chart | matplotlib figure generated from experiment data | After Section 9.7 (the 5-step investigation, before Summary) |

---

## Estimated Length

| Section | Words |
|---------|-------|
| Opening promise + chapter bridge | 500 |
| 9.1 WHY: The pixel Push problem (4 challenges, visual HER distinction, obs structure) | 1,200 |
| 9.2 Build It: Visual observation pipeline (render_and_resize, PixelObsWrapper, pixel replay buffer) | 1,200 |
| 9.3 Build It: Encoder architecture (NatureCNN, ManipulationCNN, SpatialSoftmax, ManipulationExtractor, proprio passthrough) | 1,500 |
| 9.4 Build It: Data augmentation (DrQ random shift, DrQ replay buffer, HER+DrQ buffer) | 800 |
| 9.5 Build It: Gradient routing (the problem, DrQ-v2 pattern, CriticEncoderCritic, CriticEncoderActor, DrQv2SACPolicy) | 1,000 |
| 9.6 Bridge: From-scratch to SB3 (verification, metric mapping) | 500 |
| 9.7 Run It: The 5-step investigation (Steps 0-4, experiment cards, results tables) | 2,000 |
| 9.8 Reading the training curve (three-phase loss signature, training budget rule, hockey-stick sidebar) | 800 |
| 9.9 What Can Go Wrong (10+ failure modes) | 800 |
| 9.10 Summary + bridge to Ch10 | 400 |
| Reproduce It block | 200 |
| Exercises (5 exercises) | 400 |
| **Total** | **~10,300** |

(Target range: 6,000-10,000 words. At ~10,300 this is at the upper bound.
The chapter has 12 Build It components and a 5-step investigation, which
justifies the length. Code listings counted separately by Manning but included
in overall page count estimate. If needed, the DrQ Build It section (9.4) can
be compressed to ~500 words since DrQ is a negative result; or the hockey-stick
sidebar can be trimmed.)

---

## Concept Registry Additions

Terms this chapter introduces (to be added to the Concept Registry under Ch9):

- **pixel observation wrapper**: wraps a Gymnasium env to replace vector obs with (C, H, W) uint8 pixel arrays from MuJoCo camera rendering
- **goal mode (none/desired/both)**: controls which goal vectors appear alongside pixel observations; "both" = achieved_goal + desired_goal as 3D vectors, pixels for control
- **render_and_resize**: function that calls MuJoCo camera, transposes HWC->CHW, stacks frames, produces uint8 tensor
- **NatureCNN encoder (Mnih et al. 2015)**: 3-layer CNN with 8x8 stride-4 first layer; designed for Atari; destroys small manipulation objects
- **sample-efficiency ratio (rho)**: ratio of training steps needed for pixel vs state agents to reach equivalent performance; typically 2-4x for manipulation
- **DrQ (random shift augmentation)**: pad image by k pixels (replicate padding), random crop back to original size; augments obs and next_obs independently at sample time
- **replicate padding**: padding mode that copies edge pixels (vs zero padding); preserves image statistics at boundaries
- **DrQ replay buffer**: replay buffer that applies random shift augmentation at sample time, not at store time
- **uint8 pixel storage**: storing pixel observations as 8-bit unsigned integers (1 byte/pixel vs 4 bytes for float32); critical for pixel replay buffer memory
- **native resolution rendering**: rendering MuJoCo at target resolution (84x84) directly instead of rendering high-res then downsampling
- **SubprocVecEnv (parallel envs)**: SB3 vectorized environment that runs each env in a separate process; needed for pixel envs where rendering is slow
- **replay ratio / gradient steps**: number of gradient updates per environment step; default 1 for SAC
- **deceptively dense reward**: a dense reward whose gradient points toward the goal but does not require solving the manipulation task (e.g., moving arm toward goal without pushing object)
- **HerDrQDictReplayBuffer**: composed replay buffer combining HER goal relabeling with DrQ augmentation
- **visual HER synthesis**: architecture where policy sees pixels but HER operates on 3D goal vectors; information asymmetry by design
- **information asymmetry (policy sees pixels, HER sees vectors)**: the deliberate design choice to test pixel control capability while keeping goal relabeling in vector space
- **value wavefront (Bellman diffusion through goal space)**: metaphor for how Q-value information propagates one Bellman backup step at a time from goals the agent can reach to goals it cannot yet reach
- **hockey-stick learning curve (geometric phase transition + positive feedback)**: the characteristic curve in sparse-reward goal-conditioned RL where success is flat for many steps, then jumps rapidly once the competence region overlaps the test goal distribution
- **critical competence radius**: the radius of the agent's competence region at which it begins overlapping with the test goal distribution, triggering the hockey-stick
- **effective horizon k* (Laidlaw et al. 2024)**: the minimum number of value iteration steps needed before greedy actions become near-optimal; sample complexity is exponential in k*

---

## Dependencies

- **Lab regions needed (for Lab Engineer):**
  All regions already exist across 5 lab files:

  `scripts/labs/pixel_wrapper.py`:
  - `render_and_resize` -- MuJoCo camera -> CHW uint8
  - `pixel_obs_wrapper` -- goal modes, frame stacking, proprio passthrough
  - `pixel_replay_buffer` -- uint8 storage, float32 at sample time
  - Modes: `--verify` (all checks, < 2 min CPU)

  `scripts/labs/manipulation_encoder.py`:
  - `spatial_softmax` -- "where" not "what"
  - `manipulation_cnn` -- 3x3 stride-2 encoder
  - `manipulation_extractor` -- SB3-compatible feature extractor
  - Modes: `--verify` (shape/value checks, < 2 min CPU)

  `scripts/labs/image_augmentation.py`:
  - `random_shift_aug` -- DrQ pad-and-crop
  - `drq_replay_buffer` -- augment at sample time
  - `her_drq_replay_buffer` -- HER + DrQ composed
  - Modes: `--verify` (augmentation checks, < 2 min CPU)

  `scripts/labs/visual_encoder.py`:
  - `nature_cnn` -- the "wrong" encoder (for comparison)
  - `normalized_combined_extractor` -- normalized multi-modal extractor
  - `visual_goal_encoder`, `visual_gaussian_policy`, `visual_twin_q_network` -- visual network components
  - Modes: `--verify` (shape/value checks, < 2 min CPU)

  `scripts/labs/drqv2_sac_policy.py`:
  - `critic_encoder_critic` -- always enables gradients through encoder
  - `critic_encoder_actor` -- detach features before policy MLP
  - `drqv2_sac_policy` -- encoder in critic optimizer, shared encoder routing
  - Modes: `--verify` (5 gradient routing tests, < 2 min CPU), `--probe` (SB3 gradient probe for comparison)

  **Lab Engineer note:** The existing lab code appears to have all regions
  in place. The Lab Engineer should verify:
  1. All `--verify` modes run and pass on CPU in < 2 min each
  2. Consider adding a `--demo` mode to `drqv2_sac_policy.py` that runs
     a short training (~5K steps) with ManipulationExtractor + DrQv2SACPolicy
     on FetchPush-v4 to confirm the full pipeline initializes and produces
     non-NaN losses. (This is NOT expected to learn -- 5K steps is far too
     few. It just tests that the wiring works.)
  3. Consider adding a `--bridge` alias mode that combines the verify and
     probe outputs into a single comparison summary.

- **Pretrained checkpoints needed (for Reproduce It):**
  - `checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed0.zip` -- primary (seed 0 complete at 95%+)
  - `checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed1.zip` -- queued
  - `checkpoints/ch09_manip_noDrQ_criticEnc_FetchPush-v4_seed2.zip` -- queued
  - Periodic checkpoints at 500K intervals (for checkpoint track readers who want to evaluate at different training stages)
  - NatureCNN baseline checkpoint (for Step 0 comparison): existing from earlier runs
  - DrQ + SpatialSoftmax checkpoint (for Step 4 comparison): Cell D data at 1.54M

- **Previous chapter concepts used:**
  From Ch1 (Setup): Docker dev.sh workflow, proof of life, rendering backend (EGL)
  From Ch2 (Env Anatomy): goal-conditioned observation, dictionary observation structure, compute_reward API, dense vs sparse reward
  From Ch3 (PPO): actor-critic architecture, value function V(s), Q-function Q(s,a)
  From Ch4 (SAC): off-policy learning, replay buffer, maximum entropy objective, temperature alpha, twin Q-networks, soft update / Polyak averaging, squashed Gaussian policy, SAC, Bellman target y
  From Ch5 (HER): hindsight experience replay, goal relabeling, goal sampling strategies (future), n_sampled_goal, off-policy requirement for HER
  From Ch6 (PickAndPlace): multi-phase control (approach, contact, push)
  From Ch7 (Robustness): degradation curve, noise injection
  From Ch8 (Tuning): training budget, hyperparameter selection

- **Production script used:** `scripts/ch09_pixel_push.py` (existing; flags: `--critic-encoder`, `--no-drq`, `--buffer-size`, `--her-n-sampled-goal`, `--full-state`, `--resume`, `--checkpoint-freq`)

---

## Exercises

**1. (Verify) Run the full-state control baseline.**

Confirm that the Ch9 script wiring is correct independently of pixel processing:

```bash
bash docker/dev.sh python scripts/ch09_pixel_push.py train \
  --seed 0 --full-state --total-steps 2000000
```

Expected: success_rate >= 0.85 at 2M steps, matching Ch5's state Push results.
If this fails, the problem is in the script, not the visual pipeline.

**2. (Tweak) HER relabeling strength.**

Change `--her-n-sampled-goal` from 8 to 4 (the Ch5 default). Run for 3M steps
with the winning config. Does the hockey-stick onset shift later? What about
final success rate?

Expected: Later inflection (more steps needed to build value wavefront with
fewer relabeled transitions). Final success may be similar but convergence is
slower. Quantify: at what step does success first exceed 20% with HER-4 vs
HER-8?

**3. (Explore) SpatialSoftmax ablation.**

Run with `--no-spatial-softmax` (flatten + linear instead of SpatialSoftmax).
Does the agent still learn? The flatten pathway produces 50D features instead of
64D spatial coordinates. What does this tell you about whether SpatialSoftmax is
a necessary component or a helpful inductive bias?

Expected: Likely still learns (ManipulationCNN + critic-encoder may be
sufficient), but possibly slower or lower final success. The result tells you
whether the "where not what" inductive bias is critical or supplementary.

**4. (Challenge) Test DrQ with flat features.**

Run WITH DrQ but WITHOUT SpatialSoftmax (`--no-spatial-softmax`, without
`--no-drq`). DrQ was designed for flat CNN features, not spatial coordinates.
Does removing SpatialSoftmax make DrQ useful again?

Expected: If DrQ helps with flat features, the narrative is "DrQ and
SpatialSoftmax are separately good ideas but fundamentally incompatible." If
DrQ still hurts, the narrative is "DrQ's random shift is harmful for
manipulation regardless of feature type."

**5. (Challenge) Measure gradient magnitudes.**

Read the `verify_gradient_flow()` function in
`scripts/labs/drqv2_sac_policy.py`. Modify it to measure the actual gradient
MAGNITUDE (L2 norm) flowing through the encoder during critic vs actor training.
How much larger is the critic gradient? This quantifies the argument in Section
9.5 that the critic provides a richer learning signal than the actor.

Expected: Critic gradient magnitude should be substantially larger than actor
gradient (before the detach). The ratio quantifies why critic-encoder routing
matters. Report the ratio as a concrete number.
