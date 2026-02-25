 Current Read (Repo)

  - Training script uses ManipulationExtractor (DrQ‑v2‑style convs + optional SpatialSoftmax), PixelObservationWrapper with optional proprio passthrough, HER, and DrQ random‑shift
    aug in a custom replay buffer. Flags include --share-encoder and --native-render. See:
      - scripts/ch09_pixel_push.py:497 and :365 for passing share_features_extractor into SB3 policy
      - scripts/labs/manipulation_encoder.py:220 for SpatialSoftmax; :120 for ManipulationCNN; :316 for ManipulationExtractor
      - scripts/labs/pixel_wrapper.py:118 for wrapper design; :170 for proprio_indices handling
  - Handoff summary matches code and docs: NatureCNN replaced; SpatialSoftmax + proprio + DrQ implemented; smoke tests pass; primary pixel run stalled around 3% at ~2M steps;
    vector‑only POMDP baseline fails (by design).
  - Key pending decisions: (1) Run 2 (share encoder) not launched; (2) Full‑state control not done to validate pipeline; (3) Gating Run 3 on Run 2.

  What The Literature Says (salient to your setup)

  - Pixel RL baselines: DrQ (data-regularized Q) and DrQ‑v2 (improvements incl. shared encoder, low‑variance updates, random shift aug) are state of the art for DMControl from
    pixels; they emphasize gentle downsampling (3×3 kernels, stride 2 first layer) and strong augmentation. (arxiv.org (https://arxiv.org/abs/2004.13649))
  - Spatial keying for manipulation: Spatial Softmax produces explicit (x,y) feature coordinates and has a long record in visuomotor control (e.g., PR2 insertion, screwing,
    pushing) with end‑to‑end training. (arxiv.org (https://arxiv.org/abs/1504.00702))
  - Sparse‑reward credit assignment: HER is the canonical method, but almost all successful “visual HER” variants avoid learning control purely end‑to‑end from raw pixels + sparse
    rewards. Common patterns: learn a latent (VAE/contrastive) and do HER in that latent (RIG/Skew‑Fit line). (arxiv.org (https://arxiv.org/abs/1707.01495))
  - Representation learning that helps pixels: CURL (contrastive), RAD/DrQ (augmentations), and SPR (self‑predictive) improve sample‑efficiency by stabilizing encoders. These do
    not by themselves solve sparse GCRL but are strong ingredients. (arxiv.org (https://arxiv.org/abs/2004.04136))
  - Pretrained visual encoders for manipulation: R3M/VIP/VC‑1 provide frozen or finetuned features that substantially improve robotic manipulation from pixels; they are widely
    used when end‑to‑end training is brittle. (arxiv.org (https://arxiv.org/abs/2203.12601))
  - Gymnasium‑Robotics multi‑goal API: FetchPush observations are dicts with 25D “observation” plus achieved/desired goals (HER support via compute_reward). Useful for verifying
    your control runs and that you’re not accidentally changing the MDP. (robotics.farama.org (https://robotics.farama.org/content/multi-goal_api/))
  - Important SB3 nuance on “shared encoder”: in SB3, “sharing” the feature extractor between actor and critic is primarily an architectural and compute choice; by default the
    shared extractor is optimized via the policy loss, not the critic loss (this was confirmed by maintainers in issues). That means simply setting share_features_extractor=True
    may not route critic gradients into the CNN the way DrQ implementations do. (github.com (https://github.com/DLR-RM/stable-baselines3/blob/master/docs/guide/custom_policy.rst))

  Gaps vs. Your Setup (rooted in findings)

  - Encoder architecture: You already fixed NatureCNN’s stride‑4 issue with a DrQ‑v2‑like stem and offered SpatialSoftmax—good alignment with literature.
  - Gradient flow: Your “H1: gradient flow” hypothesis is plausible; however, SB3’s default shared‑extractor behavior likely won’t realize “critic‑updates‑CNN” unless you
    explicitly change which loss updates the shared trunk or freeze one path. (github.com (https://github.com/DLR-RM/stable-baselines3/issues/691))
  - Pipeline control: The vector‑only 16D run is a POMDP (missing object dynamics). A fair control needs the full 25D state + goals to validate Chapter 9 wiring against your
    known‑good Ch4 behavior. (robotics.farama.org (https://robotics.farama.org/envs/fetch/push/))
  - Visual HER from scratch: As your research notes, few (if any) published results show end‑to‑end HER + pixels succeeding on manipulation without auxiliary representation
    learning. RIG/Skew‑Fit style methods change goal handling to latent images, which differs from your narrative but is a pragmatic fallback. (arxiv.org (https://arxiv.org/
    abs/1807.04742))

  Decisions (recommend)

  - D1. Verify pipeline first with a full‑state control run (FetchPush full 25D + goals) to reproduce the hockey‑stick signature; this isolates training/wiring from perception.
    (robotics.farama.org (https://robotics.farama.org/envs/fetch/push/))
  - D2. Test your gradient‑flow hypothesis, but not only via --share-encoder: implement “critic‑updates‑encoder” specifically (actor gradients to encoder off, critic gradients on)
    to match DrQ/DrQ‑v2 practice. Cite/borrow the drqv2 pattern if needed. (github.com (https://github.com/facebookresearch/drqv2))
  - D3. If D2 fails to produce an inflection, introduce an auxiliary representation objective (pick one: DrQ‑style strong aug only; or CURL/SPR; keep HER) before resorting to
    latent‑goal methods. (arxiv.org (https://arxiv.org/abs/2004.04136))
  - D4. As a higher‑leverage fallback (narrative‑changing), try a frozen pretrained encoder (R3M or VC‑1) with pixels+HER. This is the fastest path to success if pure end‑to‑end
    remains brittle. (arxiv.org (https://arxiv.org/abs/2203.12601))

  Two‑Week Experiment Plan

  - Phase 0: Full‑State Control (day 1–2)
      - Run Ch4 or Ch9 with full 25D observation (not 16D). Expect ~hockey‑stick rise by ~1.5–2.0M steps; use it as a control time‑to‑inflection. (robotics.farama.org (https://
        robotics.farama.org/envs/fetch/push/))
  - Phase 1: Critic‑Updates‑Encoder (days 2–6)
      - P1.1 Implement custom policy so shared CNN is updated by critic loss only; actor reads the shared CNN with stop‑grad on that path.
      - P1.2 Launch two 8M‑step runs on FetchPush:
          - A: SpatialSoftmax + proprio + DrQ aug + fs=4 + critic‑updates‑CNN
          - B: Same but no SpatialSoftmax (flatten+LN+Tanh)
      - Stop rule: abort at 1.0M if success ≤10% and critic loss monotonically decreases (your failure signature).
  - Phase 2: Aux Representation (days 6–10)
      - P2.1 DrQ baseline (keep your encoder, keep HER, use strong random shift aug).
      - P2.2 CURL head (contrastive on encoder latents using frame‑shifted views) + HER.
      - Expect faster critic shaping; watch for first increase in critic loss followed by reward drop toward −30/−20. (proceedings.neurips.cc (https://proceedings.neurips.cc/
        paper_files/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf))
  - Phase 3: Pretrained Encoder Fallback (days 10–12)
      - P3.1 Frozen R3M/VC‑1 backbone to 84×84 (or 224→project to 50D) + HER; train only SB3 heads. This often yields immediate stability and success on robot manipulation.
        (arxiv.org (https://arxiv.org/abs/2203.12601))
  - Phase 4: Cleanups & Checks (days 12–14)
      - P4.1 Native render vs PIL resize ablation (speed + invariance); P4.2 CoordConv first layer (cheap positional prior). (arxiv.org (https://arxiv.org/abs/1807.03247))

  Ablation Matrix (minimal, highest signal first)

  - Gradient routing: shared vs critic‑only vs actor‑only update to CNN.
  - Spatial head: SpatialSoftmax on/off (keep DrQ‑v2 stem constant). (arxiv.org (https://arxiv.org/abs/1504.00702))
  - Proprioception: on/off (expect on > off).
  - Augmentation: DrQ random shift on/off. (openreview.net (https://openreview.net/pdf?id=GY6-6sTvGaf))
  - Frame stacking: 1 vs 4 (velocity signal).
  - Pretrained features: frozen R3M/VC‑1 vs learned‑from‑scratch. (arxiv.org (https://arxiv.org/abs/2203.12601))

  Stop Rules and Success Metrics

  - Early abort: if at 1.0M steps success ≤10% AND critic loss is flat/declining, stop (your failure signature).
  - Working signature: critic loss rises as the policy discovers new states, then stabilizes; reward improves toward −30/−20; sustained success >15% by ~1.2–1.5M, rising
    afterward.
  - Final target: ≥80–90% by 2–3M on FetchPush (mirrors state baseline shape, acknowledging pixels may lag).

  Implementation Pointers (surgical)

  - Pipeline control run
      - Either run scripts/ch04_her_sparse_reach_push.py as-is (preferred) or adapt ch09 to pass the full 25D “observation” (not 16D) in vector mode to match MDP used in Ch4.
        FetchPush obs spec: see robotics.farama.org push docs. (robotics.farama.org (https://robotics.farama.org/envs/fetch/push/))
  - Critic‑updates‑encoder in SB3
      - Current shared flag is wired at scripts/ch09_pixel_push.py:497 and :365. SB3’s default behavior does not ensure critic gradients dominate (see maintainer note). Implement
        a custom SAC policy that:
          - Builds a single features extractor module (CNN) shared by actor and critic
          - Feeds features to actor through a stop‑gradient (detach) path
          - Optimizes CNN parameters only in the critic optimizer step
          - Keeps everything else (HER buffer, DrQ aug) unchanged
      - Rationale and caution: SB3 issues indicate the default shared path uses policy loss—or, at least, not clearly the critic—for the shared trunk. Explicitly control gradient
        flow to match DrQ‑v2 practice. (github.com (https://github.com/DLR-RM/stable-baselines3/issues/691))
  - SpatialSoftmax and dims
      - Your ManipulationExtractor produces 64 spatial + vectors (10 + 3 + 3) = 80D when SpatialSoftmax is on; confirm via scripts/labs/manipulation_encoder.py:366 verification.
  - Native render switch
      - You already expose --native-render; it can reduce aliasing and speed up renders (no PIL). Keep it as a secondary ablation; do not entangle it with gradient‑flow tests.

  If Everything Above Fails

  - Switch goal handling to a latent (RIG‑style): train a VAE on images, encode observations and goals to z, and run HER in z‑space with a learned distance. This departs from your
    current narrative but is the pragmatic “visual HER” solution shown to work. (arxiv.org (https://arxiv.org/abs/1807.04742))

  Why this should work (tie‑back)

  - You already fixed the spatial bottleneck (NatureCNN stride‑4) by adopting a DrQ‑v2 style stem; literature suggests the next bottleneck is optimization pressure on the encoder
    (critic signal + augmentation). The plan first validates the pipeline, then enforces the right gradient routing, then adds the minimal rep‑learning auxiliary (DrQ/CURL), and
    finally uses a robust frozen encoder if needed. (arxiv.org (https://arxiv.org/abs/2107.09645))

--------

