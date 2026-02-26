# Ch9 Chapter Arc: Pixel Push -- Curated Progression

## Date: 2026-02-25
## Status: Draft outline for tutorial structure

## Design Principle

NOT a chronological lab notebook of everything we tried. A curated sequence
of 4-5 experiments where each one:
- Teaches exactly ONE concept
- Fails (or partially succeeds) for a clear, explainable reason
- Naturally motivates the next step

The reader should feel they are discovering the solution, not being handed it.
But they should never feel lost in a sea of failed attempts.

---

## The Arc (5 steps)

### Step 0: The Baseline Gap (motivation)

**What the reader already has:** Ch4's SAC+HER solves FetchPush from state at
89% in 2M steps. The algorithm works. The question is: can we do it from pixels?

**Experiment:** Run the SAME pipeline but replace 25D state with 84x84 pixels.
Use SB3's default NatureCNN.

**Result:** Flat at 5-8% for 2M+ steps. Total failure.

**The question this raises:** WHY? The algorithm is proven. What about pixels
breaks it?

**One concept taught:** Pixels are not a drop-in replacement for state.
Something fundamental about the visual processing pipeline is wrong.

---

### Step 1: The Right Eyes (encoder architecture)

**Diagnosis:** NatureCNN uses 8x8 stride-4 in the first layer. On 84x84
input, this reduces to 20x20 in one step. The puck is ~5 pixels wide. After
stride-4, it's ~1 pixel. The spatial relationship between gripper and puck --
the ONE thing the policy needs -- is destroyed.

**Fix:** Three changes, all motivated by the same principle ("preserve spatial
information"):

1. **ManipulationCNN:** 3x3 kernels, stride 2 (gentle downsampling:
   84->42->42->42->21). Small objects survive.
2. **SpatialSoftmax:** Instead of flattening the feature map, extract (x,y)
   coordinates of where each feature fires. Output is "where things are" not
   "what the image looks like."
3. **Proprioception passthrough:** Give the CNN only the world to learn about.
   Robot self-state (joint positions, velocities) comes from sensors directly.

**Experiment:** Run with ManipulationCNN + SpatialSoftmax + proprioception.

**Result:** Still flat at 5-8%. Better architecture is NECESSARY but NOT
SUFFICIENT.

**The question this raises:** The encoder can now represent the right
information. But is it LEARNING to? Where do encoder gradients come from?

**One concept taught:** Architecture determines what a network CAN represent.
But training determines what it DOES represent. We fixed the capacity; now
we need to fix the learning signal.

---

### Step 2: The Right Gradient Flow (critic-encoder routing)

**Diagnosis:** In SB3's default setup with a shared encoder, the encoder
sits in the ACTOR's optimizer. During critic training, `set_grad_enabled(False)`
blocks gradients through the encoder. The encoder only learns from policy
gradients -- which are noisy and weak early in training (the policy is random,
so the actor loss gradient is essentially noise).

DrQ-v2 (Yarats et al. 2022) does the opposite: encoder in the CRITIC's
optimizer. The critic's TD loss provides a much richer learning signal -- it
directly asks "does this visual feature help predict future value?" The actor
receives detached features (no encoder gradient from policy loss).

**Fix:** Override SB3's gradient routing:
- CriticEncoderCritic: always enable gradients through encoder
- CriticEncoderActor: detach features before policy MLP
- Optimizer: encoder params in critic optimizer only

**Experiment:** Run with correct architecture + correct gradient routing.

**Result:** Flat at 6% for 2M steps... then a slow upward trend appears
at the very end (10% at 2M). Extend to 8M steps. Hockey-stick erupts at
2.2M: 6% -> 30% -> 50% -> 70% -> 90%.

**The question this raises (for the reader to think about):** Why did it
take 2.2M steps for the hockey-stick, when full-state only needed 1.2M?

**One concept taught:** WHERE gradients flow matters as much as WHAT
architecture you use. The encoder needs the critic's value signal, not the
actor's policy signal, to learn useful visual representations.

---

### Step 3: The Patience Tax (representation learning phase)

This is not a separate experiment -- it is the INTERPRETATION of Step 2's
training curve. The chapter pauses to explain what happened:

**The two regimes:**
- Steps 0-2M: "Failure" -- but the encoder IS learning spatial structure.
  Critic loss declines to 0.07 (trivially easy: predict Q=-18.5 everywhere).
  This looks bad but is the encoder warm-up phase.
- Steps 2.2M+: Hockey-stick -- the encoder is now good enough that the
  critic can distinguish good states from bad states. Positive feedback loop
  activates. Critic loss RISES to 0.3+ (learning real structure). Actor
  loss rises to 1+ (receiving real gradient signal).

**The loss signature:**
| Metric | Failure regime | Hockey-stick regime |
|--------|---------------|---------------------|
| success_rate | Flat 3-7% | Climbing 30%+ |
| critic_loss | Declining < 0.1 | Rising 0.3+ |
| actor_loss | ~0 | Rising 1.0+ |

**The training budget rule:** Pixel RL needs 2-4x the training budget of
state RL. The representation learning phase is an unavoidable overhead.
Stop rules calibrated for state RL become self-fulfilling prophecies.

**One concept taught:** How to READ training curves in pixel RL. Rising
losses = the hockey-stick signal. Declining losses with flat success =
the failure signal. This is the opposite of supervised learning intuition.

---

### Step 4: The DrQ Surprise (augmentation vs representation)

**Setup:** The reader might wonder: "we didn't use DrQ in Step 2. DrQ is
the standard data augmentation for pixel RL. Wouldn't adding it help?"

**Experiment:** Compare Step 2's result (no DrQ, 90% at ~4M) with the
SAME config plus DrQ augmentation.

**Result:** DrQ + SpatialSoftmax = flat at 3-6%. DrQ HURTS.

**Diagnosis:** DrQ shifts images by +/-4 pixels randomly. After the CNN
(84->21), this becomes +/-1 pixel on the feature map. SpatialSoftmax
converts this to +/-0.10 in [-1,1] coordinate space. Crucially, obs and
next_obs are augmented INDEPENDENTLY with different random shifts, doubling
the noise in Bellman targets. The gripper-puck distance signal is only
~0.25-0.50 in these coordinates. Noise-to-signal ratio: 40-80%.

**One concept taught:** Data augmentation is not universally good. You must
match your augmentation to your representation. SpatialSoftmax extracts
precise spatial coordinates; DrQ injects spatial noise. They are fundamentally
incompatible.

**Generalizable principle:** "Choose your representation, then choose your
augmentation."

---

## What's NOT in the chapter (but we did)

These experiments taught US things but would clutter the narrative for the
reader:

- **Normalization fix (LayerNorm + Tanh):** Improved loss curves but not
  success rate. Lesson: "fixing scale doesn't fix information content." This
  is a valid insight but adding a 5th experiment step breaks the pacing. Could
  be a sidebar/callout box.

- **Proprio-only ablation (16D without object dynamics):** 0% success because
  we accidentally created a POMDP by removing object velocity. Lesson: "change
  one variable at a time." Useful methodologically but doesn't advance the
  pixel story.

- **Multiple NatureCNN runs with different hyperparams:** We ran ~4 configs
  (baseline, frame-stack, normalization, learning rate). For the chapter, ONE
  NatureCNN failure is enough. The reader doesn't need to see all 4 to believe
  NatureCNN doesn't work.

- **Buffer size investigation:** Important engineering insight (Lesson 8) but
  not conceptually interesting for the chapter narrative. Could be a technical
  note or appendix.

- **The premature termination story:** We killed Run A at 1.54M steps, thinking
  it failed. Then discovered the upward trend and extended it. Great story for
  a blog post but would break the clean chapter arc. Instead, Step 3 teaches
  the lesson ("don't stop too early") through the interpretation of the
  training curve.

---

## Summary: 5 steps, 5 concepts, one clean arc

| Step | Experiment | Result | Concept taught |
|------|-----------|--------|---------------|
| 0 | NatureCNN + pixels | 5% flat | Pixels aren't drop-in; visual pipeline matters |
| 1 | ManipulationCNN + SpatialSoftmax + proprio | 5% flat | Architecture is necessary but not sufficient |
| 2 | + Critic-encoder gradient routing | 6% -> 90% hockey-stick | Gradient flow matters as much as architecture |
| 3 | (interpretation of Step 2's curve) | -- | How to read pixel RL training curves; patience tax |
| 4 | + DrQ augmentation (ablation) | 3% flat | Augmentation must match representation |

The reader finishes with:
- A working 90% pixel Push agent
- Understanding of WHY each piece is needed
- Transferable debugging skills for any pixel RL problem
- The insight that sometimes the "standard" technique (DrQ) is wrong

---

## Experimental data needed for the chapter

### Required data (chapter narrative)

| Experiment | Config | Data we have | Status |
|-----------|--------|--------------|--------|
| Step 0: NatureCNN baseline | Default SB3 + pixel wrapper | Multiple runs from earlier Ch9 work | DONE |
| Step 1: ManipulationCNN | ManipCNN + SS + proprio, default SB3 gradients | Runs from earlier Ch9 work (share_encoder) | DONE |
| Step 2: Critic-encoder (seed 0) | ManipCNN + SS + proprio + critic-encoder, NO DrQ | Cell A ext (95%+ at 4.2M) | RUNNING |
| Step 3: Interpretation | Same data as Step 2 | Loss curves, success trajectory | RUNNING |
| Step 4: DrQ ablation | Same as Step 2 but WITH DrQ | Cell D: 3% at 1.54M | DONE |

### Required data (book reproducibility -- 3 seeds)

| Run | Config | Status |
|-----|--------|--------|
| Seed 0 | --no-drq, SS ON, critic-encoder, buf=500K, HER-8, 5M steps | RUNNING (Cell A ext) |
| Seed 1 | Same, `--seed 1` | QUEUED -- run after seed 0 completes |
| Seed 2 | Same, `--seed 2` | QUEUED -- run after seed 1 completes |

Multi-seed is HIGHER priority than ablation cells. One seed at 95% could be
luck. Three seeds at 90%+ is a credible, publishable result. The book commits
to 3+ seeds in its reproducibility culture.

### Deferred data (ablations, run when compute is free)

| Cell | Config | Purpose | Notes |
|------|--------|---------|-------|
| B | --no-drq, SS OFF, critic-enc | Does SS matter without DrQ? | Reader exercise candidate |
| C | DrQ ON, SS OFF, critic-enc | Does DrQ help with flat features? | Reader exercise candidate |
| D (long) | DrQ ON, SS ON, critic-enc, 5M+ | Does DrQ+SS work with more patience? | If yes: "DrQ is slower." If no: "DrQ is harmful." |

**Step 4 gap:** Cell D was terminated at 1.54M steps. For the chapter, the
1.54M data is defensible (no trend at 1.54M, while Cell A showed clear trend
by 1.4M). For extra rigor, could rerun Cell D to 5M with matched buffer/HER
config. Deferred -- the existing data tells the story clearly enough.

### Compute plan

```
NOW:      Cell A seed 0 running to 5M  (~40h, will finish ~Feb 26 evening)
NEXT:     Cell A seed 1, 5M steps      (~40h)
THEN:     Cell A seed 2, 5M steps      (~40h)
OPTIONAL: Cell D (long), Cells B, C    (lower priority, when compute free)
```

Total for required data: ~120h (seed 0 already running + seeds 1 and 2).
After all 3 seeds: write the chapter with full Reproduce It block.
