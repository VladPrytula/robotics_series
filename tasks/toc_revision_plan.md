# Part 5 Revision: Why and How

**Date:** 2026-02-19
**Status:** Approved -- executing phased implementation

---

## The Problem With the Current ToC

The current Part 5 spreads visual RL across four chapters:

```
Part 5 -- Pixels and the Reality Gap
  10. Pixels, no cheating: visual Reach in the same task family
  11. Visual robustness that matters: augmentation as a tool
  12. Visual goals (optional advanced): "make it look like this"
  13. A reality-gap playbook: stress tests before hardware
```

Having drafted Ch10 and run the experiments, we found a structural problem:
**Ch10 introduces a tension it cannot resolve on its own.**

The chapter demonstrates that pixel observations are harder than state on
FetchReachDense -- but only modestly so (98% vs 100% success). DrQ closes the
gap. The honest assessment is that FetchReachDense is "too easy" to reveal the
real cost of pixels. We observe this, note that harder tasks (Push,
PickAndPlace) need HER + pixels, and leave it as "future work."

This violates the book's core pattern. Every chapter so far introduces a
tension and resolves it:

| Chapter | Tension | Resolution |
|---------|---------|------------|
| Ch3 (PPO) | Can we train a policy? | 100% on dense Reach |
| Ch4 (SAC) | Can we be more efficient? | SAC converges faster |
| Ch5 (HER) | Can we learn from sparse rewards? | HER: 99.4% on Push |
| Ch6 (PickAndPlace) | Can we scale to multi-phase? | PickAndPlace works |
| Ch7 (Actions) | Are policies good controllers? | Smoothness metrics + filtering |
| Ch8 (Robustness) | How brittle are they? | Degradation curves + augmented training |

Ch10 as currently drafted introduces the pixel penalty, partially closes it on
an easy task, then opens harder questions (Push, PickAndPlace, visual HER)
without closing them. The reader walks away with more open questions than
answers.

Meanwhile, Ch9 (ablations/sweeps), Ch11 (augmentation), and Ch12 (visual
goals) are thin as standalone chapters:

- **Ch9:** 70-75% overlap with existing chapters. Unique content (experiment
  cards) absorbed into Ch8.
- **Ch11:** DrQ is already demonstrated in Ch10. "Which augmentation is best?"
  is a table, not a narrative arc.
- **Ch12:** Pure visual goals is a research frontier. Doesn't resolve a tension
  the reader already feels.

---

## Approved Revision: 10 Chapters

### Decision: consolidate from 13 to 10 chapters

Every chapter resolves its tension. 10 chapters = 10 weeks in the syllabus.

### Revised Full ToC

```
Part 1 -- Start Running, Start Measuring
  1. Proof of life: a reproducible robotics RL loop
  2. What the robot actually sees: observations, rewards, and success

Part 2 -- Baselines That Debug Your Pipeline
  3. PPO as a lie detector (dense Reach)
  4. Off-policy without mystery (SAC on dense Reach)

Part 3 -- Sparse Goals, Real Progress
  5. Learning from failure with HER (sparse Reach and Push)
  6. Capstone manipulation: PickAndPlace with an honest report card

Part 4 -- Engineering-Grade Robotics RL
  7. Policies as controllers: stability, smoothness, and action interfaces
  8. Robustness curves: quantify brittleness

Part 5 -- Pixels and the Reality Gap
  9. Pixels, no cheating: from Reach to Push
  10. The reality gap: stress tests before hardware
```

### What changed and why

| Old Chapter | Disposition | Rationale |
|-------------|------------|-----------|
| Ch9 (Ablations/Sweeps) | **Absorbed** into Ch8 + distributed | 70-75% overlap with Ch5/Ch7/Ch10; unique content (experiment cards) -> brief section in Ch8 |
| Ch10 (Pixel Reach only) | **Expanded** -> new Ch9 (Reach + Push) | Three-act narrative: measure cost -> recognize gap -> overcome it via HER+pixels+DrQ synthesis |
| Ch11 (Augmentation) | **Absorbed** into new Ch9 (DrQ) + new Ch10 (ablations section) | DrQ inseparable from pixel pipeline; "which crop is best?" is a table, not a chapter |
| Ch12 (Visual goals) | **Dropped** | Research frontier, not pedagogy; mention in Ch9 "further reading" |
| Ch13 (Reality gap) | **Promoted** -> new Ch10 | Natural next question after "pixels work in sim" |

---

## The Narrative Arc of Revised Ch9 (Pixels)

**Act 1 -- Measure the cost (Reach, sections 9.1-9.4)**

"Everything we built in Ch1-8 assumed privileged state. What breaks when we
use cameras instead?"

Run three experiments on FetchReachDense:
- State SAC (baseline ceiling): 100% success, 500K steps
- Pixel SAC (no augmentation): 98% success, 2M steps
- Pixel SAC + DrQ: 100% success, 2M steps

Show learning curves to compute the sample-efficiency ratio (rho).

**Act 2 -- The real test (Push, sections 9.5-9.6)**

"Reach is too easy. What about a task that requires object interaction?"

FetchPushDense from state SAC (no HER) fails: ~8% success at 1M steps. The
"deceptively dense" reward is constant before contact (entropy collapses,
structural failure). We already know (Ch5) that HER solves this from state.
But HER needs goal vectors, and pure pixel mode strips them.

**The hockey-stick learning curve (sidebar/box in section 9.6):**

HER on sparse Push exhibits a characteristic phase transition -- flat near 0%
for ~1.2M steps, then rapid spike to 90%+ by 1.84M steps. Three mechanisms,
each with independent formal backing, explain this:

1. **Value propagation bottleneck (flat phase):** Bellman backups propagate
   value through goal space as a discrete diffusion process. HER seeds the
   critic with relabeled near-object successes; each backup extends knowledge
   outward by one step. The flat phase duration is governed by the **effective
   horizon** k* -- the number of backup steps needed before reward signal
   reaches the initial state distribution. Sample complexity is exponential
   in k* (Laidlaw et al. 2024, ICLR Spotlight, arXiv:2312.08369).

2. **Geometric / quasimetric threshold (inflection):** The optimal
   goal-conditioned Q-function has a quasimetric structure (Wang & Isola 2022,
   ICLR, arXiv:2206.15478). Once enough local distances are learned correctly,
   the triangle inequality forces global consistency -- analogous to
   percolation. Simplified: test goals are uniform over the table (2D), so
   overlap with the competence region scales as ~pi*r^2/L^2. Small r -> ~0%
   success. Once r crosses a critical threshold, success jumps sharply.

3. **Positive feedback / relay mechanism (steep phase):** Once real goals are
   reached, gradient quality jumps (real vs relabeled reward). Each success
   expands the competence region, unlocking more successes. Huang et al. (2025,
   arXiv:2602.14872) formalize this as a "relay mechanism" using Fourier
   analysis on difficulty spectra: gradient signals from simpler problems
   progressively unlock harder ones.

The nucleation metaphor: HER seeds are microscopic crystals in a supercooled
liquid. Below critical nucleus size, surface energy dominates (the gap between
relabeled and test goals). Above critical size, bulk energy dominates and
crystallization (the positive feedback loop) races through the system.
This is a metaphor, not a formal mapping -- we use it for intuition.

**Experimental evidence:**

| Steps | Success | Phase |
|------:|--------:|-------|
| 0-1.2M | 3-10% | Flat (wavefront building) |
| 1.28M | 14% | Inflection (quasimetric percolation) |
| 1.40M | 32% | Positive feedback active |
| 1.84M | 90% | Near-saturation |
| 2.0M | 89% | Plateau |

Same config at 1M steps: ~2% success (pre-inflection). The entire jump from
10% to 90% occurred in ~600K steps (30% of total budget).

**Honest caveat for the book:** "The full picture is assembled from three
independent theoretical results, not derived from a single theorem. No paper
provides a closed-form expression for the inflection point. We find this
honest framing more useful than pretending the theory is complete."

**Practical diagnostic:** If the HER learning curve is flat, check whether
relabeled-goal Q-values are improving even when test-goal success is flat.
If so, the wavefront is growing -- give it more time.

**Full research notes:** tasks/hockey_stick_research.md

The bridge: goal_mode="both". The policy sees pixels through NatureCNN (honest
visual learning). HER sees achieved_goal/desired_goal vectors (for relabeling).

**Act 3 -- Resolution (Push from pixels, section 9.7)**

Train SAC + HER + DrQ with pixel observations on FetchPush (sparse rewards).
Show that the combination works: the reader has gone from "pixels break
everything" to "pixels + HER + augmentation solve a real manipulation task."

---

## Execution Plan

### Phase 1: Documentation Updates [DONE]
- [x] Rewrite manning_proposal/toc.md (10-chapter structure)
- [x] Update tasks/toc_revision_plan.md (this file)
- [x] Update CLAUDE.md concept registry (renumber, add new concepts)
- [x] Update syllabus.md (week 8-10 mapping)

### Phase 2: Technical Implementation [DONE]
- [x] Implement HerDrQDictReplayBuffer in scripts/labs/image_augmentation.py
- [x] Add Push training modes to scripts/ch10_visual_reach.py

### Phase 3: Smoke Tests [DONE]
- [x] HER+DrQ buffer: train-pixel-her-drq --pixel-total-steps 10000
- [x] Push state no-HER: train-push-state --state-total-steps 10000
- [x] Push state+HER: train-push-her --total-steps 10000

### Phase 4: Training Runs [IN PROGRESS]
- [x] Push state, no HER (FetchPushDense-v4, SAC, 1M steps) -- eval: 5% success, 0.184m final distance
- [x] Push state + HER (FetchPush-v4, SAC+HER, **2M** steps) -- eval: **100%** success, 0.025m final distance
      NOTE: originally 1M steps (2% success, pre-inflection), increased to 2M to match Ch5.
      Hockey-stick inflection at ~1.28M steps. See tasks/hockey_stick_research.md for analysis.
- [ ] Push pixel + HER (FetchPush-v4, SAC+HER+pixels, **8M** steps, 1M buffer, gradient_steps=1, 8 envs)
      RUNNING: ~160 fps, started 2026-02-20. First attempt failed (200K buffer + gradient_steps=3 -> overfitting).
- [ ] Push pixel + HER + DrQ (FetchPush-v4, SAC+HER+DrQ+pixels, **8M** steps, 1M buffer, gradient_steps=1, 8 envs)
      RUNNING: ~160 fps, started 2026-02-20. Same fix as above.

### Phase 5: Tutorial Restructure + Cleanup
- [ ] Restructure tutorials/ch10_visual_reach.md (three-act narrative)
- [ ] Add experiment card section to Ch8 tutorial
- [ ] File renumbering (separate commit, deferred)
- [ ] Create ch09_reality_gap placeholder scaffold
