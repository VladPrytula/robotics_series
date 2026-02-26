# Book Review: Tutorials vs Manning Chapters -- Assessment and Recommendations

**Date:** 2026-02-25 (updated same day with Ch9 from-scratch decision
and expanded Fetch-only environment analysis)
**Scope:** Full review of Manning/chapters/ (ch01-ch03), tutorials/ (ch00-ch10),
scripts/labs/ (14 files), syllabus, ToC, production protocol, writer persona.

---

## 1. Executive Summary

The project has two products: a tutorial site (10 chapters, complete) and a
Manning book (3 chapters drafted, 7 remaining). The core concern -- that the
tutorials rely heavily on SB3 while the book must teach RL -- is valid but
partially addressed. The three existing book chapters (Proof of Life, Env
Anatomy, PPO) successfully teach RL fundamentals with SB3 in a secondary
"scaling engine" role. The risk lies in the 7 unwritten chapters.

**Key finding:** The from-scratch lab code (~330KB across 14 files) is the
book's strongest asset. Every core RL concept has a standalone implementation
with `--verify` and `--compare-sb3` modes. The book chapters should lean
harder on these labs as the narrative spine.

**Decision (2026-02-25):** Ch9 (Visual RL) will rebuild ALL SB3 subclasses
from scratch. The five SB3 subclasses (CombinedExtractor, DrQDictReplayBuffer,
HerDrQDictReplayBuffer, ManipulationExtractor, DrQv2SACPolicy) are thin
compositions of components already built in Ch3-5. Rebuilding them (~135
lines total) makes Ch9 a genuine capstone that demonstrates compositional
understanding, not SB3 API knowledge.

---

## 2. What Works Well

### 2.1 The "Build It" / "Run It" / "Bridge" Pattern

Chapter 3 (PPO) demonstrates the ideal pattern:
1. Derive the math (policy gradient, advantage, GAE, clipping)
2. Build each component from scratch (~200 lines of PyTorch)
3. Verify with concrete checks (shapes, values, learning curves)
4. Bridge: prove from-scratch == SB3 (GAE agreement within float precision)
5. Use SB3 for the production training run

This makes SB3 feel like a consequence, not a dependency. A reader who
completes Ch3 could rewrite PPO in JAX without referencing SB3.

### 2.2 The Lab Codebase

The scripts/labs/ directory contains genuinely substantial from-scratch RL:

| Lab file | Lines | What it implements from scratch |
|----------|-------|-------------------------------|
| ppo_from_scratch.py | 890 | Full PPO: network, GAE, clipped loss, update loop |
| sac_from_scratch.py | 776 | Full SAC: twin Q, squashed Gaussian, auto temp, Polyak |
| her_relabeler.py | 483 | Full HER: goal sampling, relabeling, episode processing |
| visual_encoder.py | 857 | NatureCNN, visual SAC actor/critic, visual_sac_update |
| image_augmentation.py | 757 | DrQ random shift, augmented replay buffer |
| manipulation_encoder.py | 633 | ManipulationCNN, SpatialSoftmax |
| action_interface.py | 800 | Wrappers, PD controller, metrics, eval loop |
| robustness.py | 655 | Noise injection, degradation curves, summary metrics |

This is not a toy codebase. The total from-scratch code (~330KB) is
comparable in scope to what SpinningUp or CleanRL provide, but with
verification suites and SB3 bridging proofs.

### 2.3 Voice and Pedagogy

The Manning writer persona successfully bridges the tutorial's academic tone
and Manning's practitioner register. Ch3 reads like an experienced colleague
explaining PPO over a whiteboard -- rigorous but conversational. The 5-step
definition template (motivating problem, intuition, formal definition,
grounding example, non-example) produces definitions that actually teach.

### 2.4 Honest Experimental Culture

The "no vibes" rule, experiment cards, Reproduce It blocks, and multi-seed
results are genuinely distinctive. Most RL books report a single cherry-picked
run. This book commits to 3+ seeds, JSON artifacts, and exact reproduction
commands. This is credible and differentiating.

---

## 3. Current State: What Exists vs What's Needed

### 3.1 Chapter Status Map

| Book Ch | Title | Manning Draft | Tutorial | Lab Code | Status |
|---------|-------|--------------|----------|----------|--------|
| 1 | Proof of Life | DONE | ch00 | -- | Ready for review |
| 2 | Env Anatomy | DONE | ch01 | env_anatomy.py | Ready for review |
| 3 | PPO Dense Reach | DONE | ch02 | ppo_from_scratch.py | Ready for review |
| 4 | SAC Dense Reach | NOT STARTED | ch03 | sac_from_scratch.py | Lab code ready |
| 5 | HER Sparse Reach/Push | NOT STARTED | ch04 | her_relabeler.py | Lab code ready |
| 6 | PickAndPlace | NOT STARTED | ch05 | curriculum_wrapper.py | Lab code ready |
| 7 | Action Interface | NOT STARTED | ch06 | action_interface.py | Lab code ready |
| 8 | Robustness Curves | NOT STARTED | ch07 | robustness.py | Lab code ready |
| 9 | Visual Reach to Push | NOT STARTED | ch10 | pixel_wrapper.py, visual_encoder.py, image_augmentation.py, manipulation_encoder.py, drqv2_sac_policy.py | Lab code ready, training runs in progress |
| 10 | Reality Gap | NOT STARTED | (none) | (none) | No tutorial or lab yet |

### 3.2 The SB3 Dependency Spectrum

Each chapter has a different level of SB3 dependency. This matters for
the "teaching RL vs teaching SB3" question:

**Low SB3 dependency (algorithm chapters):**
- Ch3 (PPO): Full from-scratch implementation. SB3 only in Run It.
- Ch4 (SAC): Same pattern. sac_from_scratch.py is complete.
- Ch5 (HER): her_relabeler.py is standalone. SB3 for training.
- Ch9 (Visual): visual_encoder.py has visual_sac_update(). But pixel
  wrappers and DrQ buffer are SB3 subclasses.

**Moderate SB3 dependency (methodology chapters):**
- Ch6 (PickAndPlace): No new algorithm. Loads SB3 models for eval.
  But curriculum_wrapper.py is standalone.
- Ch7 (Action Interface): From-scratch wrappers and PD controller.
  SB3 models loaded for eval only.
- Ch8 (Robustness): From-scratch noise injection and degradation
  analysis. SB3 models loaded for eval only.

**~~High~~ Low SB3 dependency after from-scratch rebuild (decided 2026-02-25):**
- Ch9 (Visual): Originally had 5 SB3 subclasses. Decision: rebuild all
  from scratch as compositions of Ch3-5 components. SB3 only in Run It
  for scaled training. See Section 4.3 for the full rebuild plan.

### 3.3 The "Category B" Challenge: Methodology Chapters

Chapters 6-8 (PickAndPlace, Action Interface, Robustness) teach
experimental methodology, not new RL algorithms. This is valuable but
creates a different pedagogical challenge:

- The "Build It" content is evaluation tools (wrappers, metrics,
  noise injection), not learning algorithms.
- The "from-scratch" promise is about building diagnostic tools,
  not RL update functions.
- The reader may feel a plateau after learning PPO, SAC, and HER.

**Recommendation:** These chapters should explicitly frame themselves as
"engineering the learned policy" -- the point is that training is only
half the job. Evaluation, robustness testing, and controller analysis
are where rubber meets road. This positioning makes the methodology
feel essential rather than supplementary.

---

## 4. Concerns and Recommendations

### 4.1 SB3 API Concepts in Explanations

Throughout the tutorials, SB3-specific terms appear in explanatory prose:
- `MultiInputPolicy` -- SB3's name for dict-observation policy networks
- `HerReplayBuffer` -- SB3's HER implementation class
- `BaseFeaturesExtractor` -- SB3's CNN integration point
- `DictReplayBuffer` -- SB3's dict-observation replay buffer

**Recommendation:** In book chapters, treat these as implementation details.
The chapter should explain the RL concept first ("a policy network that
processes dictionary observations by encoding each key separately"), then
note the SB3 class name in passing ("SB3 calls this `MultiInputPolicy`").
Never let the SB3 class name drive the explanation.

### 4.2 The HER Tutorial Is Too Empirical for the Book

The ch04 tutorial is 1360 lines, with a large portion devoted to a 120-run
hyperparameter sweep and factorial analysis. This is excellent research
documentation but wrong for a book chapter.

**Recommendation for Book Ch5 (HER):**
- Lead with the geometric intuition (the "imagined goal" reframing)
- Build the relabeling algorithm from scratch (her_relabeler.py)
- Show 3-4 key configurations, not 24
- Spend more space on WHY HER works (value propagation, effective
  horizon) and WHEN it fails (no exploration, wrong reward structure)
- The hockey-stick learning curve sidebar is excellent -- keep it
- Move the full 120-run sweep to an online appendix

### 4.3 Ch9 Visual RL: Full From-Scratch Rebuild (DECIDED)

**Decision (2026-02-25):** Rebuild all five SB3 subclasses from scratch
in Ch9's Build It. Each is a thin composition of components the reader
already built in earlier chapters.

**The five pieces and their from-scratch equivalents:**

| SB3 Subclass | From-scratch replacement | Lines | What it teaches |
|---|---|---|---|
| `NormalizedCombinedExtractor` | `CombinedEncoder`: route pixels->CNN, vectors->flatten, concat | ~20 | Multi-modal observation processing |
| `DrQDictReplayBuffer` | `PixelDrQReplayBuffer`: store uint8, augment at sample time | ~40 | Memory-efficient pixel storage + augmentation timing |
| `HerDrQDictReplayBuffer` | `HerPixelDrQBuffer`: compose Ch5's relabeler + DrQ augmentation | ~60 | Component composition -- the synthesis moment |
| `ManipulationExtractor` | Same as CombinedEncoder with ManipulationCNN swapped in | trivial | Architecture as a hyperparameter |
| `DrQv2SACPolicy` | Explicit optimizer setup: encoder in critic, detached from actor | ~15 | **Gradient routing -- the key insight of DrQ-v2** |

**Total new from-scratch code: ~135 lines of PyTorch.**

Each piece composes existing building blocks:

```
Ch4's ReplayBuffer + Ch9's uint8 storage + Ch9's DrQ augmentation
= PixelDrQReplayBuffer (from scratch)

Ch5's HER relabeling + the above
= HerPixelDrQBuffer (from scratch)

Ch9's NatureCNN + CombinedEncoder + the above
= Complete visual HER+DrQ pipeline (from scratch)
```

**Why this matters pedagogically:**

The most valuable new component is the gradient routing (~15 lines):

```python
# The key decision -- which optimizer owns the CNN encoder?
# DrQ-v2 answer: the critic optimizer.

critic_optimizer = Adam(
    list(q_network.parameters()) + list(encoder.parameters()),
    lr=3e-4
)
actor_optimizer = Adam(
    actor.parameters(),  # encoder NOT here
    lr=3e-4
)

# In the actor update: detach encoder features
with torch.no_grad():
    features = encoder(pixels)  # no encoder gradients from actor loss
```

This teaches something genuinely important about visual RL that is invisible
inside an SB3 `SACPolicy` subclass. From scratch, the reader sees the
decision and understands why it matters: the encoder should learn
state features from Bellman error (critic), not learn to fool the
critic (actor).

**The compositional insight:** A reader who builds HER in Ch5 and DrQ in
Ch9 separately, then wires them together in 60 lines, understands that
these techniques are orthogonal. HER transforms goals in the replay
buffer. DrQ transforms pixels at sample time. They compose because they
operate on different parts of the transition. This insight -- that good
RL engineering is about composable components -- is worth more than any
single technique.

**Impact on Ch9 structure:** Build It now covers the full visual RL
pipeline from scratch. SB3 only appears in Run It as the scaling engine
for 2M-8M step training runs. The Bridge section validates from-scratch
vs SB3 agreement (NatureCNN weight-copy test, DrQ augmentation shape
checks, replay buffer sampling equivalence).

### 4.4 Chapter Ordering: Ch5 and Ch9 Are Good Next Targets

Ch5 (HER) and Ch9 (Visual) are strong choices to draft next because:
- Ch5 is the methodological turning point (sparse rewards, the core
  problem the book was written to solve)
- Ch9 is the capstone synthesis (pixels + HER + DrQ)
- Both have complete lab code ready
- Ch4 (SAC) follows the exact Ch3 pattern and could be drafted
  quickly in between

**Risk:** Skipping Ch4 (SAC) means Ch5 (HER) references algorithms the
reader hasn't built from scratch yet. The from-scratch SAC lab exists but
the book narrative doesn't. Consider doing Ch4 first or concurrently.

### 4.5 The "From Scratch" Promise Now Scales to Ch9

With the from-scratch rebuild decision (Section 4.3), Ch9's Build It
covers the FULL visual RL pipeline in pure PyTorch:

| Component | Source | Status |
|---|---|---|
| NatureCNN / ManipulationCNN | visual_encoder.py, manipulation_encoder.py | Already from-scratch |
| DrQ random shift augmentation | image_augmentation.py | Already from-scratch |
| PixelObservationWrapper | pixel_wrapper.py | Already from-scratch (gym.Wrapper) |
| visual_sac_update() | visual_encoder.py | Already from-scratch |
| CombinedEncoder (multi-modal) | NEW | ~20 lines |
| PixelDrQReplayBuffer (uint8 + augment) | NEW | ~40 lines |
| HerPixelDrQBuffer (HER + DrQ composed) | NEW | ~60 lines |
| Encoder gradient routing | NEW | ~15 lines |

The "from scratch" promise now holds across the entire book:
- Ch3: full PPO from scratch
- Ch4: full SAC from scratch
- Ch5: full HER from scratch
- Ch9: full visual pipeline from scratch, composing Ch3-5 components

SB3 is the scaling engine in Run It only. The reader never encounters
an SB3 subclass they haven't already built the equivalent of.

---

## 5. Manning Reviewer Perspective

Based on the manning_proposal/review_notes.md and standard Manning
reviewer concerns:

### 5.1 What Reviewers Will Like

- **"From scratch" is genuine.** The lab code is real, verified, and
  bridged to SB3. This isn't hand-waving.
- **Reproducibility culture.** Multi-seed, Docker, JSON artifacts.
  Distinctive and credible.
- **Clear structure.** WHY-HOW-BUILD-BRIDGE-RUN is predictable and
  satisfying.
- **Honest about difficulty.** "This took us three attempts" is
  refreshing in a technical book.

### 5.2 What Reviewers Will Question

- **"Do I need SB3 to use this book?"** The answer should be
  emphatically "no for understanding, yes for scale." Build It runs on
  pure PyTorch + NumPy. Run It uses SB3. Make this boundary crisp.

- **"Chapters 6-8 feel like a different book."** The methodology
  chapters (Action Interface, Robustness) are valuable but feel different
  from the algorithm chapters. Reviewers may suggest condensing them.
  Counter: frame them as "Part 4: Engineering-Grade Robotics RL" --
  the engineering IS the point for practitioners.

- **"Only one robot, only one simulator."** The book uses only Fetch in
  MuJoCo. See Section 8 for the full analysis and mitigation strategy.

- **"Where are the latest methods?"** No Dreamer, no diffusion policies,
  no foundation models. Counter: the book teaches fundamentals (PPO, SAC,
  HER) that underpin everything newer. Appendix D (world models) provides
  a pointer.

- **"Ch9 is very long."** The three-act visual RL chapter (Reach ->
  Push diagnostic -> Push from pixels) may be 12,000+ words. Consider
  splitting or trimming.

### 5.3 Strongest Differentiators

1. **From-scratch implementations with bridging proofs to production code.**
   No other RL book does this. SpinningUp teaches from scratch but doesn't
   bridge. SB3 docs teach the library but don't show the math.

2. **Single task family, increasing difficulty.** The Fetch ladder
   (Reach -> Push -> PickAndPlace, dense -> sparse, state -> pixels)
   creates a coherent arc. Most RL books hop between CartPole, Atari,
   and MuJoCo locomotion with no connective tissue.

3. **Honest experimental methodology.** Multi-seed, JSON artifacts,
   degradation curves, noise robustness. This is what a real RL
   research lab does; no other introductory book teaches it.

---

## 6. Specific Suggestions for Ch5 (HER) and Ch9 (Visual)

### 6.1 Book Ch5: Learning from Failure with HER

**Source material:** Tutorial ch04 (1360 lines), lab her_relabeler.py (483
lines), lab sb3_compare.py (HER comparison section).

**Recommended structure:**

```
5.1 WHY: The sparse reward wall
    - Show SAC without HER on FetchReach-v4 (sparse): flatline
    - The exploration problem: 5cm success region in 60x70x20cm workspace
    - What fraction of random transitions are successes? (~0.1%)

5.2 HOW: Hindsight relabeling
    - The key insight: "what goal WOULD this have achieved?"
    - The three sampling strategies (future, final, episode) with
      geometric intuition
    - The off-policy requirement: why PPO can't do this

5.3 BUILD IT: HER relabeling from scratch
    - 5.3.1: Data structures (Transition, Episode)
    - 5.3.2: Goal sampling (future strategy)
    - 5.3.3: Transition relabeling + reward recomputation
    - 5.3.4: Full episode processing (original + relabeled)
    - 5.3.5: Data amplification analysis (k=4 -> 5x transitions)
    - 5.3.6: Verification (--verify)
    - 5.3.7: SB3 comparison (--compare-sb3)

5.4 BRIDGE: From-scratch HER vs SB3 HerReplayBuffer
    - Same episode, same strategy -> same relabeled transitions
    - Reward recomputation agreement within float precision

5.5 RUN IT: SAC + HER on Sparse Reach and Push
    - Experiment Card: Sparse Reach (fast path, ~5 min)
    - Experiment Card: Sparse Push (2M steps, ~40 min GPU)
    - The hockey-stick learning curve (sidebar)
    - 3 key configurations: gamma, entropy, n_sampled_goal

5.6 WHAT CAN GO WRONG
    - Success rate stuck at 0% (wrong gamma, too few relabeled goals)
    - Success rate stuck at 5-10% for >1M steps (patience -- hockey stick)
    - Rising critic/actor loss (this is normal with HER + sparse)

5.7 Summary

REPRODUCE IT: 3 seeds x FetchReach-v4 + FetchPush-v4
EXERCISES: strategy comparison, k ablation, HER ratio analysis
```

**Key cuts from tutorial ch04:**
- Remove the 120-run factorial sweep (move to online appendix)
- Remove the detailed factor-level marginal analysis
- Keep the hockey-stick sidebar (it's excellent pedagogy)
- Keep the 3 most important configurations, not all 24

**Key additions for the book:**
- More geometric intuition (draw the goal space, show the relabeling)
- Explicit connection to Ch4's replay buffer (HER augments it)
- Figure showing relabeled transitions in goal space

### 6.2 Book Ch9: Pixels, No Cheating -- From Reach to Push

**Source material:** Tutorial ch10 (large), labs pixel_wrapper.py,
visual_encoder.py, image_augmentation.py, manipulation_encoder.py.
Training runs in progress (Push pixel + HER + DrQ).

**Recommended structure (updated with full from-scratch Build It):**

```
9.1 WHY: What changes when you remove privileged state
    - The observation design space (goal modes, frame stacking, proprio)
    - The pixel penalty: dimensionality, partial observability, rendering cost
    - Hadamard check: can a CNN learn manipulation from 84x84 images?

9.2 BUILD IT: The visual observation pipeline (all from scratch)
    - 9.2.1: render_and_resize (MuJoCo -> CHW uint8)          [exists]
    - 9.2.2: PixelObservationWrapper (goal modes, stacking)    [exists]
    - 9.2.3: NatureCNN encoder (pure PyTorch)                  [exists]
    - 9.2.4: DrQ random shift augmentation (pure PyTorch)      [exists]
    - 9.2.5: CombinedEncoder (pixels + vectors -> features)    [NEW ~20 lines]
    - 9.2.6: PixelDrQReplayBuffer (uint8 + augment at sample)  [NEW ~40 lines]
    - 9.2.7: HerPixelDrQBuffer (compose Ch5 relabeling + DrQ)  [NEW ~60 lines]
    - 9.2.8: Encoder gradient routing (critic owns encoder)     [NEW ~15 lines]
    - 9.2.9: Verification (--verify)
    - 9.2.10: SB3 comparison (--compare-sb3)

9.3 BRIDGE: From-scratch visual pipeline vs SB3
    - NatureCNN weight-copy test: same input -> same output
    - DrQ augmentation: shape preservation, distribution uniformity
    - Replay buffer: sampling equivalence (uint8->float32->augment)
    - Gradient routing: confirm encoder params in critic optimizer only

9.4 RUN IT ACT 1: Measuring the pixel cost on Reach
    - Experiment Card: State SAC (baseline, 500K steps)
    - Experiment Card: Pixel SAC (2M steps)
    - Experiment Card: Pixel SAC + DrQ (2M steps)
    - Sample-efficiency ratio: 4x cost for pixels
    - Reach is "too easy" -- the real test is ahead

9.5 RUN IT ACT 2: Why Push breaks everything
    - FetchPushDense from state SAC (no HER): 5% success
    - The "deceptively dense" reward problem
    - The bridge: goal_mode="both" -- policy sees pixels, HER sees vectors
    - Information asymmetry as a design choice

9.6 RUN IT ACT 3: Push from pixels with HER + DrQ
    - Experiment Card: SAC + HER + DrQ on FetchPush-v4 (8M steps)
    - The hockey-stick revisited (now with pixels)
    - Resolution: pixels + HER + augmentation solve a real manipulation task

9.7 Making it fast (sidebar or appendix)
    - Profiling: where time goes (render vs train vs step)
    - Three optimizations (native resolution, SubprocVecEnv, replay ratio)

9.8 WHAT CAN GO WRONG
    - NatureCNN destroys small objects (8x8 stride-4)
    - Pixel buffer OOM (compute memory before launching)
    - Hockey-stick patience (rising losses are the signal, not the problem)
    - Wrong gradient routing (encoder in actor optimizer -> overfitting)

9.9 Summary

REPRODUCE IT: State vs pixel vs pixel+DrQ on Reach; Push from pixels
EXERCISES: encoder comparison, augmentation ablation, goal mode analysis,
           gradient routing ablation (swap encoder to actor optimizer)
```

**Key decisions for the book (updated):**
- Build the ENTIRE visual pipeline from scratch (pure PyTorch, ~135 new
  lines on top of existing labs)
- The Build It is a genuine capstone: compose Ch4 (SAC), Ch5 (HER), and
  Ch9 (visual components) into a unified pipeline
- SB3 appears ONLY in Run It for scaled training (2M-8M steps)
- Bridge section validates ALL from-scratch components against SB3
- The three-act structure (Reach -> Push diagnostic -> Push resolution)
  is excellent narrative -- keep it
- Gradient routing exercise: ablation that shows what breaks when encoder
  is in the wrong optimizer (this is the kind of insight only from-scratch
  builds can teach)

---

## 7. Bottom Line Assessment

**Strengths:** The book's "from scratch -> verify -> bridge -> scale" pattern
is genuinely distinctive. The lab codebase is substantial. The three existing
chapters deliver on the promise. The experimental culture (multi-seed,
artifacts, honest reporting) is a real differentiator.

**Primary risk:** The 7 unwritten chapters. If they maintain Ch3's quality,
this is a strong book. If they slide toward "SB3 tutorial with math sidebar,"
the book loses its identity.

**The SB3 concern is real but manageable.** SB3 appears in every production
run, but the from-scratch labs ensure the reader understands what SB3 computes.
The bridging proofs are the structural element that makes this work. Every
chapter that has a bridging proof is a chapter that teaches RL, not SB3.

**Recommendation:** Prioritize Ch4 (SAC) and Ch5 (HER) as the next
drafts -- they follow the proven Ch3 pattern and complete the algorithm
arc. Ch9 (Visual) can follow, but depends on training runs completing.
The methodology chapters (Ch6-8) can be drafted in parallel since they
don't introduce new algorithms.

---

## 8. The Fetch-Only Environment Question

### 8.1 The Concern

The book uses a single environment family (Gymnasium-Robotics Fetch) in a
single simulator (MuJoCo) for all 10 chapters. A Manning reviewer will
ask: "Will readers feel trapped? Can they apply this to their own robots?"

This is a legitimate concern. Most RL books rotate between CartPole,
Atari, MuJoCo locomotion, and maybe a custom environment. They sacrifice
depth for breadth. This book does the opposite.

### 8.2 Why Depth Over Breadth Is the Right Call

The philosophical position: it is better to understand one environment
deeply -- where you can diagnose every failure, explain every metric,
and know what 94% success rate *means* physically -- than to touch many
environments and never understand what is going on.

**Concrete arguments:**

1. **Controlled variable elimination.** When PPO fails on FetchPush
   (sparse), is it the algorithm, the reward, the exploration, or the
   environment? Because the reader already solved FetchReach (dense) and
   FetchReach (sparse), they can isolate the variable. With a new
   environment each chapter, every failure has five possible causes.

2. **Transferable methodology, not transferable hyperparameters.** The
   book teaches: formulate the problem -> derive the algorithm from
   constraints -> implement from scratch -> verify -> evaluate honestly.
   This methodology transfers to any environment. The hyperparameters
   do not, and the book is explicit about this.

3. **The difficulty ladder is within Fetch.** The environment family
   provides a natural curriculum:

   ```
   FetchReachDense  (Ch3-4)   -- easiest: move arm to target, continuous signal
   FetchReach sparse (Ch5)    -- same task, binary reward: the HER motivation
   FetchPush sparse  (Ch5)    -- harder: object interaction, multi-phase control
   FetchPickAndPlace (Ch6)    -- hardest: grasp + lift + place, air goals
   FetchPush pixels  (Ch9)    -- same task, but from camera: the visual RL test
   ```

   This is five qualitatively different challenges within one API. The
   reader never has to learn new observation dictionaries, new action
   semantics, or new success criteria -- they accumulate understanding.

4. **Honest about the limitation.** The book should own this choice:
   "We use one environment family because we believe you learn more from
   understanding one task deeply than from running many tasks superficially.
   The cost is obvious: you haven't seen locomotion, multi-agent, or
   real hardware. We address portability in the appendices."

### 8.3 Mitigation Strategy: The "Beyond Fetch" Arc

The book already plans three appendices that address environment diversity.
Here is the current state and what needs work:

**Appendix C: PyBullet Port (PandaGym)**
- Status: Planned, not implemented
- What it proves: the experiment contract (train/eval/artifacts) is
  simulator-agnostic. Same SAC+HER, different physics, same JSON reports.
- Effort: Low. PandaGym has Fetch-like goal-conditioned tasks.
  The reader runs the same commands with a different env ID.
- Value: High. "Your methodology ports to new robots in 30 minutes"
  is a powerful claim.

**Appendix E: Isaac Peg-in-Hole (GPU-only)**
- Status: Planned, needs environment selection
- What it proves: the methodology works on a qualitatively different
  task (tight tolerances, contact-rich, jamming failure modes).
- Effort: Medium-high. Needs an Isaac environment that:
  (a) is goal-conditioned or can be wrapped as such,
  (b) is solvable with SAC+HER at the book's compute budget,
  (c) teaches something Fetch doesn't (tight tolerances, contact forces).
- **Research needed:** Find specific Isaac environments that work.
  Candidates:
  - `IsaacGymEnvs/FrankaPeg` -- peg insertion with Franka arm
  - `OmniIsaacGymEnvs` factory tasks -- if any have goal-conditioned API
  - Isaac Lab (newer) manipulation environments
  - Key constraint: must be solvable in <= 500K steps (fast path) or
    have a checkpoint track
- Value: Very high. Isaac is the industry-standard GPU simulator.
  Showing the same methodology on Isaac hardware says "this book
  prepares you for production robotics RL."

**Appendix D: World Models (VAE demo)**
- Status: Planned, minimal scope
- What it proves: the rendering bottleneck from Ch9 motivates latent
  imagination methods.
- Value: Moderate. More of a "further reading" pointer than a hands-on
  exercise.

### 8.4 Recommended "Beyond Fetch" Content for the Main Text

The appendices help, but they're optional and online-only. The main text
should also signal that Fetch is a teaching choice, not a limitation.

**Option A: "Beyond Fetch" sidebar in Ch9 or Ch10 (~0.5 page)**

After the reader has seen the full Fetch ladder (dense -> sparse ->
pick-and-place -> pixels), include a sidebar:

> "Everything we've built -- SAC, HER, DrQ, degradation curves, the
> experiment card pattern -- transfers to other environments. The
> methodology is environment-agnostic; only the hyperparameters are
> Fetch-specific. Here's what changes and what stays the same when you
> move to a new task:
>
> | What stays the same | What changes |
> |---|---|
> | SAC + HER for sparse goal-conditioned tasks | Hyperparameters (gamma, n_sampled_goal, buffer size) |
> | DrQ for pixel observations | CNN architecture (receptive field for your object scale) |
> | Degradation curves for robustness | Noise model (what noise is realistic for your hardware) |
> | Experiment cards and JSON artifacts | Success criteria and metrics |
> | Dense-first debugging pattern | The specific dense reward to use |
>
> Appendix C demonstrates this with PandaGym (PyBullet); Appendix E
> with Isaac (NVIDIA GPU). Readers who want to apply this to custom
> environments should start with the experiment contract from Ch1 and
> the Hadamard check from Ch2 -- those are the most portable tools in
> this book."

**Option B: "Porting checklist" exercise in Ch10 (~1 page)**

A structured exercise where the reader maps the book's methodology
to a hypothetical new task:

> Exercise: Choose an environment you care about (your own robot, a
> different simulator, a different Gymnasium task). Answer:
> 1. Is it goal-conditioned? If not, can you define goals?
> 2. Can you recompute rewards from (achieved_goal, desired_goal)?
>    (This is the HER requirement from Ch5.)
> 3. What is your dense-reward debugging version? (Ch6 pattern.)
> 4. What noise model is realistic for your hardware? (Ch8 pattern.)
> 5. What resolution do you need for pixel observations? (Ch9 pattern.)

**Recommendation:** Do both. The sidebar is cheap (half a page) and
signals awareness. The exercise is pedagogically valuable and makes the
"depth over breadth" choice feel intentional rather than narrow.

### 8.5 The Isaac Research Task

Finding the right Isaac environment for Appendix E is an open research
task. Requirements:

1. **Goal-conditioned or easily wrappable as such.** The reader should
   be able to use SAC+HER without fundamental API changes.
2. **Solvable at book-scale compute.** Fast path: ~500K steps on GPU.
   Full run: ~2-5M steps. If it needs 50M steps, it's wrong for the book.
3. **Qualitatively different from Fetch.** Tight tolerances, contact
   forces, or assembly -- something where "close enough" is NOT success.
4. **Available without proprietary assets.** Open-source Isaac environment.
5. **Stable API.** Isaac Gym -> Isaac Lab migration is ongoing. Choose
   an environment that won't break in 6 months.

Candidates to investigate:
- `IsaacGymEnvs` FrankaCabinet or FrankaPeg tasks
- Isaac Lab `manipulation` environments (newer, more maintained)
- OmniIsaac factory tasks (if any are small enough)
- Potential fallback: just use PandaGym for Appendix E too (different
  robot, same simulator class) and note that Isaac integration is
  a "next step" for readers with GPU access.

### 8.6 Addressing the Reviewer Directly

When a Manning reviewer raises the Fetch-only concern, the answer is:

> "The book uses one environment family intentionally. The Fetch ladder
> (Reach -> Push -> PickAndPlace, dense -> sparse, state -> pixels)
> provides five qualitatively different challenges with a single API.
> This allows the reader to isolate algorithm effects from environment
> effects -- when HER improves success from 0% to 99%, they know it's
> HER, not a different environment's dynamics. The cost is obvious:
> no locomotion, no multi-agent, no real hardware. Appendices C and E
> address portability (PyBullet and Isaac respectively), and Chapter 10
> includes a 'porting checklist' exercise that maps the methodology to
> new tasks. We believe this is the right tradeoff for a practitioner
> book -- depth that teaches transferable methodology, rather than breadth
> that teaches environment-specific recipes."
