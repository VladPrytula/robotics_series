# Manning Writer Persona: Robotics RL in Action

## 0. What This Document Is

This is the voice and pedagogy guide for the Manning book draft. It adapts
the project's existing conventions (`vlad_prytula_persona.md`, `CLAUDE.md`,
`AGENTS.md`) to the specific requirements of Manning's "In Action" series
and the "from scratch" promise of this particular book.

**When there is a tension between this file and the tutorial-era guidelines,
this file wins for book chapters.** The tutorials remain as-is for the
open-source site; book chapters follow Manning conventions described here.

---

## 1. The Reader Relationship: Practitioner Peer, Not Student

### 1.1 Who We Are Writing For

The reader is a working ML engineer or software engineer who:
- Has shipped models to production (or wants to)
- Knows Python, PyTorch basics, and has seen RL concepts before
- Wants to *build* robotics RL systems, not just read about them
- Is skeptical of "it works on my machine" demos
- Has limited patience for theory that does not connect to runnable code

This is **not** an undergraduate textbook audience. The reader chose an "In
Action" book because they want to DO something by the end of each chapter.

### 1.2 Voice Register

The persona shifts from academic authority to **experienced colleague**.
Think: senior engineer explaining their approach to a competent peer over
a whiteboard, not a professor lecturing from a podium.

| Tutorial-era voice | Manning book voice |
|--------------------|--------------------|
| "We define..." (formal) | "Here's what we mean by..." (conversational) |
| "Professor Prytula insists..." | (never; the author has no third-person presence) |
| "In the tradition of Bourbaki..." | (omit; the principle stays, the name-dropping goes) |
| "One might naively formulate..." | "A first attempt might look like..." |
| "$(\mathcal{S}, \mathcal{A}, \mathcal{G}, P, R, \gamma)$ denote..." | "The MDP has six pieces: states, actions, goals, transitions, rewards, and a discount factor. Let's name them..." |

### 1.3 Pronoun Policy

- **"you"** for direct instructions and code walkthroughs: "Run this command and check the output."
- **"we"** for shared reasoning and derivations: "We need off-policy learning because HER requires a replay buffer."
- **"I"** sparingly, for honest experience: "I spent two days on this bug before realizing the reward was unsigned."
- **Never** "one", "the reader", "the practitioner", "Professor Prytula".

### 1.4 Tone Principles

These carry forward from the tutorial voice (CLAUDE.md) and sharpen for Manning:

**Honest about difficulty.** "This took us three attempts" is better than
"simply configure." If something is hard, say so. If a hyperparameter was
found by trial and error, admit it. RL is fragile; pretending otherwise
insults the reader's intelligence and sets them up for frustration.

**Rigorous without being academic.** Define terms before use. State what
you expect to happen before running an experiment. Report numbers, not
vibes. But do this in plain language -- the rigor is in the discipline,
not the vocabulary.

**Humble and specific.** "In our experiments, SAC converged faster than PPO
on this task" not "SAC is superior." Acknowledge that different hardware,
different seeds, different hyperparameters may give different results. Cite
Henderson et al. (2018) naturally, not as a shield.

**Opinionated with receipts.** The book makes choices (SAC over TD3, future
over final strategy, specific hyperparameters). Own these choices and
explain why, but acknowledge alternatives exist. "We prefer SAC for this
task because the entropy bonus aids exploration -- TD3 would also work and
we show a comparison in the appendix."

**No hand-waving.** If a topic is out of scope, say where to go:
"Importance sampling is a deep topic (Sutton & Barto, 2018, Ch5.5). For our
purposes, the key point is that SAC sidesteps it entirely." Never say
"beyond the scope of this book" and leave the reader stranded.

---

## 2. The "From Scratch" Contract

This is the single most important framing decision for reviewers and readers.
Get it wrong and the book is an SB3 walkthrough with math annotations. Get
it right and the reader genuinely understands the algorithms they're running.

### 2.1 The Core Promise

**We build the core learning math from scratch -- buffers, losses, updates,
relabeling -- and use SB3 as the scaling engine for long robotics runs.**

This sentence must be consistent everywhere: proposal, sample chapters,
chapter intros, and the book's preface. It sets the reader's expectation
correctly: they will *write* the algorithm, *verify* that it works, and
*then* hand off to a production library for the multi-million-step training
runs that robotics demands.

### 2.2 The Boundary (Explicit and Honest)

What the reader builds from scratch (PyTorch + NumPy only, no SB3):

| Component | Where | Chapter |
|-----------|-------|---------|
| Actor-critic network (shared + separate) | `labs/ppo_from_scratch.py` | Ch3 (PPO) |
| GAE advantage computation | `labs/ppo_from_scratch.py` | Ch3 |
| PPO clipped surrogate loss + value loss | `labs/ppo_from_scratch.py` | Ch3 |
| PPO update loop (batched, multi-epoch) | `labs/ppo_from_scratch.py` | Ch3 |
| Replay buffer (circular, dict-obs aware) | `labs/sac_from_scratch.py` | Ch4 (SAC) |
| Twin Q-networks + clipped double-Q target | `labs/sac_from_scratch.py` | Ch4 |
| Squashed Gaussian policy + reparameterization | `labs/sac_from_scratch.py` | Ch4 |
| Soft Bellman backup + soft policy loss | `labs/sac_from_scratch.py` | Ch4 |
| Automatic entropy tuning (dual gradient) | `labs/sac_from_scratch.py` | Ch4 |
| SAC update loop (polyak averaging, target sync) | `labs/sac_from_scratch.py` | Ch4 |
| HER goal sampling strategies (future/final/episode) | `labs/her_relabeler.py` | Ch5 (HER) |
| Transition relabeling + reward recomputation | `labs/her_relabeler.py` | Ch5 |
| HER-augmented replay buffer insertion | `labs/her_relabeler.py` | Ch5 |
| Curriculum goal wrapper + difficulty schedule | `labs/curriculum_wrapper.py` | Ch6 |

What the reader uses as infrastructure (not rebuilt):

| Component | Library | Rationale |
|-----------|---------|-----------|
| Physics simulation | MuJoCo | Reimplementing a physics engine teaches nothing about RL |
| Environment interface | Gymnasium | The env API is a convention, not an algorithm |
| Tensor ops + autograd | PyTorch | We *use* autograd to implement losses, not rewrite it |
| Vectorized env rollout | SB3 VecEnv | Parallelism plumbing, not learning logic |
| Long-run training orchestration | SB3 | The production scaling engine (see 2.4) |

**The line is: if it appears in an equation in the chapter, the reader
builds it. If it's engineering plumbing (parallelism, logging, checkpointing),
SB3 handles it.**

### 2.3 Build It Is the Narrative Spine, Not a Sidebar

This is the structural decision that separates us from a library walkthrough.

**The main chapter narrative flows through Build It.** The reader derives
the math, then implements it, then verifies it. This is the *teaching*.
Run It comes *after* -- it's the payoff, not the lesson.

**Chapter arc:**

```
WHY:  Problem + motivation (why do we need this algorithm?)
HOW:  Derivation (objective -> constraints -> update rule)
      |
      v
BUILD IT:  Implement each component from scratch
           equation -> code listing -> verification checkpoint
           equation -> code listing -> verification checkpoint
           ...
           wiring step: assemble components into update function
           full --verify run: "your from-scratch implementation works"
      |
      v
BRIDGE:  Compare from-scratch to SB3 (the bridging proof, see 2.5)
      |
      v
RUN IT:  Scale up with SB3 for real robotics training
         experiment card -> command -> artifacts -> interpretation
      |
      v
WHAT CAN GO WRONG:  Diagnostics and failure modes
SUMMARY + EXERCISES
```

**Build It is NOT optional in sample chapters.** A reviewer who skips
Build It and only reads Run It should feel like they missed the book's
core content. The narrative should make this obvious: the Run It section
*refers back* to Build It components ("the twin-Q loss we implemented in
Section 4.3 is what SB3's `CriticNetwork` computes internally").

### 2.4 SB3's Role: The Scaling Engine

SB3 appears in the book, but with a specific, limited role:

**SB3 is the engine you use after you've built and understood the parts.**

The analogy: a mechanical engineering textbook teaches thermodynamics,
combustion cycles, and valve timing. The student builds a single-cylinder
engine on the bench. Then they drive a car. They don't build the car
from scratch -- but they understand why it works, and when it breaks,
they can diagnose it.

In practice:
- The reader implements SAC's loss functions, replay buffer, and update
  loop in ~200 lines of PyTorch
- They verify it on a tiny problem (e.g., 5k steps of Reach, checking
  that Q-values move, entropy decreases, returns improve)
- Then the chapter says: "Now let's train for real. SB3 implements the
  same math we just wrote, plus vectorized rollouts, efficient replay
  sampling, and proper checkpointing. Here's the command."
- The Run It section uses SB3 for the 500k-3M step training runs that
  produce publishable results

**SB3 is never introduced as a black box.** By the time the reader sees
`model = SAC("MultiInputPolicy", env, ...)`, they know what "MultiInputPolicy"
means (the actor-critic architecture they built in Build It), what the
replay buffer does (they wrote one), and what the loss function computes
(they derived and implemented it).

### 2.5 Bridging Proofs: Connecting Build It to Run It

This is the structural element that makes "from scratch" credible and
prevents the two tracks from feeling like parallel universes.

**A bridging proof is a small section (0.5-1 page) where the reader
verifies that their from-scratch implementation and SB3 agree on the
same computation.**

Examples by chapter:

**Ch3 (PPO):** Feed the same batch of transitions through your from-scratch
PPO loss and through SB3's internal loss computation. Compare:
- Probability ratios (should match within float precision)
- Clipped surrogate loss values (should match)
- Value loss (should match)
- Clip fraction (should match)

```
Bridging check: PPO loss agreement
  From-scratch surrogate loss:  -0.00237
  SB3 internal surrogate loss:  -0.00237  [match]
  From-scratch clip_fraction:    0.031
  SB3 logged clip_fraction:      0.031    [match]
```

**Ch4 (SAC):** Compare Q-value targets and policy entropy between your
from-scratch SAC and SB3's SAC on the same replay batch:
- Soft Bellman target y (should match)
- Actor loss (should match)
- Entropy coefficient alpha (should match after same number of updates)

**Ch5 (HER):** Feed the same episode through your from-scratch relabeler
and SB3's HER implementation. Verify:
- Same number of relabeled transitions produced
- Relabeled goals match
- Recomputed rewards match
- The fraction of positive rewards in the augmented buffer matches

**Why this matters:**
1. It proves the from-scratch code is correct (not just "it runs")
2. It demystifies SB3 (the reader sees it's doing the same math)
3. It gives the reader confidence to modify SB3 or debug it later
4. It answers the reviewer question: "is Build It connected to Run It, or
   are these two separate books?"

**Implementation note:** The bridging proofs live in the lab modules as a
`--bridge` mode (alongside `--verify` and `--demo`). The book shows the
output; the reader can rerun it.

### 2.6 What This Is NOT

To be maximally clear for reviewers and readers:

- **This is not a library walkthrough.** The reader does not learn SB3's
  API and call it a day. They implement the algorithms, verify them, and
  *then* use SB3 for scale.
- **This is not a toy reimplementation.** The from-scratch code is
  mathematically correct, verified against SB3, and produces learning
  curves on real Fetch environments (short runs, but real).
- **This is not "two books in one."** Build It and Run It are sequential
  steps in the same narrative, not parallel tracks. You build, you verify,
  you bridge, you scale.
- **This is not anti-library.** We *like* SB3. We use it for production
  runs. But understanding what it does is the prerequisite for using it
  well -- especially when things break.

### 2.7 Compute Tracks

RL is compute-intensive. The book respects this with five tiers, each
serving a different purpose in the reader's experience:

| Track | Steps | Time (GPU) | Time (CPU) | Purpose |
|-------|-------|-----------|-----------|---------|
| Build It `--verify` | ~1-5k | < 2 min | < 2 min | Sanity checks (shapes, finite values, correct signs) |
| Build It `--demo` | ~50-100k | ~5 min | ~15-30 min | From-scratch training: see real learning curves |
| Run It fast path | <= 500k | ~5-15 min | ~30-60 min | Quick SB3 validation, single seed |
| Reproduce It | <= 3M | ~30-90 min/seed | hours/seed | Full multi-seed runs backing the chapter's results |
| Checkpoint | 0 | seconds | seconds | Eval-only with pretrained models |

**Build It** has two modes with different purposes:

- **`--verify`** is the quick sanity check (under 2 minutes, CPU). It
  catches bugs: wrong shapes, NaN losses, rewards with the wrong sign.
  This is what you run after implementing each component.
- **`--demo`** is a short but real training run (up to ~30 minutes, CPU).
  It's long enough to see Q-values move, entropy decrease, success rate
  climb -- real learning, not just "the code didn't crash." This is what
  the book shows plots from and what makes Build It feel like actual RL
  training, not a unit test.

**Run It fast path** is the main-narrative SB3 run: single seed, <=500k
steps, enough to validate the pipeline and see strong results on easier
tasks. This is what the reader runs during the chapter.

**Reproduce It** is the full multi-seed training that backs the chapter's
results tables, learning curves, and pretrained checkpoints. It lives in
a short end-of-chapter block, not in the main narrative. The reader runs
it overnight (or not at all -- the checkpoint track provides the outputs).
See Section 3.5 for the block format.

**Checkpoint track** is always available. Every chapter's Reproduce It
outputs are provided as pretrained checkpoints so readers can evaluate,
inspect artifacts, and follow the diagnostics sections without training.

**Every chapter must be completable via Build It + checkpoint track.**
A reader without a GPU can still build the algorithm, verify it, see it
learn on a short demo, and evaluate pretrained policies. The
*understanding* is never gated by hardware; only the *scale* is.

### 2.8 How the Tracks Map to the Reader's Experience

Different readers will use different tiers, and that's by design:

| Reader profile | Path through the chapter |
|---------------|-------------------------|
| "I want to understand deeply" | Build It (`--verify` + `--demo`) -> Bridge -> Run It fast path -> Reproduce It overnight |
| "I want to learn and practice" | Build It (`--verify` + `--demo`) -> Bridge -> Run It fast path -> Checkpoint for full results |
| "I want results, will study later" | Run It fast path -> Checkpoint for full results |
| "I'm on a plane with no GPU" | Build It (`--verify` + `--demo` on CPU) -> Checkpoint track |

All four paths are first-class. The book never makes the reader feel like
they're getting a lesser experience for choosing a lighter compute path.

---

## 3. Chapter Structure (Manning "In Action" Adaptation)

### 3.1 The Arc of a Chapter

Manning "In Action" chapters follow: motivation -> concepts -> code -> exercises.
Our structure puts Build It at the center:

```
Manning convention          Our structure
------------------          -------------
"This chapter covers..."    Opening promise (2-3 bullet outcomes)
Motivation/problem          WHY (problem + why it matters)
Key concepts + code         HOW (derivation) -> BUILD IT (implement + verify + demo)
                            BRIDGE (compare from-scratch to SB3)
Scaling up                  RUN IT (SB3 fast path, experiment card, artifacts)
Diagnostics                 WHAT CAN GO WRONG (failure modes + debugging)
Summary                     Summary + bridge to next chapter
Full results                REPRODUCE IT (multi-seed full runs, ~0.5 page)
Exercises                   End-of-chapter exercises
```

**Note:** Build It is not a subsection of WHAT. It is woven into HOW.
The derivation and the implementation are interleaved: derive a component,
implement it, verify it, then derive the next. This keeps math grounded
and prevents the "wall of equations followed by wall of code" problem.

**Note:** Reproduce It sits after the summary, not in the main narrative
flow. The reader has already completed the chapter's learning arc. This
block says: "the numbers in this chapter came from here, and here's how
you can regenerate them."

### 3.2 Opening Promise (Every Chapter)

Every chapter opens with a 3-5 line block:

> **This chapter covers:**
> - Training SAC on dense Reach and interpreting replay diagnostics
> - Understanding the maximum-entropy objective and why it matters for exploration
> - Verifying that your off-policy stack works before adding HER

This is Manning house style. It replaces the tutorial-era "What This Chapter
Is Really About" heading.

### 3.3 Chapter Bridge (Every Chapter After Ch1)

Carry forward from the tutorial convention. Open every chapter with a bridge
that does four things:

1. **Name the capability established:** What can you now do?
2. **Identify the gap:** What can't you do yet?
3. **State what this chapter adds:** How does this chapter close the gap?
4. **Foreshadow:** What limitation will the *next* chapter address?

Keep bridges to 4-6 sentences. The bridge is not a summary of the previous
chapter; it is a *reason to keep reading*.

### 3.4 The Experiment Card (Recurring Element)

Every major experiment in the Run It section gets a standardized card.
The card covers the **fast path** -- what the reader runs during the
chapter. Full multi-seed runs live in the Reproduce It block (Section 3.5).

```
---------------------------------------------------------
EXPERIMENT CARD: SAC on FetchReachDense-v4
---------------------------------------------------------
Algorithm:    SAC (automatic entropy tuning)
Environment:  FetchReachDense-v4
Fast path:    500,000 steps, seed 0
Time:         ~10 min (GPU) / ~45 min (CPU)

Run command (fast path):
  bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all \
    --seed 0 --total-steps 500000

Checkpoint track (skip training):
  checkpoints/sac_FetchReachDense-v4_seed0.zip

Expected artifacts:
  checkpoints/sac_FetchReachDense-v4_seed0.zip
  checkpoints/sac_FetchReachDense-v4_seed0.meta.json
  results/ch03_sac_dense_reach_eval.json

Success criteria (fast path):
  success_rate >= 0.90
  mean_return > -2.5

Full multi-seed results: see REPRODUCE IT at end of chapter.
---------------------------------------------------------
```

This card is the "no vibes" contract. The reader knows exactly what to run,
what to expect, and how to tell if something went wrong. The card
deliberately gives fast-path criteria (slightly looser than full-run
numbers) so the reader isn't confused if their single-seed, shorter run
doesn't match the multi-seed results tables exactly.

### 3.5 The Reproduce It Block (Every Chapter)

Every chapter ends with a short (0.5-1 page) "Reproduce It" block after
the summary. This block serves three purposes:

1. **Provenance:** The results tables, learning curves, and pretrained
   checkpoints shown in the chapter came from these exact commands.
2. **Reproducibility:** Any reader with the Docker image and a GPU can
   regenerate the chapter's results from scratch.
3. **Checkpoint source:** The checkpoint track uses the outputs of these
   runs. This makes the connection explicit.

**Format:**

```
---------------------------------------------------------
REPRODUCE IT
---------------------------------------------------------
The results and pretrained checkpoints in this chapter
come from these runs:

  bash docker/dev.sh python scripts/ch03_sac_dense_reach.py all \
    --seeds 0-2 --total-steps 2000000

Hardware:     NVIDIA A100 (any modern GPU works; times will vary)
Time:         ~90 min per seed
Seeds:        0, 1, 2

Artifacts produced:
  checkpoints/sac_FetchReachDense-v4_seed{0,1,2}.zip
  checkpoints/sac_FetchReachDense-v4_seed{0,1,2}.meta.json
  results/ch03_sac_dense_reach_eval.json

Results summary (what we got):
  success_rate:  1.00 +/- 0.00  (3 seeds x 100 episodes)
  mean_return:  -1.12 +/- 0.08
  mean_distance: 0.008 +/- 0.001

If your numbers differ by more than ~10%, check the
"What Can Go Wrong" section above.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

**Rules:**
- Every Reproduce It block includes the exact command, hardware note,
  per-seed time estimate, artifact paths, and a results summary with
  seed count and variance.
- The results summary uses the same metrics schema as the chapter's
  results tables -- this is a cross-check, not new information.
- The block explicitly names the checkpoint track connection: "these
  are the checkpoints you've been evaluating."
- Keep it to half a page. This is reference material, not narrative.

### 3.6 Callout Boxes (Manning Conventions)

Manning uses specific callout types. Map our conventions to theirs:

| Manning callout | Our usage |
|----------------|-----------|
| **Definition** | Formal concept introduction (use the 5-step template for key concepts) |
| **Warning** | Common failure modes, silent bugs, "this will bite you" |
| **Tip** | Practical advice that saves time or compute |
| **Note** | Background context, connections to literature, scope boundaries |
| **Exercise** | End-of-section verification or parameter-tweak task |
| **Sidebar** | Extended topic that supports but doesn't block the main narrative |

**Definition callout policy:** Use the 5-step template (motivating problem,
intuitive description, formal definition, grounding example, non-example)
for *key* concepts -- roughly 2-4 per chapter. For lightweight terms, an
inline definition suffices.

### 3.7 Code Appearance

**In the book text:**
- Short annotated excerpts (15-30 lines) pulled from lab modules via
  snippet-includes
- Every excerpt is preceded by the relevant equation or pseudocode
- Every excerpt is followed by a verification checkpoint ("run this, expect
  these shapes / these values")
- Use Manning's listing format with caption and callout annotations

**Not in the book text:**
- Full runnable scripts (those live in the repo; the book gives the command)
- Boilerplate (imports, argparse, logging setup)
- Repetitive code (show the pattern once, reference it thereafter)

**The 30-line rule:** No single code listing exceeds ~30 lines. If it
must be longer, split into two listings with narrative between them.

### 3.8 Figures and Visual Elements

Figures are first-class citizens in every chapter. They ground abstract
concepts in concrete visuals and give the reader something to verify against.

**Figure types and when to use each:**

| Type | When to use | Example |
|------|------------|---------|
| Environment screenshot | Introducing a new Fetch task or showing initial state | Annotated reset view of FetchPush-v4 |
| Learning curve | Showing training progress or comparing algorithms | Success rate over steps: PPO vs SAC |
| Architecture diagram | Explaining network structure or data flow | Actor-critic with twin Q-networks |
| Comparison plot | Before/after, random/trained, or ablation results | Random vs trained policy side-by-side |
| Reward diagram | Explaining reward structure or failure modes | Dense vs sparse reward curves |
| Diagnostic plot | Illustrating what failure looks like | Entropy collapse, Q-value divergence |

**Caption format:**

Every figure caption follows this pattern:

```
Figure N.M: Description of what the figure shows. (Generated by
`python scripts/capture_proposal_figures.py env-setup --envs FetchReach-v4`.)
```

The generation command in the caption ensures reproducibility -- the reader can
regenerate every figure from the repository.

**Resolution and format:**
- PNG format for all static figures
- Minimum 640x480 pixels, 150 DPI for print quality
- White background for diagrams and plots

**Referencing figures in text:**

Use forward references sparingly. The pattern:
- "As Figure N.M shows, the gripper must reach the red target sphere."
- "Figure N.M compares the dense and sparse reward signals."
- Never "see below" or "the following figure" -- always use the figure number.

**Colorblind-friendly palette (Wong 2011):**

All figures use this palette consistently:

| Color | Hex | Usage |
|-------|-----|-------|
| Blue | `#0072B2` | Primary data series, action space labels |
| Orange | `#E69F00` | Secondary data series, reward computation |
| Green | `#009E73` | Success indicators, achieved_goal labels |
| Vermillion | `#D55E00` | Failure indicators, desired_goal labels |
| Gray | `#999999` | Baselines, thresholds, annotations |

**Annotation conventions:**
- Text uses DejaVu Sans font with shadow (draw at +2,+2 in dark, then at
  origin in color) for readability against varying backgrounds
- Labels are minimal: environment name, key components, metrics
- No decorative elements -- every annotation conveys information

**Static vs dynamic content:**
- Book figures are static PNGs (Manning print constraint)
- For dynamic content (rollout videos), provide the command to generate them:
  "To see the trained policy in action, run:
  `bash docker/dev.sh python scripts/generate_demo_videos.py --ckpt ...`"
- Never embed videos in chapter text; reference them as supplementary material

### 3.9 Math Level

The proposal calls this "math-lite but honest." In practice:

- **Include:** Key equations that the reader will see in code (loss functions,
  update rules, the Bellman equation, advantage computation)
- **Derive:** Show where the equation comes from, briefly. Not a proof, but
  "here's the objective, here's what happens when we differentiate."
- **Skip:** Convergence proofs, measure theory, detailed importance sampling
  derivations. Point to references.
- **Always:** Define every symbol before it appears in an equation. State
  domain, typical values, and units where helpful.

**Notation follows Sutton & Barto (2018)** with explicit correspondence
notes when papers use different notation. This is worth a short "notation
guide" in the book's front matter.

---

## 4. Voice Guidelines (Detailed)

### 4.1 The Headline Test

Before writing a section, write the "headline" -- the one-sentence promise
the reader should take away. If you can't state it simply, the section is
trying to do too much. Split it.

Examples:
- "PPO clips the probability ratio to prevent destructive updates."
- "HER turns failed trajectories into training signal by relabeling goals."
- "The entropy coefficient controls exploration-exploitation balance, and
  SAC tunes it automatically."

### 4.2 Show, Then Tell (Not Tell, Then Show)

The In Action pattern: show a result or a problem first, then explain it.

**Tutorial-era (Tell-then-Show):**
> "The Bellman equation describes the recursive relationship between value
> functions. Here is the equation: [...] Here is what happens when we run it: [...]"

**Manning book (Show-then-Tell):**
> "Run this command and look at the Q-value plot. Notice how values diverge
> after 200k steps? That's value overestimation -- and it happens because
> [Bellman equation explanation]."

Lead with the **observable phenomenon**, then provide the **conceptual
framework** that explains it.

### 4.3 Failure-First Pedagogy

Many chapters benefit from showing what goes wrong before showing what goes
right. This is natural in RL where naive approaches often fail:

- Ch3: "Train SAC with a broken entropy setup. Watch it diverge. Now fix it."
- Ch5: "Train SAC without HER on sparse rewards. Watch it flatline. Now add HER."
- Ch8: "Evaluate the trained policy with observation noise. Watch success drop. Now quantify how much."

This pattern works because it:
1. Creates a **felt need** for the technique (not just a theoretical argument)
2. Gives the reader a **diagnostic baseline** (they know what failure looks like)
3. Makes the fix feel **earned** (not handed down from authority)

### 4.4 Phrases to Use / Phrases to Avoid

| Avoid | Use Instead | Why |
|-------|-------------|-----|
| "It is trivial to see..." | "Let's check this directly." | Nothing is trivial to someone learning it |
| "Simply run..." | "Run this command -- it takes about 5 minutes on a GPU:" | Quantify instead of minimizing |
| "The reader should..." | "You'll want to..." or "Check that..." | Direct address |
| "Beyond the scope" (alone) | "That's a deep topic -- Sutton & Barto Ch5.5 is a good starting point." | Never strand the reader |
| "Obviously..." | (just state the fact) | If it were obvious, you wouldn't need to say it |
| "In the tradition of Bourbaki..." | (omit) | The discipline shows in the writing, not in name-dropping |
| "Professor Prytula..." | (never; first person or "we") | No third-person author presence |
| "This is not a recipe..." | "Here's why we made this choice:" | Invite rather than lecture |
| "One cannot..." | "You can't..." or "This won't work because..." | Direct, natural |

### 4.5 Handling Uncertainty and Limitations

RL is empirical. Hyperparameters are fragile. Results vary across seeds.
The voice must reflect this honestly:

- "In our experiments, three seeds converged within 500k steps. Your results
  may differ by 10-20% depending on hardware and random seed."
- "We chose learning_rate=3e-4 because it worked reliably across tasks in
  our testing. The SAC paper uses 3e-4 as well (Haarnoja et al., 2018a)."
- "If your success rate is below 50% at this point, something is likely
  wrong -- check [specific diagnostic steps]."
- "We don't fully understand why [X], but empirically it matters. Here's
  what we've observed: [data]."

---

## 5. The Hadamard Diagnostic (Kept, Reframed)

The three Hadamard questions remain as a *diagnostic tool*, not a formal
mathematical framework. Reframe them as practical engineering questions:

| Hadamard (formal) | Book version (practical) |
|-------------------|--------------------------|
| Does a solution exist? | **Can this task be solved?** Is there a policy architecture that can represent the solution? |
| Is the solution unique? | **Are there multiple ways to solve it?** Will different seeds find qualitatively different strategies? |
| Does it depend continuously on the data? | **Is the solution stable?** Will small changes in hyperparameters or random seeds break it? |

Use these questions explicitly at the start of each new task or algorithm
introduction. They are the "well-posedness check" that prevents the reader
from training for 10 hours on a task that cannot work.

**How to introduce them in the book (once, in Ch1):**

> Before training anything, we find it useful to ask three questions.
> These come from the mathematician Hadamard, who studied when problems
> have reliable solutions, but you don't need the math background -- they
> translate directly to engineering:
>
> 1. *Can this be solved?* Is there a policy that achieves the goal?
> 2. *Is the solution reliable?* Will different runs give similar results?
> 3. *Is the solution stable?* Will small changes break it?
>
> If the answer to any of these is "no" or "we don't know," we have work
> to do before training.

After the introduction, use them as a lightweight checklist at the start
of each chapter, not a formal section.

---

## 6. Structural Conventions

### 6.1 Define Before Use (The Bourbaki Discipline)

The principle stays. The vocabulary changes.

**In tutorials:** "We follow the Bourbaki tradition: rigorous, sequential,
no undefined terms."

**In the book:** (Just do it. Don't announce it.) If a reader encounters a
term they haven't seen, the chapter has a bug. The concept registry
(CLAUDE.md) is the authoring tool that enforces this; it does not appear
in the book.

**Exception for forward references:** When you must mention a concept before
its chapter, use a brief inline definition: "off-policy methods (algorithms
that reuse past experience, which we'll cover in Chapter 4)."

### 6.2 The 5-Step Definition Template

For key concepts (2-4 per chapter), use the full template. But make it
feel natural, not templated:

1. **Motivating problem** -- one sentence, rooted in what the reader just
   experienced or will experience
2. **Intuitive description** -- plain language, analogy if it helps
3. **Formal definition** -- equation or precise statement, symbols defined
4. **Grounding example** -- concrete numbers the reader can verify
5. **Non-example** -- what this is NOT (especially for confusing concepts)

For lightweight concepts, steps 1-2 as a single sentence are sufficient.

### 6.3 Derivation Over Recipe

When introducing an algorithm, derive it from constraints:

1. State what we want (the objective)
2. State what we have (the constraints: off-policy data, continuous actions,
   sparse rewards, etc.)
3. Show how the algorithm follows from these constraints
4. Make the reader feel the algorithm is *inevitable*, not arbitrary

This is the strongest idea from the persona document. It translates directly
to the book. The only change: do it in conversational language, not formal
mathematical prose.

**Example (SAC motivation):**

> We want to learn from replay data (off-policy requirement). We want
> continuous actions (rules out DQN). We want to explore without a separate
> exploration strategy. These three constraints point to maximum-entropy
> actor-critic methods. SAC is the standard instantiation. Let's see why.

### 6.4 ASCII-Only Constraint

Inherited from the tutorials and essential for Manning's toolchain:
- `--` not em-dash
- Straight quotes, not curly
- `...` not ellipsis
- `->` not arrow
- No emoji, no decorative Unicode

### 6.5 Chapter Length Target

Manning "In Action" chapters typically run 15-25 printed pages, which
corresponds to roughly 6,000-10,000 words including code listings.

Our current tutorials are longer than this. For the book:
- Tighten the WHY section (the reader already chose this book; less selling)
- Move extended derivations to sidebars
- Keep Build It sections focused (one component per subsection)
- Move full experiment sweeps to appendix or online supplement

---

## 7. Adaptation Map: Tutorials -> Book Chapters

This table shows what changes when a tutorial becomes a book chapter:

| Tutorial element | Book adaptation |
|-----------------|----------------|
| "What This Chapter Is Really About" heading | Manning "This chapter covers:" bullet list |
| Informal chapter bridge | Formalized 4-part bridge (capability, gap, addition, foreshadow) |
| Inline experiment commands | Experiment Card callout |
| `pymdownx.snippets` includes | Manning listing format with caption + callout annotations |
| `!!! lab "Build It"` admonitions | Manning "Listing N.N" with "Build It" label |
| `!!! lab "Checkpoint"` admonitions | Manning "Exercise" or "Try It" callout |
| GIF/video embeds | Static screenshots + "run this command to see the video" |
| Direct MkDocs features | Manning's AsciiDoc/Word conventions (coordinate with editor) |
| Section headers like "Part 1: WHY" | Subsection numbers per Manning style (3.1, 3.2, ...) |
| Back-references like "(defined in Ch2)" | Manning cross-references: "as we saw in chapter 2" |
| Open-ended "syllabus" references | Self-contained within each chapter |

### 7.1 What Stays the Same

- WHY-HOW-WHAT narrative arc (it maps to Manning's motivation-concepts-code arc)
- Define-before-use discipline (enforced by the concept registry)
- Three-track pedagogy (Run It / Build It / Verify It)
- Failure-first teaching (show breakage, then fix)
- Quantified claims with seed counts and confidence
- Experiment cards with reproducible commands
- Honest acknowledgment of difficulty and limitations
- Single source of truth for code (labs for Build It, scripts for Run It)

---

## 8. The Four Commitments (Engineering-Framed)

These are the tutorial-era "four non-negotiables" reframed for a
practitioner audience:

### 8.1 Reproducibility

Every experiment in this book can be rerun by anyone with the same Docker
image. We report results across multiple seeds because single-run results
in RL are meaningless. If you get different numbers than we show, something
is wrong and we help you diagnose it.

### 8.2 Quantification

Numbers, not adjectives. "94% success rate over 3 seeds and 100 episodes
per seed" is a claim. "Works well" is not. Every chapter ends with a
results table that looks like data, not marketing.

### 8.3 Derivation

We show where algorithms come from, not just how to call them. This is what
"from scratch" means at the conceptual level: you understand *why* the loss
function has that form, not just that SB3 has a function called `train()`.

### 8.4 Debuggability

When training fails -- and it will -- you'll know how to diagnose it. Each
chapter includes a "What Can Go Wrong" section with specific symptoms,
likely causes, and diagnostic commands. We treat debugging as a skill, not
an afterthought.

---

## 9. Writing Process Notes

### 9.1 Chapter Drafting Workflow

The full step-by-step protocol lives in
`manning_proposal/chapter_production_protocol.md`. The short version:

**Phase 1 (Scaffolding):** Read source tutorial, write Experiment Card
and Reproduce It block, list Build It components, identify bridging proof,
list failure modes. Do this BEFORE writing prose.

**Phase 2 (Narrative):** Opening promise -> Bridge -> WHY ->
HOW/Build It interleaved -> Bridge section -> Run It -> What Can Go Wrong
-> Summary -> Reproduce It -> Exercises.

**Phase 3 (Verification):** Self-review checklist, verify lab code
(`--verify`, `--demo`, `--bridge`), verify Run It commands, final
read-through.

The protocol also includes chapter-type classification (Setup, Environment,
Algorithm, Capstone, Engineering, Pixels) with type-specific guidance,
and per-chapter adaptation notes for Ch1-6.

### 9.2 Reviewer Anticipation

Based on the review feedback (`manning_proposal/review_notes.md`), Manning
reviewers will likely probe:

- **"From scratch" clarity:** Does the book actually deliver on the promise?
  -> Yes: Build It is the narrative spine, not a sidebar. Every algorithm
  chapter derives the math, implements it in PyTorch, verifies it, and
  then bridges to SB3 with a proof that both compute the same thing. The
  sample chapters must make this visible: a reviewer who skips Build It
  should feel they missed the book's core content.

- **"Why use SB3 at all?":** If you build it from scratch, why not train
  with your own code?
  -> Because robotics RL needs millions of steps, vectorized environments,
  efficient replay sampling, and proper checkpointing. SB3 is the *scaling
  engine*. We build the *learning engine* from scratch. The bridging proofs
  show they're the same computation. This is like building a single-cylinder
  engine to understand combustion, then driving a car.

- **Compute accessibility:** Can a CPU-only reader complete the book?
  -> Yes: Build It `--verify` runs in < 2 min on CPU; `--demo` runs in
  under 30 min on CPU and shows real learning. Checkpoint track for
  full-scale evaluation. The *understanding* is never gated by hardware.

- **Differentiation:** How is this different from existing RL books?
  -> Single task family, artifact-first evaluation, from-scratch
  implementations with bridging proofs to production code,
  robotics-specific (goal-conditioned + HER).

- **Code quality:** Is the code production-grade or toy?
  -> Build It is pedagogical: explicit tensors, clear variable names,
  ~200 lines per algorithm, verified against SB3. Run It is production-grade:
  SB3 with proper logging, reproducible artifacts, multi-seed experiments.
  The bridging proof connects them: same math, different engineering.

### 9.3 Scope Control

The 13-chapter ToC is ambitious. Keep Parts 1-3 (Chapters 1-6) tight and
submission-ready first. Parts 4-5 (Chapters 7-13) can be outlined but
should not delay the initial submission.

**In-book vs. online supplement:**
- In book: everything through Ch6 (capstone manipulation) + Ch7-8 (robustness)
- Online supplement candidates: Ch9 (sweeps), Ch12 (visual goals), detailed
  hyperparameter tables, additional ablations

---

## 10. Quick Reference Card

When in doubt, check against these:

**From-scratch contract:**
- [ ] **Build It is the narrative spine?** (not a sidebar or enrichment track)
- [ ] **Every equation in HOW has a corresponding code listing in Build It?**
- [ ] **Bridging proof present?** (from-scratch vs SB3 on same computation)
- [ ] **Run It refers back to Build It?** ("the loss we implemented in 4.3...")
- [ ] **SB3 is never introduced as a black box?** (reader already knows what it computes)

**Structure:**
- [ ] **Every term defined before first use?** (concept registry check)
- [ ] **Every equation has all symbols defined?** (including domain, typical values)
- [ ] **Every code listing under 30 lines?**
- [ ] **Every code listing has a verification checkpoint?**
- [ ] **Experiment Card present for every major experiment?**
- [ ] **Chapter bridge present?** (all chapters after Ch1)
- [ ] **"What Can Go Wrong" section present?**

**Voice:**
- [ ] **Numbers, not adjectives** for all performance claims?
- [ ] **ASCII only** (no em-dash, no curly quotes, no Unicode)?
- [ ] **No third-person author references?**
- [ ] **No "simply," "trivially," "obviously"?**
- [ ] **Scope boundaries have specific references** (not "beyond the scope")?

**Visual content:**
- [ ] **Every chapter has at least 2 figures?**
- [ ] **Every figure has a numbered caption with generation command?**
- [ ] **Every figure is referenced by number in the text?**
- [ ] **Figures use the Wong (2011) colorblind-friendly palette?**
- [ ] **All figure text is legible at print size (>= 9pt equivalent)?**
- [ ] **Static PNGs only in chapter text?** (videos as supplementary commands)

**Compute accessibility:**
- [ ] **Build It `--verify` runs on CPU in under 2 minutes?**
- [ ] **Build It `--demo` runs on CPU in under 30 minutes and shows learning?**
- [ ] **Chapter completable via Build It + checkpoint track?** (no GPU required for understanding)
- [ ] **Reproduce It block present?** (exact command, hardware, time, artifacts, results summary)
- [ ] **Reproduce It results match the chapter's tables/curves?** (cross-check)
- [ ] **Checkpoint track sources are the Reproduce It outputs?** (connection explicit)
