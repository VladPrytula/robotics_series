# Chapter Production Protocol

A step-by-step, reusable protocol for producing Manning book chapters from
existing tutorials. Follow this for every chapter.

---

## How to Use This Protocol

Four specialized agents execute this protocol. Their prompts live in
`manning_proposal/agents/`. Each agent reads only the context it needs.

| Agent | Prompt file | Phase | Input | Output |
|-------|------------|-------|-------|--------|
| **Scaffolder** | `agents/scaffolder.md` | Phase 0-1 | Source tutorial + this protocol | `Manning/scaffolds/chNN_scaffold.md` |
| **Lab Engineer** | `agents/lab_engineer.md` | Phase 2 (code) | Scaffold + root `CLAUDE.md` | Updated `scripts/labs/*.py` |
| **Book Writer** | `agents/writer.md` | Phase 2 (prose) | Scaffold + persona | `Manning/chapters/chNN_<topic>.md` |
| **Reviewer** | `agents/reviewer.md` | Phase 3 | Chapter + scaffold + persona + labs | `Manning/reviews/chNN_review.md` |
| **Revisor** | `agents/revisor.md` | Phase 3.5 | Chapter + scaffold + review + revision scope | Updated chapter + scaffold + `Manning/revisions/chNN_revision_NNN.md` |

### Workflow

```
Step 1:  Spawn Scaffolder agent
         Input:  chapter number + source tutorial path
         Output: Manning/scaffolds/chNN_scaffold.md
              |
         [User reviews and approves scaffold]
              |
Step 2:  Spawn Lab Engineer + Book Writer agents (can be parallel)
         Lab Engineer: reads scaffold -> updates scripts/labs/*.py
         Book Writer:  reads scaffold + persona -> writes Manning/chapters/chNN.md
              |
         [Note: if Writer needs lab regions that don't exist yet,
          run Lab Engineer first, then Writer]
              |
Step 3:  Spawn Reviewer agent
         Input:  chapter draft + scaffold + persona + lab code
         Output: Manning/reviews/chNN_review.md
              |
         [User reviews findings]
              |
Step 3.5 (optional): Spawn Revisor agent for targeted fixes
         Input:  chapter + scaffold + review + revision scope
         Output: Updated chapter + scaffold + Manning/revisions/chNN_revision_NNN.md
              |
         [Re-review if changes were substantial]
              |
Step 4:  Verification (manual or scripted)
         Run Experiment Card commands, check artifacts, final read-through
```

### Spawning Agents

Use the Task tool to spawn each agent. Pass the agent prompt file as
context and include specific instructions:

```
Task(subagent_type="general-purpose", prompt="""
Read manning_proposal/agents/scaffolder.md for your instructions.
Then produce the scaffold for Chapter 1.
Source tutorial: tutorials/ch00_containerized_dgx_proof_of_life.md
""")
```

### Key Rules

- **Always scaffold first.** Never skip Phase 1.
- **The scaffold is the contract.** Lab Engineer and Book Writer both
  work from the same scaffold. If the scaffold changes, both must re-run.
- **Lab Engineer before Writer when lab code is new.** The Writer
  references lab regions by name. If those regions don't exist yet,
  the Lab Engineer must create them first.
- **Review before verification.** The Reviewer catches structural and
  voice issues cheaply (just reading). Verification catches runtime
  issues (running code). Do cheap checks first.

---

## Phase 0: Classify the Chapter Type

Not all chapters are alike. The protocol adapts based on chapter type:

| Type | Chapters | Build It focus | Bridging proof | Reproduce It |
|------|----------|---------------|----------------|--------------|
| **Setup** | Ch1 (Proof of Life) | Env inspection, reward semantics | Manual compute_reward == env reward | Smoke test artifacts |
| **Environment** | Ch2 (Env Anatomy) | Obs/action spaces, reward structure, goal mapping | Manual reward/goal checks == env behavior | N/A (no training) |
| **Algorithm** | Ch3 (PPO), Ch4 (SAC), Ch5 (HER) | Full from-scratch implementation (losses, updates, buffers) | Loss/output comparison vs SB3 on same batch | Multi-seed full runs |
| **Capstone** | Ch6 (PickAndPlace) | Curriculum wrapper, stress-test wrapper | Curriculum schedule matches env difficulty | Multi-seed + stress results |
| **Engineering** | Ch7-8 (Robustness, Tuning) | Noise wrappers, degradation metrics | Noise injection matches expected distributions | Degradation curve runs |
| **Pixels** | Ch10-12 (Visual RL) | CNN encoder, augmentation pipeline | Encoder output shapes, augmentation statistics | Visual training runs |

Identify your chapter type before starting. The protocol below marks
type-specific steps with [Algorithm], [Setup], etc.

### Visual Content by Chapter Type

Each chapter type has a target figure count. These are minimums -- add more
if they aid understanding, but every figure must earn its place.

| Chapter type | Target figures | Typical figure types |
|---|---|---|
| **Setup** (Ch1) | 1-2 | Environment screenshot, smoke test output |
| **Environment** (Ch2) | 4-6 | Env screenshots (all tasks), obs structure diagram, action space |
| **Algorithm** (Ch3-5) | 3-5 | Learning curves, architecture diagram, before/after comparison |
| **Capstone** (Ch6) | 3-5 | Difficulty progression, curriculum schedule, stress-test results |
| **Engineering** (Ch7-8) | 3-4 | Degradation curves, noise injection effects, diagnostic plots |
| **Pixels** (Ch10-12) | 3-5 | Raw vs augmented frames, CNN architecture, visual learning curves |

---

## Phase 1: Scaffolding (Before Writing Prose)

Do these steps FIRST. They force clarity about what the chapter delivers
before you commit to narrative structure. The Scaffolder agent produces
`Manning/scaffolds/chNN_scaffold.md` with all six deliverables below.

### Step 1: Read the Source Tutorial

Read `tutorials/chNN_*.md` end-to-end. Note:

- [ ] What concepts does it introduce? (Check against the Concept Registry in CLAUDE.md)
- [ ] What commands does it run?
- [ ] What artifacts does it produce?
- [ ] What voice/framing needs to change for Manning? (See adaptation map, persona Section 7)
- [ ] What content is missing that the book needs? (Build It, Bridge, Experiment Card, etc.)
- [ ] What content is excessive and should be cut or moved to a sidebar?

### Step 2: Write the Experiment Card

This forces you to commit to what the chapter actually delivers.

```
---------------------------------------------------------
EXPERIMENT CARD: <Algorithm> on <Environment>
---------------------------------------------------------
Algorithm:    <name> (<key detail>)
Environment:  <env-id>
Fast path:    <steps>, <seed(s)>
Time:         ~<N> min (GPU) / ~<N> min (CPU)

Run command (fast path):
  bash docker/dev.sh python scripts/chNN_<topic>.py <subcommand> \
    --seed 0 --total-steps <N>

Checkpoint track (skip training):
  checkpoints/<algo>_<env>_seed0.zip

Expected artifacts:
  checkpoints/<algo>_<env>_seed0.zip
  checkpoints/<algo>_<env>_seed0.meta.json
  results/chNN_<topic>_eval.json

Success criteria (fast path):
  success_rate >= <threshold>
  mean_return > <threshold>

Full multi-seed results: see REPRODUCE IT at end of chapter.
---------------------------------------------------------
```

**[Setup chapters]:** The experiment card is simpler (smoke test, not
training). Success criteria = "artifacts exist and are valid."

### Step 3: Write the Reproduce It Block

This forces you to commit to provenance. What exact command generated
the numbers in your tables?

```
---------------------------------------------------------
REPRODUCE IT
---------------------------------------------------------
The results and pretrained checkpoints in this chapter
come from these runs:

  <exact command, with --seeds and --total-steps>

Hardware:     <GPU model> (any modern GPU works; times will vary)
Time:         ~<N> min per seed
Seeds:        <list>

Artifacts produced:
  <artifact paths with seed placeholders>

Results summary (what we got):
  <metric>: <mean> +/- <std>  (<N> seeds x <N> episodes)
  <metric>: <mean> +/- <std>
  ...

If your numbers differ by more than ~<tolerance>%, check the
"What Can Go Wrong" section above.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

**[Setup chapters]:** Reproduce It may be identical to the fast path
(the smoke test is quick enough to always run in full). Still include
the block for consistency.

**[Environment chapters]:** No training, so no Reproduce It block.
Replace with a "Verify It" summary showing expected inspection outputs.

### Step 4: List the Build It Components

Enumerate what the reader implements from scratch. For each component:

| # | Component | Equation / concept it implements | Lab region | Verify check |
|---|-----------|--------------------------------|------------|--------------|
| 1 | ... | ... | `labs/<file>.py:<region>` | "expect shape (N,4), values in [-1,1]" |
| 2 | ... | ... | ... | ... |

**[Algorithm chapters]:** This is the big one. Every equation in the HOW
section maps to a component here. Typical count: 4-8 components per chapter.

**[Setup chapters]:** Build It is lighter -- env inspection, manual reward
calls, observation structure. Typical count: 2-4 components.

**Ordering rule:** Foundation first. Show the data structures and networks
before the losses that operate on them. Each component has a verification
checkpoint before the next one starts.

### Step 5: Identify the Bridging Proof

What specific comparison connects Build It to Run It?

| Chapter type | Bridging proof |
|-------------|----------------|
| **Algorithm** | Feed same batch through from-scratch and SB3; compare loss values, Q-targets, gradients, or logged metrics |
| **Setup** | Manual `compute_reward(ag, dg, info)` == reward from `env.step()` |
| **HER** | Same episode through from-scratch relabeler and SB3 HER; compare relabeled goals and reward counts |
| **Capstone** | Curriculum wrapper produces expected goal distributions at each difficulty level |

Write down:
- [ ] What inputs both implementations receive (same batch, same episode, same env state)
- [ ] What outputs you compare (loss values, rewards, goals, metrics)
- [ ] What "match" means (exact float match, within tolerance, same distribution)
- [ ] Where this lives in the lab code (`--bridge` mode)

### Step 6: List "What Can Go Wrong"

Enumerate failure modes with symptoms, causes, and diagnostics:

| Symptom | Likely cause | Diagnostic command or check |
|---------|-------------|----------------------------|
| ... | ... | ... |

Pull from:
- Tutorial's troubleshooting sections
- Known bugs encountered during development
- Common RL failure patterns (flat curves, diverging Q-values, entropy collapse)

### Step 7: Plan Figures

Every chapter needs a figure plan before prose begins. For each figure:

| # | Description | Type | Source command | Chapter location |
|---|------------|------|---------------|-----------------|
| 1 | ... | screenshot / curve / diagram / comparison | `python scripts/...` or "matplotlib in lab" | After Section N.M |
| 2 | ... | ... | ... | ... |

**Per-figure checklist:**
- [ ] Caption drafted (Figure N.M format with generation command)
- [ ] Source command tested or confirmed possible
- [ ] Alt-text written (for accessibility)
- [ ] Uses Wong (2011) colorblind-friendly palette
- [ ] Minimum 640x480, PNG format

**[Algorithm chapters]:** Plan at least one learning curve, one architecture
or data-flow diagram, and one before/after comparison.

**[Setup/Environment chapters]:** Plan at least one annotated environment
screenshot per task introduced.

---

## Phase 2: Narrative (Writing the Chapter)

Write sections in this order. Each section has a target length and a
completion check.

### Writer Span Protocol (Long Chapters)

Some chapters exceed the output capacity of a single writer agent. The
scaffold's **Estimated Length** table (Phase 1) tells you in advance.

**Rule:** If the scaffold's total estimated word count exceeds **6,000 words**,
split the chapter into sequential **spans** -- each assigned to a separate
writer agent instance.

**How to split:**

1. Read the scaffold's Estimated Length table.
2. Accumulate sections top-to-bottom until the running total approaches
   5,000-6,000 words. That is the end of Span 1.
3. The remaining sections form Span 2 (and Span 3 if needed, though most
   chapters fit in 2 spans).
4. Never split inside a section. The smallest unit is one `##` section.
5. If the final span would be under 2,000 words, merge it into the previous
   span.

**Split point guidelines:**

| Preferred split points | Why |
|----------------------|-----|
| Between last Build It section and Bridge section | Natural shift from implementation to verification |
| Between Bridge section and Run It | Shift from conceptual to procedural |
| After WHY, before first Build It | Shift from motivation to implementation |

**How to spawn span agents:**

Spans run **sequentially** (Span 2 depends on Span 1's output). For each span:

```
Task(subagent_type="general-purpose", prompt="""
Read manning_proposal/agents/writer.md for your instructions.

You are writing **Span N of M** for Chapter NN.

Your assigned sections: [list sections with estimated word counts]

[If Span 1:]
No previous context -- start fresh with Opening Promise.

[If Span 2+:]
Previous span's handoff context:
--- HANDOFF FROM SPAN N-1 ---
[paste the previous span's HANDOFF block]
---
Previous span's closing text (last ~1000 words):
--- CLOSING TEXT ---
[paste last ~1000 words of previous span's output]
---

Write ONLY your assigned sections. End with a <!-- HANDOFF --> block
(see writer.md Span Mode for format).

Scaffold: Manning/scaffolds/chNN_scaffold.md
Source tutorial: tutorials/chNN_<topic>.md
""")
```

**Assembly:**

After all spans complete:
1. Strip the `<!-- HANDOFF ... -->` block from each span's output.
2. Concatenate the cleaned spans in order.
3. Write the result to `Manning/chapters/chNN_<topic>.md`.
4. Verify the assembled chapter has all sections from the scaffold.

### Step 7: Opening Promise

**Write:**
```
> **This chapter covers:**
> - <outcome 1>
> - <outcome 2>
> - <outcome 3>
```

**Target:** 3-5 bullet points, each starting with a gerund or verb.
**Check:** Could someone read just this block and know what they'll be able
to do after the chapter?

### Step 8: Chapter Bridge

**[Skip for Ch1.]**

**Write:** 4-6 sentences covering:
1. What capability the previous chapter established
2. What gap remains
3. What this chapter adds
4. (Optional) What limitation the *next* chapter addresses

**Check:** Does this create a *reason to keep reading*, not just a summary?

### Step 9: WHY Section

**Write:** The problem this chapter solves and why it matters.

**Source:** Tutorial's WHY / "The Problem" section.

**Adaptation:**
- Drop formal Definition blocks (unless this is a key concept worth the 5-step template)
- Reframe Hadamard questions as the 3 practical engineering questions (can it be solved? is it reliable? is it stable?) -- use as a lightweight checklist, not a formal section
- Lead with the felt problem ("you train for 10 hours and success is 0% -- now what?"), not the mathematical formulation
- Keep it to ~1,500 words max

**Check:** Does this section make the reader *feel* why the technique matters, not just understand it intellectually?

### Step 10: HOW / Build It (Interleaved)

This is the narrative spine. The reader derives, implements, and verifies
one component at a time.

**Pattern for each component:**

```
10.N.1  Restate the relevant equation (1-3 lines of math, symbols defined)
10.N.2  Code listing from lab (15-30 lines, annotated)
10.N.3  Verification checkpoint:
        - What to run (--verify or inline snippet)
        - What shapes / values to expect
        - What "correct" looks like (e.g., "loss near 0 at init",
          "Q-values in [-10, 0] range")
```

**After all components:**

```
10.W    Wiring step: the orchestration function that assembles components
        (e.g., ppo_update, sac_update, her_relabel_episode)
10.D    Demo run: --demo mode, show that the from-scratch implementation
        actually learns (~50-100k steps, show a plot or metric summary)
```

**[Algorithm chapters]:** 4-8 components, ~3,000-4,000 words total.
**[Setup chapters]:** 2-4 components (env inspection, reward calls),
~2,000-2,500 words.

**Check:**
- Every equation has a corresponding code listing?
- Every code listing has a verification checkpoint?
- The wiring step is present (not just isolated components)?
- `--demo` shows real learning, not just "it didn't crash"?

### Step 11: Bridge Section

**Write:** ~500 words showing the from-scratch vs SB3 comparison.

**Structure:**
1. "We've built X from scratch. SB3 implements the same math with
   production engineering. Let's verify they agree."
2. What inputs both receive (same batch / episode / env state)
3. The comparison output (expected values, [match] annotations)
4. What this means for the reader ("when you see SB3's SAC logging
   `ent_coef: 0.15`, that's the alpha we implemented in Section N.N")

**Check:** Does a reader finishing this section trust that SB3 is doing
what they just built? Can they map SB3 log metrics to their code?

### Step 12: Run It

**Write:** ~1,500 words covering:
1. The Experiment Card (from Phase 1, Step 2)
2. The command to run
3. How to interpret the artifacts (what files, what the numbers mean)
4. A results table or plot from the fast path
5. Cross-reference to Reproduce It for full results

**Tone shift:** This section is more procedural than HOW/Build It. The
reader is executing, not learning. Keep instructions crisp.

**Check:**
- Experiment Card is present and complete?
- A reader following only this section + checkpoint track can still
  get meaningful outputs?
- The results shown here are from the fast path, not the full run?
  (Full run numbers live in Reproduce It)

### Step 13: What Can Go Wrong

**Write:** ~1,000 words covering failure modes from Phase 1, Step 6.

**Structure:** Table or bulleted list:
- **Symptom:** What the reader sees (error message, flat curve, diverging metric)
- **Likely cause:** What's probably wrong
- **Fix:** Specific diagnostic step or command

**Source:** Tutorial's troubleshooting appendices + known RL failure modes.

**Check:** If a reader's training fails, can they find their symptom in
this section and fix it without external help?

### Step 14: Summary + Bridge to Next Chapter

**Write:** ~500 words.

**Structure:**
1. What the reader can now do (concrete capabilities, not abstractions)
2. What numbers they should have (success rate, artifacts)
3. What limitation this chapter did NOT solve (bridge to next chapter)
4. One sentence foreshadowing the next chapter's approach

**Check:** Does this summary make the reader want to turn the page?

### Step 15: Reproduce It Block

Insert the block from Phase 1, Step 3.

### Step 16: Exercises

**Write:** 3-5 exercises, graduated in difficulty:

| Level | Type | Example |
|-------|------|---------|
| **Verify** | Confirm understanding by checking a specific output | "Change the seed and confirm training still produces artifacts" |
| **Tweak** | Modify one parameter and interpret the outcome | "Set n_sampled_goal=8 instead of 4; does success improve?" |
| **Extend** | Small implementation or analysis task | "Add action smoothness to the evaluation and compare across seeds" |
| **Challenge** | Open-ended exploration | "Replace 'future' strategy with 'final'; explain why results differ" |

Each exercise should be completable in 5-30 minutes. Include expected
outcomes where possible ("you should see success_rate drop to ~0.60").

---

## Phase 3: Review and Verification

**Spawn the Reviewer agent** (`manning_proposal/agents/reviewer.md`) to
produce `Manning/reviews/chNN_review.md`. The Reviewer runs the checklist
below systematically plus voice compliance, technical accuracy, structural
completeness, content gap analysis, and reader experience simulation.
See the agent prompt for the full review structure.

After the review, address findings and then run the manual verification
steps (Steps 18-20) which require actually running code.

### Step 17: Review Checklist

The Reviewer agent checks these (and more -- see `agents/reviewer.md`):

**From-scratch contract:**
- [ ] Build It is the narrative spine (not a sidebar)?
- [ ] Every equation in HOW has a corresponding code listing in Build It?
- [ ] Bridging proof present (from-scratch vs SB3)?
- [ ] Run It refers back to Build It ("the loss we implemented in N.N...")?
- [ ] SB3 is never introduced as a black box?

**Structure:**
- [ ] Every term defined before first use?
- [ ] Every equation has all symbols defined?
- [ ] Every code listing under 30 lines?
- [ ] Every code listing has a verification checkpoint?
- [ ] Experiment Card present?
- [ ] Chapter bridge present (except Ch1)?
- [ ] "What Can Go Wrong" section present?

**Voice:**
- [ ] Numbers, not adjectives for all performance claims?
- [ ] ASCII only?
- [ ] No third-person author references?
- [ ] No "simply," "trivially," "obviously"?
- [ ] Scope boundaries have specific references?

**Compute accessibility:**
- [ ] Build It `--verify` runs on CPU in under 2 minutes?
- [ ] Build It `--demo` runs on CPU in under 30 minutes and shows learning?
- [ ] Chapter completable via Build It + checkpoint track?
- [ ] Reproduce It block present with exact commands and results?

**Visual content:**
- [ ] Every figure referenced by number in the text?
- [ ] Every figure has a caption with generation command?
- [ ] Alt-text present for all figures?
- [ ] Figures legible at print size (text >= 9pt equivalent)?
- [ ] Colorblind-friendly palette used consistently?
- [ ] All figures are static PNGs (no embedded videos)?
- [ ] Figure count meets chapter-type minimum (see Phase 0 table)?

---

## Phase 3.5: Revision (Targeted Post-Review Fixes)

**When to use:** After Phase 3 produces a review, some chapters need targeted
fixes that do not warrant re-running the full Writer agent. The Revisor agent
makes surgical changes to reviewed chapters.

**Spawn the Revisor agent** (`manning_proposal/agents/revisor.md`) with a
specific revision scope:

| Scope | When to use | Example |
|-------|-------------|---------|
| `figures` | Chapter has no figures or is below the chapter-type minimum | "Add figures to Ch02 per the scaffold's Figure Plan" |
| `review-fixes` | Review flagged specific issues (FAIL or High severity) | "Fix the 3 FAIL items in the Ch03 review" |
| `naming` | Script names, chapter numbers, or env IDs are inconsistent | "Fix ch02 script references in Ch03" |
| `general` | User-specified targeted changes | "Add a sidebar about MPS support in Ch01" |

**Key constraint:** The Revisor preserves all prose that passed review. It
inserts, fixes, or corrects -- it does not rewrite. If a change requires
restructuring the chapter, return to the Writer agent instead.

**Outputs:**
- Updated `Manning/chapters/chNN_<topic>.md` (in-place)
- Updated `Manning/scaffolds/chNN_scaffold.md` (if scaffold needs additions
  like a Figure Plan)
- New `Manning/revisions/chNN_revision_NNN.md` (change log)

**Re-review:** For `figures` and `naming` scopes, re-review is usually not
needed (changes are additive and mechanical). For `review-fixes` and
`general` scopes, consider re-running the Reviewer on modified sections.

---

## Figure and Video Generation

All figures in the book are self-generated from code -- no external licensing
issues, and readers can regenerate them.

### Generating Figures

The primary figure generation script is `scripts/capture_proposal_figures.py`.
It runs inside Docker and produces annotated PNGs in `figures/`.

```bash
# Generate all figures (env screenshots + diagrams + PPO figures)
bash docker/dev.sh python scripts/capture_proposal_figures.py all

# Generate specific figure types
bash docker/dev.sh python scripts/capture_proposal_figures.py env-setup
bash docker/dev.sh python scripts/capture_proposal_figures.py reward-diagram
bash docker/dev.sh python scripts/capture_proposal_figures.py ppo-clipping
bash docker/dev.sh python scripts/capture_proposal_figures.py ppo-demo-curve

# Side-by-side comparison (requires trained checkpoint)
bash docker/dev.sh python scripts/capture_proposal_figures.py compare \
  --env FetchReachDense-v4 --ckpt checkpoints/ppo_FetchReachDense-v4_seed0.zip
```

**Figure output directory:** `figures/` (at repo root). This directory is
referenced by relative paths in chapter markdown (e.g., `figures/fetch_reach_setup.png`).

**Figures currently generated:**

| File | Source subcommand | Used in |
|------|------------------|---------|
| `fetch_reach_setup.png` | `env-setup` | Ch01, Ch02, Ch03 |
| `fetch_push_setup.png` | `env-setup` | Ch02 |
| `fetch_pick_and_place_setup.png` | `env-setup` | Ch02 |
| `obs_dict_structure.png` | `reward-diagram` | Ch02 |
| `dense_vs_sparse_reward.png` | `reward-diagram` | Ch02 |
| `ppo_clipping_diagram.png` | `ppo-clipping` | Ch03 |
| `ppo_demo_learning_curve.png` | `ppo-demo-curve` | Ch03 |

### Generating Videos

Evaluation videos (MP4/GIF) are produced by `scripts/generate_demo_videos.py`
using trained checkpoints:

```bash
# Generate demo video from a trained checkpoint
bash docker/dev.sh python scripts/generate_demo_videos.py \
  --ckpt checkpoints/ppo_FetchReachDense-v4_seed0.zip

# Quick test (fewer episodes)
bash docker/dev.sh python scripts/generate_demo_videos.py \
  --ckpt checkpoints/ppo_FetchReachDense-v4_seed0.zip --n-episodes 3
```

Videos are saved to `videos/` by default.

### Build Pipeline Integration

The build script (`scripts/build_book.py`) resolves local image paths
relative to the repo root via pandoc's `--resource-path`. Figures in
`figures/` are automatically found during PDF/DOCX generation.

To verify all images resolve before building:

```bash
python scripts/build_book.py --validate-only --verbose
```

The validator reports any missing local images as warnings.

---

### Step 18: Verify Lab Code

- [ ] All snippet-include regions referenced in the chapter exist in `scripts/labs/`
- [ ] `--verify` mode runs and passes
- [ ] `--demo` mode runs and shows learning (or correct behavior for non-training chapters)
- [ ] `--bridge` mode runs and shows agreement with SB3
- [ ] No lab code requires GPU (CPU-only by design)

### Step 19: Verify Run It Commands

- [ ] Fast-path command from Experiment Card runs to completion
- [ ] Expected artifacts are produced at the documented paths
- [ ] Success criteria from the Experiment Card are met
- [ ] Reproduce It command runs to completion (can be overnight)
- [ ] Reproduce It results match the chapter's tables within stated tolerance

### Step 20: Final Read-Through

Read the chapter as a reader would, start to finish. Ask:

- [ ] Could I complete this chapter if I only had Build It + checkpoints (no GPU)?
- [ ] Does the chapter flow naturally, or are there jarring transitions?
- [ ] Is there any section where I'd think "why am I reading this?" (cut it)
- [ ] Is there any point where I'd be confused about what to do next? (add guidance)
- [ ] Does the chapter land? Does the summary make me want to read the next one?

---

## Adaptation Notes by Chapter

### Chapter 1: Proof of Life (Setup)

**Source tutorial:** `tutorials/ch00_containerized_dgx_proof_of_life.md` (445 lines, formal voice)

**Key adaptations:**
- Drop Hadamard well-posedness formalism for reproducibility (2 pages -> 2 sentences)
- Drop formal Definition blocks for Container, Image
- Introduce the 3 Hadamard engineering questions here (practical framing, ~1 paragraph), used as a lightweight checklist in all later chapters
- Build It: env inspection (obs dict structure, shapes, values) + manual `compute_reward`
- Bridge: manual compute_reward == env step reward (the critical invariant)
- Merge Appendix A (troubleshooting) into What Can Go Wrong
- Compress Mac M4 section into a sidebar
- Cut Appendix B (env variable reference) into an inline tip

**Visual requirements:** 1-2 figures. Annotated FetchReach screenshot showing
obs dict components. Optional: smoke test artifact screenshot.

**Estimated length:** ~8,500 words (~20 pages)

### Chapter 2: Environment Anatomy (Environment)

**Source tutorial:** `tutorials/ch01_fetch_env_anatomy.md`

**Key adaptations:**
- This is a concepts chapter, not a training chapter
- Build It: inspect all observation components, visualize goal space, test action effects
- No Reproduce It (no training runs)
- Replace with a "Verify It" summary of expected inspection outputs
- Introduce the concept registry entries for Ch2 (goal-conditioned MDP, dense/sparse reward, etc.)

**Visual requirements:** 4-6 figures. Annotated screenshots of all three Fetch
environments, obs dict structure diagram, dense vs sparse reward comparison,
action space visualization.

### Chapter 3: PPO on Dense Reach (Algorithm)

**Source tutorial:** `tutorials/ch02_ppo_dense_reach.md`

**Key adaptations:**
- Full algorithm chapter: derive PPO, implement from scratch, bridge to SB3
- Build It components: actor-critic network, GAE, PPO clipped loss, value loss, PPO update loop
- Bridge: PPO loss values on same batch, clip_fraction comparison
- Reproduce It: 3-seed, 1M-step runs

**Visual requirements:** 3-4 figures. Actor-critic architecture diagram,
learning curve (success rate over steps), PPO clipping visualization,
random vs trained comparison.

### Chapter 4: SAC on Dense Reach (Algorithm)

**Source tutorial:** `tutorials/ch03_sac_dense_reach.md`

**Key adaptations:**
- Full algorithm chapter: derive SAC from maximum-entropy objective
- Build It components: replay buffer, twin Q-networks, squashed Gaussian policy, soft Bellman backup, actor loss, temperature loss, SAC update loop
- Bridge: Q-targets, actor loss, entropy coefficient on same replay batch
- Reproduce It: 3-seed, 2M-step runs

**Visual requirements:** 3-4 figures. SAC architecture (twin Q + actor +
temperature), entropy coefficient over training, learning curve comparison
(PPO vs SAC), replay buffer diagnostics.

### Chapter 5: HER on Sparse Reach/Push (Algorithm)

**Source tutorial:** `tutorials/ch04_her_sparse_reach_push.md`

**Key adaptations:**
- Full algorithm chapter: derive HER from the sparse reward problem
- Build It components: goal sampling strategies, transition relabeling, reward recomputation, HER buffer insertion
- Bridge: same episode through from-scratch relabeler and SB3 HER
- Failure-first pedagogy: show SAC-without-HER flatline before introducing HER
- Reproduce It: 3-seed runs for both HER and no-HER baselines on Reach and Push

**Visual requirements:** 4-5 figures. Sparse reward flatline (no-HER baseline),
HER learning curve comparison, goal relabeling diagram, Push environment
annotated screenshot, HER ratio visualization.

### Chapter 6: PickAndPlace Capstone (Capstone)

**Source tutorial:** `tutorials/ch05_pick_and_place.md`

**Key adaptations:**
- Capstone chapter: applies everything from Ch1-5 to a harder task
- Build It: curriculum wrapper, stress-test evaluation wrapper
- Bridge: curriculum schedule produces expected goal distributions
- Reproduce It: multi-seed runs with curriculum and stress evaluation
- Dense-first debugging strategy as a structural element

**Visual requirements:** 3-5 figures. PickAndPlace annotated screenshot,
difficulty progression across all three tasks, curriculum schedule plot,
stress-test degradation curve, success rate comparison table as figure.

---

## Quick Reference: Production Checklist

For each chapter, confirm all boxes before marking complete:

```
CHAPTER NN: <title>

Phase 1 -- Scaffolding (Scaffolder agent):
  [ ] Source tutorial read
  [ ] Chapter type classified
  [ ] Experiment Card written
  [ ] Reproduce It block written
  [ ] Build It components listed
  [ ] Bridging proof identified
  [ ] What Can Go Wrong items listed
  [ ] Figure plan created (>= 2 figures, each with source command)
  [ ] Scaffold approved by user

Phase 2 -- Production (Lab Engineer + Book Writer agents):
  [ ] Lab regions created (--verify, --demo, --bridge modes)
  [ ] Opening promise
  [ ] Bridge (or N/A for Ch1)
  [ ] WHY section
  [ ] HOW / Build It (interleaved)
  [ ] Bridge section
  [ ] Run It + Experiment Card
  [ ] What Can Go Wrong
  [ ] Summary + bridge to next chapter
  [ ] Reproduce It block
  [ ] Exercises (3-5)
  [ ] All figures generated, captioned, and referenced

Phase 3 -- Review (Reviewer agent):
  [ ] Reviewer agent produced Manning/reviews/chNN_review.md
  [ ] Verdict: READY / REVISE / RESTRUCTURE
  [ ] Review findings addressed

Phase 3.5 -- Revision (Revisor agent, optional):
  [ ] Revision scope identified (figures / review-fixes / naming / general)
  [ ] Revisor agent produced targeted updates
  [ ] Revision log written to Manning/revisions/chNN_revision_NNN.md
  [ ] Re-review completed (if needed for substantial changes)

Phase 4 -- Verification (manual / scripted):
  [ ] Lab code verified (--verify, --demo, --bridge all pass)
  [ ] Run It fast-path command produces expected artifacts
  [ ] Reproduce It command runs to completion
  [ ] Final read-through completed
```
