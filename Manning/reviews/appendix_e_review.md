# Review: Appendix E -- Isaac Lab Manipulation (GPU-Only)

> **Review #2** (previous verdict: RESTRUCTURE -- scaffolded stub, never written)
> **Reviewed:** 2026-02-28
> **Chapter draft:** `Manning/chapters/appendix_e_isaac_manipulation.md` (~8,269 words)
> **Scaffold:** `Manning/scaffolds/appendix_e_scaffold.md`
> **Source tutorial:** `tutorials/appendix_e_isaac_manipulation.md` (~1,178 lines)
> **Lab code:** `scripts/labs/isaac_sac_minimal.py`, `scripts/labs/isaac_goal_relabeler.py`
> **Previous review:** RESTRUCTURE (chapter was a scaffolded stub with zero prose)

---

## 1. Checklist Audit

### From-scratch contract

- [x] **Build It is the narrative spine** -- PASS. Build It spans sections E.1-E.9 across ~2,800 words. Each component has math preamble, annotated code listing, explanatory prose, and a verification checkpoint. The chapter's HOW section weaves derivation and implementation together rather than dumping code listings.
- [x] **Every equation has corresponding code listing** -- PASS. Four equations appear (log-prob correction, Q-target, actor loss, temperature loss) and all have corresponding code in Listings E.2, E.4. The squashing correction formula maps directly to the `log_prob -=` line; the Bellman target maps to `critic_loss()`; the actor and temperature losses map to `actor_loss()` and `temperature_loss()`.
- [x] **Bridging proof present** -- PASS. The "Bridge: From Scratch to SB3" section provides a 6-row mapping table from Build It components to SB3 internals, references the `--bridge` mode, and explains what the structural bridge confirms. The bridge is described as structural rather than numerical (different initializations), which is honest and appropriate for an appendix that reuses Ch3-4 SAC math.
- [x] **Run It refers back to Build It** -- PASS. The Bridge section explicitly states "the twin-Q loss we implemented in Section N.N is what SB3's CriticNetwork computes internally" and "when you see SB3's `ent_coef` logged during training, you are watching the same automatic temperature tuning you implemented in Listing E.4." The Run It section opens by saying "SB3 implements the same computations at scale."
- [x] **SB3 is never introduced as a black box** -- PASS. The Bridge section maps every from-scratch component to its SB3 counterpart before any SB3 commands appear. The reader knows what `MlpPolicy`, `SAC.actor`, `SAC.critic`, and `SAC.train()` do internally because they built the equivalent code.

### Structure

- [x] **Every term defined before first use** -- PASS. "Staged dense reward," "CurriculumManager," "replay buffer stationarity," "TiledCamera" are all defined or explained before use. Cross-references to earlier chapters are present ("the same clipped double-Q technique from Chapter 3," "the same curriculum pattern from Chapter 5").
- [x] **Every equation has all symbols defined** -- PASS. The log-prob correction defines $d$, $u$, $a$, $\tanh$. The critic loss defines $\bar{\theta}$, $\gamma$, $d$, $\alpha$. The actor loss and temperature loss define $\bar{H}$ and give specific values ($\bar{H} = -8$ for Lift-Cube, $-4$ for Fetch).
- [x] **Every code listing under 30 lines** -- PASS. Listings range from 15-30 lines. Two listings (E.2 and E.5) are exactly 30 lines -- at the limit but within bounds.
- [x] **Every code listing has a verification checkpoint** -- PASS. All 9 listings have blockquoted "Checkpoint:" sections with specific expectations (shapes, value ranges, invariants).
- [x] **Experiment Card present** -- PASS. Complete Experiment Card with algorithm, environment, fast path, full run, commands, checkpoint track, expected artifacts, and success criteria for both state-based and pixel variants.
- [x] **Chapter bridge present** -- PASS. The "Chapter Bridge" section covers all 4 parts: capability established (Ch1-6 workflow on Fetch), gap (does methodology transfer?), what this appendix adds (portability evidence), and scope constraints (GPU-only, dense reward).
- [x] **"What Can Go Wrong" section present** -- PASS. Eight failure modes with specific symptoms, causes, and fixes.

### Voice

- [x] **Numbers, not adjectives for all performance claims** -- PASS. All claims are quantified: "9,377 fps," "+0.54 +/- 0.05 return," "100/100 positive-return episodes," "540x critic loss explosion," "1000x penalty scaling," "24x faster than MuJoCo pixel."
- [x] **ASCII only** -- PASS. No Unicode characters detected. Uses `--` for dashes, straight quotes, `$...$` for math.
- [x] **No third-person author references** -- PASS. No "the reader," "the practitioner," "one might," or "Professor" references.
- [x] **No "simply," "trivially," "obviously"** -- PASS. None detected.
- [x] **Scope boundaries have specific references** -- PASS. Scope boundaries in the summary explicitly name what was NOT covered (sparse-reward Isaac, POMDP contact tasks, multi-GPU scaling) with reasons. The Honest Difficulty Comparison section is a model of scope honesty.

### Compute accessibility

- [x] **Build It `--verify` runs on CPU in under 2 minutes** -- PASS. The "Verification Summary" section states "Both lab files run on CPU without Isaac Sim" and gives the commands. The checkpoint text says "~10-20 seconds" and "~1-5 seconds."
- [ ] **Build It `--demo` runs on CPU in under 30 minutes and shows learning** -- WARN: `isaac_sac_minimal.py` has no `--demo` mode. The goal relabeler has `--demo` but the chapter does not reference it. The chapter's Build It track relies entirely on `--verify` for the from-scratch demonstration. This is a minor gap -- the SAC math was already demonstrated in Ch3-4's demo modes, and an appendix may reasonably defer to those. However, the protocol technically requires it.
- [x] **Chapter completable via Build It + checkpoint track** -- PASS. The chapter explicitly states "you can verify the math even if you do not have a Linux GPU" for Build It, and the Experiment Card includes a checkpoint track path. The Reproduce It block references the companion repository.
- [x] **Reproduce It block present with exact commands and results** -- PASS. Complete Reproduce It block with state-based command, pixel command, evaluation command, hardware note, time estimates, artifact paths, results summary with metrics, and wall-clock comparison table.

### Visual content

- [ ] **Every figure referenced by number in the text** -- FAIL: No figures exist in the chapter. Zero figure references.
- [ ] **Every figure has a caption with generation command** -- FAIL: Zero figures.
- [ ] **Alt-text present for all figures** -- FAIL: Zero figures.
- [ ] **Colorblind-friendly palette used consistently** -- N/A: No figures to evaluate.
- [ ] **All figures are static PNGs** -- N/A.
- [ ] **Figure count meets chapter-type minimum** -- FAIL: The scaffold's Figure Plan specifies 4 figures (state learning curve, wall-clock comparison bar chart, pixel hockey-stick curve, curriculum crash diagnostic). Zero are present. For an appendix with Algorithm + Engineering hybrid classification, 3-4 figures is the minimum.

---

## 2. Voice Compliance

### Section-by-section analysis

| Section | Assessment | Severity | Notes |
|---------|-----------|----------|-------|
| Opening Promise | Good | -- | Five concrete bullets, each actionable and quantified. The "170x speedup" number in the last bullet grounds the promise concretely. |
| Chapter Bridge | Good | -- | Covers all 4 bridge elements. Tone is collaborative ("the natural next question is whether the methodology transfers"). Scope constraints stated honestly ("GPU-only -- no Mac, no CPU fallback"). |
| WHY: The Task | Excellent | -- | Scene-setting passage is vivid and specific ("matte gray links tapering from a heavy shoulder to a slender wrist," "small red cube, roughly 5 cm on a side"). This is exactly what the persona's Section 4.6 asks for. |
| WHY: Staged Dense Reward | Good | -- | Connects clearly to Ch5 curriculum concepts. Weight table is well explained. |
| WHY: Hidden Curriculum | Good, 1 minor flag | Low | The analogy "like training a regression model where half the labels are in meters and the other half are in kilometers" is effective and concrete. The Warning callout uses "incompatible with off-policy replay buffers" which is a factual statement, not doom-saying. |
| WHY: Observation Space | Good | -- | Clean comparison between Gymnasium-Robotics and Isaac Lab conventions. No jargon without explanation. |
| WHY: Why SAC Works | Good | -- | Derivation-from-constraints reasoning is tight and connects to Ch3-4. |
| HOW: Build It E.1-E.5 | Good | -- | Each component has math, code, and checkpoint. Prose explains *why* each design choice matters, not just what the code does. |
| HOW: Build It E.6-E.9 | Good | -- | Honestly notes Lift-Cube does not use HER and explains why the relabeling components are included anyway ("demonstrate how HER would work on a goal-conditioned Isaac Lab task"). |
| Bridge: From Scratch to SB3 | Good | -- | The mapping table is clear. The explanation of what "structural rather than numerical" means is honest. |
| Run It: Experiment Card | Good | -- | Complete and quantified. Includes both fast path and full run. |
| Run It: Training Progression | Excellent | -- | Phase annotations on the training progression table are exactly the kind of interpretive context the persona demands. |
| Run It: Curriculum Crash | Excellent | -- | Failure-first pedagogy done well. Shows the crash data, explains the root cause, gives the fix. No doom-saying -- just factual diagnosis. |
| Run It: Honest Difficulty | Excellent | -- | "What this appendix proves / does not prove" is a model of intellectual honesty. No hand-waving. |
| Run It: Pixels | Good | -- | Hockey-stick curve explained with cross-reference to Ch9. Three key lessons are practical and specific. |
| What Can Go Wrong | Good | -- | Eight failure modes, all with specific symptoms, causes, and fixes. The format is paragraph-based rather than tabular, which reads well but is slightly less scannable than the tutorial's format. |
| Summary | Good | -- | Quantified recap with explicit scope boundaries. Forward-looking without over-promising. |
| Reproduce It | Good | -- | Complete with all required elements. Single-seed scope is honestly acknowledged as "appendix scope." |
| Exercises | Good | -- | 5 exercises spanning warm-up to advanced. Each has a clear expected outcome or prediction. E.3 (reproduce the crash) is pedagogically strong. |

### Tone anti-pattern check (persona Section 4.5)

| # | Anti-pattern | Found? | Details |
|---|-------------|--------|---------|
| 1 | Doom-saying ("unless...no way") | No | -- |
| 2 | Harsh absolutism ("not a result") | No | -- |
| 3 | "No vibes" scolding | No | -- |
| 4 | Deficit statements ("You do not yet know...") | No | -- |
| 5 | Defensive "this is not..." | No | One instance of "This is not the only way" thinking in scope boundaries, but expressed positively as "what this proves / does not prove" |
| 6 | Patronizing qualifiers | No | -- |
| 7 | Predicting reader failure | No | Warning callout about curriculum crash is factual, not prophetic |
| 8 | Repetitive hammering | Borderline | The curriculum crash lesson appears in WHY, Run It, What Can Go Wrong, and the Warning callout -- four times. Each appearance adds context, but the core message (replay buffers assume stationarity) could be stated once with cross-references. See Issue #3 below. |
| 9 | Imperative scolding | No | -- |
| 10 | Over-emphasis with italics | No | Minimal italic usage |

**Voice verdict: Strong.** The chapter reads as a competent colleague explaining a porting experience. The scene-setting passage is one of the best in the manuscript. The "honest difficulty comparison" section demonstrates the kind of intellectual integrity the persona demands. The only voice concern is mild repetition of the curriculum crash lesson across four sections.

---

## 3. Technical Accuracy

### Equations and math

| Location | Issue | Impact |
|----------|-------|--------|
| E.2 log-prob correction | Correct. Formula matches the lab code exactly: `log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)` | None |
| E.4 critic loss | Correct. Matches lab code. The `F.mse_loss(q1, target) + F.mse_loss(q2, target)` matches the equation's $\sum_{i=1}^{2}$ | None |
| E.4 actor loss | Correct. `(alpha * logp - q).mean()` matches $\mathbb{E}[\alpha \log\pi - \min Q]$ | None |
| E.4 temperature loss | Correct. `-(alpha * (logp.detach() + target_entropy)).mean()` matches $-\alpha(\log\pi + \bar{H})$ | None |
| E.3 Q-target equation | Minor: the equation $Q_{\text{target}} = \min(Q_1(s', a'), Q_2(s', a'))$ is incomplete -- it shows the min operation but not the full soft Bellman target. The full target appears in the critic loss equation (E.4), so no information is lost, but this partial equation could confuse readers who expect the complete expression here. | Low |

### Code listings vs lab source

| Listing | Lab region | Match? | Notes |
|---------|-----------|--------|-------|
| E.1 (DictFlattenEncoder) | `dict_flatten_encoder` | Matches with minor formatting differences | Chapter version adds a comment header and slightly compressed docstring. Core logic identical. |
| E.2 (SquashedGaussianActor) | `squashed_gaussian_actor` | Matches with formatting differences | Chapter compresses `forward()` slightly. Core logic identical. |
| E.3 (TwinQCritic) | `twin_q_critic` | Matches | Core logic identical. |
| E.4 (SAC losses) | `sac_losses` | Matches with compression | Chapter compresses type annotations. Core logic identical. |
| E.5 (SAC update step) | `sac_update_step` | Formatting difference | Chapter uses semicolons (`critic_opt.zero_grad(); c_loss.backward(); critic_opt.step()`) while lab uses separate lines. This is a deliberate editorial choice for space (30-line limit), but semicolons reduce readability. Functionally identical. |
| E.6 (GoalTransition) | `goal_transition_structs` | Matches | |
| E.7 (goal sampling) | `isaac_goal_sampling` | Matches | |
| E.8 (relabel_transition) | `isaac_relabel_transition` | Matches | |
| E.9 (episode processing) | `isaac_her_episode_processing` | Matches | |

### Numerical claims

| Claim | Source verification | Status |
|-------|-------------------|--------|
| "9,377 fps at 256 envs" | Tutorial reports 9,377 fps | Consistent |
| "+0.54 +/- 0.05 return" | Tutorial reports +0.54 +/- 0.05 | Consistent |
| "100/100 positive-return episodes" | Tutorial reports 100/100 | Consistent |
| "1,181 fps pixel training" | Tutorial reports ~1,181 fps | Consistent |
| "540x critic loss explosion" | Tutorial reports "0.05 -> 26.2 (a 540x increase)" | Consistent (26.2/0.05 = 524x, rounded to 540x; close enough) |
| "170x speedup vs MuJoCo pixel" | Derived: 40 hours / 14 min = ~171x | Consistent |
| "24x faster than MuJoCo pixel" | Derived: 1,181 fps / ~40 fps (midpoint of 30-50) = ~30x for fps; wall-time comparison: 56 min for 4M vs ~40 hours for 8M = different step counts. The 24x claim needs more care. | WARN -- see below |
| "15x faster than MuJoCo state" | 9,377 / 550 (midpoint of 500-600) = ~17x | Approximately correct |
| Reward progression table (state-based) | Matches tutorial exactly (64K:-27.2, 1M:-24.8, ..., 8M:-1.4) | Consistent |
| Reward progression table (pixel) | Matches tutorial (64K:-27.2, 800K:-26.5, 1M:-22.0, ..., 4M:-1.07) | Consistent |

**Note on the "24x faster" claim:** The chapter says TiledCamera is "24x faster than MuJoCo pixel training." The throughput comparison table shows 1,181 fps vs 30-50 fps (Ch9 MuJoCo pixel), giving a ratio of ~24-39x in fps. However, the wall-clock comparison is at different step counts (4M Isaac pixel vs 8M MuJoCo pixel). This is acknowledged implicitly in the comparison table but could be stated more precisely. Not a blocking issue -- the fps comparison is fair.

### Commands

All commands in the Experiment Card, Run It, and Reproduce It sections reference `scripts/appendix_e_isaac_manipulation.py` with subcommands (`train`, `eval`, `smoke`, etc.) and appropriate flags (`--headless`, `--seed`, `--num-envs`, `--total-steps`, `--pixel`, `--checkpoint-freq`). The commands are consistent with the tutorial and scaffold.

---

## 4. Visual Content

**Figure count: 0.** The scaffold's Figure Plan specifies 4 figures. None are present.

| Scaffold figure | Present? | Severity |
|----------------|----------|----------|
| Figure E.1: State-based learning curve (8M steps) with phase annotations | No | High -- this is the primary result visualization |
| Figure E.2: Wall-clock comparison bar chart | No | High -- the speedup is the core value proposition and demands visual emphasis |
| Figure E.3: Pixel learning curve (hockey-stick) | No | Medium -- supplementary but reinforces the Ch9 connection |
| Figure E.4: Curriculum crash diagnostic (clean vs crashed run overlay) | No | Medium -- powerful pedagogically but the text description is already vivid |

**Impact:** The chapter relies entirely on prose and tables to convey training dynamics, speedup comparisons, and the curriculum crash. For a Manning "In Action" chapter about physical manipulation and dramatic performance differences, this is a significant gap. The training progression tables are effective substitutes but cannot replace a visual learning curve for showing phase transitions and hockey-stick shapes.

**Recommendation:** At minimum, Figures E.1 (state learning curve) and E.2 (wall-clock comparison) should be added. These are the two visualizations that would most help a reader quickly grasp the chapter's central claims. Figures E.3 and E.4 are desirable but not blocking.

---

## 5. Structural Completeness

| Section | Present? | Approx. length | Notes |
|---------|----------|----------------|-------|
| Opening Promise | Yes | 5 bullets, ~120 words | Complete. All bullets quantified and actionable. |
| Chapter Bridge | Yes | ~250 words | Complete. Covers all 4 bridge elements (capability, gap, addition, scope). |
| WHY: The Task | Yes | ~350 words | Strong scene-setting. Multi-phase breakdown clear. |
| WHY: Staged Dense Reward | Yes | ~250 words | Reward table well explained. Ch5 connection made. |
| WHY: Hidden Curriculum | Yes | ~400 words | The centerpiece diagnostic lesson. Warning callout is well-placed. |
| WHY: Observation + Action Space | Yes | ~500 words | Clean comparison of conventions. Engineering detail on infinite bounds is valuable. |
| WHY: Why SAC Works | Yes | ~200 words | Tight derivation-from-constraints. |
| HOW / Build It (E.1-E.9) | Yes | ~2,800 words, 9 components | All components from scaffold present. Each has math, code, checkpoint. |
| Verification Summary | Yes | ~100 words | CPU-only commands for both lab files. |
| Bridge (from-scratch to SB3) | Yes | ~400 words | 6-row mapping table plus `--bridge` command. |
| Run It: Experiment Card | Yes | ~350 words | Complete with fast path, full run, pixel variant. |
| Run It: State-Based Training | Yes | ~400 words | Phase progression table with interpretation. |
| Run It: Curriculum Crash | Yes | ~350 words | Failure-first pedagogy. Data, diagnosis, fix. |
| Run It: Honest Difficulty | Yes | ~350 words | Comparison table + proves/does-not-prove framing. |
| Run It: Pixels on Isaac | Yes | ~400 words | TiledCamera, throughput table, hockey-stick curve, three lessons. |
| Run It: Wall-Clock Comparison | Yes | ~200 words | 5-row comparison table with interpretation. |
| What Can Go Wrong | Yes | ~700 words, 8 items | All 8 failure modes from scaffold present. |
| Summary | Yes | ~250 words | Quantified recap with explicit scope boundaries. |
| Reproduce It | Yes | ~400 words | Complete block with both state and pixel commands. |
| Exercises | Yes | 5 exercises, ~400 words | Graduated difficulty (warm-up to advanced). |

**Total estimated prose: ~8,200 words** (consistent with the 8,269-word count). This is within the scaffold's ~8,000-word estimate and the Manning target of 6,000-10,000 words.

**Structural verdict: Complete.** Every section specified in the scaffold is present at approximately the estimated length. The chapter follows the WHY-HOW-BUILD IT-BRIDGE-RUN IT-WHAT CAN GO WRONG-SUMMARY-REPRODUCE IT-EXERCISES arc faithfully.

---

## 6. Content Gap Analysis

### Lost from tutorial

| Tutorial content | In Manning chapter? | Assessment |
|-----------------|---------------------|------------|
| Prerequisites and Setup (~500 words): hardware requirements, software requirements, Isaac image build, `dev-isaac.sh` vs `dev.sh` table | Partially. The chapter mentions `docker/dev-isaac.sh` in commands but does not explain how it differs from `dev.sh`, nor does it cover build instructions or first-launch expectations. | **Minor gap.** For an appendix, the reader is expected to have Docker working from Ch1. The `dev-isaac.sh` differences are operational, and the "What Can Go Wrong" section covers the key pitfalls (shader compilation, headless flag). A sidebar on "Getting Isaac Lab Running" would help but is not blocking. |
| Verification Protocol (4-test dependency chain, ~800 words) | No. The tutorial's structured 4-test protocol (Container Boots -> Tasks Register -> Env Steps -> Training Completes) is replaced by a brief "Verification Summary" with the two Build It `--verify` commands. | **Acceptable cut.** The verification protocol is valuable for a standalone tutorial but would slow down a Manning appendix. The Experiment Card's `smoke` command and "What Can Go Wrong" section cover the same ground more efficiently. |
| `dev-isaac.sh` vs `dev.sh` comparison table | No. | **Acceptable cut.** Operational detail better suited to a sidebar or companion repo README. |
| Docker volume caching explanation | Mentioned in "What Can Go Wrong" (shader compilation item). | **Adequate coverage.** |
| Video progression section (~300 words) | No. The tutorial describes recording videos at 200K, 3M, and 8M checkpoints. The Manning chapter mentions videos only in the Reproduce It artifacts list. | **Minor gap.** Video commands could be mentioned in a 1-2 sentence "To see the trained policy in action" note, per persona Section 3.8 static-vs-dynamic guidance. |
| Extended `Sb3VecEnvWrapper` rationale | Covered in the WHY section's observation space discussion and the "What Can Go Wrong" infinite action bounds entry. | **Adequate coverage.** |

### Missing from scaffold

| Scaffold item | Status |
|---------------|--------|
| Build It components 1-9 | All present with prose, math, checkpoints |
| Experiment Card | Present and complete |
| Reproduce It block | Present and complete |
| What Can Go Wrong (8 items) | All 8 present |
| Figure Plan (4 figures) | **MISSING** -- zero figures |
| Sidebar: "Curriculum + Off-Policy = Danger" | Not a separate sidebar, but the content is present in the Warning callout and the "Hidden Curriculum" section. Functionally equivalent. |
| Sidebar: "When NOT to Use HER" | Addressed inline in E.6 ("While Lift-Cube itself does not use HER..."). Not a separate sidebar. |
| Concept Registry Additions | The chapter introduces the relevant terms. The CLAUDE.md registry has not been updated, but that is a project-level task, not a chapter-level one. |

### Unplanned additions

| Content | Assessment |
|---------|------------|
| The `--pixel` flag's TiledCamera setup description is more detailed in the chapter than in the scaffold | Good addition -- explains `clip=(0, 255)`, `concatenate_terms=False`, and the sensor separation principle. |
| SAC update step uses semicolons for compact formatting | Editorial choice for 30-line limit; functionally correct but slightly less readable than expanded form. |

---

## 7. Summary Verdict

**REVISE**

The chapter has transformed from a scaffolded stub (previous verdict: RESTRUCTURE) into a substantially complete, well-written Manning chapter. The narrative spine flows through Build It, the bridge connects from-scratch to SB3, the Run It section includes a complete Experiment Card and quantified results, and the voice is consistently that of an experienced colleague. The "Honest Difficulty Comparison" and curriculum crash narrative are highlights that exemplify the book's commitment to intellectual honesty.

The single blocking issue is the absence of figures. Everything else is at or near production quality.

### Top 5 Issues (Priority Order)

1. **No figures (blocking).** The scaffold specifies 4 figures and the chapter has 0. At minimum, add the state-based learning curve (with phase annotations) and the wall-clock comparison bar chart. These are the two visualizations most critical for a reader to quickly grasp the chapter's central claims. The pixel hockey-stick curve and curriculum crash overlay are desirable additions. Every figure needs a numbered caption (Figure E.N format) with a generation command, alt-text, and Wong (2011) colorblind palette.

2. **No `--demo` mode for SAC lab (minor).** The protocol expects Build It to include a `--demo` mode showing real learning. `isaac_sac_minimal.py` has `--verify` but no `--demo`. This is partially defensible for an appendix that reuses Ch3-4 SAC math, but adding a short synthetic training demo (or explicitly noting that Ch3-4's `--demo` serves this purpose) would close the gap.

3. **Curriculum crash lesson appears four times.** The replay-buffer stationarity violation is discussed in the WHY "Hidden Curriculum" section, the Warning callout, the Run It "Curriculum Crash" section, and the "What Can Go Wrong" section. Each adds context, but the core principle could be stated definitively once (in WHY) with brief cross-references from the other locations. This borders on the persona's anti-pattern #8 (repetitive hammering).

4. **Setup/prerequisites gap.** The chapter assumes the reader can build and run the Isaac Docker image without guidance. A 3-5 sentence paragraph (or a sidebar titled "Getting Isaac Lab Running") pointing to `docker/build.sh isaac` and `docker/dev-isaac.sh`, noting the ~15 GB NGC image size, and warning about the 30-90 second first-launch shader compilation would prevent a common early frustration point. The "What Can Go Wrong" section partially covers this, but a proactive note earlier would help.

5. **Listing E.5 semicolons reduce readability.** The `sac_update_step` listing uses semicolons to fit within 30 lines (`critic_opt.zero_grad(); c_loss.backward(); critic_opt.step()`). The lab source uses separate lines. For a teaching context, the expanded form is clearer -- the semicolons obscure the three-step pattern. Consider either splitting the listing into two (with narrative between) or allowing it to run to 33-35 lines with a note explaining the slight overshoot.

### Recommended Next Steps

1. **Add figures** via the Revisor agent with `scope: figures`. Use the scaffold's Figure Plan as the specification. Generate PNGs from the training data already documented in the Reproduce It block.
2. **Trim curriculum crash repetition** -- keep the full explanation in WHY and the diagnostic in "What Can Go Wrong," reduce the Run It section to a brief forward reference, and keep the Warning callout as-is (it serves a different structural purpose as a scannable callout).
3. **Add a "Getting Isaac Lab Running" sidebar or paragraph** early in the chapter, before the Build It section.
4. **Consider expanding or splitting Listing E.5** for readability.
5. After figures are added, re-run the Reviewer for a final READY check.
