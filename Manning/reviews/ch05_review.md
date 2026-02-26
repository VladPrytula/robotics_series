# Review: Chapter 5 -- HER on Sparse Reach and Push: Learning from Failure

**Reviewer:** Manning Reviewer Agent
**Date:** 2026-02-26
**Chapter:** `Manning/chapters/ch05_her_sparse_reach.md`
**Word count:** 9,098 (target: 6,000-10,000)

---

## 1. Checklist Audit

### From-scratch contract

- [x] **Build It is the narrative spine** -- PASS. Build It spans Sections 5.3-5.6 (~2,700 words), forming the core teaching arc. The HOW section (5.2) derives concepts, then Build It implements them component by component.
- [x] **Every equation has corresponding code listing** -- PASS. The sparse reward equation (Section 5.1, Section 5.5) maps to `sparse_reward` in Listing 5.5. The relabeling reward equation (Section 5.5) maps to `relabel_transition`. The data amplification arithmetic (Section 5.2) maps to `process_episode_with_her`.
- [x] **Bridging proof present** -- PASS. Section 5.7 runs `--compare-sb3`, reports `max_abs_reward_diff = 0.000e+00`, and maps SB3 parameters to from-scratch concepts.
- [x] **Run It refers back to Build It** -- PASS. Section 5.8 explicitly says "the relabeled successes from Section 5.6 -- the same `process_episode_with_her` math that SB3's `HerReplayBuffer` computes" and "the twin-Q loss from Chapter 4, Section 4.7."
- [x] **SB3 is never introduced as a black box** -- PASS. By the time `HerReplayBuffer` appears (Section 5.7), the reader has built goal sampling, relabeling, and episode processing from scratch. The bridge section maps every SB3 parameter to a concept the reader implemented.

### Structure

- [x] **Every term defined before first use** -- PASS. HER, goal relabeling, goal sampling strategies, n_sampled_goal, data amplification, off-policy requirement, effective horizon, HER ratio -- all defined before use. Previous-chapter concepts (off-policy, replay buffer, SAC, alpha, gamma) are referenced with chapter pointers.
- [x] **Every equation has all symbols defined** -- PASS. $R_{\text{dense}}$ has all symbols. $R_{\text{sparse}}$ defines $\epsilon$. The relabeling reward equation defines $g_a^t$ and $g'$. The data amplification arithmetic defines $T$, $k$, and `her_ratio`.
- [x] **Every code listing under 30 lines** -- PASS. Longest listing is `process_episode_with_her` at ~22 lines. The `sample_her_goals` function is split across two listings with narrative between them.
- [x] **Every code listing has a verification checkpoint** -- PASS. Sections 5.3, 5.4, 5.5, and 5.6 each end with a Checkpoint block containing concrete code and expected values.
- [x] **Experiment Card present** -- PASS. Section 5.8 opens with a properly formatted Experiment Card for SAC+HER on FetchPush-v4 with fast path, checkpoint track, success criteria, and artifact paths.
- [x] **Chapter bridge present** -- PASS. The second paragraph (lines 11-16) covers all four bridge elements: capability (SAC 100% on dense Reach), gap (dense rewards required hand-designed signal), addition (HER + sparse rewards), and foreshadow (Ch6 PickAndPlace).
- [x] **"What Can Go Wrong" section present** -- PASS. Section 5.9 covers 8 failure modes with symptom, cause, and diagnostic for each.

### Voice

- [x] **Numbers, not adjectives for all performance claims** -- PASS. Every result is quantified: "5% +/- 0%", "99% +/- 1%", "184.5mm", "25.7mm", etc.
- [x] **ASCII only** -- PASS. No em-dashes, no curly quotes, no Unicode symbols detected. Uses `--` consistently.
- [x] **No third-person author references** -- PASS. No "Professor Prytula", no "the author." Uses "we" and "you" throughout.
- [x] **No "simply," "trivially," "obviously"** -- PASS. No banned phrases detected.
- [x] **Scope boundaries have specific references** -- PASS. HER paper cited as "Andrychowicz et al. (2017, Section 3.3)." Entropy auto-tuning failure points back to "Chapter 4, Section 4.8." No bare "beyond the scope" occurrences.

### Compute accessibility

- [x] **Build It `--verify` runs on CPU in under 2 minutes** -- PASS (per pipeline context: all modes pass, verified).
- [x] **Build It `--demo` runs on CPU in under 30 minutes** -- PASS (per pipeline context: `--demo` runs in seconds on synthetic data).
- [x] **Chapter completable via Build It + checkpoint track** -- PASS. Section 5.8 explicitly provides the checkpoint track command: `bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py eval --ckpt checkpoints/sac_her_FetchPush-v4_seed0.zip`.
- [x] **Reproduce It block present** -- PASS. End of chapter contains properly formatted Reproduce It block with exact commands, hardware, time estimates, artifact paths, and results summary matching the chapter tables.
- [x] **Reproduce It results match the chapter's tables** -- PASS. Reproduce It: Push HER 0.99 +/- 0.01, chapter table: 99% +/- 1%. Reach HER 1.00 +/- 0.00, chapter table: 100% +/- 0%. All cross-checks match.
- [x] **Checkpoint track sources are the Reproduce It outputs** -- PASS. The connection is explicit in the Reproduce It block.

### Visual content

- [x] **Figure count meets chapter-type minimum** -- PASS. 5 figures present (Algorithm chapter minimum: 3-5).
- [x] **Every figure from the scaffold's figure plan is present** -- PASS. Scaffold planned 5 figures; all 5 are present (Figures 5.1-5.5).
- [ ] **Every figure has a numbered caption** -- WARN. See Visual Content section below for detail on Figure 5.5 numbering placement.
- [x] **Every caption includes the generation command** -- PASS. All 5 captions include generation commands or source descriptions.
- [ ] **Every figure is referenced by number in the text** -- FAIL. Figure 5.5 (FetchPush-v4 environment screenshot) is introduced in the text (Section 5.8: "Figure 5.5 shows the environment layout") AFTER Figure 5.4 has already appeared in the text. Figures 5.4 and 5.5 are out of numerical sequence in the document flow -- Figure 5.5 appears on line 614 but Figure 5.4 appears later on line 703. This means a reader encounters Figure 5.5's text reference before seeing Figure 5.4.
- [x] **Colorblind-friendly palette referenced** -- PASS. Figure 5.2 caption uses "red X" and "blue dots" which map to the Wong (2011) Vermillion and Blue. Learning curve captions reference shaded regions. However, actual palette usage cannot be verified from markdown alone.
- [x] **All figures are static PNGs** -- PASS. All figure references use `.png` format.
- [x] **Alt-text present for all figures** -- PASS. All 5 image tags include descriptive alt-text in the Markdown image syntax.

---

## 2. Voice Compliance

| Section | Issue | Severity |
|---------|-------|----------|
| Section 5.1, para 2 (line 26) | "Here are the results from training SAC without HER" -- acceptable In Action show-then-tell. No issue. | -- |
| Section 5.2, Definition block (lines 106-121) | Uses the formal 5-step definition template (motivating problem, intuitive description, formal definition, grounding example, non-example). This is well executed and appropriate for a key concept. No issue. | -- |
| Section 5.2, "Why HER Requires Off-Policy Learning" (lines 159-173) | Uses formal "Definition (Off-policy requirement for HER)" block. Three numbered conditions are clear and well-motivated. Connects explicitly to Ch4's SAC. Good. | -- |
| Section 5.8, para 1 (line 577) | Script naming note explaining ch04 vs ch05 numbering mismatch. This is practical and helpful -- the reader needs this. | -- |
| Section 5.1, line 83 | "Random exploration at each step has perhaps a 10-20% chance of being correct, so the probability of 20 consecutive correct steps is $(0.15)^{20} \approx 10^{-16}$." The 0.15 value is not clearly explained -- the text says 10-20% but uses 15%. This is a minor clarity issue, not a voice issue. | Low |
| Section 5.8, "Two key hyperparameters" (lines 620-632) | Excellent treatment. The explanation of why `ent_coef=auto` fails on sparse rewards and why `gamma=0.95` matches the task timescale is clear, quantitative, and honest about the sweep. | -- |
| General | The voice throughout is consistent with the Manning persona: conversational, peer-to-peer, honest about difficulty. "We" for reasoning, "you" for instructions. No academic stiffness detected. | -- |

**Overall voice assessment:** Strong compliance. The chapter reads as an experienced colleague explaining HER to a competent peer. No High or Medium severity voice issues found.

---

## 3. Technical Accuracy

### Code listing verification against lab source

| Listing | Chapter code | Lab source | Match? |
|---------|-------------|------------|--------|
| Data structures (Section 5.3, lines 189-213) | `Transition`, `Episode`, `GoalStrategy` | `her_relabeler.py:data_structures` (lines 68-98) | **WARN**: Chapter omits `GoalStrategy.RANDOM = "random"` and the `# =============================================================================` / `# Goal Sampling Strategies` section divider that appear in the source. The chapter also omits the docstring-level comments present in the source. This is an intentional pedagogical simplification (RANDOM strategy is not discussed in the chapter) and is acceptable, but should be noted. |
| Goal sampling (Section 5.4, lines 246-289) | `sample_her_goals` | `her_relabeler.py:goal_sampling` (lines 101-169) | **PASS**: The chapter strips the docstring and inline comments but preserves the logic accurately. The split into two listings (FUTURE/FINAL in one, EPISODE in the next) is a good pedagogical choice. The `else: raise ValueError` clause from the source is omitted in the chapter -- acceptable since only the three discussed strategies appear. |
| Relabel transition (Section 5.5, lines 331-368) | `relabel_transition` + `sparse_reward` | `her_relabeler.py:relabel_transition` (lines 176-237) | **PASS**: Code matches the source accurately. Docstrings are stripped. All logic preserved. |
| Episode processing (Section 5.6, lines 405-442) | `process_episode_with_her` + `compute_success_fraction` | `her_relabeler.py:her_buffer_insert` (lines 244-304) | **PASS**: Code matches. Docstrings stripped. Logic identical. |

### Numerical claims verification

| Claim | Chapter value | Verified value (from pipeline context) | Match? |
|-------|--------------|---------------------------------------|--------|
| Push HER success rate | 99% +/- 1% | 99% +/- 1% | PASS |
| Push no-HER success rate | 5% +/- 0% | 5% +/- 0% | PASS |
| Push HER mean return | -13.20 +/- 1.48 | -13.20 +/- 1.48 | PASS |
| Push no-HER mean return | -47.50 +/- 0.00 | -47.50 +/- 0.00 | PASS |
| Push HER final distance | 25.7mm +/- 1mm | 25.7mm | PASS |
| Push no-HER final distance | 184.5mm +/- 0mm | 184.5mm | PASS |
| Reach HER success rate | 100% +/- 0% | 100% | PASS |
| Reach no-HER success rate | 96% +/- 7% | 96% +/- 7% | PASS |
| Reach HER mean return | -1.68 +/- 0.02 | -1.68 | PASS |
| --compare-sb3 max_abs_reward_diff | 0.000e+00 | 0.000e+00 | PASS |
| --compare-sb3 success_fraction | 0.799 | 0.799 | PASS |
| --verify transitions | 50 -> ~210 | 50 -> 218 | PASS (approximate) |
| --verify success rate | ~0% -> ~16% | 0% -> 4.1% | **WARN** |

**WARN on --verify success rate:** The chapter states "HER success rate: ~16%" in the expected output block (Section 5.6, line 498), but the actual lab output shows 4.1% success rate. The grounding example (Section 5.2, line 119) also says "~16-80%". The discrepancy arises because the synthetic episode creates a trajectory that gradually approaches the goal (line 329 of lab source: `achieved = achieved + 0.02 * (desired_goal - achieved) + 0.01 * np.random.randn(goal_dim)`), but achieved goals in such a trajectory are spread over a large distance, so few relabeled transitions land within the 0.05m threshold. The 16% figure may come from a different random seed or a tighter trajectory. The `--demo` output shows strategy-specific numbers: FUTURE 8.3%, FINAL 3.9%, EPISODE 5.4% -- none near 16%.

This is a **technical accuracy issue** that needs correction. The expected output block should match actual lab output.

### Math verification

| Equation | Check | Status |
|----------|-------|--------|
| $R_{\text{dense}}$ (line 61) | Standard Fetch dense reward. Correct. | PASS |
| $R_{\text{sparse}}$ (line 67) | Binary reward with threshold $\epsilon = 0.05$. Correct. | PASS |
| Relabeling reward (line 321-323) | $r'_t = R(g_a^t, g')$ -- correct formulation. | PASS |
| Data amplification (lines 147-151) | $50 + 50 \times 0.8 \times 4 = 210$. Correct arithmetic. | PASS |
| Effective horizon (line 628) | $T_{\text{eff}} = 1/(1-\gamma)$: at $\gamma=0.95$ gives 20, at $\gamma=0.99$ gives 100. Correct. | PASS |
| Cumulative entropy bonus (line 630) | $\alpha \times T_{\text{eff}} \times \bar{\mathcal{H}}$ -- correct qualitative scaling analysis. | PASS |
| Probability estimate (line 83) | $(0.15)^{20} \approx 10^{-16}$ -- actually $(0.15)^{20} \approx 3.3 \times 10^{-17}$, which rounds to $\sim 10^{-16.5}$. Close enough for an order-of-magnitude estimate. | PASS (approximate) |

### Specific technical issues

| Location | Issue | Impact |
|----------|-------|--------|
| Section 5.3, line 209 | Chapter listing shows `GoalStrategy` with only FINAL, FUTURE, EPISODE. Lab source also has `RANDOM = "random"`. Chapter correctly omits RANDOM since it is not discussed, but the comment `# (from scripts/labs/her_relabeler.py:data_structures)` suggests this is an exact copy. | Low -- pedagogical simplification is acceptable, but the source attribution could note the omission. |
| Section 5.6, line 498 | Expected `--verify` output shows "HER success rate: ~16%" but actual output is ~4.1%. | Medium -- readers running the code will see different numbers than the book shows. |
| Section 5.6, line 499 | Expected output shows "Transitions: 50 -> ~210" -- actual lab shows 218. The chapter says "~210" which is a reasonable approximation given stochastic her_ratio=0.8. | Low -- the "~" appropriately signals approximation. |
| Section 5.7, line 541 | Bridge expected output shows "Success fraction: 0.799" -- matches actual output. | PASS |
| Section 5.2, line 119 | Grounding example says "success fraction jumps from ~0% to ~16-80%, depending on how closely the trajectory's achieved goals cluster." The lower bound of ~16% does not match actual lab output (~4-8%). | Medium -- the 16% lower bound should be revised to match actual observations. |
| Section 5.8, line 628 | "At $\gamma=0.99$, that is 100 steps" -- correct but the chapter uses 0.99 for the counter-example while Ch4 used 0.99 as default. The chapter correctly explains why 0.95 is better for Push. | Low -- no issue, just noting. |

---

## 4. Visual Content

| Figure | Present? | Numbered caption? | Generation command? | Referenced by number? | Notes |
|--------|----------|-------------------|--------------------|-----------------------|-------|
| Figure 5.1 | Yes (line 40) | Yes | Yes (TensorBoard extraction) | Yes (implied by placement after Section 5.1 heading) | No explicit text reference like "as Figure 5.1 shows." The figure appears after the results table and before the "Why does standard RL fail here?" subsection. The placement is logical but a forward reference in the text would be cleaner. |
| Figure 5.2 | Yes (line 175) | Yes | Yes ("Illustrative diagram") | Yes (implied by placement) | Same issue -- no explicit "as Figure 5.2 shows" text reference before the figure. The figure appears at the end of Section 5.2. |
| Figure 5.3 | Yes (line 517) | Yes | Yes (`--demo` command) | No explicit text reference | The figure appears at the end of Section 5.6 but is not referenced by number in the preceding text. |
| Figure 5.4 | Yes (line 703) | Yes | Yes (TensorBoard extraction) | Yes (line 701: "Figure 5.4 shows the learning curves") | Good -- explicit forward reference. |
| Figure 5.5 | Yes (line 614) | Yes | Yes (`capture_proposal_figures.py env-setup`) | Yes (line 612: "Figure 5.5 shows the environment layout") | **FAIL: Figure ordering.** Figure 5.5 appears in the document BEFORE Figure 5.4. The text on line 612 says "Figure 5.5 shows the environment layout" and the figure is on line 614. Figure 5.4 does not appear until line 703. This means figures are out of sequential order in the chapter flow. |

### Figure-specific issues

| Figure | Issue | Severity |
|--------|-------|----------|
| Figure 5.1 | No explicit forward reference in text body (e.g., "as Figure 5.1 shows"). The figure appears after the no-HER results table but before the "Why does standard RL fail here?" analysis. Adding a reference like "Figure 5.1 makes this failure visible" would strengthen the connection. | Low |
| Figure 5.2 | No explicit text reference. Same low-severity issue as Figure 5.1. | Low |
| Figure 5.3 | No explicit text reference. The figure appears after the demo section but is not called out by number in the preceding paragraph. | Low |
| Figure 5.4 | Properly referenced. Good. | -- |
| Figure 5.5 | **Out of order.** Figure 5.5 (line 614) appears before Figure 5.4 (line 703). This breaks the sequential numbering convention. Either renumber so that the Push environment screenshot is Figure 5.4 and the learning curves are Figure 5.5, or move the environment screenshot earlier in the chapter. | **High** |

---

## 5. Structural Completeness

| Section | Present? | Length | Notes |
|---------|----------|-------|-------|
| Opening promise | Yes | 5 bullets | Well-written, specific, quantified. Matches scaffold. |
| Bridge | Yes | ~6 sentences (lines 11-17) | Covers all 4 bridge elements. Connects Ch4 SAC to Ch5 HER clearly. Foreshadows Ch6 PickAndPlace. |
| 5.1 WHY | Yes | ~1,800 words | Failure-first pedagogy works well. Shows SAC no-HER flatline, explains why, contrasts Reach vs Push. |
| 5.2 HOW | Yes | ~1,600 words | Formal definition (5-step template), goal strategies table, k parameter, data amplification, off-policy requirement. |
| 5.3-5.6 Build It | Yes | ~2,700 words, 4 components | Data structures, goal sampling, relabeling, episode processing. Each has code + checkpoint. |
| 5.7 Bridge section | Yes | ~700 words | `--compare-sb3` output, SB3 parameter mapping table, insertion-time vs sample-time explanation, SB3 additions list. |
| 5.8 Run It | Yes | ~2,000 words | Experiment Card, Push env description, hyperparameter explanation, commands, results tables for Reach and Push, training milestones, TensorBoard mapping. |
| 5.9 What Can Go Wrong | Yes | ~800 words, 8 items | Good coverage. Each item has symptom, cause, diagnostic. |
| 5.10 Summary | Yes | ~700 words | Summarizes all accomplishments, lists key numbers, bridges to Ch6. |
| Reproduce It | Yes | ~400 words | Properly formatted block with exact commands, hardware, artifacts, results. |
| Exercises | Yes | 5 exercises | Graduated: Verify, Tweak x2, Extend, Challenge. Each has commands and expected outcomes. |
| Figures | Yes | 5 figures | All 5 from scaffold present (see Visual Content for ordering issue). |

---

## 6. Content Gap Analysis

### Lost from tutorial

| Content | In tutorial? | In chapter? | Assessment |
|---------|-------------|-------------|------------|
| 120-run hyperparameter sweep full table | Yes (~2,000 words, full ranked table, factor-level marginal analysis, gamma x entropy interaction table, seed sensitivity analysis) | Summarized in ~200 words (two paragraphs in Section 5.8) | **Acceptable cut.** The scaffold explicitly marked the sweep detail as "cut from tutorial" and recommended presenting only the winning configuration with a brief mention of the sweep. The chapter mentions "120-run hyperparameter sweep" and its key finding (gamma dominant, +11pp) without the full table. This is the right choice for the book -- the detail lives in the tutorial for readers who want it. |
| Debugging diary (Attempts 1-4) | Yes (~1,500 words, detailed progression from 5% to 98%) | Cut entirely | **Acceptable cut.** The scaffold marked "entropy debugging diary" for removal. The chapter integrates the lessons (fixed ent_coef, correct gamma) into the hyperparameter explanation in Section 5.8 without the narrative progression. |
| Entropy coefficient modes (auto, auto-floor, schedule, adaptive) | Yes (~400 words, CLI reference table) | Cut | **Acceptable cut.** Scaffold explicitly cut this. Only the fixed ent_coef=0.05 recommendation is presented. |
| n_sampled_goal ablation | Yes (~400 words) | Moved to Exercise 2 | **Good adaptation.** Scaffold recommended this. |
| Animated GIFs (best/worst/no-HER policies) | Yes (3 GIFs) | Replaced with static figures + text descriptions | **Good adaptation.** Manning print constraint. |
| References section | Yes (standalone section) | Inline citations throughout | **Good adaptation.** |
| "The Reality of RL Research" section (~2,000 words) | Yes | Cut | **Acceptable.** This was tutorial-specific pedagogical content about the debugging journey. The book chapter integrates the practical takeaways into Section 5.8 (hyperparameters) and Section 5.9 (What Can Go Wrong). |

### Missing from scaffold

| Scaffold item | Present in chapter? | Assessment |
|---------------|-------------------|------------|
| Experiment Card | Yes | Matches scaffold exactly. |
| Reproduce It block | Yes | Matches scaffold exactly. |
| Build It: 4 components | Yes | All 4 present in correct order with verification checkpoints. |
| Bridging proof | Yes | `--compare-sb3` output matches scaffold expected output (success_fraction differs: scaffold says 0.832, chapter says 0.799, actual lab says 0.799). |
| What Can Go Wrong: 8 items | Yes | All 8 items from scaffold table present, expanded into subsections. |
| Figure plan: 5 figures | Yes | All 5 figures present (with ordering issue noted above). |
| Chapter Bridge: 4 elements | Yes | All 4 elements present. |
| Opening Promise: 5 bullets | Yes | All 5 bullets present, matching scaffold. |
| Concept Registry additions | Covered | All 9 concepts from the scaffold's "Concept Registry Additions" are introduced and defined in the chapter. |

### Unplanned additions

| Addition | Assessment |
|----------|------------|
| Section 5.8 "A note on script naming" (lines 577-578) | **Good addition.** Necessary practical clarification about ch04 vs ch05 numbering. |
| Section 5.8 "What the mean return tells you" (lines 707-709) | **Good addition.** Helpful interpretation that connects the abstract return number (-13.20) to the task structure (13 steps to approach and push, then success). |
| Section 5.8 "Reading TensorBoard during training" table (lines 720-729) | **Good addition.** Maps SB3 log keys to chapter concepts, as recommended by the scaffold's "Add for Manning" section. |

---

## 7. Reader Experience Simulation

| Reader profile | Blocking issue? | Section | Severity |
|---------------|----------------|---------|----------|
| "Understand deeply" | The `--verify` expected output block (Section 5.6, lines 479-505) shows "HER success rate: ~16%" but actual output is ~4.1%. A reader running the code and seeing 4% will doubt their implementation is correct. This is the most likely point of confusion for a careful reader. | 5.6 | **Medium** |
| "Understand deeply" | The grounding example (Section 5.2, line 119) says success fraction "jumps from ~0% to ~16-80%." The 16% lower bound does not match lab output. A reader who has just run `--verify` and seen ~4% success will find this claim confusing. | 5.2 | **Medium** |
| "Learn and practice" | Figure 5.5 appearing before Figure 5.4 is disorienting. The reader sees a reference to "Figure 5.5" while still expecting Figure 5.4. | 5.8 | **Medium** |
| "Results first" | The Run It section (5.8) is self-contained enough for a results-first reader. The Experiment Card provides commands, the checkpoint track is documented, and results tables are clear. No blocking issue. | 5.8 | -- |
| "Plane / no GPU" | The chapter provides `--verify` (CPU, < 2 min) and `--demo` (CPU, < 10 sec on synthetic data) plus checkpoint track for evaluation. The Build It `--demo` only processes synthetic episodes -- it does not show actual RL training. This is appropriate since HER's lab is a data processing module, not a training loop, but it means the "plane reader" does not see a learning curve from Build It. This is a design constraint of the lab, not a chapter issue. | 5.3-5.6 | Low |

---

## 8. Summary Verdict

**REVISE** -- Minor-to-moderate issues in 2 areas. Needs targeted corrections before editorial submission.

### Top 5 Issues (Prioritized)

1. **[High] Figure ordering: Figure 5.5 appears before Figure 5.4.** The FetchPush-v4 environment screenshot (Figure 5.5, line 614) appears in the chapter flow before the HER vs no-HER learning curves (Figure 5.4, line 703). This breaks sequential numbering. **Fix:** Renumber figures so they appear in sequential order in the text flow. The simplest fix is to swap the numbers: make the Push environment screenshot Figure 5.4 and the learning curves Figure 5.5.

2. **[Medium] Expected `--verify` output does not match actual lab output.** The chapter's expected output block (Section 5.6, lines 496-505) shows "HER success rate: ~16%" but the actual `--verify` output produces approximately 4.1% success. A reader running the code will see numbers that disagree with the book. **Fix:** Update the expected output block to show the actual `--verify` output. Also update the tilde estimates in the narrative to match (e.g., "~4-8% success" instead of "~16%").

3. **[Medium] Grounding example success fraction lower bound.** The HER formal definition's grounding example (Section 5.2, line 119) states "success fraction in the augmented data jumps from ~0% to ~16-80%." The 16% lower bound does not match actual lab output (~4-8% on random trajectories). **Fix:** Revise to "~4-8%" for random trajectories, or "~5-80%" to cover the actual observed range. The key pedagogical point (success fraction increases) is correct; the specific numbers need alignment.

4. **[Low] Figures 5.1, 5.2, and 5.3 lack explicit numbered text references.** While all three figures appear at logical positions in the chapter flow, none are referenced by number in the preceding text body (e.g., "as Figure 5.1 shows"). Figure 5.4 correctly uses an explicit reference. **Fix:** Add brief text references for Figures 5.1-5.3 in the paragraph immediately before each figure.

5. **[Low] GoalStrategy.RANDOM omitted from data structures listing without note.** The chapter listing (Section 5.3) attributes the code to `scripts/labs/her_relabeler.py:data_structures` but omits the `RANDOM = "random"` member that exists in the source. This is a correct pedagogical choice (RANDOM is not discussed), but readers who open the source file will notice the difference. **Fix:** Either add a brief note ("the source also includes a RANDOM strategy not used in this chapter") or adjust the attribution to indicate it is an excerpt rather than a verbatim copy.

### What Works Well

- **Failure-first pedagogy (Section 5.1):** Opening with the SAC no-HER flatline at 5% is effective. The reader feels the problem before learning the solution. The Reach vs Push comparison with effective horizon analysis is well-motivated.

- **Build It arc:** The four-component progression (data structures -> goal sampling -> relabeling -> episode processing) is well-ordered. Each component builds on the previous one. Verification checkpoints are concrete and useful.

- **Bridge section (5.7):** The from-scratch to SB3 comparison is convincing. The `max_abs_reward_diff = 0.000e+00` result is strong evidence. The insertion-time vs sample-time distinction and the SB3 parameter mapping table add real understanding.

- **Hyperparameter explanation (Section 5.8):** The treatment of `ent_coef=0.05` and `gamma=0.95` is honest, quantitative, and connects to the 120-run sweep without drowning the reader in sweep details. The cumulative entropy bonus scaling argument is clear.

- **What Can Go Wrong (Section 5.9):** Eight well-structured failure modes with actionable diagnostics. The inclusion of "this is normal, not a bug" entries (hockey-stick learning curve, Reach marginal improvement) shows honest assessment.

- **Overall voice:** Consistently conversational, honest about difficulty, quantified throughout. No persona violations detected.
