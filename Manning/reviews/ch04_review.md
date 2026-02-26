# Review: Chapter 4 -- SAC on Dense Reach

**Reviewer:** Automated (Reviewer agent)
**Date:** 2026-02-26
**Chapter:** Manning/chapters/ch04_sac_dense_reach.md
**Scaffold:** Manning/scaffolds/ch04_scaffold.md
**Lab code:** scripts/labs/sac_from_scratch.py
**Word count:** 9,054 (target: 6,000-10,000)

---

## 1. Checklist Audit

### From-scratch contract

- [x] **Build It is the narrative spine** -- PASS. Sections 4.3-4.9 contain all 7 Build It components, spanning roughly 3,500 words (excluding code). The chapter's weight is clearly on Build It.
- [x] **Every equation has a corresponding code listing** -- PASS. Maximum entropy objective (Sec 4.1) -> actor loss (Sec 4.7); Bellman target (Sec 4.6) -> compute_q_loss (Sec 4.6); temperature loss (Sec 4.1, 4.8) -> compute_temperature_loss (Sec 4.8); Polyak averaging (Sec 4.2) -> sac_update (Sec 4.9). All covered.
- [x] **Bridging proof present** -- PASS. Section 4.10 compares squashed Gaussian log-probabilities between from-scratch and SB3. Includes command, expected output, and explanation of the ~0.02 nat discrepancy.
- [x] **Run It refers back to Build It** -- PASS. Section 4.11 explicitly states "these are the same quantities you implemented in the Build It sections" and maps TensorBoard metrics to from-scratch functions.
- [x] **SB3 is never introduced as a black box** -- PASS. SB3 appears only after all components are built, verified, and bridged. The TensorBoard metric mapping table (Sec 4.10) ensures the reader understands every logged quantity.

### Structure

- [x] **Every term defined before first use** -- PASS. All 16 concept registry terms are present and defined before use. Cross-checked against scaffold's concept registry.
- [x] **Every equation has all symbols defined** -- PASS. Symbols are defined inline or in preceding text. Discount factor gamma, horizon T, and other recurring symbols have brief reminders linking to Chapter 3.
- [ ] **Every code listing under 30 lines** -- WARN. Block 7 (sac_update, Sec 4.9) is 41 lines. The 30-line rule states: "If it must be longer, split into two listings with narrative between them." This is the wiring step and is inherently longer, but it exceeds the guideline.
- [x] **Every code listing has a verification checkpoint** -- PASS. All 7 Build It components have a Checkpoint block with concrete shapes, value ranges, and expected behavior.
- [x] **Experiment Card present** -- PASS. Section 4.11 contains a complete Experiment Card matching the scaffold.
- [x] **Chapter bridge present** -- PASS. Opening paragraphs (lines 11-17) perform all four bridge functions: capability established (PPO 100% success), gap (on-policy discards data), this chapter adds (SAC, off-policy, replay buffer), foreshadow (HER requires off-policy).
- [x] **"What Can Go Wrong" section present** -- PASS. Section 4.12 has 7 failure modes with symptoms, causes, and diagnostics.

### Voice

- [x] **Numbers, not adjectives** -- PASS. All performance claims are quantified: "100% success rate", "18.6mm", "594 fps", "0.02 nats".
- [x] **ASCII only** -- PASS. No em-dashes, curly quotes, or non-ASCII Unicode detected. Uses `--` throughout.
- [x] **No third-person author references** -- PASS. No "Professor", "the author", or "the reader" found.
- [x] **No banned phrases** -- PASS. No "simply", "trivially", "obviously", "beyond the scope" found.
- [x] **Scope boundaries have specific references** -- PASS. Citations to Haarnoja et al. (2018a, 2018b) and Fujimoto et al. (2018) are inline with specific claims.

### Compute accessibility

- [ ] **Build It `--verify` command shown** -- FAIL. The chapter never shows `python scripts/labs/sac_from_scratch.py --verify`. It shows `--demo` (Sec 4.9) and `--compare-sb3` (Sec 4.10), but the quick verification command is missing. The tutorial had a dedicated section (2.5.8) for this. Without it, readers completing each component have no command to run the full end-to-end sanity check.
- [x] **Build It `--demo` shown** -- PASS. Section 4.9 shows the demo command with expected output table and learning curve figure.
- [x] **Chapter completable via Build It + checkpoint track** -- PASS. Checkpoint path given in Experiment Card; eval command provided.
- [x] **Reproduce It block present** -- PASS. Matches scaffold exactly: 3 seeds, 1M steps, correct artifact paths, correct results summary.
- [x] **Reproduce It results match chapter tables** -- PASS. success_rate 1.00, return_mean -1.06, final_distance_mean 0.019 appear in both.

### Visual Content

- [x] **Figure count meets chapter-type minimum** -- PASS. 4 figures present (minimum for Algorithm chapters: 3-5).
- [x] **Every figure from scaffold's figure plan is present** -- PASS. Architecture diagram (4.1), entropy curve (4.2), PPO vs SAC comparison (4.3), demo learning curve (4.4).
- [x] **Every figure has a numbered caption** -- PASS.
- [x] **Every caption includes generation command** -- PASS. Each caption ends with a generation note (TensorBoard extraction or script command, or "Illustrative diagram").
- [x] **Every figure is referenced by number in the text** -- PASS. Each figure has exactly 2 references (intro mention + caption).
- [ ] **Colorblind-friendly palette noted** -- WARN. Captions do not mention the Wong (2011) palette. This is a production concern for actual figure generation, not a chapter text issue, but the scaffold should specify palette in figure commands.
- [x] **All figures are static PNGs** -- PASS. No GIF or video embeds.

---

## 2. Voice Compliance

| Line/Section | Issue | Severity |
|-------------|-------|----------|
| Section 4.1, "standard RL objective (review)" heading | Uses "(review)" which is slightly academic; Manning convention would be "a quick refresher" or just fold it into narrative | Low |
| Section 4.2, "SAC maintains five networks" | Tell-then-show pattern. The Manning voice prefers show-then-tell (persona Sec 4.2). Consider leading with a concrete result or observable phenomenon | Low |
| Section 4.5, bullet "State-dependent std" | Uses "Unlike PPO in Chapter 3 (which used a state-independent `log_std` parameter)" -- good cross-reference, maintains peer tone | N/A (positive) |
| Section 4.9, "This is the same algorithm SB3 runs" | Strong bridge sentence. Good. | N/A (positive) |
| Section 4.11, script naming note | The note about ch03 vs ch04 naming is honest and helpful. Good transparency. | N/A (positive) |

**Overall voice assessment:** The chapter maintains a consistent practitioner-peer tone throughout. It uses "you" for instructions and "we" for shared reasoning. The five-step definition template is used for the maximum entropy objective (Section 4.1) and off-policy learning (Section 4.1). No significant voice violations detected.

---

## 3. Technical Accuracy

### Symbols and definitions

- [x] Every symbol in every equation is defined before use
- [x] Code listings match the lab source in logic and structure
- [x] Numerical claims match scaffold's expected values
- [x] Success criteria in Experiment Card are reasonable
- [x] Reproduce It block has exact commands that would work
- [x] No silent assumptions (all hyperparameters are tabled)

### Issues found

| Location | Issue | Impact |
|----------|-------|--------|
| Section 4.6, code listing | Chapter uses variable names `tq1, tq2` for target Q-values, but the math-to-code mapping table in the same section references `target_q1, target_q2`. The lab source uses `target_q1, target_q2`. | Medium -- notation mismatch within the same section confuses readers cross-referencing code and table |
| Section 4.5, code listing | Chapter writes `dist = Normal(mean, log_std.exp())` (inline exp), but lab source uses intermediate `std = log_std.exp()` then `dist = Normal(mean, std)`. Functionally equivalent but a cosmetic discrepancy between chapter listing and actual lab code. | Low -- reader running the lab will see slightly different code |
| Section 4.5, code listing | Chapter code omits the intermediate `std` variable and the `# Reparameterization trick` comment from the lab. Minor pedagogical loss since the chapter already explains reparameterization in the surrounding text. | Low |
| Section 4.9, code listing | The `sac_update` function listing is 41 lines, exceeding the 30-line guideline. This is the wiring step and combines all three update phases plus Polyak averaging. | Medium -- long listing may fatigue readers; persona Sec 3.7 says to split into two listings with narrative between |
| Section 4.1, entropy formula | The chapter defines entropy as $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ (standard Shannon entropy). The source tutorial (Section 1.3) used the Gaussian entropy formula with determinant notation. The chapter's version is more general and correct for the SAC context. | N/A (improvement) |
| Section 4.11, training time | Experiment Card says "~14 min (GPU)" for fast path (500k steps). Reproduce It says "~28 min per seed (GPU)" for full runs (1M steps). Results table also says "Training Time: ~28 min" and "Throughput: ~594 fps". 500k / 594 = ~840 sec = ~14 min, consistent. | N/A (correct) |

---

## 4. Visual Content

| Figure | Issue | Severity |
|--------|-------|----------|
| Figure 4.4 (demo curve) | Appears on line 574 (Section 4.9) but is numbered 4.4. Figure 4.3 (PPO vs SAC) appears on line 739 (Section 4.11). Figures should be numbered in order of appearance: the demo curve should be 4.3 and the PPO vs SAC comparison should be 4.4. | Medium -- out-of-order numbering confuses readers |
| All figures | Alt text is present and descriptive for each `![...]` image tag -- good for accessibility | N/A (positive) |
| Figure 4.1 | Caption says "(Illustrative diagram.)" -- this is appropriate for a hand-drawn/diagrammatic figure. | N/A |

---

## 5. Structural Completeness

| Section | Present? | Length | Notes |
|---------|----------|-------|-------|
| Opening promise | Yes | 5 bullets | Matches scaffold exactly |
| Bridge | Yes | 3 paragraphs (~220 words) | All 4 bridge elements present: capability, gap, addition, foreshadow |
| WHY (4.1) | Yes | ~1,600 words | Standard RL review, determinism problems, max entropy definition (5-step template), Boltzmann policy, off-policy definition, auto temperature, robotics motivation |
| HOW (4.2) | Yes | ~800 words | Component table, training loop pseudocode, PPO vs SAC comparison, hyperparameters |
| Build It (4.3-4.9) | Yes | ~3,500 words, 7 components | All 7 scaffold components present with checkpoints |
| Bridge section (4.10) | Yes | ~600 words | Log-prob comparison, SB3 additions, TensorBoard mapping |
| Run It + Experiment Card (4.11) | Yes | ~1,300 words | Card, milestones, TensorBoard guide, results comparison |
| What Can Go Wrong (4.12) | Yes | ~700 words, 7 items | All 8 scaffold failure modes present |
| Summary (4.13) | Yes | ~400 words | 5 bullet recap, bridge to Ch5 (HER) |
| Reproduce It | Yes | Matches scaffold | 3 seeds, 1M steps, correct metrics |
| Exercises | Yes | 5 exercises | Graduated: Verify, Tweak, Tweak, Extend, Challenge |
| Figures | Yes | 4 figures | Architecture, entropy curve, PPO vs SAC, demo curve |

---

## 6. Content Gap Analysis

### Lost from tutorial

- **`--verify` command and expected output (Tutorial Section 2.5.8):** The tutorial had a dedicated section showing the full verification command and its expected terminal output. The chapter omits this entirely. The `--demo` and `--compare-sb3` commands are present, but `--verify` -- the quickest Build It sanity check -- is not shown as a runnable command. Readers need this as the capstone of the Build It sections.

- **Exercise on tau variation (Tutorial Exercise 2.5.4):** The tutorial had an exercise exploring `tau` values (0.001, 0.01, 1.0) for Polyak averaging. The chapter's exercises cover twin-Q ablation, fixed vs auto alpha, Q-divergence monitoring, and target entropy variation, but not tau. This is a minor loss -- the chapter's exercises are strong and the tau concept is explained in the main text.

- **Throughput scaling section (Tutorial Section 3.5):** The tutorial included a throughput experiment varying n_envs. The chapter mentions throughput (594 fps) and notes GPU utilization, but does not include the scaling experiment or command. This was planned for cutting in the scaffold adaptation notes, so this is intentional.

### Missing from scaffold

- Nothing missing. All scaffold deliverables are present.

### Unplanned additions

- **Section 4.11 script naming note:** The chapter adds a note explaining why the production script is named `ch03_sac_dense_reach.py` despite being Manning Chapter 4. This is a useful addition that prevents reader confusion.

- **Section 4.1 "Why this matters for robotics":** Expanded from the scaffold's brief mention into a three-point subsection. Appropriate depth for the WHY section.

---

## 7. Reader Experience Simulation

| Reader profile | Blocking issue | Section |
|---------------|----------------|---------|
| "Understand deeply" | No `--verify` command shown. Reader builds all 7 components with manual checkpoints but has no single command to run the full verification suite before proceeding to Bridge. | After Section 4.9 |
| "Learn and practice" | The MLP helper class is used in TwinQNetwork and GaussianPolicy code listings but never defined or shown. Reader copying code from the chapter cannot run it without also finding the MLP class from the lab file. The chapter describes it textually ("two hidden layers of 256 units, ReLU activations") but the class definition is absent. | Section 4.4, 4.5 |
| "Results first" | Run It section (4.11) is navigable without Build It. The Experiment Card is self-contained with commands and success criteria. No blocking issue for this reader. | N/A |
| "Plane / no GPU" | `--demo` command includes `--steps 50000` but no CPU time estimate is given for the demo. The Experiment Card gives CPU time for the Run It fast path (~60 min) but not for `--demo`. The lab file says ~5 min for 50k steps on CPU, but the chapter does not state this. | Section 4.9 |

---

## 8. Summary Verdict

### **REVISE**

The chapter is structurally complete and well-written. All scaffold deliverables are present, the voice is consistently in the Manning peer-practitioner register, and the from-scratch contract is delivered. The issues are targeted and fixable.

### Top 5 issues to fix (prioritized)

1. **Missing `--verify` command (blocking).** Add the `python scripts/labs/sac_from_scratch.py --verify` command with expected output after Section 4.9, mirroring the tutorial's Section 2.5.8. This is the Build It capstone -- the reader needs it to confirm all components work before proceeding to Bridge. Without it, the Build It track feels incomplete.

2. **Figure numbering out of order (medium).** Swap figure numbers so they appear sequentially: the demo learning curve (currently 4.4, appears in Sec 4.9) should be 4.3, and the PPO vs SAC comparison (currently 4.3, appears in Sec 4.11) should be 4.4. Update all references.

3. **Variable name mismatch in Section 4.6 (medium).** The code listing uses `tq1, tq2` but the math-to-code mapping table references `target_q1, target_q2` (matching the lab source). Either rename the code variables to `target_q1, target_q2` to match both the table and the lab, or update the table to match the code. Recommend matching the lab source (`target_q1, target_q2`) since that is what the reader will see when running the code.

4. **sac_update listing exceeds 30-line limit (medium).** Block 7 is 41 lines. Consider splitting into two listings: (a) the three update steps (Q-networks, policy, temperature), and (b) the Polyak averaging, with a short narrative paragraph between them explaining the target network update.

5. **MLP class undefined (low-medium).** The `MLP` helper class is used in TwinQNetwork and GaussianPolicy listings but never shown. Add a brief note (1-2 sentences + optional collapsed code block) explaining that MLP is a standard feedforward network available in the lab file. Alternatively, show the 10-line MLP class definition before the first listing that uses it (Section 4.4). The "Learn and practice" reader who copies code needs this.

### Minor items (fix if time)

- Add CPU time estimate for `--demo --steps 50000` in Section 4.9 (~5 min CPU per the lab module).
- Section 4.5 code listing has a minor cosmetic difference from the lab (inline `log_std.exp()` vs intermediate `std` variable). Consider aligning to match the lab exactly.
- Consider noting the Wong (2011) colorblind palette in figure generation notes for production readiness.
