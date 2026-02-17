# Review: Chapter 3 -- PPO on Dense Reach

**Reviewer:** Automated Manning Reviewer Agent
**Date:** 2026-02-17
**Chapter draft:** `Manning/chapters/ch03_ppo_dense_reach.md`
**Word count:** ~7,500 words (target: 6,000-10,000)

---

## 1. Checklist Audit

### From-scratch contract

- [x] **Build It is the narrative spine** -- PASS. Build It spans sections 3.3-3.8, covering all six scaffold components (actor-critic network, GAE, PPO loss, value loss, wiring, training loop). The Build It sections constitute the majority of the chapter's technical content (~3,500 words). The chapter flows WHY -> HOW -> Build It -> Bridge -> Run It as required.
- [x] **Every equation in HOW has a corresponding code listing in Build It** -- PASS. All equations from the compact summary (Section 3.2, lines 219-234) map to code: $\delta_t$ and $\hat{A}_t$ in Section 3.4, $\rho_t$ and $L^{\text{CLIP}}$ in Section 3.5, $L_{\text{value}}$ in Section 3.6, combined $\mathcal{L}$ in Section 3.7. The return $\hat{G}_t$ is computed inline in the GAE function.
- [x] **Bridging proof present** -- PASS. Section 3.9 (lines 574-619) shows the `--compare-sb3` bridging proof with expected output and match criteria. The section also includes the SB3 TensorBoard-to-code mapping table.
- [x] **Run It refers back to Build It** -- PASS. Section 3.10, line 708: "Remember, these are the same quantities you implemented in the Build It sections. `train/value_loss` is the output of your `compute_value_loss`. `train/clip_fraction` comes from your `compute_ppo_loss`." The bridge section (3.9, line 619) also states: "You are not reading opaque metrics from a black box; you are reading quantities you have implemented and verified."
- [x] **SB3 is never introduced as a black box** -- PASS. SB3 is introduced in Section 3.9 only after all components have been built and verified. The bridge proves algorithmic equivalence. The Run It section cross-references the Build It functions.

### Structure

- [x] **Every term defined before first use** -- PASS. Policy $\pi_\theta$ (line 26), reward $r_t$ (line 34), discount factor $\gamma$ (line 36), time horizon $T$ (line 38), return $J(\theta)$ (lines 40-44), advantage $A(x,a)$ (lines 58-68), probability ratio $\rho_t$ (lines 93-101), clipping parameter $\epsilon$ (line 105), TD residual $\delta_t$ (lines 284-288), GAE $\lambda$ (lines 290-294) -- all defined before use.
- [x] **Every equation has all symbols defined** -- PASS. All symbols are defined inline before or within their first equation. The compact summary (lines 219-234) restates equations previously introduced.
- [x] **Every code listing under 30 lines** -- PASS. Actor-critic: 26 lines (243-269). GAE: 27 lines (298-325). PPO loss: 34 lines (350-384). Value loss: 20 lines (403-423). PPO update: 37 lines (444-481). Training loop: 21 lines (496-517). NOTE: The PPO loss listing (34 lines) and PPO update listing (37 lines) exceed the 30-line guideline. See issue #2 in Summary Verdict.
- [ ] **Every code listing has a verification checkpoint** -- WARN. Sections 3.3 through 3.8 all have Checkpoint blocks. However, the training loop section (3.8) places its checkpoint (the `--verify` block, lines 550-571) after the demo results table, which is a slightly unusual ordering. The checkpoint for the training loop is effectively the `--verify` command rather than a hand-runnable code snippet like the other checkpoints. Minor structural inconsistency.
- [x] **Experiment Card present** -- PASS. Lines 624-653 contain a properly formatted experiment card with algorithm, environment, fast path, time estimates, checkpoint track, expected artifacts, and success criteria.
- [x] **Chapter bridge present** -- PASS. Lines 11-17 bridge from Ch2: capability established (environment anatomy), gap (random policy 0% success), this chapter adds (PPO, from-scratch implementation, pipeline validation), foreshadow (PPO is on-policy; Ch4 introduces SAC off-policy).
- [x] **"What Can Go Wrong" section present** -- PASS. Section 3.11 (lines 753-823) covers 10 failure modes with symptom/cause/diagnostic structure. Thorough and well-organized.

### Voice

- [x] **Numbers, not adjectives** -- PASS. All performance claims are quantified: "100% success rate" (line 687), "4.6mm average goal distance" (line 687), "~1,300 steps/second" (line 687), "success threshold is 50mm" (line 738), multi-seed results "1.00 +/- 0.00" (line 871).
- [x] **ASCII only** -- PASS. No non-ASCII characters found. Dashes are `--`, quotes are straight, no Unicode symbols.
- [x] **No third-person author references** -- PASS. No "Professor", "the author", or similar constructs found.
- [x] **No "simply," "trivially," "obviously"** -- PASS. None of the banned phrases appear in the draft.
- [x] **Scope boundaries have specific references** -- PASS. Policy gradient theorem cites "Sutton & Barto, 2018, Ch13.1" (line 71). GAE cites "Schulman et al., 2015" (line 280). PPO cites "Schulman et al., 2017" (line 89). Henderson et al. (2018) cited for instability (line 79).

### Compute accessibility

- [x] **Build It `--verify` runs on CPU in under 2 minutes** -- PASS (line 571: "This runs on CPU in under 2 minutes").
- [x] **Build It `--demo` runs on CPU** -- PASS (line 535: `--demo` command shown; demo results table at lines 540-546 shows ~37k steps solved).
- [x] **Chapter completable via Build It + checkpoint track** -- PASS. Checkpoint track explicitly mentioned at lines 668-674 with eval-only command. All Build It verification runs on CPU.
- [x] **Reproduce It block present** -- PASS. Lines 846-881. Contains exact commands, hardware note, per-seed time estimates, artifact paths, and results summary with variance.
- [x] **Reproduce It results match chapter tables** -- PASS. Reproduce It states "success_rate: 1.00 +/- 0.00, return_mean: -0.40 +/- 0.05, final_distance_mean: 0.005 +/- 0.001". The Run It section states "100% success rate" and "4.6mm average goal distance" and the eval JSON shows `success_rate: 1.0, return_mean: -0.40, final_distance_mean: 0.0046`. These are consistent.

---

## 2. Voice Compliance

| Line/Section | Issue | Severity |
|-------------|-------|----------|
| Section 3.1, line 15 | "This chapter introduces PPO (Proximal Policy Optimization), an on-policy algorithm that learns by clipping likelihood ratios to prevent destructive updates." -- Reads slightly like a textbook abstract. The sentence is functional but could be more conversational. | Low |
| Section 3.1, line 48 | "The challenge: how do you take a gradient of this?" -- Good, natural Manning voice. Asks the question the reader is thinking. | N/A (positive) |
| Section 3.1, line 84 | "In our experience, unclipped policy gradient on Fetch tasks fails roughly half the time" -- Honest, quantified claim grounded in experience. Excellent. | N/A (positive) |
| Section 3.2, line 163 | "PPO maintains two neural networks (or two heads of one network)" -- Good parenthetical hedge. | N/A (positive) |
| Section 3.2, line 288 | "The value terms $V_{\text{rollout}}$ are computed when collecting the rollout and treated as constants during optimization." -- Passive voice ("are computed", "treated as constants"). Could be "We compute the value terms during rollout collection and treat them as constants." | Low |
| Section 3.2, line 329 | "The **returns** are computed as..." -- Passive voice. | Low |
| Section 3.5, line 386 | "The ratio is computed in log-space..." -- Passive voice. | Low |
| Section 3.10, line 748 | "**Metrics are computed correctly**" -- Passive voice in a list item. Acceptable in context. | Low |
| Section 3.9, line 619 | "You are not reading opaque metrics from a black box; you are reading quantities you have implemented and verified." -- Strong, effective line that reinforces the from-scratch contract. | N/A (positive) |
| Section 3.12, line 841 | "That last point is the gap that Chapter 4 addresses." -- Good, clean bridge. | N/A (positive) |

**Overall voice assessment:** The chapter is remarkably clean on voice. No banned phrases, no third-person author references, no hand-waving. There are a few pockets of passive voice in technical descriptions (lines 288, 329, 386), which is common and low-severity in algorithmic writing. The tone is consistently "experienced colleague at whiteboard" -- conversational where appropriate, precise where needed. The occasional passive voice does not disrupt the reading experience.

---

## 3. Technical Accuracy

### Symbols and equations

- [x] Every symbol in every equation is defined before use.
- [x] The goal-conditioned input $x_t := (s_t, g)$ is defined at line 28 before appearing in subsequent equations.
- [x] The advantage function definition (lines 58-68) includes a concrete non-example (positive advantage does not mean good absolute outcome), matching the persona's 5-step definition template.
- [x] The compact equation summary (lines 219-234) restates all equations from the chapter for reference. All symbols previously defined.

### Code listings vs lab source

- [x] **ActorCritic network (3.3):** The chapter listing (lines 243-269) is a simplified version of the lab source (lines 69-110). The lab source has docstrings and type annotations; the chapter strips these for readability. The core logic (backbone, actor_mean, actor_log_std, critic, forward method) matches exactly. PASS.
- [x] **compute_gae (3.4):** The chapter listing (lines 298-325) matches the lab source (lines 130-189) in logic. The chapter strips docstrings and type annotations. The recursion, done masking, and returns computation are identical. PASS.
- [x] **compute_ppo_loss (3.5):** The chapter listing (lines 350-384) matches the lab source (lines 196-267) in logic. Stripped of docstrings. The ratio computation, clipping, min operation, and diagnostic calculations (approx_kl, clip_fraction) are identical. PASS.
- [ ] **compute_value_loss (3.6):** The chapter listing (lines 403-423) is a **simplified version** of the lab source (lines 270-329). The lab source accepts optional `old_values` and `clip_range` parameters for value clipping and uses a separate `explained_variance` helper function. The chapter inlines the explained variance computation and drops the optional value clipping. The core MSE loss computation is identical. The simplification is pedagogically appropriate (value clipping is optional and rarely used with these defaults), but the comment header says "from scripts/labs/ppo_from_scratch.py:value_loss" which suggests exact correspondence. WARN -- the listing is a pedagogical simplification, not an exact excerpt.
- [x] **ppo_update (3.7):** The chapter listing (lines 444-481) matches the lab source (lines 332-410) in logic. Stripped of docstrings. The advantage normalization, individual loss computation, combined loss, gradient clipping, and info dict assembly are identical. PASS.
- [x] **collect_rollout (3.8):** The chapter listing (lines 496-517) is deliberately abbreviated with `# ... step env, store transition, handle resets` at line 511. The header notes "full code in scripts/labs/ppo_from_scratch.py". This is acceptable for a training loop where the full code would exceed 30 lines. PASS.

### Numerical claims

- [x] Success rate 100% after 500k steps -- matches scaffold expected values.
- [x] Final distance 4.6mm -- matches scaffold (0.005 +/- 0.001, which rounds to 0.0046-0.006).
- [x] ~1,300 steps/second on NVIDIA GB10 -- stated as measured value.
- [x] ~5,577 parameters -- matches lab source: `sum(p.numel() for p in model.parameters())` for obs_dim=16, act_dim=4, hidden_dim=64. Calculation: backbone (16*64 + 64 + 64*64 + 64) + actor_mean (64*4 + 4) + actor_log_std (4) + critic (64*1 + 1) = 1088 + 4160 + 260 + 4 + 65 = 5577. PASS.
- [x] CartPole demo results (lines 540-546) -- plausible and consistent with lab source demo output.

### Experiment card

- [x] Fast path: 500,000 steps, seed 0.
- [x] Time estimates: ~5 min GPU, ~30 min CPU.
- [x] Success criteria: success_rate >= 0.90, mean_return > -10.0, final_distance_mean < 0.02.
- [x] Checkpoint track: `checkpoints/ppo_FetchReachDense-v4_seed0.zip`.
- [x] Expected artifacts listed with exact paths.

### Reproduce It commands

- [x] `for seed in 0 1 2; do bash docker/dev.sh python scripts/ch02_ppo_dense_reach.py all --seed $seed --total-steps 1000000; done` -- correct command using the production script.
- [x] Results summary includes seed count, variance, and per-metric values.
- [x] Hardware note and per-seed time estimates included.

---

## 4. Structural Completeness

| Section | Present? | Length | Notes |
|---------|----------|-------|-------|
| Opening promise | Yes | 5 bullets | Lines 3-9. Covers derivation, implementation, verification, bridging, and training. Complete. |
| Chapter bridge | Yes | 4 sentences + 2 paragraphs | Lines 11-17. All four bridge elements (capability, gap, addition, foreshadow) present. Well-executed. |
| WHY (3.1) | Yes | ~1,800 words | Lines 20-157. Covers objective, policy gradient, instability, PPO clipping, concrete example, dense rewards, well-posedness. Thorough. |
| HOW (3.2) | Yes | ~600 words | Lines 159-234. Actor-critic architecture, training loop pseudocode, hyperparameter table, compact equation summary. |
| Build It (3.3-3.8) | Yes | ~3,200 words, 6 components | All six scaffold components present. Each has equation, code, checkpoint. Wiring step (3.7) and training loop (3.8) present. |
| Bridge section (3.9) | Yes | ~600 words | `--compare-sb3` proof, SB3 additions list, TensorBoard metric mapping table. |
| Run It + Experiment Card (3.10) | Yes | ~1,200 words | Experiment card, commands (full + quick + eval-only), milestones table, TensorBoard guide, JSON verification, pipeline validation discussion. |
| What Can Go Wrong (3.11) | Yes | ~800 words, 10 items | Covers all failure modes from scaffold table plus additional items (Build It verify failure, compare-sb3 mismatch). |
| Summary (3.12) | Yes | ~400 words | Six bullet recap + bridge to Ch4 (SAC off-policy). |
| Reproduce It | Yes | ~350 words | After summary. Exact commands, hardware, time, seeds, artifacts, results summary. |
| Exercises | Yes | 5 exercises | Graduated: Verify (1), Tweak (2, 3), Extend (4), Challenge (5). Each has expected outcomes. |

**All required sections present and well-formed.**

---

## 5. Content Gap Analysis

### Lost from tutorial

- **Part 0 "Setting the Stage" (Sections 0.1-0.3):** Intentionally cut per scaffold. The "Option A vs Option B" framing and "diagnostic mindset" preamble are removed. The diagnostic mindset concept is captured more concisely in the bridge and WHY section. CORRECT CUT.
- **Part 4 "Understanding What You Built" (Sections 4.1-4.3):** Intentionally cut per scaffold. Section 4.1 (what the policy learned) content is folded into Run It (lines 732-738). Section 4.3 (pipeline validation) is folded into Run It (lines 742-750). Section 4.2 (clipping in action) is folded into the PPO loss checkpoint (Section 3.5). CORRECT CUT.
- **Exercise 2.3 "Explain the Clipping (Written)":** Intentionally cut per scaffold. Replaced with code-based clip range ablation (Exercise 3). CORRECT CUT.
- **The geometric aside (collapsible details block):** Intentionally cut per scaffold. CORRECT CUT.
- **References section:** Intentionally cut per scaffold. Citations moved inline. CORRECT CUT.
- **GIF/video references:** Intentionally cut per scaffold. Replaced with description of what the trained policy does (lines 732-738). CORRECT CUT.
- **Exercise 2.5.2 "Clipping Effect" from tutorial:** Tutorial had a brief exercise asking readers to modify clip_range in verify_ppo_loss. This is partially captured in book Exercise 3 (clip range ablation) but the verify-level exercise of modifying clip_range to 0.0 and 1.0 is lost. Minor loss -- the book's Exercise 3 is more substantial.
- **Exercise 2.5.4 "Record a GIF":** The tutorial had a `--record` exercise with GIF output. This is dropped in the book. Not significant for Manning format.

### Missing from scaffold

- [x] All six Build It components present per scaffold table.
- [x] Experiment Card matches scaffold exactly.
- [x] Reproduce It block matches scaffold exactly.
- [x] All 10 "What Can Go Wrong" items from scaffold present, plus one extra.
- [x] All five exercises from scaffold present.
- [x] Concept registry additions (return, discount factor, advantage, etc.) are implicitly defined in the chapter text. The registry itself does not appear in the chapter (correct -- it is an authoring tool, not reader-facing).
- [ ] **Scaffold Section "Adaptation Notes: Add for Manning" item "Explicit on-policy definition and discussion"**: The scaffold calls for defining on-policy learning formally as a concept registry entry. The chapter defines on-policy at line 198-200 ("PPO is **on-policy**: we can only use data from the current policy. After updating, our data is stale and must be discarded.") and expands at lines 200-200. However, the formal definition is brief and embedded in the training loop description rather than receiving the full 5-step treatment. WARN -- adequate but could be more prominent given the scaffold's emphasis that "this is critical because the PPO->SAC transition in Ch4 depends on the reader understanding this limitation."

### Unplanned additions

- **Well-posedness check (lines 146-157):** Present in the chapter as Section 3.1 subsection. This was not explicitly called for in the scaffold's section plan but is consistent with the persona's Hadamard diagnostic convention. Appropriate addition.
- **"What the trained policy does" subsection (lines 732-738):** Extended description of the trained policy's behavior (velocity commands, convergence in 10-15 steps, gripper dimension irrelevant). This is content folded from the tutorial's Part 4.1. Appropriate addition.
- **"Why this validates your pipeline" subsection (lines 742-750):** Folded from tutorial's Part 4.3. Appropriate addition.

---

## 6. Reader Experience Simulation

| Reader profile | Blocking issue | Section |
|---------------|----------------|---------|
| "Understand deeply" | No blocking issues. The WHY -> Build It -> Bridge path is comprehensive. The reader can trace every equation to code, verify each component, see the `--compare-sb3` proof, and connect SB3 metrics to their own functions. Well-served by this chapter. | -- |
| "Learn and practice" | The training loop listing (Section 3.8, lines 496-517) is deliberately abbreviated with `# ...` comments. A reader who wants to run the from-scratch code on FetchReach will not find enough guidance until Exercise 5. The gap between "here's the abbreviated training loop" and "here's the `--demo` on CartPole" may frustrate a hands-on reader who expected to run from-scratch code on the actual Fetch environment. | 3.8 |
| "Results first" | The Run It section (3.10) can be reached by scrolling past Build It. Lines 660-661 give the one-command version. However, the experiment card references `scripts/ch02_ppo_dense_reach.py` -- the script name uses `ch02` (tutorial numbering) while the chapter is numbered 3 in the book. A reader who sees "Chapter 3" in the book and `ch02` in the script may be confused about whether they are running the right script. See issue #1 in Summary Verdict. | 3.10 |
| "Plane / no GPU" | Build It `--verify` runs on CPU in under 2 minutes. `--demo` trains CartPole in ~30 seconds on CPU. Checkpoint track for evaluation is documented. However, no CPU time estimate is given for the `--demo` mode specifically (the 30-second figure comes from the tutorial, not the chapter). The chapter only states "~30 min (CPU)" for the full Run It fast path. | 3.8, 3.10 |

**Additional reader experience notes:**

- The chapter is long (~7,500 words) but flows well. The Build It sections have a consistent rhythm (equation -> code -> checkpoint) that creates a satisfying pattern.
- The advantage function non-example (line 69: "positive advantage does NOT mean the action leads to a good outcome in absolute terms") is excellent pedagogy.
- The concrete PPO clipping example (Section 3.1, lines 118-133) is effective -- tracing specific numbers through the clipped objective makes the math tangible.
- The compact equation summary (Section 3.2, lines 219-234) provides a useful reference but may feel redundant to a linear reader who just saw these equations individually. Consider labeling it more explicitly as a reference card.

---

## 7. Summary Verdict

**REVISE** -- The chapter is structurally complete and technically accurate, with strong voice compliance and well-executed Build It sections. However, there are a few issues that should be addressed before submission.

### Top 5 Issues (Prioritized)

**1. Script naming mismatch (Medium -- reader confusion risk).**
The production script is `scripts/ch02_ppo_dense_reach.py` (tutorial numbering), but the book chapter is numbered 3. The experiment card, Run It section, and Reproduce It block all reference `ch02_ppo_dense_reach.py` and produce artifacts with `ch02` in their names (e.g., `results/ch02_ppo_fetchreachdense-v4_seed0_eval.json`). A Manning reader who has read "Chapter 3" will see `ch02` in every command and artifact path, which creates a numbering disconnect. This is acknowledged in the scaffold (`Production script: scripts/ch02_ppo_dense_reach.py (existing; the "Run It" reference)`) so it may be intentional, but it will confuse readers. Either rename the script or add a brief note explaining the numbering difference.

**2. Two code listings exceed 30-line guideline (Low -- Manning editorial concern).**
The PPO loss listing (Section 3.5, lines 350-384) runs 34 lines. The PPO update listing (Section 3.7, lines 444-481) runs 37 lines. The persona specifies "No single code listing exceeds ~30 lines. If it must be longer, split into two listings with narrative between them." These are modestly over and the content is cohesive, so splitting may hurt readability. Consider whether editorial will flag this.

**3. Value loss listing is simplified relative to lab source (Low -- accuracy concern).**
The chapter's `compute_value_loss` (lines 403-423) is a simplified version of the lab source -- it drops the optional `old_values` and `clip_range` parameters and inlines the `explained_variance` helper. The header comment says "from scripts/labs/ppo_from_scratch.py:value_loss" which implies exact correspondence. Either update the header to say "adapted from" or note the simplification in the text.

**4. On-policy definition could be more prominent (Low -- pedagogical concern).**
The scaffold specifically calls for an "Explicit 'on-policy' definition and discussion" as a key addition for Manning, noting it is "critical because the PPO->SAC transition in Ch4 depends on the reader understanding this limitation." The chapter defines on-policy at line 198-200 within the training loop description and expands at line 200. This is adequate but the concept is embedded in prose rather than receiving a highlighted definition or callout. Given the scaffold's emphasis on this concept's importance for the Ch3->Ch4 bridge, consider making it more prominent.

**5. `--demo` CPU time not stated in chapter (Low -- compute accessibility).**
The Build It training loop section (3.8) references `--demo` on CartPole but does not state the CPU time estimate. The tutorial says ~30 seconds. The persona requires that the "plane / no GPU" reader know what to expect. The chapter only gives the full Run It CPU time (~30 min for the fast path). Adding "~30 seconds on CPU" for the `--demo` would help.

### Positive Highlights

- The from-scratch contract is fully delivered. Build It is genuinely the narrative spine, not a sidebar.
- Technical accuracy is excellent. Code listings match lab source in logic, equations are correct, numerical claims are consistent.
- Voice is clean throughout -- no banned phrases, no hand-waving, no third-person references.
- The bridge (both chapter bridge and from-scratch-to-SB3 bridge) is well-executed.
- The "What Can Go Wrong" section is thorough (10 items with specific diagnostics).
- The TensorBoard metric-to-code mapping table (Section 3.9, lines 610-618) is a standout element that directly serves the from-scratch promise.
- Exercise graduation (Verify -> Tweak -> Extend -> Challenge) is well-calibrated.
