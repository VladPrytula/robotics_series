# Review: Chapter 2 -- Environment Anatomy

**Reviewer:** Automated Manning Reviewer Agent
**Date:** 2026-02-17
**Chapter draft:** `Manning/chapters/ch02_env_anatomy.md`
**Word count:** ~8,025 words (target: 6,000-10,000)

---

## 1. Checklist Audit

### From-scratch contract (adapted for Environment chapter)

- [x] **Build It is the narrative spine** -- PASS. Build It sections span 2.3-2.7 (sections titled "Build It: ..."), constituting the bulk of the chapter. The reader inspects observations, actions, rewards, goals, and cross-environment structure through hands-on code before reaching Run It.
- [x] **Every concept has a corresponding code listing** -- PASS. All seven scaffold components are present: obs_inspector (2.3), action_explorer (2.4), goal_space (2.3), dense_reward_check (2.5), sparse_reward_check (2.5), relabel_check (2.6), cross_env_compare (2.7).
- [x] **Bridging proof present** -- PASS. Section 2.8 provides the bridging proof connecting manual lab code to the production script, with the `--bridge` command and expected output.
- [x] **Run It refers back to Build It** -- PASS. Section 2.9 explicitly states "This is the machine-readable version of section 2.3" (line 647), "This is the automated version of sections 2.5 and 2.6" (line 649), and the summary in 2.11 bridges to Ch3 by referencing the shapes, rewards, and baseline documented in Build It.
- [x] **SB3 is never introduced as a black box** -- PASS. SB3's `MultiInputPolicy` is mentioned in 2.3 (lines 220-230) with a clear explanation of how it processes the dictionary observation. The reader already understands the dictionary structure from the Build It component before `MultiInputPolicy` is referenced.
- [x] **No Reproduce It needed** -- PASS (correct for Environment chapter type). A "Verify It" block replaces it (lines 784-839).

### Structure

- [x] **Every term defined before first use** -- PASS. Goal-conditioned MDP pieces are defined in 2.1 (lines 68-79). Dense/sparse rewards are defined in 2.5 before use. `compute_reward` API is introduced in 2.5 before the relabeling section. $\phi$ is introduced in 2.1 (line 76) and elaborated in 2.3 (line 182).
- [x] **Every equation has all symbols defined** -- PASS. Dense reward equation (line 304) defines $g_a$, $g_d$, and $\|\cdot\|_2$. Sparse reward equation (lines 351-352) defines $\epsilon = 0.05$. The GCMDP tuple (lines 70-78) defines all seven components with plain-language descriptions.
- [ ] **Every code listing under 30 lines** -- WARN. Most listings are within bounds, but `dense_reward_check` (lines 318-338, ~21 lines), `relabel_check` (lines 432-455, ~24 lines), and `cross_env_compare` (lines 494-517, ~24 lines) are all fine. The `sparse_reward_check` listing (lines 365-391, ~27 lines) is approaching the 30-line limit but still within it. PASS.
- [x] **Every code listing has a verification checkpoint** -- PASS. Each Build It component has a "> **Checkpoint.**" block with expected shapes, values, and diagnostic guidance for failures.
- [x] **Experiment Card present** -- PASS. Lines 613-641 contain a properly formatted experiment card for the inspection pipeline.
- [x] **Chapter bridge present** -- PASS. Lines 11-15 serve as the bridge from Ch1, covering all four parts: capability established (proof of life), gap (don't understand what env says), this chapter adds (complete anatomy), foreshadow (Ch3 trains PPO).
- [x] **"What Can Go Wrong" section present** -- PASS. Section 2.10 (lines 684-758) covers 8 failure modes with symptom/cause/fix structure.
- [x] **Verify It block present** -- PASS. Lines 784-839 contain a properly formatted Verify It block with exact commands, expected artifacts, and expected values.

### Voice

- [x] **Numbers, not adjectives** -- PASS. All performance claims are quantified: "success rate of 0%-10%" (line 668), "return_mean in [-25, -15]" (line 669), "threshold 0.05" (line 349), displacement magnitudes "0.005-0.02 meters per step" (line 289).
- [x] **ASCII only** -- PASS. No non-ASCII characters found. Em-dashes are `--`, quotes are straight, no Unicode.
- [x] **No third-person author references** -- PASS. No "Professor", "the author", or similar constructs.
- [x] **No "simply," "trivially," "obviously"** -- PASS (with a note). The word "trivial" appears twice (lines 438, 460) but only in the technical sense "non-trivial achieved_goal" (meaning: not the initial position), not as a minimizing phrase. This usage is acceptable.
- [x] **Scope boundaries have specific references** -- PASS. HER forward reference points to "Chapter 5" (lines 34, 479). The original HER paper is cited specifically: "Andrychowicz et al., 2017" (line 477).

### Compute accessibility

- [x] **Build It `--verify` runs on CPU in under 2 minutes** -- PASS (stated at line 828: "All checks pass in < 1 min on CPU").
- [N/A] **Build It `--demo`** -- N/A for Environment chapter (no training).
- [x] **Chapter completable via Build It + Verify It** -- PASS. No GPU required. All inspection code runs on CPU in seconds.
- [x] **Verify It block present with exact commands and results** -- PASS. Lines 784-839.

---

## 2. Voice Compliance

| Line/Section | Issue | Severity |
|-------------|-------|----------|
| 2.1, line 24 | "In our experience, environment misunderstandings are the single most common source" -- strong claim. Could be softened to "In our experience, ... are among the most common sources" | Low |
| 2.3, line 132 | "Let's look at the code" -- good, natural Manning voice | N/A (positive) |
| 2.5, line 343 | "Why do we check all three?" -- rhetorical question followed by good explanation; fits the Manning colleague voice | N/A (positive) |
| 2.6, line 416 | "This section is the pedagogical climax of the chapter" -- slightly self-conscious; the reader does not need meta-commentary about where the climax falls. Consider removing or softening. | Low |
| 2.7, line 558 | "This is worth appreciating." -- slightly patronizing. The reader will appreciate it (or not) without being told to. | Low |
| 2.8, line 606 | "never trust a pipeline you have not verified by hand" -- good, strong, practitioner-oriented. | N/A (positive) |
| 2.1, lines 65-79 | GCMDP presentation is conversational as the scaffold requested ("seven pieces"), not formal Definition block. | N/A (positive, matches scaffold guidance) |

**Overall voice assessment:** Strong. The chapter consistently uses "you" for instructions and "we" for shared reasoning. Tone is that of an experienced colleague. No banned phrases found. No passive-voice clusters. The few low-severity items are stylistic polish, not voice violations.

---

## 3. Technical Accuracy

### Code listing vs. lab source comparison

| Chapter listing | Lab source region | Match? | Notes |
|----------------|-------------------|--------|-------|
| `obs_inspector` (lines 138-157) | `env_anatomy.py:obs_inspector` (lines 49-67) | PARTIAL | Chapter uses `gym.make(env_id)` directly; lab uses `_make_env(env_id, seed)` which sets `MUJOCO_GL=disable` and imports `gymnasium_robotics`. Chapter also adds `import gymnasium as gym` and `import numpy as np` implicitly. The function body logic matches, but the chapter version would fail without `import gymnasium as gym` and `import gymnasium_robotics`. See Issue T1 below. |
| `action_explorer` (lines 256-280) | `env_anatomy.py:action_explorer` (lines 76-99) | PARTIAL | Same `gym.make` vs `_make_env` discrepancy. Core logic matches. |
| `goal_space` (lines 188-209) | `env_anatomy.py:goal_space` (lines 108-131) | PARTIAL | Same pattern. Logic matches. |
| `dense_reward_check` (lines 318-338) | `env_anatomy.py:dense_reward_check` (lines 140-161) | PARTIAL | Same pattern. Lab version returns `atol` in result dict; chapter version matches the lab logic. |
| `sparse_reward_check` (lines 365-391) | `env_anatomy.py:sparse_reward_check` (lines 170-196) | PARTIAL | Same pattern. Logic matches. Lab has slightly different formatting of the `cr_r` assignment (uses line continuation). |
| `relabel_check` (lines 432-455) | `env_anatomy.py:relabel_check` (lines 205-228) | PARTIAL | Same pattern. Logic matches. Lab returns `"total"` key; chapter does not. |
| `cross_env_compare` (lines 494-517) | `env_anatomy.py:cross_env_compare` (lines 237-265) | PARTIAL | Same pattern. Lab returns additional keys (`expected_dim`, `dim_ok`); chapter version is simpler. |

**Issue T1 (Medium):** All code listings in the chapter use `gym.make()` but the lab source uses `_make_env()` (a helper that sets `MUJOCO_GL=disable` and imports `gymnasium_robotics`). The chapter code says `# (from scripts/labs/env_anatomy.py:obs_inspector)` suggesting these are snippet-includes, but they are actually hand-written adaptations. This is not necessarily wrong -- simplified listings are fine for the book -- but the comment claiming provenance from the lab file is misleading since the code does not literally match. The reader who goes to the lab file expecting identical code will find differences.

**Recommendation:** Either (a) use actual snippet-includes (`--8<--` syntax) with the lab regions (which would require the lab functions to use `gym.make` or the chapter to explain the `_make_env` wrapper), or (b) change the comment from "from scripts/labs/env_anatomy.py:obs_inspector" to something like "adapted from scripts/labs/env_anatomy.py" to set expectations correctly.

**Issue T2 (Low):** The chapter code listings do not include `import gymnasium as gym` or `import numpy as np`. The listings assume these are available but never show the imports. For a Manning book, a brief note at the start of the Build It section saying "all code below assumes `import gymnasium as gym` and `import numpy as np`" would help readers who type along.

### Numerical claims

| Claim | Location | Correct? |
|-------|----------|----------|
| 10D observation for FetchReach | Lines 165, 228 | Correct |
| 25D observation for FetchPush/PickAndPlace | Lines 489, 528 | Correct |
| 4D action space in [-1,1] | Lines 239-247 | Correct |
| Dense reward = -\|\|ag - dg\|\| | Line 304 | Correct |
| Sparse reward: 0 if dist <= 0.05, else -1 | Lines 349-352 | Correct |
| Success threshold epsilon = 0.05m | Line 349 | Correct |
| Episode length = 50 steps | Line 111 | Correct |
| Random success rate 0.0-0.1 | Line 668 | Correct (consistent with scaffold) |
| Random return_mean -25 to -15 | Line 669 | Correct (scaffold says -25 to -10; chapter says -25 to -15 in one place and -15 to -25 in another. Minor inconsistency -- see T3.) |
| Workspace bounds x:[1.1,1.5], y:[0.5,1.0], z:[0.35,0.75] | Lines 90-103 | Consistent with lab code workspace check |

**Issue T3 (Low):** The chapter states `return_mean: -25 to -15` in the Run It table (line 669) but `return_mean in [-25, -10]` is stated in the scaffold experiment card. The chapter's range is tighter. This is fine if intentional (tighter is more accurate), but it introduces a minor inconsistency with the experiment card shown in the chapter itself (line 638), which says "return_mean in [-25, -10]." The experiment card inside the chapter should use the same range as the body text or vice versa.

### Symbol definitions

All symbols in equations are defined before or at the point of use:

- $g_a$ (achieved goal) -- defined line 306
- $g_d$ (desired goal) -- defined line 306
- $\|\cdot\|_2$ (Euclidean norm) -- defined line 306
- $\epsilon$ (success threshold) -- defined line 349 with value 0.05
- $\phi$ (goal-achievement mapping) -- defined line 76, elaborated line 182
- $\mathcal{S}, \mathcal{A}, \mathcal{G}, P, R, \gamma$ -- all defined lines 70-78

No undefined symbols found.

---

## 4. Structural Completeness

| Section | Present? | Length (approx words) | Notes |
|---------|----------|-----------------------|-------|
| Opening promise | Yes | ~100 (5 bullets) | Good; each bullet starts with a gerund as required |
| Chapter bridge | Yes | ~100 (3 sentences, lines 11-15) | Covers capability, gap, addition, foreshadow. Slightly compressed -- could be 1-2 sentences longer to develop the gap more. |
| 2.1 WHY | Yes | ~1,100 | Strong. Opens with felt problem (10-hour wasted training). Includes the cost table. Includes GCMDP description. |
| 2.2 Fetch task family | Yes | ~550 | Good. ASCII workspace diagram, episode structure, reward variants. |
| 2.3-2.7 Build It (5 sections) | Yes | ~3,600 total | All 7 scaffold components present across 5 sections. Observation + goal space merged into 2.3 (as scaffold suggested). |
| 2.8 Bridge | Yes | ~450 | Connects manual lab to production script. Shows `--bridge` command and expected output. |
| 2.9 Run It + Experiment Card | Yes | ~750 | Experiment card present. Three-stage pipeline described. Artifact interpretation included. |
| 2.10 What Can Go Wrong | Yes | ~700 | 8 failure modes with symptom/cause/fix. Comprehensive. |
| 2.11 Summary | Yes | ~400 | 7-bullet summary + bridge to Ch3. |
| Verify It | Yes | ~350 | Replaces Reproduce It. Exact commands, expected outputs, lab verification commands. |
| Exercises | Yes | 4 exercises (~500) | Graduated: Verify, Tweak, Extend, Challenge. Expected outcomes provided. |
| **Total** | -- | **~8,025** | Within 6,000-10,000 target range |

**Structural assessment:** All required sections present. Lengths are appropriate. No section feels bloated or vestigial.

---

## 5. Content Gap Analysis

### Lost from tutorial

| Tutorial content | Present in chapter? | Assessment |
|-----------------|--------------------| -----------|
| Formal abstract | No | Correct (cut per scaffold, replaced by Opening Promise) |
| Formal Definition blocks (GCMDP, Goal-Conditioned Observation, Reward Recomputation Proposition) | No (replaced by conversational "seven pieces" description) | Correct (cut per scaffold) |
| Part IV formal propositions (HER sufficient conditions, Corollary about non-goal envs) | Partially kept (section 2.6 discusses why relabeling works informally) | Correct -- formal propositions cut, key insight retained |
| Appendix B "Formal Verification of HER Requirements" | No | Correct (redundant with Build It component 6) |
| "Part" structure (six parts) | Flattened to 2.1-2.11 | Correct |
| Section 0.2 robot spec table and real robot photos | No | Correct (cut per scaffold -- "compress to ~2-3 sentences") |
| Section 0.2 MuJoCo physics description (~400 words) | Compressed to ~2 sentences in 2.2 (line 86) | Correct |
| Section 0.1 Docker command explanation | No | Correct (already covered in Ch1 Manning chapter) |
| Tutorial section 0.4 four-task table | Kept briefly in 2.2 (line 82 ff.) | Correct |
| Appendix A observation component tables | Kept in 2.3 (10D table) and 2.7 (25D table) | Correct |
| `list-envs` subcommand | Cut | Correct (not needed when env IDs are stated directly) |
| Sparse vs Dense trade-off discussion (tutorial 4.2) | Partially present in 2.5 (dense/sparse explanations) and 2.6 (relabeling motivation) | **WARN**: The tutorial's explicit pro/con table for dense vs sparse rewards (tutorial lines 463-478) is not present in the chapter. The chapter explains both reward types but does not present the trade-off explicitly. This is a minor gap -- the information is distributed across sections 2.5 and 2.6 but never consolidated. |
| FetchSlide environment | Not mentioned except implicitly ("the Fetch family includes Push, PickAndPlace, and Slide" at line 484) | **WARN**: Slide is mentioned in passing in section 2.7 but not included in the cross-environment comparison code. The scaffold also omits it (only Reach/Push/PickAndPlace). This is probably intentional (Slide is not used until later chapters) but worth noting. |

### Missing from scaffold

| Scaffold item | Present? | Notes |
|--------------|----------|-------|
| 7 Build It components | Yes, all 7 | Components mapped to sections 2.3-2.7 |
| Experiment Card | Yes | Section 2.9 |
| Verify It block | Yes | After section 2.11 |
| Chapter Bridge (4 parts) | Yes | Lines 11-15 |
| Opening Promise | Yes | Lines 4-9 |
| What Can Go Wrong | Yes | Section 2.10 |
| Bridging proof | Yes | Section 2.8 |
| Cross-environment comparison | Yes | Section 2.7 |
| ASCII workspace diagram | Yes | Section 2.2, lines 90-103 |
| "What does the policy network see?" summary table | Yes | Section 2.3, lines 222-230 |
| 4 exercises | Yes | Lines 842-875 |
| Scaffold "is_success is True immediately after reset" failure mode | No | Not in What Can Go Wrong. **Minor gap** -- this is an edge case the scaffold lists but the chapter omits. |

### Unplanned additions

| Addition | Location | Assessment |
|----------|----------|-----------|
| Workspace scale analysis ("8% of workspace width") | Section 2.2, line 105 | Good addition -- gives concrete intuition for why random baseline is low |
| Episode return number walkthrough (sparse: $40 \times (-1) + 10 \times 0 = -40$) | Section 2.5, lines 354-357 | Good addition -- helps reader understand return arithmetic |
| Action non-linearity warning | Section 2.4, line 293 | Good addition -- prevents a common misconception |
| "Random baseline as diagnostic tool" subsection | Section 2.9, lines 676-681 | Good addition -- makes the baseline feel purposeful, not just a formality |

All unplanned additions are valuable. None detract from the chapter.

---

## 6. Reader Experience Simulation

| Reader profile | Path | Blocking issue? | Section |
|---------------|------|-----------------|---------|
| "Understand deeply" | Build It (all sections) -> Bridge -> Run It -> Exercises | **No blocking issues.** The interleaved concept->code->checkpoint flow works well. The reader can follow the derivation (GCMDP -> observation -> actions -> rewards -> relabeling -> cross-env) and verify at each step. The bridge section explicitly connects manual work to the production script. | -- |
| "Learn and practice" | Build It + checkpoints -> Run It | **Minor issue:** The chapter does not explicitly state that all Build It code can be run via `--verify` until the Verify It block at the very end. A reader who wants to run code as they read section 2.3 would benefit from an earlier mention of `python scripts/labs/env_anatomy.py --verify` -- perhaps in a tip at the start of section 2.3. | 2.3 |
| "Results first" | Run It -> skim Build It | **No blocking issues.** Run It (section 2.9) is self-contained enough. The experiment card gives exact commands and expected outputs. A reader skipping Build It would miss the *why* but could still produce artifacts. | -- |
| "Plane / no GPU" | Build It on CPU | **No blocking issues.** All code runs on CPU. The Verify It block explicitly states "Any machine with Docker (no GPU required)" (line 798). Time estimates are given (< 2 min for --verify, < 1 min for lab checks). | -- |

**Additional observation:** A reader typing code directly from the listings (rather than running the lab file) would encounter issues because the listings use `gym.make()` but do not show the `import gymnasium as gym` statement. This is a minor friction point for the "type-along" reader (see Issue T2 above).

---

## 7. Summary Verdict

### **READY** -- Minor issues only. Can go to editorial after fixes.

The chapter is well-structured, voice-compliant, technically accurate, and delivers on all scaffold commitments. Build It is genuinely the narrative spine. The interleaved concept-code-checkpoint pattern works effectively for an Environment chapter. The bridge section connects manual inspection to the production pipeline cleanly. All seven Build It components are present with verification checkpoints.

### Top 5 issues to fix (prioritized)

**1. Code listing provenance comments are misleading (T1, Medium).**
All seven code listings say `# (from scripts/labs/env_anatomy.py:<region>)` but the code uses `gym.make()` while the lab uses `_make_env()`. The listings are adaptations, not verbatim includes. Either use actual snippet-includes or change the comment to "adapted from" to avoid confusing readers who check the lab file.

**2. Return range inconsistency between body and experiment card (T3, Low).**
The Run It metrics table says `return_mean: -25 to -15` (line 669) but the experiment card says `return_mean in [-25, -10]` (line 638). Pick one range and use it consistently.

**3. Missing "is_success True at reset" failure mode from scaffold.**
The scaffold lists this edge case ("Goal happened to be sampled at gripper position -- rare, < 5% of the time") in its What Can Go Wrong table. The chapter's section 2.10 covers 8 other failure modes but omits this one. Add it for completeness.

**4. Missing imports note for type-along readers (T2, Low).**
The Build It listings omit `import gymnasium as gym` and `import numpy as np`. Add a brief note before the first listing (e.g., "All code below assumes `import gymnasium as gym` and `import numpy as np`") or show the imports once.

**5. Self-conscious meta-commentary (Low, two instances).**
Line 416 ("This section is the pedagogical climax of the chapter") and line 558 ("This is worth appreciating") are mildly self-conscious. The content should speak for itself. Consider removing or rewording.

### Items that are NOT issues

- The word "non-trivial" (lines 438, 460) is used in its technical sense, not as a minimizing phrase. No action needed.
- The chapter is ~8,025 words, comfortably within the 6,000-10,000 target. No length concerns.
- FetchSlide omission from cross-environment comparison is intentional and fine for this chapter.
- The sparse vs. dense trade-off is not consolidated into a single table (as in the tutorial) but the information is present across sections 2.5-2.6. This is a stylistic choice, not a gap.
