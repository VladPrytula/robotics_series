# Review: Chapter 1 -- Proof of Life

**Reviewer:** Manning Reviewer Agent (Phase 3)
**Draft reviewed:** `Manning/chapters/ch01_proof_of_life.md`
**Scaffold reference:** No formal scaffold file; adaptation notes in `manning_proposal/chapter_production_protocol.md` (Chapter 1 section) used as scaffold reference.
**Source tutorial:** `tutorials/ch00_containerized_dgx_proof_of_life.md`
**Date:** 2026-02-17
**Word count:** ~8,256 words (~20 printed pages)

---

## 1. Checklist Audit

### From-scratch contract

- [x] **Build It is the narrative spine** -- PASS. Build It spans section 1.5 (subsections 1.5.1-1.5.4), approximately 1,800 words. For a Setup chapter (not an Algorithm chapter), this is appropriate. The chapter explicitly frames Build It as the "understand the pieces" step before the "run the pipeline" step (section 1.0 intro, line 17).
- [x] **Every equation has corresponding code listing** -- PASS (with caveat). There are no formal equations in this chapter (appropriate for a Setup chapter). The "equations" are the reward computation (`-distance`) and the success threshold check (`distance < 0.05`), both of which have code listings in sections 1.5.2 and 1.5.3.
- [x] **Bridging proof present** -- PASS. Section 1.5.2 verifies `compute_reward(ag, dg, info) == env.step() reward`. Section 1.5.4 ("The bridge: Build It meets Run It") explicitly connects Build It findings to the production pipeline. This is the correct bridging proof for a Setup chapter per the protocol.
- [x] **Run It refers back to Build It** -- PASS. Section 1.6 ("What 'proof of life' means") at line 508 explicitly lists the Build It findings alongside the Run It test results. Section 1.5.4 bridges forward to Run It. The connection is clear.
- [x] **SB3 is never introduced as a black box** -- PASS. When PPO and SB3 appear in section 1.6 (Test 4), the chapter says PPO is "a policy gradient method we derive and implement from scratch in Chapter 3" (line 445). When `MultiInputPolicy` appears (line 483), the chapter explains what it means in terms of the observation structure the reader just inspected. SB3 is introduced as a tool whose internals the reader will later understand, not as a black box.

### Structure

- [x] **Every term defined before first use** -- PASS. Terms like PPO, SAC, HER, SB3, MuJoCo, EGL, OSMesa, dense/sparse rewards, and `compute_reward` are all introduced with inline definitions or forward references before being used substantively. Forward references use the "we introduce X in Chapter N" pattern correctly.
- [x] **Every equation has all symbols defined** -- N/A (no formal equations). The distance computation (`np.linalg.norm(...)`) and threshold comparison are self-evident from the code.
- [x] **Every code listing under 30 lines** -- PASS. The longest listing is the `compute_reward` verification at approximately 12 lines. All listings are well within the 30-line limit.
- [x] **Every code listing has a verification checkpoint** -- PASS. Sections 1.5.1, 1.5.2, and 1.5.3 each end with a "Checkpoint" block (lines 252, 295, 344) that tells the reader what to check and what values to expect.
- [x] **Experiment Card present** -- PASS. Section 1.6 opens with a properly formatted Experiment Card (lines 371-389).
- [x] **Chapter bridge present (except Ch1)** -- N/A. Ch1 correctly has no bridge (the protocol says "Skip for Ch1").
- [x] **"What Can Go Wrong" section present** -- PASS. Section 1.7 covers 7 failure modes with Symptom/Cause/Fix structure (lines 517-600), approximately 1,000 words.

### Voice

- [x] **Numbers, not adjectives for all performance claims** -- PASS. The chapter avoids vague performance claims. Where numbers appear (50,000 steps, 5 cm threshold, 500,000 steps for full training, ~600 fps vs ~60-100 fps), they are concrete. No claim of "works well" or "performs nicely."
- [x] **ASCII only** -- PASS. No non-ASCII characters detected. The chapter uses `--` for dashes, straight quotes throughout, and `...` for ellipsis.
- [x] **No third-person author references** -- PASS. No "Professor Prytula," no "the author," no "one."
- [ ] **No "simply," "trivially," "obviously"** -- WARN. Line 54 uses "less obvious" in context: "the answers become less obvious -- and the questions become more valuable." This is borderline acceptable -- it is not saying something IS obvious, but rather that something becomes LESS obvious. The word is used naturally, not dismissively. Low severity.
- [x] **Scope boundaries have specific references** -- PASS. Forward references to later chapters are specific: "Chapter 2" for reward formalization, "Chapter 3" for PPO, "Chapter 4" for SAC, "Chapter 5" for HER. Henderson et al. (2018) is cited specifically for the reproducibility crisis. No orphaned "beyond the scope" statements found.

### Compute accessibility

- [ ] **Build It `--verify` runs on CPU in under 2 minutes** -- WARN. There is no `--verify` mode for this chapter. The Build It content is inline Python snippets, not a lab module with `--verify`/`--demo`/`--bridge` modes. This is noted as appropriate for a Setup chapter in the protocol (Build It is "env inspection, manual reward calls"), but the absence means there is no single command a reader can run to execute all Build It checks. See Content Gap Analysis.
- [ ] **Build It `--demo` runs on CPU in under 30 minutes** -- WARN. Same as above -- no `--demo` mode. The Run It pipeline (`ch00_proof_of_life.py all`) serves as the functional equivalent (~5-10 min on CPU), but it is not labeled as "demo."
- [x] **Chapter completable via Build It + checkpoint track** -- PASS. The Build It section requires only `gym.make()` + basic Python (no GPU needed). The Run It section runs in ~5-10 min on CPU. No checkpoint track is needed because the pipeline is fast enough to always run in full (the Reproduce It block says this explicitly, line 644).
- [x] **Reproduce It block present** -- PASS. Lines 621-647. Includes the exact command, hardware note, time estimate, artifact paths, and a results summary. Correctly notes that no checkpoint track is needed for this chapter.

---

## 2. Voice Compliance

| Line/Section | Issue | Severity |
|---|---|---|
| Line 54 (Section 1.1) | "the answers become less obvious" -- uses "obvious" but in a natural comparative form, not dismissively. Not a banned-phrase violation, but worth noting. | Low |
| Section 1.2, line 59 | "originally developed at the University of Washington and now maintained by Google DeepMind" -- informational detail about MuJoCo's provenance. Accurate but slightly academic; "In Action" readers are unlikely to care about institutional history. | Low |
| Section 1.3, line 100 | "a web-based visualization dashboard for monitoring training runs" -- parenthetical definition of TensorBoard is slightly condescending for the target audience (ML engineers who likely know TensorBoard). Could be dropped. | Low |
| Section 1.3, line 113 | Henderson et al. (2018) is cited a second time (first at line 36-37). Repetition is acceptable because the two citations serve different purposes (first: silent failure; second: provenance), but the reader may notice the overlap. | Low |
| Section 1.4, "Docker as environment specification" | The 94%/12% scenario (line 146) is vivid and effective. Good voice. | N/A (positive) |
| Section 1.4, "The two-layer architecture" | The explanation of the Docker layer architecture (lines 152-157) is clear and well-pitched for the audience. | N/A (positive) |
| Section 1.5.2, line 293 | "The policy learns from corrupted data, and training may silently fail or converge to a bad policy." -- Strong, specific, correctly motivates the invariant check. Good voice. | N/A (positive) |
| Section 1.6, line 445 | "a policy gradient method we derive and implement from scratch in Chapter 3" -- Correct forward reference pattern. Sets reader expectations without black-boxing. | N/A (positive) |
| Section 1.8 (Summary), lines 603-616 | The summary is strong. It lists concrete capabilities, names what the reader does NOT yet know, and bridges to Chapter 2 with a clear gap statement. Very effective. | N/A (positive) |

**Overall voice assessment:** The chapter reads as an experienced colleague explaining the setup process. The voice is consistently conversational, specific, and honest. No major persona violations detected. The tone is slightly more detailed than typical "In Action" chapters (see Structural Completeness), but the voice register is correct.

---

## 3. Technical Accuracy

| Location | Issue | Impact |
|---|---|---|
| Section 1.2, line 67 | Chapter reference table maps `FetchReachDense-v4` to "Ch1-4" and `FetchReach-v4` to "Ch5". This is consistent with the book's stated plan, but note that the Manning book chapter numbering differs from the tutorial numbering. Verify this mapping is correct against the final ToC. | Low (editorial) |
| Section 1.5.1, line 244 | Observation shape `(10,)` for FetchReachDense-v4. This is correct for gymnasium-robotics Fetch environments (gripper_pos 3 + gripper_vel 3 + finger_pos 2 + finger_vel 2 = 10). | N/A (verified correct) |
| Section 1.5.2, lines 283-289 | Expected output shows `step_reward: -1.058329`. This is a seed-dependent value (seed=42, first random action). It will be correct if the reader uses the exact same gymnasium-robotics version and seed. The chapter should note that the exact numerical value may differ across versions -- only the `Match: True` result matters. | Medium |
| Section 1.5.2, line 272 | Uses `env.unwrapped.compute_reward(...)` -- the `.unwrapped` is necessary because `gym.make()` wraps the environment. This is correct but not explained. A reader who tries `env.compute_reward(...)` without `.unwrapped` will get an AttributeError. | Medium |
| Section 1.5.3, line 322 | Claims threshold is "5 cm (0.05 meters)" for FetchReach. This is correct per gymnasium-robotics defaults (`distance_threshold=0.05`). | N/A (verified correct) |
| Section 1.6, Experiment Card (line 376) | Lists `Fast path: ~5 min (GPU) / ~10 min (CPU)`. This is for the entire `all` pipeline including build. Reasonable estimate. | N/A (reasonable) |
| Section 1.6, line 447 | States PPO smoke test uses "50,000 timesteps on FetchReachDense-v4 with 8 parallel environments." The script (`ch00_proof_of_life.py`) defaults to these values. Consistent. | N/A (verified correct) |
| Section 1.4, line 196 | States Mac uses `python:3.11-slim` base. The script does auto-detect platform. Consistent with `CLAUDE.md`. | N/A (verified correct) |
| Section 1.6, line 483 | Observation space print output shows `Box(-inf, inf, (3,), ...)`. The `...` is informal -- the actual dtype would be `float64`. Acceptable for illustration. | Low |
| Exercise 1, line 657-658 | The `--seed` flag passed to `ch00_proof_of_life.py all` -- verified that the script accepts this flag. | N/A (verified correct) |
| Exercise 2, lines 668-669 | Environment variable override pattern `MUJOCO_GL=egl bash docker/dev.sh ...` -- this sets the variable on the HOST, which gets passed into the container via `docker/dev.sh`. Need to verify that `dev.sh` passes through this env var. The script does set `MUJOCO_GL` inside the container, so the host-side override may be overwritten. | Medium |

---

## 4. Structural Completeness

| Section | Present? | Length | Notes |
|---|---|---|---|
| Opening promise | Yes | 5 bullets | Lines 3-9. All bullets start with gerunds. Clear, actionable outcomes. Good. |
| Bridge | N/A (Ch1) | -- | Correctly omitted per protocol. |
| WHY | Yes | ~1,400 words | Sections 1.1 (silent failures + Hadamard questions). Effective problem motivation. Within the ~1,500 word target. |
| Fetch task family (background) | Yes | ~900 words | Section 1.2. Provides necessary context about the environments. Well-structured table. |
| Experiment contract | Yes | ~700 words | Section 1.3. Introduces the "no vibes" rule and artifact contract. Good structural element. |
| Setup / prerequisites | Yes | ~2,100 words | Section 1.4. Docker, dev.sh, venv, Mac support. Comprehensive. |
| HOW / Build It (interleaved) | Yes | ~1,800 words, 4 components | Section 1.5 (1.5.1-1.5.4). Components: obs dict, compute_reward, success signal, bridge. Each has verification checkpoint. |
| Bridge section | Yes | ~400 words | Section 1.5.4. Connects Build It findings to Run It pipeline. |
| Run It + Experiment Card | Yes | ~2,000 words | Section 1.6. Experiment Card + 4-test walkthrough + artifact interpretation + dependency chain. |
| What Can Go Wrong | Yes | ~1,000 words | Section 1.7. 7 failure modes with Symptom/Cause/Fix. |
| Summary | Yes | ~350 words | Section 1.8. Lists concrete capabilities, names gaps, bridges to Ch2. |
| Reproduce It | Yes | ~150 words | Lines 621-647. Correct format. Notes no checkpoint track needed. |
| Exercises | Yes | 4 exercises | Lines 650-699. Verify, Tweak, Extend, Challenge. Good graduation. |

**Structural assessment:** All required sections are present and well-formed. The chapter is at the upper end of the recommended length (~8,250 words vs. the protocol's estimate of ~8,500). The WHY section is within target. The Build It section is appropriately lighter than what Algorithm chapters will have (4 components vs. the 4-8 typical for Algorithm chapters).

One structural note: Section 1.2 (Fetch task family) and Section 1.3 (experiment contract) sit between WHY and Build It. These are background/context sections that are necessary but somewhat extend the "preamble before hands-on work." The reader does not touch code until section 1.5, which is roughly 3,500 words into the chapter. For an "In Action" reader who wants to DO something, this may feel like a long runway. See Reader Experience Simulation.

---

## 5. Content Gap Analysis

### Lost from tutorial

- **Formal Definition blocks for Container and Image.** The tutorial (lines 62-67) includes formal "Definition (Container)" and "Definition (Image)" blocks. The book chapter drops these per the protocol's adaptation notes: "Drop formal Definition blocks for Container, Image." This is correct.
- **Well-posedness formalism.** The tutorial devotes ~2 pages to formal well-posedness (Definition, three conditions, mathematical notation). The book chapter reduces this to a practical framing via the three Hadamard questions (~1 paragraph). This is correct per protocol: "Drop Hadamard well-posedness formalism for reproducibility (2 pages -> 2 sentences)."
- **Appendix B (Environment Variable Reference).** The tutorial includes a standalone table of env vars (MUJOCO_GL, PYOPENGL_PLATFORM, NVIDIA_DRIVER_CAPABILITIES). The book chapter integrates this as an inline TIP in section 1.4 (line 184). This is correct per protocol: "Cut Appendix B (env variable reference) into an inline tip."
- **Detailed Mac platform comparison tables (collapsible).** The tutorial has a detailed collapsible table comparing Mac vs DGX results. The book chapter has a static comparison table (lines 200-206) plus inline narrative. Adequate -- the collapsible `<details>` format is a web convention that does not translate to print.
- **Appendix A troubleshooting.** Merged into section 1.7 "What Can Go Wrong." Correct per protocol.

**Nothing important was lost.** All intentional cuts follow the protocol's adaptation notes.

### Missing from scaffold (adaptation notes)

- [ ] **"Compress Mac M4 section into a sidebar."** The protocol says to compress the Mac section into a sidebar. The chapter instead includes it as a full subsection of section 1.4 (lines 195-212), running approximately 500 words. This is not compressed into a sidebar -- it is inline narrative. **WARN:** The Mac section is concise relative to the tutorial (which had ~100 lines on Mac), but it is still a full subsection rather than a sidebar callout. For a Manning print layout, a sidebar would keep the main narrative flow cleaner for the 80%+ of readers who are on Linux/NVIDIA. This is a medium-severity structural issue.
- [x] **"Introduce the 3 Hadamard engineering questions here."** Done in section 1.1 (lines 44-54). Uses the practical framing from the persona (Section 5). Includes all three questions and notes they will be reused as a checklist. Correct.
- [x] **"Build It: env inspection + manual compute_reward."** Done in section 1.5. Four components: obs dict, compute_reward, success signal, bridge. Matches the protocol's specification.
- [x] **"Bridge: manual compute_reward == env step reward."** Done in section 1.5.2 and reinforced in 1.5.4. The critical invariant is clearly stated and verified.
- [x] **"Merge Appendix A into What Can Go Wrong."** Done. Section 1.7 covers all tutorial troubleshooting items plus additional ones.

### Unplanned additions

- **Section 1.2 (Task family overview).** The tutorial does not include a substantial overview of the Fetch environment family -- it defers this to Ch1 (tutorial numbering). The book chapter includes a ~900-word section with a difficulty ladder table, observation/action/reward descriptions, and workspace bounds. This is a **good addition** -- it grounds the reader in what they are working with before the Build It section. However, it partially overlaps with what would be Chapter 2's content (Environment Anatomy). The overlap is manageable because the treatment here is introductory rather than thorough, but the Ch2 author should be aware.
- **Section 1.3 (Experiment contract / "no vibes" rule).** The tutorial does not have a dedicated section on the experiment contract. The book chapter introduces the artifact table, the "no vibes" rule, and the train/eval CLI contract. This is a **good addition** -- it establishes a recurring element that pays off in every later chapter. It belongs in Ch1.
- **Section 1.5.3 (Success signal parsing).** The tutorial's verification tests check observation structure and rendering but do not include a dedicated section on the `is_success` signal. The book chapter adds this as a Build It component. This is a **good addition** -- the success signal is the metric the entire book is built around. Inspecting it early is valuable.
- **Relationship paragraph at end of 1.5.3 (lines 336-343).** The chapter draws a three-part relationship (observations -> rewards -> success) that is pedagogically strong. Good unplanned addition.

---

## 6. Reader Experience Simulation

| Reader profile | Blocking issue | Section |
|---|---|---|
| "Understand deeply" | No blocking issue. The Build It section provides hands-on inspection with checkpoints. The bridge section (1.5.4) connects pieces to the production pipeline. The exercises provide further exploration. **Good path.** | -- |
| "Understand deeply" | Minor: Section 1.5.2 expected output (line 283-289) shows specific floating-point values that are seed- and version-dependent. A meticulous reader who gets different values may worry something is wrong, even though the `Match: True` result is what matters. The chapter should note that exact values vary. | 1.5.2 |
| "Learn and practice" | The reader does not touch code until section 1.5, which is ~3,500 words in. Sections 1.2 (task family) and 1.3 (experiment contract) are valuable context but delay the first hands-on moment. A "Learn and practice" reader may skim these and feel uncertain about what they missed. | 1.2-1.3 |
| "Learn and practice" | Minor: Build It code snippets are standalone (not connected to a single script). The reader must either run them interactively (`python -c "..."`) or create a scratch script. No guidance on which approach to use. The Checkpoint block in 1.5.1 (line 252) mentions `bash docker/dev.sh python -c "..."` but the others do not repeat this. | 1.5.1-1.5.3 |
| "Results first" | This reader skips Build It and goes to Run It (section 1.6). The Run It section is **self-contained for this chapter** -- the Experiment Card, the four-test walkthrough, and the artifact interpretation do not require Build It knowledge. The "What proof of life means" subsection (line 499) summarizes Build It results alongside Run It results, but a skipping reader will not be confused. **Good path.** | -- |
| "Results first" | Minor: The Run It section refers to "`MultiInputPolicy` -- not a `MlpPolicy`" (line 483) and explains why, but a results-first reader who skipped 1.5.1 may not fully appreciate the distinction. This is acceptable -- the explanation is self-contained. | 1.6 |
| "Plane / no GPU" | The Build It section runs entirely on CPU (just `gym.make()` + basic Python). The Run It section's proof-of-life script also works on CPU (~10 min). **Good path.** No GPU required for any part of this chapter. | -- |
| "Plane / no GPU" | Minor: Exercise 2 (force rendering backend) requires Docker, which means the reader cannot do it on a plane without Docker pre-pulled. This is inherent to the Docker-first approach and not a real problem. | Exercise 2 |

---

## 7. Summary Verdict

### **READY** -- Minor issues only. Can go to editorial after fixes.

The chapter is well-written, structurally complete, and faithful to the Manning persona. It successfully adapts the formal tutorial voice to a conversational "experienced colleague" register. The Build It section is appropriately scoped for a Setup chapter. The bridge between Build It and Run It is clear and well-motivated. The What Can Go Wrong section is practical and comprehensive. The exercises are well-graduated.

### Top 5 issues to fix (prioritized)

1. **Expected output values in section 1.5.2 are version-dependent (Medium).** The `step_reward: -1.058329` value will differ across gymnasium-robotics versions and even minor dependency changes. Add a note: "Your exact values will differ -- the important result is that `Match: True`. The specific reward value depends on the random action sampled, which varies across library versions." This prevents a meticulous reader from thinking something is broken when they get `-0.973241` instead of `-1.058329`.

2. **`.unwrapped` in `compute_reward` call not explained (Medium).** Line 272 uses `env.unwrapped.compute_reward(...)` but does not explain why `.unwrapped` is needed. A reader who tries `env.compute_reward(...)` will get an `AttributeError`. Add one sentence: "We use `env.unwrapped` to access the base Fetch environment directly, bypassing Gymnasium's wrapper layers that do not expose `compute_reward`."

3. **Exercise 2 env var override may not work as written (Medium).** The command `MUJOCO_GL=egl bash docker/dev.sh python scripts/ch00_proof_of_life.py render` sets the env var on the host side. Whether `docker/dev.sh` passes this through to the container depends on its implementation. If it does not, the exercise silently does nothing. Verify this works or adjust the exercise to run the override inside the container: `bash docker/dev.sh bash -c 'MUJOCO_GL=egl python scripts/ch00_proof_of_life.py render'`.

4. **Mac section should be a sidebar, not inline narrative (Medium-Low).** The protocol specifies "Compress Mac M4 section into a sidebar." Currently it is a full inline subsection (~500 words). For the majority of readers (Linux/NVIDIA), this interrupts the setup narrative. Convert to a Manning sidebar or NOTE callout. This is a layout concern more than a content concern -- the content is good and concise.

5. **Long runway before first hands-on code (Low).** The reader hits ~3,500 words of context (sections 1.1-1.4) before the first code snippet in section 1.5. For "In Action" readers who want to DO something, consider adding a short "Quick start" callout after section 1.1 or in the chapter intro: "If you want to verify your setup immediately, jump to section 1.6 and run `bash docker/dev.sh python scripts/ch00_proof_of_life.py all`. Then come back here to understand what it checked." This gives impatient readers an escape hatch without disrupting the narrative.

### Minor items (fix if time)

- Line 54: "less obvious" -- technically uses the word "obvious" but in a natural comparative context. Not a real violation. No action needed.
- Line 100: TensorBoard parenthetical definition may be unnecessary for the target audience. Consider dropping "(a web-based visualization dashboard for monitoring training runs)."
- Line 113: Henderson et al. cited twice in the chapter (lines 36-37 and 113). The second citation adds the provenance angle, which is distinct from the first (silent failure). Acceptable but the reader may notice the repetition.
- Section 1.2 partially overlaps with what Ch2 (Environment Anatomy) will cover in more depth. Ensure Ch2 does not repeat the same overview -- it should go deeper, building on what Ch1 established.
- Build It code snippets are not connected to a single lab file. For a Setup chapter this is acceptable (the snippets are short and standalone), but it means there is no `scripts/labs/ch01_*.py` file that a reader can run with `--verify`. Consider whether a lightweight lab module would add value.

---

**End of review.**
