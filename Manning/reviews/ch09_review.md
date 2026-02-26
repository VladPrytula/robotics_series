# Review: Chapter 9 -- Pixels, No Cheating

## Summary Verdict

**REVISE.** The chapter is strong in its core narrative arc and technical substance. The 5-step investigation is well-paced and genuinely feels like discovery rather than recipe. The Build It sections are thorough, with 13 listings covering the full visual pipeline, and the three-phase loss signature analysis is original content that adds real value. The voice is clean -- no banned phrases, no third-person author references, proper pronoun usage throughout.

However, the chapter has several issues that need targeted fixing before editorial submission. The most significant: (1) the step numbering skips Step 3 entirely in the Run It section, creating a confusing gap since Step 3 is defined in the chapter arc document as the interpretation of Step 2's training curve -- that content exists in Section 9.8 but is not labeled as Step 3; (2) Figure 9.3 (three-phase loss) appears AFTER Figure 9.4 (investigation summary) in the text, violating sequential figure ordering; (3) the Reproduce It block claims "0.95 +/- 0.03 (3 seeds x 100 episodes)" but the training data shows only seed 0 completed -- seeds 1 and 2 were still queued at time of writing; (4) one of the five scaffold lab files (`scripts/labs/visual_encoder.py`) is never referenced in the chapter, and the NatureCNN listing appears to come from scratch rather than from that lab file; (5) there is no `--bridge` mode reference connecting Build It to SB3, which the persona document calls a required structural element.

## Structural Completeness

| Section | Present? | Length | Notes |
|---------|----------|-------|-------|
| Opening promise | Yes | 5 bullets | Well-crafted, specific, matches scaffold exactly |
| Bridge (from previous chapter) | Yes | 1 paragraph (lines 11-13) | Covers all 4 bridge elements in a single dense paragraph. Could benefit from slightly more breathing room. |
| 9.1 WHY | Yes | ~800 words | Four challenges, observation structure, visual HER, Hadamard check -- all present and well-organized |
| 9.2 Build It: Visual pipeline | Yes | ~600 words, 3 components | render_and_resize, PixelObservationWrapper, PixelReplayBuffer |
| 9.3 Build It: Encoder architecture | Yes | ~1100 words, 5 components | NatureCNN, ManipulationCNN, SpatialSoftmax, ManipulationExtractor, proprioception passthrough |
| 9.4 Build It: Data augmentation | Yes | ~600 words, 3 components | RandomShiftAug, DrQDictReplayBuffer, HerDrQDictReplayBuffer |
| 9.5 Build It: Gradient routing | Yes | ~900 words, 3 components | CriticEncoderCritic, CriticEncoderActor, DrQv2SACPolicy |
| 9.6 Bridge: From scratch to SB3 | Yes | ~400 words | Verification commands, TensorBoard metric mapping. No `--bridge` mode. |
| 9.7 Run It + Experiment Card | Yes | ~1200 words | 4 experimental steps (0, 1, 2, 4). Experiment Card present for Step 2. |
| 9.8 Reading the training curve | Yes | ~800 words | Three-phase loss table, hockey-stick sidebar, patience tax -- strong original content |
| 9.9 What Can Go Wrong | Yes | 10 items | Comprehensive, well-structured table |
| 9.10 Summary | Yes | ~500 words | 12 components listed, 3 principles, looking ahead |
| Reproduce It | Yes | ~350 words | Exact commands, hardware, time, artifacts, results |
| Exercises | Yes | 5 exercises | Good graduation: verify, tweak, explore, 2x challenge |
| Figures | 4 referenced | See visual content section below |

## Checklist Audit

### From-scratch contract
- [x] Build It is the narrative spine -- PASS (Sections 9.2-9.5, ~3100 words, 13 listings)
- [x] Every SB3 class preceded by from-scratch equivalent -- PASS (ManipulationExtractor before SB3 policy, CriticEncoderCritic before DrQv2SACPolicy)
- [ ] Bridging proof present -- WARN: Section 9.6 connects Build It to SB3 via `--verify` and `--probe` commands, and maps TensorBoard metrics to components. But there is no `--bridge` mode that directly compares from-scratch output to SB3 output on the same data. The persona (Section 2.5) specifies bridging proofs as "same computation, same data, same output." The current bridge is structural (showing correspondence) rather than numerical (demonstrating agreement).
- [x] Run It refers back to Build It -- PASS (e.g., "the encoder we built in Section 9.3" pattern used throughout Run It)
- [x] SB3 is never introduced as a black box -- PASS (reader builds every component SB3 uses)
- [ ] Can reader rewrite pipeline in JAX without SB3 -- WARN: The Build It components are complete for the visual pipeline and gradient routing. However, the SAC training loop itself (which chapter references to Ch4) is not re-derived here. A reader coming specifically to Ch9 would need Ch4's Build It to have the full picture. This is acceptable given the book's sequential structure, but worth noting.

### Voice compliance
- [x] No "simply/trivially/obviously" -- PASS (grep found zero instances)
- [x] ASCII only -- PASS (no Unicode characters found)
- [x] "you"/"we" pronouns only -- PASS (no "one", "the reader", "the practitioner", "Professor" found)
- [x] Numbers not adjectives -- PASS (all performance claims are quantified: "5-8%", "95%", "2.2x overhead")
- [x] No third-person author references -- PASS
- [x] Scope boundaries have specific references -- PASS (Nair et al. 2018, Kostrikov et al. 2020, Yarats et al. 2021, Laidlaw et al. 2024 all cited with specifics)

### Structure
- [x] Every term defined before first use -- PASS (sensor separation principle defined at first use, hockey-stick defined at first appearance, all Ch1-8 concepts used with implicit reminder)
- [x] Every equation has all symbols defined -- PASS (SpatialSoftmax equations define alpha, tau, pos_x, pos_y; Bellman backup defines all terms; memory formula is self-contained)
- [ ] Every code listing under 30 lines -- WARN: Listing 9.7 (ManipulationExtractor) is ~18 lines which is fine, but Listing 9.13 (DrQv2SACPolicy) is ~20 lines. All are under 30. PASS.
- [x] Every code listing has a verification checkpoint -- PASS (all 5 Build It subsections end with checkpoints)
- [x] Experiment Card present for major experiment -- PASS (Step 2 has full experiment card)
- [x] Chapter bridge present -- PASS
- [x] "What Can Go Wrong" section present -- PASS (10 items)
- [x] Reproduce It block present -- PASS (with exact commands, seeds, artifacts)

## From-Scratch Contract

The Build It sections carry the narrative effectively. The progression is logical: observation pipeline (how images become tensors) -> encoder architecture (how tensors become features) -> data augmentation (how features are regularized) -> gradient routing (how the encoder learns). Each section builds on the previous, and the math-before-code pattern is followed consistently.

The 13 listings across 4 Build It sections provide a complete implementation that could be ported to another framework. The key components -- ManipulationCNN, SpatialSoftmax, gradient routing overrides -- are fully self-contained. A reader who finishes Build It would understand:
- How to capture and process pixel observations
- Why stride-4 fails and stride-2 works for small objects
- How SpatialSoftmax extracts spatial coordinates
- Why the encoder must be in the critic optimizer, not the actor
- How DrQ augmentation works and when it conflicts with spatial features

The one gap is that the `scripts/labs/visual_encoder.py` lab file (listed in the scaffold with 5 regions: nature_cnn, normalized_combined_extractor, visual_goal_encoder, visual_gaussian_policy, visual_twin_q_network) is never referenced in the chapter. The NatureCNN listing (Listing 9.4) appears as inline code with no lab file reference. The scaffold assigns NatureCNN to `visual_encoder.py:nature_cnn`. This disconnect means the chapter's NatureCNN might not match the actual lab code, and readers looking for the lab file will not find a reference in the text.

## 5-Step Investigation Arc

The investigation arc is the chapter's strongest feature. Each step teaches exactly one concept and fails for a clear reason that motivates the next step. The pacing is excellent: Steps 0 and 1 are brisk (one paragraph each), Step 2 gets the full experiment card treatment (it is the breakthrough), and Step 4 provides a surprising negative result that deepens understanding.

However, there are structural issues:

1. **Missing Step 3.** The chapter arc document defines Step 3 as "the interpretation of Step 2's training curve" -- not a new experiment, but a pause to explain the three-phase loss signature. In the chapter, this content appears in Section 9.8 but is not labeled as Step 3. The investigation jumps from "Step 2" to "Step 4" in Section 9.7, which is confusing. Either label the interpretation as Step 3 within the investigation, or renumber Step 4 to Step 3 (and update the summary table and figure captions accordingly).

2. **Investigation summary inconsistencies.** The summary table (line 723-729) lists 5 rows (full-state, Step 0, 1, 2, 4) but the figure caption (line 721) says "five-step investigation" while there are only 4 numbered steps. The Figure 9.4 caption says "Steps 0 and 1 fail because the architecture and gradients are wrong" -- but Step 1 fails because of gradient routing, not architecture. Step 0 fails because of architecture AND gradient routing.

3. **Step 4 claim needs qualification.** The DrQ result (3% at 1.54M steps for the original Cell D, which used buffer=200K and HER-4) is compared against Step 2 (which used buffer=500K and HER-8). The chapter narrates this as "DrQ makes things worse" but the configs differ in buffer size and HER strength, not just DrQ. The Step 4 run command shown at line 702-704 does not include `--no-drq`, suggesting DrQ is the default -- which means this IS the correct comparison (same config as Step 2 but with DrQ on). However, the training data file reveals the original Cell D used buffer=200K and HER-4, not 500K and HER-8. If Step 4's actual run used the new matched config, this is fine. If it used the original Cell D data (200K buffer, HER-4), the comparison is confounded.

## Technical Accuracy

### Three-phase loss table

The chapter's three-phase loss table (line 744-748) matches the training data in `critic_encoder_status.md` (lines 427-432) well:

| Metric | Chapter | Training Data | Match? |
|--------|---------|---------------|--------|
| Phase 1 success_rate | 3-7% | Flat 3-7% | YES |
| Phase 1 critic_loss | Declining to 0.07 | 0.07 declining | YES |
| Phase 2 success_rate | 10-90% | Rising 10% -> 90% | YES |
| Phase 2 critic_loss | Rising to 0.3+ | 0.3-0.8 RISING | YES -- chapter says "0.3+" which is conservative; data says 0.3-0.8 |
| Phase 2 actor_loss | Rising to 1.0+ | 0.5-1.3 RISING | YES |
| Phase 3 success_rate | 95%+ | Saturating 95%+ | YES |
| Phase 3 critic_loss | Declining to 0.15-0.35 | 0.15-0.35 declining | YES |
| Phase 3 actor_loss | Negative (-0.2 to -0.7) | -0.2 to -0.7 NEGATIVE | YES |

### Hockey-stick onset

Chapter claims ~2.2M steps. Training data shows the upward trend appeared at ~1.4M-2M in the initial 2M run, with the clear hockey-stick confirmed at 2.44M (25-34%). The "~2.2M" claim is reasonable -- the onset is gradual and depends on what threshold you use. PASS.

### DrQ noise analysis

Chapter (lines 709-712): "+/-4 pixels via pad-and-crop. After CNN (84->21), this becomes +/-1 pixel on 21x21 feature map. SpatialSoftmax converts this to +/-0.10 in [-1,1] coordinate space."

The training data (line 131-133) says "+/-0.42 noise" and "57% noise-to-signal ratio." The chapter says "+/-0.10" and "40-80%." These differ. The discrepancy: the training data uses a different calculation (possibly including both obs and next_obs noise combined, or a different mapping from pixels to coordinates). The chapter's analysis is self-consistent (+/-4px on 84px = +/-1px on 21px; 1/21 * 2 range = 0.095 ~ 0.10 per axis), and the 40-80% range for the combined obs+next_obs noise-to-signal ratio is plausible. The training data's "57%" falls within the chapter's "40-80%" range. PASS, though the discrepancy between documents should be reconciled.

### Multi-seed results

The Reproduce It block claims "success_rate: 0.95 +/- 0.03 (3 seeds x 100 episodes)." The training data shows seed 0 RUNNING at 95%+ and seeds 1, 2 QUEUED. The scaffold's Reproduce It template had "+/- [PENDING]." The chapter reports a specific variance of 0.03 that cannot have been measured yet. This is a fabricated number.

### 95% vs 89% claim

The chapter claims "95%+ success" (line 687: "By 4.4M, it reaches 95%") and the summary says "matching state-based performance." State-based Push reached 89%. The pixel agent reaching 95% would exceed state-based performance, which is possible but worth noting. If this 95% is from a single seed and the true multi-seed mean is lower, the claim may need adjustment once seeds 1-2 complete.

## Visual Content

| Figure | Present? | Caption with generation command? | Referenced by number in text? | Notes |
|--------|----------|--------------------------------|------------------------------|-------|
| 9.1 (pixel observation) | Yes (line 24) | Yes | Yes (line 22) | Good |
| 9.2 (encoder comparison) | Yes (line 279) | Yes | Yes (line 277) | Good |
| 9.3 (three-phase loss) | Yes (line 740) | Yes | Yes (line 738) | PROBLEM: appears AFTER Figure 9.4 |
| 9.4 (investigation summary) | Yes (line 719) | Yes | Yes (line 717) | PROBLEM: appears BEFORE Figure 9.3 |

Figure ordering issue: Figure 9.4 (investigation summary) is in Section 9.7 (line 717-721) and Figure 9.3 (three-phase loss) is in Section 9.8 (line 738-742). The figures are numbered out of sequence in the text. Either renumber them (swap to 9.3 and 9.4) or reorder the sections so Figure 9.3 appears before Figure 9.4.

Colorblind-friendly palette: Cannot verify without seeing the actual figure files. The captions include generation commands, which is good. Note that Figures 9.1, 9.2, and 9.4 cite scripts as generation sources; Figure 9.3 cites TensorBoard logs directly, which is less reproducible (the reader would need to know how to export from TensorBoard).

## Specific Findings

### BLOCKER

1. **[BLOCKER] Fabricated multi-seed variance in Reproduce It (line 855).**
   Location: Reproduce It block, line 855.
   Issue: The chapter reports "success_rate: 0.95 +/- 0.03 (3 seeds x 100 episodes)" but training data shows only seed 0 completed. Seeds 1 and 2 were queued. The scaffold template had "+/- [PENDING]." Reporting a specific variance from data that does not exist violates the book's core commitment to quantified honesty.
   Fix: Either mark the variance as pending/TBD until seeds complete, or report only the single-seed result with appropriate caveats.

### MAJOR

2. **[MAJOR] Step numbering gap: Step 3 missing from investigation (Section 9.7).**
   Location: Section 9.7, lines 617-729.
   Issue: The investigation jumps from Step 2 (line 648) to Step 4 (line 695). The chapter arc document defines Step 3 as interpretation of the training curve, which appears in Section 9.8 but is not labeled as a step. The summary table (line 723) also skips Step 3. This makes the "five-step investigation" (promised in the opening and figure caption) actually four steps.
   Fix: Either incorporate Step 3 (interpretation/patience tax) into Section 9.7 before Step 4, or renumber Step 4 to Step 3 and update all references (summary table, figure captions, opening promise).

3. **[MAJOR] Figure numbering out of sequence (Figures 9.3 and 9.4).**
   Location: Figure 9.4 at line 719, Figure 9.3 at line 738.
   Issue: Figure 9.4 appears before Figure 9.3 in the text. Manning requires sequential figure numbering matching order of appearance.
   Fix: Swap the figure numbers so the investigation summary is Figure 9.3 and the three-phase loss is Figure 9.4, or reorder the sections.

4. **[MAJOR] Lab file `visual_encoder.py` unreferenced.**
   Location: Scaffold lists 5 lab files; chapter references only 4.
   Issue: The scaffold assigns NatureCNN to `scripts/labs/visual_encoder.py:nature_cnn` (component #4 in the scaffold table). The chapter's Listing 9.4 shows NatureCNN as inline code with no lab file reference. The other 4 regions in `visual_encoder.py` (normalized_combined_extractor, visual_goal_encoder, visual_gaussian_policy, visual_twin_q_network) are also unreferenced.
   Fix: Either reference the lab file for the NatureCNN listing, or update the scaffold to reflect that NatureCNN lives inline. The other 4 regions in `visual_encoder.py` may be out of scope for this chapter -- if so, the scaffold's lab file list should be corrected.

5. **[MAJOR] No `--bridge` mode in Section 9.6.**
   Location: Section 9.6, lines 581-609.
   Issue: The persona document (Section 2.5) specifies that each chapter needs a bridging proof where from-scratch output and SB3 output are compared on the same computation. Section 9.6 shows `--verify` and `--probe` commands but no numerical bridging proof (e.g., "feed the same observation through our ManipulationExtractor and SB3's features_extractor, compare output tensors"). The scaffold's Bridging Proof section (lines 152-196) describes a `--bridge` mode but the chapter does not reference it.
   Fix: Add a `--bridge` comparison (either as a lab mode or as inline output) showing that the from-scratch components produce identical results to what SB3 uses internally.

6. **[MAJOR] Step 4 (DrQ ablation) potentially confounded.**
   Location: Section 9.7, Step 4 (lines 695-713).
   Issue: The training data reveals Cell D (the DrQ run) used buffer_size=200K and HER-4, while Cell A (the no-DrQ run) used buffer_size=500K and HER-8. If Step 4's "result" uses Cell D data, the comparison is confounded by buffer size and HER strength, not just DrQ. The run command shown in the chapter (lines 702-704) does NOT include `--no-drq` but DOES include `--buffer-size 500000 --her-n-sampled-goal 8`, implying a properly matched run -- but it is unclear whether this run was actually executed.
   Fix: Clarify whether Step 4 used matched hyperparameters (buffer=500K, HER-8, DrQ ON) or the original Cell D data (buffer=200K, HER-4, DrQ ON). If the latter, either rerun with matched config or caveat the comparison.

### MINOR

7. **[MINOR] Figure 9.4 caption says "five-step" but shows four labeled steps.**
   Location: Line 721.
   Issue: Caption says "The five-step investigation" but the figure shows Steps 0, 1, 2, 4 (four steps). Related to finding #2 above.
   Fix: Reconcile caption with actual step count once Step 3 issue is resolved.

8. **[MINOR] Listing count: chapter claims "12 components" but has 13 listings.**
   Location: Section 9.10 (line 803: "What you built (12 components)") and the text lists 13 bullets (Listing 9.1 through 9.13).
   Issue: The count 12 may exclude the PixelReplayBuffer (Listing 9.3, which is a from-scratch teaching example not directly used by SB3). But the listing numbering goes to 13, and the summary lists 13 bullets. The scaffold table has 12 rows. There is an off-by-one confusion.
   Fix: Reconcile the component count. If 13, update "12 components" references. If 12, clarify which listing is supplementary.

9. **[MINOR] `--demo` mode only mentioned once in passing (line 661).**
   Location: Experiment Card, line 661.
   Issue: The persona specifies Build It should have `--demo` mode for short real training (Section 2.7). The chapter mentions `--demo: ~10 min CPU` parenthetically in the experiment card but never describes what it does or invites the reader to run it.
   Fix: Add a brief mention of `--demo` mode in the Bridge section (9.6) or after the Build It sections, explaining that readers can see the pipeline produce actual (short) learning curves on CPU.

10. **[MINOR] Memory sidebar gives ~80 GB for 500K buffer; scaffold says ~40 GB.**
    Location: Line 164 says "500,000 x 169,344 ~ 80 GB." Scaffold line 205 says "500K buffer needs ~40 GB."
    Issue: The chapter's math is correct (500K x 169KB ~ 80 GB). The scaffold's number is wrong (probably a stale estimate from before the full memory analysis). However, this means readers with <80 GB RAM cannot use the recommended buffer size. The What Can Go Wrong table (line 786) says "500K buffer needs ~85 GB" which is more accurate.
    Fix: No change needed in the chapter (the math is correct). The scaffold should be updated to match, but that is not the chapter author's concern.

11. **[MINOR] No CPU time estimate for Reproduce It.**
    Location: Reproduce It block, line 843-845.
    Issue: The block says "~40 hours per seed at ~30 fps" but does not mention that this is GPU time, nor give a CPU estimate. The persona (Section 2.7) says chapters should be completable via Build It + checkpoint track -- but if a reader tries Reproduce It on CPU, they need to know it is infeasible.
    Fix: Add "(GPU; not feasible on CPU -- use checkpoint track)" after the time estimate.

12. **[MINOR] `image_augmentation.py` lab file referenced only in a checkpoint line.**
    Location: Line 482 is the only reference to `scripts/labs/image_augmentation.py`.
    Issue: Unlike the other lab files (pixel_wrapper.py and manipulation_encoder.py, which are referenced in both checkpoints and the Bridge section), image_augmentation.py appears only in the Section 9.4 checkpoint. It is not mentioned in the Bridge section (9.6).
    Fix: Add `image_augmentation.py --verify` to the Bridge section's verification command list for completeness.

### NIT

13. **[NIT] Inconsistent "roughly" usage for puck size.**
    Location: Lines 22, 32, 234, 253.
    Issue: The puck is described as "roughly 5 pixels wide" (line 22), "roughly 5 pixels wide" (line 32), "roughly 1 pixel" after stride-4 (line 234), and "roughly 3 pixels" after stride-2 (line 253). The gripper alternates between "roughly 4 pixels" (line 26) and "roughly 3-4 pixels" (line 32). Minor inconsistency in the gripper size.
    Fix: Pick one and use it consistently (4 pixels seems to be the more common one).

14. **[NIT] Reference to "Ch4" in Section 9.5.4 (line 607).**
    Location: Line 607: "SAC automatic tuning (Ch4)."
    Issue: The entropy temperature tuning is introduced in Ch3 (SAC chapter) according to the concept registry, not Ch4 (HER chapter). If the book's chapter numbering has shifted, this reference may be wrong.
    Fix: Verify which book chapter introduces SAC automatic entropy tuning and update the cross-reference.

15. **[NIT] Missing `--verify` reference for `image_augmentation.py` in Bridge section.**
    Location: Section 9.6, lines 587-591.
    Issue: The Bridge section shows three verify commands (pixel_wrapper, manipulation_encoder, drqv2_sac_policy) but omits `image_augmentation.py --verify`. Since DrQ augmentation is a Build It component (Section 9.4), its verification should appear in the bridge.
    Fix: Add `bash docker/dev.sh python scripts/labs/image_augmentation.py --verify` to the Bridge section.

16. **[NIT] Listing 9.3 (PixelReplayBuffer) is a from-scratch teaching component, not SB3-integrated.**
    Location: Section 9.2.3, lines 168-193.
    Issue: The PixelReplayBuffer listing shows a from-scratch buffer that is not what SB3 actually uses (SB3 uses DictReplayBuffer with uint8 dtype). The text correctly explains this (line 166: "SB3's DictReplayBuffer already stores pixels as uint8"), but the listing creates the impression that the reader is building a component SB3 will use. This is the one listing that breaks the "Build It components are what SB3 uses" contract.
    Fix: Consider adding a note after the listing: "This illustrates the principle. SB3's DictReplayBuffer implements the same uint8 storage pattern automatically when the observation space dtype is np.uint8."

## Reader Experience Simulation

| Reader profile | Path | Potential issue |
|---------------|------|-----------------|
| "Understand deeply" | Build It + Bridge + Run It + Reproduce It | The `--bridge` mode is missing, so the numerical bridging proof (persona Section 2.5) is absent. Reader must trust that Build It components = SB3 internals without a direct comparison. |
| "Learn and practice" | Build It `--verify` + Run It fast path + Checkpoint | No `--demo` walkthrough means the reader never sees from-scratch components produce actual learning. They verify shapes and types but not training dynamics. |
| "Results first" | Run It + Checkpoint | Works well. Step 2's experiment card is clear. The checkpoint track path is explicitly mentioned. |
| "Plane / no GPU" | Build It `--verify` + Checkpoint | Works well. All `--verify` commands are CPU-compatible (< 2 min each). But the `--demo` mode (short training showing learning) is mentioned only in passing and no CPU time estimate is given. |

## Content Gap Analysis

### Lost from scaffold

- **`scripts/labs/visual_encoder.py` with 5 regions.** The scaffold lists this file with nature_cnn, normalized_combined_extractor, visual_goal_encoder, visual_gaussian_policy, and visual_twin_q_network. None are referenced in the chapter. NatureCNN appears as inline code. The other 4 regions are absent entirely.
- **`--bridge` mode.** The scaffold's Bridging Proof section (lines 152-196) describes a detailed bridge protocol. The chapter's Section 9.6 is weaker -- it has verify + probe commands but no numerical bridging proof.
- **`--demo` mode walkthrough.** The scaffold mentions it (line 28: "--demo: ~10 min CPU"). The chapter mentions it once in passing (line 661) but never invites the reader to use it.
- **Checkpoint track intermediate checkpoints.** The scaffold (line 42) mentions "500K interval checkpoints" for readers who want to evaluate at different training stages. The chapter's Reproduce It block includes `--checkpoint-freq 500000` in the command but does not explain the intermediate checkpoint use case.

### Missing from scaffold

- Nothing significant. The chapter follows the scaffold closely.

### Unplanned additions

- The chapter adds inline NatureCNN code (Listing 9.4) rather than using a lab file reference. This is acceptable pedagogically (showing the "wrong" encoder before the "right" one) but creates a discrepancy with the scaffold's lab file mapping.
- The proprioception passthrough (Section 9.3.5) gets its own subsection. The scaffold lists it as part of ManipulationExtractor (component #7). Giving it a dedicated subsection is a good pedagogical choice -- the sensor separation principle deserves emphasis.

## Recommended Actions

Priority-ordered list of fixes before editorial submission:

1. **Fix the fabricated multi-seed variance (BLOCKER).** Replace "0.95 +/- 0.03" with actual data. Either report the single-seed result with a note that multi-seed results are pending, or wait for seeds 1-2 to complete. The book's commitment to quantified honesty makes this non-negotiable.

2. **Resolve the Step 3 gap (MAJOR).** Either add Step 3 (interpretation/patience tax) as a labeled step within Section 9.7 before Step 4, or renumber to Steps 0-3 and update all references (summary table, figure captions, opening promise "5-step" claim). The current skip from Step 2 to Step 4 is confusing.

3. **Fix figure numbering (MAJOR).** Swap Figure 9.3 and 9.4 numbers so they appear in sequential order, or reorder the sections. Currently Figure 9.4 appears before Figure 9.3 in the text.

4. **Clarify Step 4 (DrQ ablation) config match (MAJOR).** Confirm the DrQ run used matched hyperparameters (buffer=500K, HER-8). If it used the old Cell D data (buffer=200K, HER-4), either rerun with matched config or explicitly caveat that the comparison is confounded.

5. **Add bridging proof or `--bridge` reference (MAJOR).** Section 9.6 needs either a numerical comparison (from-scratch vs SB3 on same input) or at minimum a reference to a `--bridge` lab mode that readers can run.

6. **Reference or reconcile `visual_encoder.py` (MAJOR).** Either add a reference to the lab file for NatureCNN, or update the scaffold to reflect the actual lab file mapping used in the chapter.

7. **Add `--demo` mode invitation (MINOR).** Tell readers they can run `--demo` to see short training curves on CPU. This addresses the "learn and practice" reader profile.

8. **Add `image_augmentation.py --verify` to Bridge section (NIT).** Complete the verification command list.
