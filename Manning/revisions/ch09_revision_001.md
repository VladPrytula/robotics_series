# Revision Log: Chapter 09 -- Revision 001

**Date:** 2026-02-26
**Scope:** review-fixes
**Chapter:** Manning/chapters/ch09_pixel_push.md
**Review:** Manning/reviews/ch09_review.md

## Changes Made

| # | Severity | Location | Change | Reason |
|---|----------|----------|--------|--------|
| 1 | BLOCKER | Reproduce It block (~line 855) | Replaced fabricated "0.95 +/- 0.03 (3 seeds x 100 episodes)" with honest "0.95 (seed 0, 100 episodes)" + pending note for seeds 1-2 | Only seed 0 completed; seeds 1-2 queued. Reporting variance from nonexistent data violates quantified honesty commitment. |
| 2 | MAJOR | Section 9.7, between Steps 2 and 4 | Added "Step 3: Reading the training curve (the patience tax)" sub-section with ~3 paragraphs linking to Section 9.8 | Investigation jumped from Step 2 to Step 4, making the "five-step" claim a lie. Step 3 (interpretation) is defined in ch09_chapter_arc.md. |
| 3 | MAJOR | Section 9.7 summary + Section 9.8 | Swapped Figure 9.3 and 9.4 numbers: investigation summary is now Figure 9.3 (appears first in text), three-phase loss is now Figure 9.4 (appears second) | Figures must appear in sequential order. Figure 9.4 appeared before 9.3 in the text. |
| 4 | MAJOR | Section 9.7 summary table | Added Step 3 row: "Interpret Step 2's loss signature / (see Section 9.8) / Rising losses = good news" | Summary table skipped Step 3, inconsistent with "five-step" claim. |
| 5 | MAJOR | Section 9.7 Figure 9.3 caption | Updated caption to mention all 5 steps; corrected claim about Steps 0/1 failure causes (architecture wrong AND gradient routing wrong, not just one) | Original caption was inaccurate: Step 0 fails because of both architecture and gradient routing, Step 1 fails because of gradient routing only. |
| 6 | MAJOR | Listing 9.4 caption | Added "(from scripts/labs/visual_encoder.py:nature_cnn)" lab file reference | visual_encoder.py was unreferenced in the chapter despite being in the scaffold's lab file list. |
| 7 | MAJOR | Section 9.6 Bridge | Added --bridge command reference and description of numerical bridging proof (same observation through from-scratch vs SB3, max_diff < 1e-6) | Bridge section lacked the numerical bridging proof required by the persona (Section 2.5). |
| 8 | MAJOR | Section 9.7 Step 4 | Clarified that Step 4 uses identical config to Step 2 except DrQ is enabled; added parenthetical noting Step 2 showed upward trend by 1.4M for comparison | Review flagged potential confound: original Cell D used different buffer/HER config. The run command already had matched params, but text did not explicitly state the match. |
| 9 | MINOR | Section 9.10 | Changed "12 components" to "13 components across 5 lab files" | Chapter has 13 listings (9.1-9.13) and 13 bullet points in the summary. |
| 10 | MINOR | Section 9.6 Bridge | Changed "12 components" to "13 components" | Consistency with Section 9.10 fix. |
| 11 | MINOR | Section 9.6 Bridge | Added --demo mode invitation with run command and explanation (~4 lines) | Persona requires --demo mode walkthrough; chapter mentioned it only in passing. |
| 12 | MINOR | Reproduce It block | Added "(GPU; not feasible on CPU -- use the checkpoint track instead)" to time estimate | Reader on CPU needs to know Reproduce It is GPU-only. |
| 13 | MINOR | Section 9.6 Bridge verify commands | Added `image_augmentation.py --verify` to the four-command verification list | image_augmentation.py was missing from Bridge verification, unlike the other lab files. |
| 14 | NIT | Section 9.1, challenge #1 | Changed "roughly 3-4 pixels" to "roughly 4 pixels" for gripper | Inconsistent gripper size across the chapter. Standardized to 4 pixels (matches Figure 9.1 caption). |
| 15 | NIT | Section 9.6 TensorBoard metric table | Changed "SAC automatic tuning (Ch4)" to "SAC automatic tuning (Ch3)" | SAC/entropy tuning is introduced in Ch3 (SAC chapter), not Ch4 (HER chapter). |
| 16 | NIT | After Listing 9.3 (PixelReplayBuffer) | Added clarification: "This illustrates the principle. SB3's DictReplayBuffer implements the same uint8 storage pattern automatically..." | PixelReplayBuffer is a teaching example, not SB3-integrated. Needed clarification to avoid confusing readers about what SB3 actually uses. |

## Items NOT Addressed (by design)

| # | Severity | Item | Why deferred |
|---|----------|------|-------------|
| 10 (review) | MINOR | Memory sidebar 80 GB vs scaffold 40 GB | Chapter math is correct (80 GB). Scaffold needs updating, not the chapter. |
| -- | -- | visual_encoder.py regions beyond nature_cnn | Other regions (normalized_combined_extractor, visual_goal_encoder, etc.) are out of scope for this chapter. Scaffold should be updated to reflect this. |

## Files Modified

- `Manning/chapters/ch09_pixel_push.md` -- 16 targeted edits (insertions + modifications)
- `Manning/revisions/ch09_revision_001.md` -- This file (new)

## Verification

- [x] Every addressed issue is documented with its review reference
- [x] Only BLOCKER, MAJOR, MINOR, and NIT items addressed
- [x] Changes are minimal (fix the issue, nothing more)
- [x] No existing prose was modified outside the revision scope
- [x] ASCII compliance maintained (no em-dash, no curly quotes)
- [x] All changes documented in this revision log
- [x] Files modified list is accurate
