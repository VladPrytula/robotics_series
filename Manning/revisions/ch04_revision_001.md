# Revision Log: Chapter 4 -- Revision 001

**Date:** 2026-02-26
**Scope:** Review fixes + multi-seed results update
**Chapter:** Manning/chapters/ch04_sac_dense_reach.md
**Review:** Manning/reviews/ch04_review.md (verdict: REVISE)

## Changes Made

### From review (5 priority issues)

| # | Location | Change | Reason |
|---|----------|--------|--------|
| 1 | After Section 4.9 checkpoint | Added `--verify` command section with expected output block | Review issue #1 (blocking): Build It capstone was missing |
| 2 | Sections 4.9 and 4.11 | Swapped figure numbering: demo curve 4.4->4.3, PPO vs SAC 4.3->4.4; updated all references | Review issue #2 (medium): figures numbered out of document order |
| 3 | Section 4.6 code listing | Changed `tq1, tq2` to `target_q1, target_q2` | Review issue #3 (medium): variable names now match math-to-code table and lab source |
| 4 | Section 4.9 sac_update listing | Split 41-line listing into two: (a) gradient steps (25 lines), (b) Polyak averaging (10 lines), with narrative paragraph between | Review issue #4 (medium): exceeded 30-line listing guideline |
| 5 | Before Section 4.4 | Added MLP helper class note explaining the `MLP` class used in subsequent listings | Review issue #5 (low-medium): class used but never shown |

### From review (minor items)

| # | Location | Change | Reason |
|---|----------|--------|--------|
| 6 | Section 4.9 demo command | Added CPU time estimate (~5 min for 50k steps) | Review minor: "Plane / no GPU" reader had no CPU estimate |
| 7 | Section 4.5 code listing | Added intermediate `std = log_std.exp()` variable | Review minor: aligns chapter code with lab source exactly |

### From multi-seed verification (Phase 4)

| # | Location | Change | Reason |
|---|----------|--------|--------|
| 8 | Reproduce It block (line 914-915) | Updated return_mean from `-1.06 +/- 0.15` to `-0.93 +/- 0.13`; final_distance_mean from `0.019 +/- 0.003` to `0.016 +/- 0.003` | Actual 3-seed results computed from eval JSONs |

## Multi-Seed Results

Seed 0 was trained before the chapter was written; seeds 1 and 2 were trained
during the writing session. All 3 seeds converged to 100% success.

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean +/- Std |
|--------|--------|--------|--------|-------------|
| success_rate | 1.00 | 1.00 | 1.00 | 1.00 +/- 0.00 |
| return_mean | -1.056 | -0.792 | -0.944 | -0.93 +/- 0.13 |
| final_distance_mean | 0.0186 | 0.0129 | 0.0160 | 0.016 +/- 0.003 |

Note: The Run It section's results table (line 777) still shows seed 0 values
(-1.06 return, 18.6mm distance) because it documents the single-seed experience
the reader will get when they first run the training script. The Reproduce It
block now reflects the true multi-seed aggregate.

## Files Modified

- `Manning/chapters/ch04_sac_dense_reach.md` -- 7 review fixes + 1 results update
- `Manning/scaffolds/ch04_scaffold.md` -- Updated Reproduce It results to match actuals
- `Manning/revisions/ch04_revision_001.md` -- This file (new)

## Verification

- [x] All 5 review priority issues addressed
- [x] All 2 review minor items addressed
- [x] Multi-seed results computed from actual eval JSONs (3 seeds x 100 episodes)
- [x] Reproduce It block matches actual cross-seed statistics
- [x] Run It section correctly shows seed 0 single-run results
- [x] Figure numbering is sequential (4.1, 4.2, 4.3, 4.4)
- [x] Variable names in Section 4.6 match lab source and math-to-code table
- [x] sac_update listing split into two listings under 30 lines each
- [x] `--verify` command present with expected output
- [x] ASCII compliance maintained throughout
