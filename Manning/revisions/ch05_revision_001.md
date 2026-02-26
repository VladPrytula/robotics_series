# Revision Log: Chapter 05 -- Revision 001

**Date:** 2026-02-26
**Scope:** review-fixes
**Chapter:** Manning/chapters/ch05_her_sparse_reach.md

## Changes Made

| # | Location | Change | Reason |
|---|----------|--------|--------|
| 1 | Section 5.8, line 612 | Renamed "Figure 5.5" to "Figure 5.4" (Push env screenshot) | Figure ordering: was out of sequence (appeared before Figure 5.4) |
| 2 | Section 5.8, line 616 | Renamed caption "Figure 5.5:" to "Figure 5.4:" | Same as above |
| 3 | Section 5.8, line 701 | Renamed "Figure 5.4" to "Figure 5.5" (learning curves) | Figure ordering: swapped to restore sequential order |
| 4 | Section 5.8, line 705 | Renamed caption "Figure 5.4:" to "Figure 5.5:" | Same as above |
| 5 | Section 5.6, line 498 | Changed "--verify" expected output "~16%" to "~4%" | Actual lab output shows 4.1% success, not 16% |
| 6 | Section 5.6, line 499 | Changed "50 -> ~210" to "50 -> ~218" | Actual lab output shows 218 transitions |
| 7 | Section 5.2, line 119 | Changed "~16-80%" to "~4-80%" with clarification | Lower bound did not match actual lab output |
| 8 | Section 5.2, line 155 | Changed "~16%" to "~4-8%" | Same consistency fix |
| 9 | Section 5.6, line 469 | Changed "~16%" to "~4-8%" with expanded range note | Same consistency fix |
| 10 | Section 5.10, line 799 | Changed "16-80%" to "4-80%" | Same consistency fix |
| 11 | Section 5.3, line 519 | Changed Figure 5.3 caption "~16%" to "~4-8%" | Same consistency fix |
| 12 | Section 5.1, line 38 | Added "Figure 5.1 makes this failure visible" text reference | Reviewer flagged missing explicit figure references |
| 13 | Section 5.2, line 173 | Added "Figure 5.2 illustrates how relabeling transforms..." text reference | Same as above |
| 14 | Section 5.6, line 515 | Added "Figure 5.3 visualizes the amplification effect" text reference | Same as above |
| 15 | Section 5.3, line 215 | Added note about RANDOM strategy omission from listing | Reviewer flagged that source file has RANDOM member not in chapter |

## Files Modified

- `Manning/chapters/ch05_her_sparse_reach.md` -- 15 modifications
- `Manning/revisions/ch05_revision_001.md` -- This file (new)

## Verification

- [x] All figures now in sequential order (5.1 -> 5.2 -> 5.3 -> 5.4 -> 5.5)
- [x] All figures have explicit text references by number
- [x] No "~16%" references remain in the chapter
- [x] Expected --verify output matches actual lab output (~4%, 218 transitions)
- [x] GoalStrategy.RANDOM omission noted in the text
- [x] ASCII compliance maintained (no em-dash, curly quotes, etc.)
- [x] No existing prose modified outside the revision scope
