# Revision Log: Chapter 3 -- Revision 001

**Date:** 2026-02-17
**Scope:** figures
**Chapter:** Manning/chapters/ch03_ppo_dense_reach.md

## Changes Made

| # | Location | Change | Reason |
|---|----------|--------|--------|
| 1 | Opening paragraph, "train a policy that reaches 100% success" | Added "(see Figure 3.1)" reference | Connect opening to environment screenshot |
| 2 | After opening bridge, before Section 3.1 | Inserted Figure 3.1 (FetchReachDense-v4 annotated screenshot with caption and alt text) | Figure plan item 1: establishes visual context |
| 3 | Section 3.1, clipping table introduction | Changed "What this does:" to "What this does (also illustrated in Figure 3.2):" | Connect prose to clipping diagram |
| 4 | Section 3.5, after PPO loss checkpoint block | Inserted Figure 3.2 (PPO clipping diagram with caption and alt text) | Figure plan item 2: visualize clipping mechanism |
| 5 | Section 3.8, demo mode paragraph | Added "Figure 3.3 shows the resulting learning curve:" | Connect prose to learning curve figure |
| 6 | Section 3.8, after demo results table | Inserted Figure 3.3 (PPO demo learning curve with caption and alt text) | Figure plan item 3: validate from-scratch implementation |

## Files Modified

- `Manning/chapters/ch03_ppo_dense_reach.md` -- 3 figure insertions, 3 text reference additions
- `Manning/scaffolds/ch03_scaffold.md` -- Added Figure Plan section (3 figures)
- `Manning/revisions/ch03_revision_001.md` -- This file (new)

## Verification

- [x] All inserted figures have numbered captions with generation commands
- [x] All figures are referenced by number in the text
- [x] No existing prose was modified outside the revision scope
- [x] ASCII compliance maintained
- [x] Figure count meets chapter-type minimum (Algorithm: 3-5 figures; we have 3)
