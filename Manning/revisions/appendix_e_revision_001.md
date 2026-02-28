# Revision Log: Appendix E -- Revision 001

**Date:** 2026-02-28
**Scope:** review-fixes
**Chapter:** Manning/chapters/appendix_e_isaac_manipulation.md
**Review:** Manning/reviews/appendix_e_review.md (verdict: REVISE)

## Changes Made

| # | Location | Change | Reason |
|---|----------|--------|--------|
| 1a | After training progression table in "State-Based Training (8M Steps)" | Inserted Figure E.1 (state-based learning curve with phase annotations) -- placeholder image reference, caption with generation command, alt-text, and text reference sentence | Review Issue #1: No figures (blocking). Scaffold Figure Plan item 1. |
| 1b | After wall-clock comparison table in "Wall-Clock Comparison" | Inserted Figure E.2 (wall-clock comparison bar chart, log-scale) -- placeholder image reference, caption with generation command, alt-text, and text reference sentence | Review Issue #1: Scaffold Figure Plan item 2. |
| 1c | After pixel training progression table in "Pixels on Isaac" | Inserted Figure E.3 (pixel hockey-stick learning curve) -- placeholder image reference, caption with generation command, alt-text, and text reference sentence | Review Issue #1: Scaffold Figure Plan item 3. |
| 1d | After curriculum crash data in "The Curriculum Crash" | Inserted Figure E.4 (curriculum crash diagnostic overlay) -- placeholder image reference, caption with generation command, alt-text, and text reference sentence | Review Issue #1: Scaffold Figure Plan item 4. |
| 2 | After "Verification Summary" section, before Bridge section | Added single sentence noting Ch3-4 `--demo` modes cover the SAC training demonstration | Review Issue #2: No `--demo` mode for SAC lab (minor). |
| 3 | "The Curriculum Crash" section in Run It | Replaced the detailed root-cause paragraph (CurriculumManager scaling explanation, reward magnitude analysis, Q-function divergence) with a brief cross-reference to the WHY "Hidden Curriculum" section. Kept the crash data, diagnostic tip, and fix. | Review Issue #3: Curriculum crash lesson appeared in 4 places (anti-pattern #8: repetitive hammering). WHY section retains full explanation. |
| 4 | Between "Why SAC Works on This Task" and "HOW: Build It Components" | Inserted blockquote "Getting started with Isaac Lab" sidebar covering build command, `dev-isaac.sh` entry point, NGC image size, and shader compilation warning | Review Issue #4: Setup/prerequisites gap. |
| 5 | Listing E.5 (SAC update step) | Expanded three semicolon-compressed optimizer sequences into separate lines (`zero_grad()`, `backward()`, `step()` each on own line). Listing grows from ~30 to ~36 lines. | Review Issue #5: Semicolons reduce readability in a teaching context. |

## Files Modified

- `Manning/chapters/appendix_e_isaac_manipulation.md` -- 5 issues addressed: 4 figure insertions, 1 sentence addition, 1 paragraph trimmed and replaced with cross-reference, 1 blockquote sidebar inserted, 1 code listing expanded
- `Manning/revisions/appendix_e_revision_001.md` -- This file (new)

## Verification

- [x] All 4 inserted figures have numbered captions (Figure E.1-E.4 format) with generation commands
- [x] All 4 figures are referenced by number in the text ("As Figure E.1 shows...", "Figure E.2 makes...", "Figure E.3 plots...", "Figure E.4 overlays...")
- [x] All 4 figures have descriptive alt-text
- [x] Figure count meets chapter-type minimum (4 figures for Algorithm + Engineering hybrid)
- [x] No existing prose was modified outside the revision scope
- [x] ASCII compliance maintained (no em-dash, no curly quotes, straight quotes only)
- [x] Every addressed issue is documented with its review reference
- [x] Only items from the review's top 5 issues were addressed
- [x] Changes are minimal (fix the issue, nothing more)
- [x] Curriculum crash explanation preserved in WHY section (unchanged) and Warning callout (unchanged)
- [x] Listing E.5 expanded without changing logic or variable names
