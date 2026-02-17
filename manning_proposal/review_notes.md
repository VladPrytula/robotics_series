# Proposal Review Notes (Codex)

This document records findings, feedback, and concrete next steps after reviewing `manning_proposal/` in the context of the existing `tutorials/` + runnable repo.

## High-level assessment

The core idea is strong for a Manning "In Action" book: a single, coherent robotics RL task family (Fetch) plus an engineering-grade experiment contract (reproducible runs, fixed-seed evaluation, JSON artifacts). The draft already differentiates on "no vibes" evaluation and debuggability, which is a real market gap.

The main improvement needed for a book proposal is to make the "from scratch" / hands-on promise explicit and believable: not just "here are scripts", but a structured pathway where readers can (a) reproduce results, (b) implement core pieces themselves, and (c) verify correctness with targeted checks.

## What already works (keep)

- Clear positioning and audience: "robotics RL engineering playbook" for ML/software engineers, not a theory textbook.
- Stable spine: Fetch tasks stay constant while capabilities evolve (dense -> off-policy -> HER -> capstone -> robustness -> pixels).
- Evaluation-first narrative: artifacts (checkpoints + metadata + metrics JSON) as the unit of progress.
- Proposal structure is already close to what Manning expects: topic, audience, why now, competition, ToC, sample chapter outlines.

## Tutorial + repo reuse: what is usable for a book

The current repo already supports a three-lane pedagogy that maps well to a book:

- Run It: `scripts/chNN_*.py` provide end-to-end runs with artifacts.
- Build It: `scripts/labs/*.py` provide minimal from-scratch implementations (PPO/SAC/HER/curriculum) with snippet-able regions.
- Verify It: labs include `--verify` sanity checks with expected outputs; chapter scripts produce measurable eval JSON.

This is a strong foundation for a "from scratch" claim -- as long as the proposal and sample chapters explicitly foreground it (and define what "from scratch" means).

## Gaps / risks to address before submitting

1) Define "from scratch" precisely (avoid reviewer mismatch).
   - If a reviewer interprets "from scratch" as "no SB3", the proposal must say: we build the *core algorithm mechanics* from scratch (losses, updates, buffers, relabeling) but use a stable harness (Gymnasium/MuJoCo/PyTorch) and SB3-backed scripts for full-scale robotics training runs.
   - Make this a feature: Build It teaches internals; Run It achieves deliverable robotics results; Verify It prevents silent failure.

2) Reduce platform-specific framing in early sample chapters.
   - The tutorial content is DGX-forward in tone. In the book, prefer "Linux + NVIDIA GPU is recommended; CPU/Mac can run early chapters; later chapters provide checkpoint track."
   - Keep Docker-first, but present it as portable rather than cluster-specific.

3) Evidence and compute budget.
   - Your fast path / full run / checkpoint track concept is exactly the right mitigation for RL compute cost; it should be explained as a core product decision in the proposal, not a side note.
   - In sample chapters, include at least one small table of "expected outcomes" (not exact numbers, but directional + typical ranges) to build credibility.

4) Rights/licensing for visuals.
   - Some tutorial chapters embed externally sourced images. For a book submission, plan to replace those with your own diagrams/screenshots or confirm licensing.

5) Missing proposal-template items.
   - Manning's template typically expects: author bio/credibility, competing courses/online resources (not just books), marketing plan/audience reach, and a writing schedule. These can live in separate proposal files, but they should exist somewhere in `manning_proposal/` before submission.

## How to frame "Run It / Build It / Verify It" in the proposal

Recommended phrasing (paraphrase, keep consistent everywhere):

- Run It: reproducible end-to-end pipelines that produce auditable artifacts.
- Build It: minimal implementations of core ideas (PPO/SAC/HER) with explicit tensors and readable code.
- Verify It: small correctness checks with expected outputs, plus evaluation protocols that generate comparable JSON metrics across seeds.

Crucially: present this as a reader choice of depth, not three separate books.

## Suggested updates to `manning_proposal/` package (concrete)

1) Add a short "Pedagogy Tracks" subsection (or bullet list) to the proposal overview.
   - This has been started in `manning_proposal/overview.md`; keep it consistent across sample chapter prose.

2) Update the sample chapter deliverables to include all three tracks explicitly.
   - Chapter 1 sample: Run It (proof-of-life script), Build It (a tiny wrapper/metric computation), Verify It (reward/obs checks + expected files).
   - Chapter 5 sample: Run It (HER vs no-HER experiment), Build It (HER relabeling core), Verify It (relabeling correctness + "positive transition fraction increases" check).

3) Add one file: `manning_proposal/submission_checklist.md`.
   - Checklist: target reader, prerequisites, repo link, reproducibility story, compute tracks, sample chapters, author bio, marketing plan, schedule, competing titles.
   - This makes the submission package easy to audit.

4) Add one file: `manning_proposal/from_scratch_definition.md`.
   - One page that defines what is and is not rebuilt from scratch, to preempt reviewer confusion.

5) Add one file: `manning_proposal/author_and_promotion.md`.
   - Bio, why you, where readers come from (blog, Twitter/X, GitHub, talks, courses), and how you will support readers during launch.

## What to draft next (recommended order)

1) Draft Chapter 1 in full prose (sample submission) with a strong Run It / Build It / Verify It arc.
   - Goal: reviewers should finish Chapter 1 believing the book will be runnable and hands-on.

2) Draft one "algorithm chapter" sample in full prose (either PPO or HER).
   - If the mandate is "from scratch", PPO is the cleanest place to demonstrate "math -> tensors -> update step" before robotics complexity.
   - HER is the strongest differentiator for goal-conditioned robotics; it is a good second sample if you want to maximize uniqueness.

3) Only after samples: iterate the ToC language for consistency.
   - Ensure each chapter promise reads like a deliverable, not just topics.

## Optional, high-leverage refinements (if time)

- Create a single recurring "Experiment Card" template (inputs, command, artifacts, expected result, failure modes). Use it in every chapter.
- Add one page in the proposal about "failure-driven pedagogy" (how readers debug flat curves and silent failures).
- Decide early what content is "in the book" vs "online appendix" to keep page count under control (pixels + sim-to-real can balloon).

