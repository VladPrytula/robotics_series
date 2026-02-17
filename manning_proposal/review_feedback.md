# Proposal Review: Tutorials -> Manning Book Drafts

## Executive Summary
- The proposal aligns with the tutorials and codebase: Fetch task family, artifact-first workflow, and evaluation discipline are consistent across `manning_proposal/` and `tutorials/`.
- Readers can build from scratch using `scripts/labs/` modules (PyTorch/NumPy only) referenced via snippet-includes in tutorials.
- Recommendation: draft two sample chapters now. Strong pairs: (1) Proof of Life and (5) HER on sparse tasks, or (1) Proof of Life and (3) PPO (dense Reach) if you prefer a lighter compute sample.

## Tutorials -> Chapters Mapping (ready now)
- Ch1 Proof of Life -> `tutorials/ch00_containerized_dgx_proof_of_life.md`
- Ch2 Env Anatomy -> `tutorials/ch01_fetch_env_anatomy.md`
- Ch3 PPO on dense Reach -> `tutorials/ch02_ppo_dense_reach.md`
- Ch4 SAC on dense Reach -> `tutorials/ch03_sac_dense_reach.md`
- Ch5 HER on sparse Reach/Push -> `tutorials/ch04_her_sparse_reach_push.md`
- Ch6 PickAndPlace (capstone tasks) -> `tutorials/ch05_pick_and_place.md`

These map cleanly to `manning_proposal/toc.md` Parts 1–3.

## From-Scratch Readiness
- Labs are standalone, pedagogical implementations with verification:
  - PPO: `scripts/labs/ppo_from_scratch.py` (regions: actor_critic_network, gae_computation, ppo_loss, value_loss, ppo_update)
  - SAC: `scripts/labs/sac_from_scratch.py` (replay_buffer, twin_q_network, gaussian_policy, twin_q_loss, actor_loss, temperature_loss, sac_update)
  - HER: `scripts/labs/her_relabeler.py` (data_structures, goal_sampling, relabel_transition, her_buffer_insert)
  - Curriculum: `scripts/labs/curriculum_wrapper.py` (curriculum_wrapper, curriculum_schedule, integration_example)
- Tutorials include these regions via `pymdownx.snippets`, keeping code shown in-book short and sourced from labs (single source of truth).
- Each lab exposes `--verify` and small `--demo` modes, making the Build It pathway runnable in minutes.

## Feedback on Proposal Documents
- Positioning and differentiators (artifact-first, no-vibes evaluation, single coherent task family) are strong and consistent with the repo. See `manning_proposal/overview.md`.
- ToC is cohesive and achievable with your current codebase. See `manning_proposal/toc.md`.
- Sample outlines emphasize measurable objectives and artifacts. Excellent fit for Manning’s style. See `manning_proposal/sample_chapters_outline.md`.

## Recommendations: Drafting Strategy
1) Write two sample chapters now (submission-ready):
   - Chapter 1 Proof of Life (short, accessible; establishes environment + artifact contract).
   - Chapter 5 HER on sparse goals (high-signal chapter demonstrating methodology and evaluation rigor).

2) Make the Build It subsections explicit inside each sample draft:
   - Call out the exact lab region names being included.
   - Include the `--verify` command and an "Expected output" block (numbers + 1–2 lines of interpretation).
   - End with a 5–10 minute exercise (tweak one parameter and interpret outcome).

3) Keep a strict two-track contract per chapter:
   - Build It: labs-only, minimal compute, quick verification (`--verify`, tiny `--demo`).
   - Run It: SB3-based scripts for full experiments, artifacts (checkpoints, JSON metrics, optional videos).

4) Preserve the evaluation protocol in every chapter:
   - Deterministic evaluation, fixed seeds or seed ranges.
   - JSON artifacts for success, return, distance, time-to-success, and smoothness.
   - Stress tests reported with degradation curves (where applicable).

## Suggested Edits/Enhancements Before Submission
- In `manning_proposal/sample_chapters_outline.md`, add explicit Build It subsections per chapter that list:
  - The lab regions to include (by name).
  - The verify/demo commands and expected outputs.
  - A short end-of-chapter exercise.
- In Chapter 1 (sample draft), add a clear hardware note: GPU recommended; CPU-only readers can use the checkpoint track and fast paths.
- Add a short non-interactive Docker note (no TTY) for CI, mirroring your pattern in `AGENTS.md` and tutorials.
- Keep snippet-excerpts to <= 30–40 lines; move runnable details to labs.
- Up front, state the two-track principle explicitly: Build It teaches mechanics; Run It delivers reproducible, scale-ready results.

## Risks and Mitigations
- Pixels chapters (ToC 10–12) are compute-heavy. Mitigation: provide a checkpoint-only track and minimal fast path (reduced steps), and clearly state expectations.
- Reader confusion between SB3 and from-scratch code. Mitigation: keep Build It sections strictly labs only; use SB3 only in Run It sections and clearly label them.
- Overlong code excerpts. Mitigation: enforce snippet length; rely on labs for full code and link to regions.

## Proposed Sample Chapter Skeletons

Chapter 1 (Proof of Life)
- Objectives: environment boot, single rendered frame, short PPO run, artifact check.
- Build It snippets: minimal render or metrics sanity check (small function from a lab if helpful).
- Run It command:
  bash docker/dev.sh python scripts/ch00_proof_of_life.py all
- Artifacts to show: `smoke_frame.png`, `ppo_smoke.zip`, minimal evaluation JSON.
- Exercise: change seed; force EGL vs OSMesa; document any changes.

Chapter 5 (HER on Sparse Reach/Push)
- Objectives: define sparse reward and HER, run HER vs no-HER under fixed protocol, verify relabeling.
- Build It snippets from `scripts/labs/her_relabeler.py`: `data_structures`, `goal_sampling`, `relabel_transition`, `her_buffer_insert`.
- Quick verify:
  bash docker/dev.sh python scripts/labs/her_relabeler.py --verify
- Run It fast path (single seed, <=500k steps):
  bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchReach-v4 --seeds 0 --total-steps 500000
  bash docker/dev.sh python scripts/ch04_her_sparse_reach_push.py env-all --env FetchPush-v4  --seeds 0 --ent-coef 0.1 --total-steps 500000
- Artifacts to show: `checkpoints/sac_*`, `checkpoints/sac_her_*`, `results/ch04_*_eval.json`.
- Exercise: sweep `n_sampled_goal` (2/4/8) on sparse Push and interpret success vs relabel rate.

## Immediate Action Items (Next 1–2 days)
- Draft Chapter 1 and Chapter 5 using the skeletons above.
- Update `manning_proposal/sample_chapters_outline.md` to include explicit Build It subsections and expected-output blocks.
- Prepare a small set of pretrained checkpoints for the checkpoint-only track (HER and no-HER for Reach/Push) and note their paths.
- Add a short, reusable sidebar for non-interactive Docker execution to reference across chapters.

## Longer-Horizon Prep (for later ToC parts)
- Pixels: define a minimal visual encoder baseline and a checkpoint track; plan ablations for augmentations.
- Robustness: finalize degradation-curve plotting and metrics schema for controlled perturbations.
- Reporting: standardize the "experiment card" template for consistent results sections.

---
This review reflects the repository as of the current commit set and aligns with the conventions in `AGENTS.md` and `CLAUDE.md`. It is intentionally scoped to actionable guidance for proposal submission and early drafting.

