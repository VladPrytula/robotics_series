# Scaffold: Appendix E -- Isaac Peg-In-Hole (GPU-Only)

## Classification
Type: Appendix (portability / simulator transfer)
Source tutorial: tutorials/appendix_e_isaac_peg.md
Book chapter output: Manning/chapters/appendix_e_isaac_peg.md (to be authored)
Lab files:
  - scripts/labs/isaac_sac_minimal.py (5 regions)
    - dict_flatten_encoder
    - squashed_gaussian_actor
    - twin_q_critic
    - sac_losses
    - sac_update_step
  - scripts/labs/isaac_goal_relabeler.py (4 regions)
    - goal_transition_structs
    - isaac_goal_sampling
    - isaac_relabel_transition
    - isaac_her_episode_processing
Production script:
  - scripts/appendix_e_isaac_peg.py

---

## Scope Boundary

This appendix demonstrates **method transfer**, not exhaustive Isaac tuning.
We keep the loop state-first and goal-conditioned where possible.

Required deliverables remain aligned with main chapters:

- checkpoint `.zip`
- checkpoint `.meta.json`
- eval `.json`
- comparison `.json`

---

## Build It Components

| # | Component | Lab region | Verify command |
|---|-----------|------------|----------------|
| 1 | Dict observation flattening | `isaac_sac_minimal.py:dict_flatten_encoder` | `bash docker/dev-isaac.sh python3 scripts/labs/isaac_sac_minimal.py --verify` |
| 2 | Squashed Gaussian actor | `isaac_sac_minimal.py:squashed_gaussian_actor` | same as above |
| 3 | Twin critic | `isaac_sac_minimal.py:twin_q_critic` | same as above |
| 4 | SAC losses | `isaac_sac_minimal.py:sac_losses` | same as above |
| 5 | SAC update step | `isaac_sac_minimal.py:sac_update_step` | same as above |
| 6 | Goal transition schema | `isaac_goal_relabeler.py:goal_transition_structs` | `bash docker/dev-isaac.sh python3 scripts/labs/isaac_goal_relabeler.py --verify` |
| 7 | Goal sampling | `isaac_goal_relabeler.py:isaac_goal_sampling` | same as above |
| 8 | Transition relabeling | `isaac_goal_relabeler.py:isaac_relabel_transition` | same as above |
| 9 | Episode HER processing | `isaac_goal_relabeler.py:isaac_her_episode_processing` | same as above |

---

## Run It Commands

```bash
# 1) Discover available env IDs and peg candidates
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py discover-envs --headless

# 2) Dense-first smoke run
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke --headless --seed 0

# 3) Train insertion task (auto-selected or explicit env-id)
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train --headless --seed 0

# 4) Evaluate checkpoint
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py eval \
  --headless --ckpt checkpoints/appendix_e_sac_<env>_seed0.zip

# 5) Compare result reports
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py compare \
  --result results/appendix_e_sac_<env>_seed0_eval.json \
  --result results/appendix_e_sac_<env>_seed1_eval.json
```

---

## Experiment Card (Template)

```
---------------------------------------------------------
EXPERIMENT CARD: Appendix E -- Isaac insertion
---------------------------------------------------------
Algorithm:   SAC (+ HER when env is goal-conditioned)
Environment: <resolved env id>
Seed:        <0>
Steps:       <500000 fast path>

Run command:
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train \
    --headless --seed <seed> [--env-id <id>]

Expected artifacts:
  checkpoints/appendix_e_sac_<env>_seed<seed>.zip
  checkpoints/appendix_e_sac_<env>_seed<seed>.meta.json
  results/appendix_e_sac_<env>_seed<seed>_eval.json
  results/appendix_e_sac_<env>_comparison.json

Primary metrics:
  success_rate
  final_goal_distance_mean
  ep_len_mean
---------------------------------------------------------
```

