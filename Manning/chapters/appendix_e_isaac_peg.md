# Appendix E: Isaac Peg-In-Hole (GPU-Only)

> Status: scaffolded draft for Appendix E Build It/Run It wiring.
> This file is intentionally concise and will be expanded in writer/reviewer phases.

## E.1 Why This Appendix Exists

This appendix demonstrates portability: the same experiment contract from the
main text (reproducible commands, metadata, evaluation JSON) transferred to
Isaac Lab insertion tasks.

## E.2 Build It: SAC Core for Dict Observations

```python
# (from scripts/labs/isaac_sac_minimal.py:dict_flatten_encoder)
--8<-- "scripts/labs/isaac_sac_minimal.py:dict_flatten_encoder"
```

```python
# (from scripts/labs/isaac_sac_minimal.py:squashed_gaussian_actor)
--8<-- "scripts/labs/isaac_sac_minimal.py:squashed_gaussian_actor"
```

```python
# (from scripts/labs/isaac_sac_minimal.py:twin_q_critic)
--8<-- "scripts/labs/isaac_sac_minimal.py:twin_q_critic"
```

```python
# (from scripts/labs/isaac_sac_minimal.py:sac_losses)
--8<-- "scripts/labs/isaac_sac_minimal.py:sac_losses"
```

```python
# (from scripts/labs/isaac_sac_minimal.py:sac_update_step)
--8<-- "scripts/labs/isaac_sac_minimal.py:sac_update_step"
```

## E.3 Build It: Goal Relabeling for Precision Tasks

```python
# (from scripts/labs/isaac_goal_relabeler.py:goal_transition_structs)
--8<-- "scripts/labs/isaac_goal_relabeler.py:goal_transition_structs"
```

```python
# (from scripts/labs/isaac_goal_relabeler.py:isaac_goal_sampling)
--8<-- "scripts/labs/isaac_goal_relabeler.py:isaac_goal_sampling"
```

```python
# (from scripts/labs/isaac_goal_relabeler.py:isaac_relabel_transition)
--8<-- "scripts/labs/isaac_goal_relabeler.py:isaac_relabel_transition"
```

```python
# (from scripts/labs/isaac_goal_relabeler.py:isaac_her_episode_processing)
--8<-- "scripts/labs/isaac_goal_relabeler.py:isaac_her_episode_processing"
```

## E.4 Run It: Production Pipeline

Run script: `scripts/appendix_e_isaac_peg.py`

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py discover-envs --headless
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py smoke --headless --seed 0
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py train --headless --seed 0
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_peg.py eval \
  --headless --ckpt checkpoints/appendix_e_sac_<env>_seed0.zip
```

## E.5 Deliverables

- `checkpoints/appendix_e_sac_<env>_seed<N>.zip`
- `checkpoints/appendix_e_sac_<env>_seed<N>.meta.json`
- `results/appendix_e_sac_<env>_seed<N>_eval.json`
- `results/appendix_e_sac_<env>_comparison.json`
- `results/appendix_e_isaac_env_catalog.json`

