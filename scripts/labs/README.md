# Lab Modules

Minimal, readable "from-scratch" implementations for teaching reinforcement learning concepts.

## Purpose

These modules show **how equations map to code**. They are:

- **Pedagogical:** Fewer features, explicit tensor operations, clear variable naming
- **Not for production:** Use `scripts/chNN_*.py` (SB3-backed) for actual experiments
- **Snippet sources:** Tutorials include labeled regions from these files via `pymdownx.snippets`

## Files

| File | Algorithm | Key Concepts |
|------|-----------|--------------|
| `ppo_from_scratch.py` | PPO | GAE, clipped ratio, value loss |
| `sac_from_scratch.py` | SAC | Twin critics, entropy bonus, temperature tuning |
| `isaac_sac_minimal.py` | SAC (Appendix E) | Dict-observation encoding, squashed Gaussian policy, SAC update step |
| `her_relabeler.py` | HER | Goal relabeling, reward recomputation |
| `isaac_goal_relabeler.py` | HER (Appendix E) | Insertion-style sparse reward, future-goal sampling, HER episode processing |
| `curriculum_wrapper.py` | Curriculum | Goal difficulty scheduling, air/table control |

## Region Markers

Each file exports labeled regions for snippet-includes. Use these markers:

```python
# --8<-- [start:region_name]
def some_function():
    """This code will be included in tutorials."""
    pass
# --8<-- [end:region_name]
```

In tutorials, include with:

```markdown
```python
--8<-- "scripts/labs/ppo_from_scratch.py:region_name"
```  # (remove the space before the backticks)
```

## Verification

Each lab includes lightweight sanity checks runnable in under 2 minutes:

```bash
# PPO lab: verify advantages are finite, KL is bounded, value loss decreases
bash docker/dev.sh python scripts/labs/ppo_from_scratch.py --verify

# SAC lab: verify Q-targets finite, entropy coef trends down
bash docker/dev.sh python scripts/labs/sac_from_scratch.py --verify

# Appendix E SAC lab: verify dict-obs SAC update wiring
bash docker/dev.sh python scripts/labs/isaac_sac_minimal.py --verify

# HER lab: verify relabeling increases non-negative reward fraction
bash docker/dev.sh python scripts/labs/her_relabeler.py --verify

# Appendix E HER lab: verify insertion-style relabeling invariants
bash docker/dev.sh python scripts/labs/isaac_goal_relabeler.py --verify

# Curriculum lab: verify goal distributions at easy/hard difficulty
bash docker/dev.sh python scripts/labs/curriculum_wrapper.py --verify
```

## Conventions

1. **Explicit over implicit:** Name intermediate tensors (`advantages`, `ratio`, `clipped_ratio`)
2. **Comments map to math:** Reference equation numbers from tutorials
3. **Core code stays minimal:** Algorithm implementations use only PyTorch/NumPy. Optional `--compare-sb3` modes may import SB3 for cross-checks.
4. **Type hints:** Help readers understand shapes and types
5. **~200-400 lines per file:** Long enough to be complete, short enough to read in one sitting
