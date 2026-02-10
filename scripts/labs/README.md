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
| `her_relabeler.py` | HER | Goal relabeling, reward recomputation |

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

# HER lab: verify relabeling increases non-negative reward fraction
bash docker/dev.sh python scripts/labs/her_relabeler.py --verify
```

## Conventions

1. **Explicit over implicit:** Name intermediate tensors (`advantages`, `ratio`, `clipped_ratio`)
2. **Comments map to math:** Reference equation numbers from tutorials
3. **No external dependencies beyond PyTorch/NumPy:** No SB3 imports
4. **Type hints:** Help readers understand shapes and types
5. **~200-400 lines per file:** Long enough to be complete, short enough to read in one sitting
