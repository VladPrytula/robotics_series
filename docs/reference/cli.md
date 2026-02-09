# CLI Reference

## Training: `train.py`

Train RL agents on Fetch environments.

```bash
bash docker/dev.sh python train.py --algo sac --her --env FetchReach-v4 --seed 0
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algo` | `{ppo,sac,td3}` | required | Algorithm |
| `--env` | string | required | Gymnasium env ID |
| `--seed` | int | 0 | Random seed |
| `--device` | `{auto,cpu,cuda}` | auto | Compute device |
| `--n-envs` | int | 8 | Parallel environments |
| `--total-steps` | int | 1000000 | Training timesteps |
| `--her` | flag | false | Enable HER (SAC/TD3 only) |
| `--track` | `{none,tb}` | tb | Logging backend |

### PPO-Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--ppo-n-steps` | 1024 | Steps per rollout |
| `--ppo-batch-size` | 256 | Minibatch size |

### SAC/TD3-Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--off-batch-size` | 256 | Batch size |
| `--off-buffer-size` | 1000000 | Replay buffer size |
| `--off-learning-starts` | 10000 | Steps before learning |

### HER-Specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--her-n-sampled-goal` | 4 | Relabeled goals per transition |
| `--her-goal-selection-strategy` | future | `{future,final,episode}` |

## Evaluation: `eval.py`

Evaluate trained checkpoints.

```bash
bash docker/dev.sh python eval.py --ckpt checkpoints/model.zip --env FetchReach-v4 --n-episodes 100
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | path | required | Checkpoint file |
| `--env` | string | required | Gymnasium env ID |
| `--n-episodes` | int | 100 | Evaluation episodes |
| `--seeds` | string | "0" | Seeds (e.g., "0-9" or "0,1,2") |
| `--deterministic` | flag | false | Deterministic actions |
| `--json-out` | path | none | Output JSON path |
| `--video` | flag | false | Record videos |

## Chapter Scripts

### `scripts/ch00_proof_of_life.py`

Verify environment functionality.

```bash
# Run all tests
bash docker/dev.sh python scripts/ch00_proof_of_life.py all

# Individual tests
bash docker/dev.sh python scripts/ch00_proof_of_life.py list-envs
bash docker/dev.sh python scripts/ch00_proof_of_life.py render
bash docker/dev.sh python scripts/ch00_proof_of_life.py ppo-smoke
```

### `scripts/ch01_env_anatomy.py`

Inspect Fetch environment structure.

```bash
# List available environments
bash docker/dev.sh python scripts/ch01_env_anatomy.py list-envs

# Describe observation/action spaces
bash docker/dev.sh python scripts/ch01_env_anatomy.py describe --json-out results/env.json

# Verify reward consistency
bash docker/dev.sh python scripts/ch01_env_anatomy.py reward-check

# Random baseline
bash docker/dev.sh python scripts/ch01_env_anatomy.py random-episodes --n-episodes 10
```
