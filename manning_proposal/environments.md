# Fetch Environment Visual Guide

This document provides a visual reference for the Gymnasium-Robotics Fetch
task family used throughout the book. Every figure is self-generated from
MuJoCo renders -- no external assets, no licensing concerns.

Figures are produced by `scripts/capture_proposal_figures.py` and written to
`figures/`. To regenerate:

```bash
bash docker/dev.sh python scripts/capture_proposal_figures.py all --out-dir figures
```

---

## 1. The Fetch Task Family

The book uses three Fetch environments, ordered by difficulty. Each environment
shares the same 7-DoF robotic arm but differs in what the agent must accomplish.

| Environment | Task | Object | Reward | Obs dim | Goal dim | Figure |
|---|---|---|---|---|---|---|
| FetchReach-v4 | Move gripper to target | None | Sparse | 10 | 3 | `fetch_reach_setup.png` |
| FetchPush-v4 | Push block to target | Block | Sparse | 25 | 3 | `fetch_push_setup.png` |
| FetchPickAndPlace-v4 | Pick up block, place at target | Block | Sparse | 25 | 3 | `fetch_pick_and_place_setup.png` |

Dense-reward variants (e.g., `FetchReachDense-v4`) use the same environments
but replace the sparse reward with a continuous distance signal.

### Environment Screenshots

Each screenshot shows the initial state after `env.reset()`, annotated with:

- **Environment name** (title)
- **Task description** (what the agent must do)
- **desired_goal** label (red sphere -- the target position)
- **achieved_goal** label (gripper tip or block -- current position)
- **Action space** format: `[dx, dy, dz, grip]`
- **Distance and reward** at reset

See `figures/fetch_reach_setup.png`, `figures/fetch_push_setup.png`, and
`figures/fetch_pick_and_place_setup.png`.

---

## 2. Observation Structure

Fetch environments return dictionary observations, not flat vectors. This is
central to how HER works -- it needs separate access to achieved and desired
goals to perform relabeling.

See `figures/obs_dict_structure.png` for a visual diagram.

### FetchReach-v4

| Component | Key | Shape | Contents |
|---|---|---|---|
| Robot state | `"observation"` | (10,) | Gripper position (3), gripper velocity (3), finger positions (2), finger velocities (2) |
| Target | `"desired_goal"` | (3,) | Target xyz position |
| Current | `"achieved_goal"` | (3,) | Gripper tip xyz position |

### FetchPush-v4 / FetchPickAndPlace-v4

| Component | Key | Shape | Contents |
|---|---|---|---|
| Robot + object state | `"observation"` | (25,) | Gripper pos (3), object pos (3), object relative pos (3), gripper state (2), object rotation (3), object velocities (6), gripper vel (3), finger vel (2) |
| Target | `"desired_goal"` | (3,) | Target xyz position |
| Current | `"achieved_goal"` | (3,) | Object xyz position |

Note: In Reach, `achieved_goal` tracks the gripper. In Push and PickAndPlace,
it tracks the object. This distinction matters for HER -- the relabeled goal
must match the entity the agent is trying to move.

---

## 3. Action Space

All Fetch environments share a 4-dimensional continuous action space:

| Index | Action | Range | Description |
|---|---|---|---|
| 0 | dx | [-1, 1] | Gripper displacement along x-axis |
| 1 | dy | [-1, 1] | Gripper displacement along y-axis |
| 2 | dz | [-1, 1] | Gripper displacement along z-axis |
| 3 | grip | [-1, 1] | Gripper open/close (-1 = open, 1 = close) |

Actions are Cartesian deltas scaled by `action_scale` (default 0.05m per
step). The gripper dimension is only meaningful for PickAndPlace -- in Reach
and Push, it has no effect on the task.

---

## 4. Difficulty Progression

The three environments form a natural curriculum:

**Reach** (Ch3-4): The simplest task. The agent only needs to move its gripper
to the target position. No object manipulation. This is the proving ground for
every algorithm -- if it cannot solve Reach, something is fundamentally wrong.

**Push** (Ch5-6): The agent must push a block to the target. This introduces
object dynamics -- the agent must predict how the block will slide. The block
starts on the table, and the target is also on the table. The gripper finger
dimension is irrelevant.

**PickAndPlace** (Ch6): The hardest task. The block can start on the table, but
the target may be in the air. The agent must learn to grip the block, lift it,
and place it. This requires coordinated gripper control -- a qualitatively
different skill from pushing.

---

## 5. Dense vs Sparse Rewards

See `figures/dense_vs_sparse_reward.png` for a side-by-side comparison.

**Dense reward:**

$$r = -\|g - g'\|$$

where $g$ is the desired goal and $g'$ is the achieved goal. The agent receives
continuous feedback proportional to its distance from the target. This is
informative but requires knowing the right distance metric.

**Sparse reward:**

$$r = -\mathbb{1}[\|g - g'\| \geq \epsilon]$$

where $\epsilon = 0.05$m (the success threshold). The agent receives $r = 0$
only when within 5cm of the target, and $r = -1$ otherwise. This is the natural
reward -- either you succeeded or you did not -- but creates a
needle-in-a-haystack exploration problem.

The book follows a methodical progression:
1. Dense rewards to validate algorithms (Ch3)
2. Sparse rewards to motivate HER (Ch4-5)
3. Sparse rewards + curriculum for harder tasks (Ch6)

---

## 6. Goal-Conditioned Structure

The `compute_reward(achieved_goal, desired_goal, info)` API is the critical
interface. Both HER and standard training call this function to compute rewards.
HER exploits it by passing *relabeled* goals -- asking "what reward would this
transition have received if the goal had been different?"

This is why HER requires off-policy algorithms: the relabeled transitions were
not generated under the current policy, so the algorithm must handle off-policy
data. On-policy methods (like PPO) assume all data comes from the current
policy and cannot use relabeled transitions.

---

## 7. Generating These Figures

All figures are generated deterministically from `scripts/capture_proposal_figures.py`:

```bash
# All figures (env screenshots + diagrams)
bash docker/dev.sh python scripts/capture_proposal_figures.py all --out-dir figures

# Only environment screenshots
bash docker/dev.sh python scripts/capture_proposal_figures.py env-setup --out-dir figures

# Only reward/obs diagrams (no MuJoCo needed)
bash docker/dev.sh python scripts/capture_proposal_figures.py reward-diagram --out-dir figures

# Specific environments
bash docker/dev.sh python scripts/capture_proposal_figures.py env-setup --envs FetchReach-v4 FetchPush-v4

# Random vs trained comparison (requires checkpoint)
bash docker/dev.sh python scripts/capture_proposal_figures.py compare \
    --env FetchReach-v4 --ckpt checkpoints/sac_FetchReach-v4_seed0.zip
```

**Output files:**

| Figure | File | Size | Content |
|---|---|---|---|
| Reach setup | `figures/fetch_reach_setup.png` | 640x480 | Annotated reset screenshot |
| Push setup | `figures/fetch_push_setup.png` | 640x480 | Annotated reset screenshot |
| PickAndPlace setup | `figures/fetch_pick_and_place_setup.png` | 640x480 | Annotated reset screenshot |
| Reward comparison | `figures/dense_vs_sparse_reward.png` | 1500x600 | Dense vs sparse reward curves |
| Obs structure | `figures/obs_dict_structure.png` | 1500x750 | Dictionary observation diagram |

All PNGs use the colorblind-friendly Wong (2011) palette:
Blue `#0072B2`, Orange `#E69F00`, Green `#009E73`, Vermillion `#D55E00`, Gray `#999999`.
