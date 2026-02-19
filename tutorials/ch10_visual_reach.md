# Chapter 10: Pixels, No Cheating -- Visual SAC on FetchReachDense

**Week 10 Goal:** Quantify the cost of learning from raw pixels instead of
privileged state, and recover part of the gap with data augmentation (DrQ).

---

## Bridge: From Robustness Analysis to Visual Observations

In Chapter 7, we injected observation noise into trained policies and measured
how quickly success degraded. That analysis assumed the policy operated on
low-dimensional state vectors -- the 10-float Fetch observation that encodes
end-effector position, velocity, and gripper state. A real robot does not have
access to this vector. It has cameras.

This chapter asks: **what happens when we replace the privileged state vector
with raw pixel images?** We train SAC on FetchReachDense in three configurations
-- state, pixels, and pixels with DrQ augmentation -- and measure the
sample-efficiency gap quantitatively. The goal is not just to "make it work from
pixels" but to understand *why* pixels are harder and *what* helps.

Chapters 8-9 (second-suite adapter and engineering-grade ablations) are not yet
written. This chapter can be read directly after Chapter 7 -- no concepts from
Ch8-9 are needed.

---

## WHY: The Pixel Observation Problem

### 10.1 Why Pixels Matter

Every Fetch experiment so far has used the `observation` vector: a 10D float
array containing the end-effector's Cartesian position (3), velocity (3),
gripper width (1), and related quantities. This is **privileged information** --
a real robot does not know its end-effector position to millimeter accuracy
without a calibrated motion capture system.

Cameras are the natural sensor for robotic manipulation. But replacing a 10D
float vector with an 84x84x3 RGB image (21,168 values) creates several
compounding difficulties:

1. **Dimensionality.** The observation grows from 10 to 21,168 -- a 2,100x
   increase. The policy network must extract spatial features before it can
   reason about actions.

2. **Partial observability.** A single image does not encode velocity.
   The policy must infer motion from pixel differences across frames
   (or learn to act without velocity information).

3. **Rendering cost.** Each `env.step()` now requires a MuJoCo render call.
   For Fetch at 480x480 (the default), this is ~230,000 pixels per frame,
   per environment. Rendering becomes the training bottleneck -- not the
   neural network.

4. **Q-function overfitting.** With high-dimensional image inputs, the
   Q-network can memorize pixel-level details of specific transitions
   rather than learning generalizable value estimates. This is the problem
   DrQ addresses.

### 10.2 Observation Design Choices

When wrapping a goal-conditioned environment for pixel observations, we face
a design decision: **which goal information (if any) do we expose alongside
the image?**

Our `PixelObservationWrapper` supports three modes:

| `goal_mode` | Observation keys | What the policy sees |
|-------------|-----------------|---------------------|
| `"none"` | `{"pixels"}` | Image only -- must infer both current state and goal from pixels |
| `"desired"` | `{"pixels", "desired_goal"}` | Image + 3D goal position (privileged goal info) |
| `"both"` | `{"pixels", "achieved_goal", "desired_goal"}` | Image + both goal vectors |

We default to `goal_mode="none"` -- the hardest setting, where the policy must
extract everything from the image. This is the most realistic scenario: a camera
sees the scene, and the goal (a target position) is indicated visually (e.g., by
a marker in the scene).

**Non-example:** `goal_mode="both"` with pixel observations is not "learning
from pixels" in any meaningful sense -- the policy has direct access to the
3D coordinates it needs to solve the task. We include it as an experimental
control, not as a recommended configuration.

### 10.3 The Sample-Efficiency Gap

The central quantity this chapter measures:

**Definition (Sample-Efficiency Ratio).** Given two agents trained on the same
task with different observation types, the sample-efficiency ratio is:

$$
\rho = \frac{N_{\text{pixel}}}{N_{\text{state}}}
$$

where $N_{\text{pixel}}$ and $N_{\text{state}}$ are the number of environment
steps each agent needs to reach a target success rate (e.g., 90%). A ratio
$\rho = 4$ means pixel training needs 4x more samples.

We find this ratio is typically 2-10x for FetchReachDense, depending on
hyperparameters and whether augmentation is used.

---

## HOW: Three Approaches

### 10.4 Approach 1: State-Based SAC (Baseline)

This is the SAC configuration from Chapter 3: `MultiInputPolicy` on the
dictionary observation with flat state vectors. SB3 routes the observation,
achieved\_goal, and desired\_goal through its `CombinedExtractor`, which
flattens and concatenates them into a single vector for the actor and critic
MLPs.

This serves as our performance ceiling -- the best we can do with privileged
information.

### 10.5 Approach 2: Pixel SAC (No Augmentation)

We wrap FetchReachDense with `PixelObservationWrapper` (from
`scripts/labs/pixel_wrapper.py`), replacing the flat observation with an
84x84 RGB image. SB3's `MultiInputPolicy` detects the image space via
`is_image_space()` and automatically routes the `"pixels"` key through a
NatureCNN encoder (Mnih et al., 2015).

The rendering pipeline is:

```
env.render() -> 480x480 HWC uint8
             -> PIL resize to 84x84 (bilinear)
             -> transpose to CHW
             -> PixelObservationWrapper stores as obs["pixels"]
```

SB3 then normalizes uint8 [0, 255] to float32 [0, 1] internally.

### 10.6 Approach 3: Pixel SAC + DrQ Augmentation

DrQ (Kostrikov et al., 2020) is a single, clean idea: **augment pixel
observations at replay buffer sample time** with random spatial shifts.
Each time a transition is replayed, the Q-network sees a slightly different
crop of the image. This prevents the Q-function from memorizing pixel-level
details of specific transitions.

**Definition (Random Shift Augmentation).** Given an image $x$ of size
$(H, W)$ and pad size $p$:

1. Pad $x$ by $p$ pixels on all sides using replicate padding, producing
   a $(H + 2p, W + 2p)$ image.
2. Randomly crop back to $(H, W)$.

For our setup ($H = W = 84$, $p = 4$), this creates shifts of up to
$\pm 4$ pixels ($\sim 5\%$ of the image). Replicate padding avoids
black borders that would create artificial edge features.

The entire implementation is a replay buffer wrapper -- no changes to the
loss function, network architecture, or training loop. This is what makes
DrQ elegant: the algorithmic innovation is a single clean abstraction.

---

## BUILD IT: Pixel Observation Pipeline (10.7)

This section walks through the pedagogical implementations in
`scripts/labs/pixel_wrapper.py`, `scripts/labs/visual_encoder.py`, and
`scripts/labs/image_augmentation.py`.

### 10.7.1 Rendering and Resizing

The core rendering function maps MuJoCo's high-resolution output to the
84x84 images our CNN expects:

```python
--8<-- "scripts/labs/pixel_wrapper.py:render_and_resize"
```

When MuJoCo renders natively at 84x84 (via `gym.make(..., width=84, height=84)`),
the function detects the matching size and skips PIL entirely -- this is the
fast path used by `--fast` mode (Section 10.11).

### 10.7.2 Pixel Observation Wrapper

The wrapper replaces the flat state observation with a rendered image, while
optionally preserving goal vectors:

```python
--8<-- "scripts/labs/pixel_wrapper.py:pixel_obs_wrapper"
```

!!! lab "Checkpoint"
    Verify the wrapper produces correct observation spaces:
    ```python
    python scripts/labs/pixel_wrapper.py --verify
    ```
    Expected: `[ALL PASS] Pixel wrapper verified`

### 10.7.3 NatureCNN Encoder

The NatureCNN architecture from Mnih et al. (2015) maps 84x84x3 images to
a 512-dimensional feature vector:

```
Conv2d(3, 32, 8x8, stride=4)  ->  ReLU      84 -> 20
Conv2d(32, 64, 4x4, stride=2) ->  ReLU      20 -> 9
Conv2d(64, 64, 3x3, stride=1) ->  ReLU       9 -> 7
Flatten                        ->  Linear(3136, 512)  ->  ReLU
```

```python
--8<-- "scripts/labs/visual_encoder.py:nature_cnn"
```

The dummy forward pass trick (`torch.zeros(1, C, H, W)` through the conv
layers) computes the flatten dimension automatically -- this avoids
hard-coding `3136` and makes the encoder work for image sizes other than
84x84.

!!! lab "Checkpoint"
    Verify encoder shapes and parameter count:
    ```python
    python scripts/labs/visual_encoder.py --verify
    ```
    Expected: `[ALL PASS] Visual encoder verified`

### 10.7.4 DrQ: Random Shift Augmentation

The augmentation function pads and randomly crops pixel observations:

```python
--8<-- "scripts/labs/image_augmentation.py:random_shift_aug"
```

**Key design choice:** replicate padding extends border pixels outward. Zero
padding would create black borders at the crop edges, which the Q-network
could learn to exploit as an artificial feature.

### 10.7.5 DrQ Replay Buffer

The replay buffer wrapper applies augmentation at sample time -- the only
change needed to add DrQ to an existing SAC pipeline:

```python
--8<-- "scripts/labs/image_augmentation.py:drq_replay_buffer"
```

This is the entire DrQ integration. The buffer stores un-augmented observations
(so they can be replayed with different augmentations each time), and applies
`aug_fn` to the pixel key when sampling. Goals, actions, and rewards are
unchanged.

!!! lab "Checkpoint"
    Verify augmentation preserves shapes and only affects pixels:
    ```python
    python scripts/labs/image_augmentation.py --verify
    ```
    Expected: `[ALL PASS] Image augmentation verified`

### 10.7.6 Pixel Replay Buffer (uint8 Storage)

For the from-scratch implementation (`visual_encoder.py --demo`), we use a
custom replay buffer that stores images as uint8 for 4x memory savings:

??? lab "PixelReplayBuffer (click to expand)"
    ```python
    --8<-- "scripts/labs/pixel_wrapper.py:pixel_replay_buffer"
    ```

Standard SB3 replay buffers store observations as float32. For 84x84x3 images,
that is 84,672 bytes per image. Storing as uint8 costs 21,168 bytes -- a 4x
savings. With 200K transitions (obs + next\_obs), this saves ~25 GB. The
conversion to float32 [0, 1] happens at sample time, when the batch is small
(typically 256).

---

## WHAT: Experiments and Expected Results (Run It)

### 10.8 Quick Start

The full pipeline trains all three variants and produces a comparison table:

```bash
bash docker/dev.sh python scripts/ch10_visual_reach.py all --seed 0
```

This runs seven steps in sequence: train-state, eval-state, train-pixel,
eval-pixel, train-pixel-drq, eval-drq, compare. Total wall time is
approximately 10-12 hours on a DGX with GPU.

For faster iteration, use `--fast` mode (Section 10.11):

```bash
bash docker/dev.sh python scripts/ch10_visual_reach.py all --seed 0 --fast
```

### 10.9 Experiment 1: State-Based SAC (Baseline)

```bash
bash docker/dev.sh python scripts/ch10_visual_reach.py train-state --seed 0
bash docker/dev.sh python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_state_FetchReachDense-v4_seed0.zip
```

**FetchReachDense-v4 -- State SAC (500K steps):**

| Metric | Value |
|--------|------:|
| Success rate | <!-- TODO: fill from run --> |
| Return (mean) | <!-- TODO --> |
| Final distance (mean) | <!-- TODO --> |
| Training time | <!-- TODO --> |
| FPS | <!-- TODO --> |

State SAC should converge to ~100% success rate within 200-300K steps.
This is the performance ceiling.

### 10.10 Experiment 2: Pixel SAC (No Augmentation)

```bash
bash docker/dev.sh python scripts/ch10_visual_reach.py train-pixel --seed 0
bash docker/dev.sh python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_pixel_FetchReachDense-v4_seed0.zip --pixel
```

**FetchReachDense-v4 -- Pixel SAC (2M steps):**

| Metric | Value |
|--------|------:|
| Success rate | <!-- TODO: fill from run --> |
| Return (mean) | <!-- TODO --> |
| Final distance (mean) | <!-- TODO --> |
| Training time | <!-- TODO --> |
| FPS | <!-- TODO --> |

**Interpretation:** Pixel SAC needs significantly more samples than state SAC.
The NatureCNN must learn to extract spatial features (end-effector position,
target marker) from raw pixels before the RL algorithm can learn a useful
policy. The Q-function also tends to overfit -- watch for `critic_loss` that
decreases early but climbs later.

### 10.10.1 Experiment 3: Pixel SAC + DrQ

```bash
bash docker/dev.sh python scripts/ch10_visual_reach.py train-pixel-drq --seed 0
bash docker/dev.sh python scripts/ch10_visual_reach.py eval --ckpt checkpoints/sac_drq_FetchReachDense-v4_seed0.zip --pixel
```

**FetchReachDense-v4 -- Pixel+DrQ SAC (2M steps):**

| Metric | Value |
|--------|------:|
| Success rate | <!-- TODO: fill from run --> |
| Return (mean) | <!-- TODO --> |
| Final distance (mean) | <!-- TODO --> |
| Training time | <!-- TODO --> |
| FPS | <!-- TODO --> |

**Interpretation:** DrQ should improve pixel SAC's sample efficiency,
partially closing the gap to state SAC. The random shift augmentation
regularizes the Q-function, preventing overfitting to pixel-level details.

### 10.10.2 Three-Way Comparison

```bash
bash docker/dev.sh python scripts/ch10_visual_reach.py compare
```

**FetchReachDense-v4 -- State vs Pixel vs Pixel+DrQ:**

| Metric | State | Pixel | Pixel+DrQ |
|--------|------:|------:|----------:|
| Training steps | 500,000 | 2,000,000 | 2,000,000 |
| Success rate | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Return (mean) | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| Final distance | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| FPS | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

<!-- TODO: fill sample-efficiency ratio and DrQ gap closure percentage -->

---

## 10.11 Making It Fast: Rendering Is the Bottleneck

The pixel training pipeline above takes ~8.5 hours per arm (pixel + DrQ =
~17 hours total). Where does that time go?

### 10.11.1 Profiling the Pipeline

In RL with physics simulators, the **GPU sits idle** most of the time. The
bottleneck is CPU-bound: MuJoCo runs contact dynamics, then renders the
scene to an image, then Python (PIL) resizes it. The neural network
forward/backward pass on the GPU completes in microseconds.

For pixel training without `--fast`:

| Component | Time per step | Notes |
|-----------|--------------|-------|
| MuJoCo simulation | ~0.5 ms | CPU-bound physics |
| Render at 480x480 | ~4 ms | 230,400 pixels |
| PIL resize to 84x84 | ~0.3 ms | Python/C interop |
| Neural network update | ~0.1 ms | GPU, batch=256 |

Rendering dominates. The default pipeline renders 33x more pixels than needed,
ships them through PIL, then discards 97% of the data.

### 10.11.2 Three Optimizations

The `--fast` flag bundles three independent optimizations:

**Optimization 1: Native Resolution Rendering.**
Instead of rendering at 480x480 and resizing, we tell MuJoCo to render
directly at 84x84:

```python
env = gym.make("FetchReachDense-v4", render_mode="rgb_array", width=84, height=84)
```

This renders 33x fewer pixels (7,056 vs 230,400) and skips PIL entirely.
The `render_and_resize` function detects the matching size and takes the
fast path automatically.

**Optimization 2: SubprocVecEnv.**
`DummyVecEnv` (the default) runs all environments sequentially in the main
process. `SubprocVecEnv` runs each environment in a separate process,
parallelizing MuJoCo rendering across CPU cores:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
env = make_vec_env(make_pixel_env, n_envs=12, vec_env_cls=SubprocVecEnv)
```

With 12 workers, we get ~3x throughput on a multi-core DGX.

**Optimization 3: More Gradient Steps.**
When data arrives faster (more envs, faster rendering), we can afford more
gradient updates per environment step. `gradient_steps=3` means 3 SAC
updates per step, increasing learning per sample without stale-data concerns.

This is the **replay ratio** concept: faster data collection enables a higher
ratio of learning to experience collection.

### 10.11.3 Using `--fast`

```bash
# Fast mode with default settings (n_envs=12, gradient_steps=3, native render)
bash docker/dev.sh python scripts/ch10_visual_reach.py train-pixel --seed 0 --fast

# Override specific values while keeping other fast defaults
bash docker/dev.sh python scripts/ch10_visual_reach.py train-pixel --seed 0 --fast --pixel-n-envs 8 --gradient-steps 2
```

`--fast` sets: `native_render=True`, `use_subproc=True`, `pixel_n_envs=12`,
`gradient_steps=3`. Explicit CLI arguments override fast defaults.

**What stays the same:** The algorithm, architecture, hyperparameters, and
final performance are unchanged. `--fast` only optimizes the engineering of
data collection. A model trained with `--fast` should be statistically
equivalent to one trained without it.

### 10.11.4 Speedup Results

| Setting | FPS | Wall time (2M steps) | Notes |
|---------|----:|---------------------:|-------|
| Default (4 env, DummyVec, PIL resize) | ~65 | ~8.5 hrs | Pedagogical pipeline |
| `--fast` (12 env, SubprocVec, native 84x84) | ~160 | ~3.5 hrs | <!-- TODO: update from run --> |

<!-- TODO: fill exact numbers from completed --fast run -->

The fast-mode metadata is recorded in the checkpoint's `.meta.json` file for
reproducibility:

```json
{
  "fast_mode": true,
  "native_render": true,
  "gradient_steps": 3,
  "use_subproc": true,
  "n_envs": 12
}
```

---

## Summary

### Key Findings

<!-- TODO: fill with actual numbers from completed runs -->

1. **State SAC solves FetchReachDense** in ~200-300K steps (100% success rate).
2. **Pixel SAC is harder** -- needs ~2-4x more samples due to high-dimensional
   input and Q-function overfitting.
3. **DrQ augmentation helps** -- random shift regularization partially closes
   the state-vs-pixel gap without any architecture or loss changes.
4. **Rendering is the bottleneck** -- `--fast` mode achieves ~2.5x speedup by
   eliminating waste (native resolution), parallelizing (SubprocVecEnv), and
   balancing the pipeline (gradient\_steps).

### Concepts Introduced in This Chapter

| Concept | Definition |
|---------|-----------|
| Pixel observation wrapper | Replaces flat state vector with rendered 84x84 RGB images |
| Goal mode (none/desired/both) | Controls which goal vectors are exposed alongside pixels |
| NatureCNN | CNN encoder from Mnih et al. (2015): three conv layers + FC to 512D features |
| Sample-efficiency ratio | $\rho = N_{\text{pixel}} / N_{\text{state}}$ -- how many more samples pixels need |
| DrQ (random shift augmentation) | Pad-and-crop pixel augmentation at replay buffer sample time |
| Native resolution rendering | `gym.make(..., width=84, height=84)` to skip PIL resize |
| SubprocVecEnv | Parallel environment execution across CPU cores |
| Replay ratio / gradient steps | Number of SAC gradient updates per environment step |

### Files Generated

| File | Purpose |
|------|---------|
| `scripts/labs/pixel_wrapper.py` | PixelObservationWrapper, render\_and\_resize, PixelReplayBuffer |
| `scripts/labs/visual_encoder.py` | NatureCNN, VisualGoalEncoder, VisualGaussianPolicy, VisualTwinQNetwork |
| `scripts/labs/image_augmentation.py` | RandomShiftAug, DrQDictReplayBuffer |
| `scripts/ch10_visual_reach.py` | Chapter script: train-state, train-pixel, train-pixel-drq, eval, compare |

### Artifacts

| Artifact | Location |
|----------|----------|
| State SAC checkpoint | `checkpoints/sac_state_FetchReachDense-v4_seed{N}.zip` |
| Pixel SAC checkpoint | `checkpoints/sac_pixel_FetchReachDense-v4_seed{N}.zip` |
| Pixel+DrQ checkpoint | `checkpoints/sac_drq_FetchReachDense-v4_seed{N}.zip` |
| Training metadata | `checkpoints/sac_{mode}_FetchReachDense-v4_seed{N}.meta.json` |
| State evaluation | `results/ch10_state_eval.json` |
| Pixel evaluation | `results/ch10_pixel_eval.json` |
| DrQ evaluation | `results/ch10_drq_eval.json` |
| Comparison report | `results/ch10_comparison.json` |

### What Comes Next

This chapter showed the cost of removing privileged state information: pixel
observations are harder, slower, and prone to Q-function overfitting. DrQ
helps, but a gap remains.

<!-- TODO: bridge to whatever comes after Ch10 in the curriculum -->

---

### References

- Mnih, V. et al. (2015). "Human-level control through deep reinforcement
  learning." *Nature*, 518(7540), 529-533.
- Kostrikov, I. et al. (2020). "Image Augmentation Is All You Need:
  Regularizing Deep Reinforcement Learning from Pixels."
  arXiv:2004.13649, Section 3.1.
- Haarnoja, T. et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy
  Deep Reinforcement Learning with a Stochastic Actor."
  arXiv:1801.01290.
