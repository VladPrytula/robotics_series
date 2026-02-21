# Pixel Push Fix Plan: CNN Feature Normalization

**Date:** 2026-02-20
**Status:** Planning
**Depends on:** Current pixel runs finishing (baseline numbers)

---

## Root Cause Analysis

### The symptom
Pixel + HER + DrQ on FetchPush-v4 shows 2-8% success after 5M+ steps (8M
budget). State + HER solves the same task at 100% in 2M steps.

### The diagnosis: CNN features drown goal vectors

SB3's `CombinedExtractor` concatenates:

```
CNN(pixels) -> 256 features  (random early in training)
achieved_goal -> 3 features  (clean: object position)
desired_goal  -> 3 features  (clean: target position)
                 ---
total:          262 features  (goal vectors = 2.3% of input)
```

Early in training, the 256 CNN features are random noise with arbitrary
magnitude (post-ReLU: non-negative, unbounded). The 6 goal features are
clean, bounded values in the Fetch workspace (~1.0-1.5). When the MLP
receives this 262D input, its gradients are dominated by the CNN features.
The clean goal signal -- which is all the policy needs to learn *where*
the object should go -- is drowned out.

This is a well-known problem. DrQ-v2 (Yarats et al. 2021) addresses it with
a specific architectural choice: `Linear -> LayerNorm -> Tanh` on the encoder
output, compressing to **50 features** bounded in [-1, 1].

### Why this didn't matter for Reach

On Reach, there are no goal vectors (goal_mode="none"). The policy sees
only pixels. The CNN features are the ONLY input, so there's nothing to
drown out. DrQ's augmentation is sufficient to regularize the CNN, and the
task is easy enough that noisy features still converge.

### Why this compounds with HER's hockey-stick

The HER value wavefront requires the critic to learn accurate Q-values for
relabeled goals. But the critic's feature extractor (same CNN) produces
garbage features. The Q-function cannot represent goal-distance structure
because the CNN noise prevents it. The wavefront doesn't just slow down --
it can't start at all.

---

## Fix: Three Interventions (Ordered by Impact)

### Intervention 1: LayerNorm + Tanh on CNN output (HIGH IMPACT)

**What:** Replace SB3's default `CombinedExtractor` with a
`NormalizedCombinedExtractor` that applies `LayerNorm -> Tanh` after
NatureCNN. This bounds CNN features to [-1, 1], matching the scale of
goal vectors.

**Why this is the key fix:** DrQ-v2 and SAC-AE both use this pattern.
It ensures balanced gradient flow from both modalities from step 1.

**Where:** New class in `scripts/labs/visual_encoder.py` (~40 lines).

### Intervention 2: Reduce CNN output dim from 256 to 50 (MEDIUM IMPACT)

**What:** Set `cnn_output_dim=50` (DrQ-v2's default) instead of SB3's 256.

**Why:** Fewer CNN features = less noise to drown goal vectors. With 50
CNN + 6 goal features = 56 total, goals are 10.7% of input (vs 2.3%).
DrQ-v2 proves 50 features are sufficient for DMControl manipulation.

**Where:** `policy_kwargs=dict(features_extractor_kwargs=dict(cnn_output_dim=50))`

### Intervention 3: Lower learning rate to 1e-4 (LOW-MEDIUM IMPACT)

**What:** Use `learning_rate=1e-4` instead of SB3's default `3e-4`.

**Why:** DrQ-v2 uses 1e-4 for all visual RL. Slower updates = less noisy
CNN feature evolution = more stable value learning for the critic.

**Where:** `learning_rate=1e-4` in SAC constructor.

---

## Experiment Plan

### Baseline (current runs, finishing overnight)

| Run | Steps | Expected result |
|-----|-------|-----------------|
| pixel+HER (b7e835c) | 8M | ~5% (flat, no inflection) |
| pixel+HER+DrQ (bc7b58a) | 8M | ~5% (flat, no inflection) |

### Fix runs (after baseline completes)

All three interventions applied together (the "DrQ-v2 trunk" fix):

| Run | Config | Steps | Expected |
|-----|--------|-------|----------|
| **pixel+HER+norm** | HER + NormalizedCombinedExtractor(50) + lr=1e-4 | 8M | >50% |
| **pixel+HER+DrQ+norm** | HER + DrQ + NormalizedCombinedExtractor(50) + lr=1e-4 | 8M | >70% |

If both fail, ablate:
| Run | Config | Steps | What it tests |
|-----|--------|-------|---------------|
| pixel+HER+norm-dim256 | HER + NormalizedCombinedExtractor(256) + lr=1e-4 | 8M | Is dim reduction or norm more important? |
| pixel+HER+DrQ+norm-lr3e4 | HER + DrQ + NormalizedCombinedExtractor(50) + lr=3e-4 | 8M | Is lr or norm more important? |

### Success criteria

- **Minimum viable:** pixel+HER+DrQ+norm > 50% at 8M steps
- **Good:** > 70% (demonstrates clear pixel Push resolution)
- **Great:** > 85% (comparable to state HER before saturation)

---

## Implementation Plan

### Step 1: Add NormalizedCombinedExtractor to visual_encoder.py

```python
class NormalizedCombinedExtractor(BaseFeaturesExtractor):
    """CombinedExtractor with LayerNorm + Tanh on CNN features.

    Follows the DrQ-v2 trunk pattern (Yarats et al. 2021): CNN features
    are normalized to [-1, 1] before concatenation with vector inputs.
    This prevents noisy early CNN features from drowning clean goal vectors
    in mixed pixel+vector observation spaces.

    Architecture per key:
        image key  -> NatureCNN(features_dim) -> LayerNorm -> Tanh
        vector key -> Flatten (identity)
    All outputs concatenated -> features
    """

    def __init__(self, observation_space, cnn_output_dim=50,
                 normalized_image=False):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                cnn = NatureCNN(subspace, features_dim=cnn_output_dim,
                                normalized_image=normalized_image)
                extractors[key] = nn.Sequential(
                    cnn, nn.LayerNorm(cnn_output_dim), nn.Tanh())
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += gym.spaces.utils.flatdim(subspace)
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations):
        parts = [ext(observations[k]) for k, ext in self.extractors.items()]
        return th.cat(parts, dim=1)
```

### Step 2: Add --norm flag to ch10_visual_reach.py

New CLI flags:
- `--norm`: Use NormalizedCombinedExtractor instead of default CombinedExtractor
- `--cnn-dim`: CNN output dimension (default 50 when --norm, 256 otherwise)
- `--lr`: Override learning rate (default 1e-4 when --norm, 3e-4 otherwise)

Wiring in push pixel modes:
```python
if cfg.norm:
    policy_kwargs = dict(
        features_extractor_class=NormalizedCombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=cfg.cnn_dim),
    )
else:
    policy_kwargs = {}
```

### Step 3: Smoke test (10K steps)

```bash
bash docker/dev.sh python scripts/ch10_visual_reach.py \
    train-push-pixel-her-drq --pixel-total-steps 10000 \
    --norm --cnn-dim 50 --lr 1e-4
```

Verify: no crashes, correct feature dimensions in model summary.

### Step 4: Full training runs

```bash
# Run 1: pixel+HER+norm
docker run ... python scripts/ch10_visual_reach.py \
    train-push-pixel-her --seed 0 --fast \
    --gradient-steps 1 --pixel-n-envs 8 \
    --push-pixel-buffer-size 1000000 \
    --push-pixel-total-steps 8000000 \
    --norm --cnn-dim 50 --lr 1e-4

# Run 2: pixel+HER+DrQ+norm
docker run ... python scripts/ch10_visual_reach.py \
    train-push-pixel-her-drq --seed 0 --fast \
    --gradient-steps 1 --pixel-n-envs 8 \
    --push-pixel-buffer-size 1000000 \
    --push-pixel-total-steps 8000000 \
    --norm --cnn-dim 50 --lr 1e-4
```

---

## Book Integration

### If the fix works (>50% success)

The Ch9 narrative gains a fourth act:

```
Act 1: Measure the pixel penalty (Reach: state vs pixel vs DrQ)
Act 2: The real test (Push fails from dense, HER solves from state)
Act 3: Naive pixel+HER fails (CNN features drown goal vectors)
Act 4: Resolution (normalize features, push from pixels works)
```

The "Build It" section in Ch9 introduces NormalizedCombinedExtractor as
the key insight: **when mixing high-dimensional learned features with
low-dimensional clean vectors, normalize the learned features first.**

This is a real engineering lesson -- not a trick, but a principle backed
by DrQ-v2, SAC-AE, and TD-MPC2.

### If the fix doesn't work (<50% success)

Ch9 resolves on Reach (pixels work) and honestly shows Push as an open
challenge. The Further Reading sidebar (already drafted) points to
pre-trained encoders (VIP, R3M) and contrastive RL (Eysenbach et al.)
as the frontier. Ch10 stays as reality-gap, scoped to Reach from pixels.

---

## References

- Yarats, D. et al. (2021). "Mastering Visual Continuous Control: Improved
  Data-Augmented Reinforcement Learning." arXiv:2107.09645. (DrQ-v2)
- Yarats, D. et al. (2019). "Improving Sample Efficiency in Model-Free
  Reinforcement Learning from Images." arXiv:1910.01741. (SAC-AE)
- Hansen, N. et al. (2023). "TD-MPC2: Scalable, Robust World Models for
  Continuous Control." arXiv:2310.16828. (SimNorm)
