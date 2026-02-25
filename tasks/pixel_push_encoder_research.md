# Pixel Push: Encoder Architecture Research

**Date:** 2026-02-21
**Status:** Research complete, implementation pending
**Context:** Normalization + frame stacking fix (Run 2) shows no improvement
over baseline at 3M+ steps. This document investigates why and proposes
encoder-level interventions grounded in the literature.

---

## 1. The Story So Far

### What we tried

| Run | Config | Result at 3M steps |
|-----|--------|--------------------|
| Baseline pixel+HER | NatureCNN(256), lr=3e-4, no stacking | 5-8% flat for 8M |
| Baseline pixel+HER+DrQ | NatureCNN(256), DrQ, lr=3e-4 | 5-8% flat for 8M |
| Fix pixel+HER+norm+fs4 | NormalizedCombinedExtractor(50), LN+Tanh, fs=4, lr=1e-4 | 4-8% flat |
| Fix pixel+HER+DrQ+norm+fs4 | Same + DrQ augmentation | 5-7% flat |

### What the normalization fix DID change

From clean TensorBoard comparison at matching steps:

- **Critic loss started lower and more stable**: 0.11 at 200K (fix) vs 0.75
  (baseline). The critic can fit Q-values more easily with bounded CNN features.
- **Actor loss was 2x higher early**: 2.28 (fix) vs 0.14 (baseline) at 200K.
  The critic provided stronger gradient signals to the actor.
- **By 2M steps, fix and baseline converged** to identical success rates (~5%),
  critic losses (~0.15), and actor losses (~1.6).

### What the normalization fix did NOT change

- Success rate trajectory: identical to baseline (5-8% oscillation)
- Episode reward: stuck at -44 to -48 (state+HER reaches -20 by 2M steps)
- No hockey-stick inflection at any point
- The critic loss never INCREASED (which is the signature of state+HER's
  inflection -- the critic faces harder values as the policy discovers new states)

### Revised diagnosis

The scale mismatch (Root Cause 1) was real and the normalization fixed it at the
gradient level. But fixing gradients didn't help because the CNN features
themselves don't contain the spatial information the critic needs. The CNN is
providing clean, bounded, well-scaled garbage -- the features are tidy but
uninformative.

---

## 2. NatureCNN: Why It's Wrong for Manipulation

### Architecture review

NatureCNN (Mnih et al. 2015, arXiv:1312.5602):

```
Layer 1: Conv2d(in, 32, kernel=8, stride=4) -> ReLU    84x84 -> 20x20
Layer 2: Conv2d(32, 64, kernel=4, stride=2) -> ReLU    20x20 -> 9x9
Layer 3: Conv2d(64, 64, kernel=3, stride=1) -> ReLU    9x9   -> 7x7
Flatten: 64 * 7 * 7 = 3,136
Linear:  3,136 -> features_dim (256 default, 50 with our fix)
```

### The first-layer problem

The 8x8 kernel with stride 4 reduces the image from 84x84 to 20x20 in a
single step. This is a **4x spatial downsampling** that discards fine spatial
detail. For Atari, this is appropriate:
- Game sprites are large (10-30 pixels)
- Decisions are coarse (move left/right/up/down)
- Background is simple, objects have high contrast

For FetchPush at 84x84, the relevant objects are:
- Gripper: ~4-6 pixels across
- Puck: ~3-5 pixels across
- Goal marker: ~3 pixels across

After the 8x8 stride-4 first layer, these objects become **1 pixel** in the
20x20 feature map. The spatial relationship between gripper and puck -- which
is the entire signal the critic needs -- is compressed to the difference between
two adjacent or identical pixels. This is effectively destroyed.

### What modern visual RL uses instead

**DrQ-v2 encoder** (Yarats et al. 2021, arXiv:2107.09645):

```
Layer 1: Conv2d(in, 32, kernel=3, stride=2) -> ReLU    84x84 -> 42x42
Layer 2: Conv2d(32, 32, kernel=3, stride=1) -> ReLU    42x42 -> 42x42
Layer 3: Conv2d(32, 32, kernel=3, stride=1) -> ReLU    42x42 -> 42x42
Layer 4: Conv2d(32, 32, kernel=3, stride=1) -> ReLU    42x42 -> 42x42
Flatten: 32 * 35 * 35 = 39,200
Linear:  39,200 -> 50 -> LayerNorm -> Tanh
```

Key differences:
- **3x3 kernels throughout** (vs 8/4/3): finer spatial processing
- **Stride 2 only in first layer** (vs 4): 84->42 (2x) vs 84->20 (4x)
- **32 channels everywhere** (vs 32->64->64): consistent width
- **4 layers** (vs 3): deeper but narrower
- **LayerNorm + Tanh on output**: bounded features (our normalization fix)

After layer 1, the feature map is 42x42. A 4-pixel object becomes ~2 pixels --
still localized and distinguishable from its neighbors. The spatial relationship
between two objects 10 pixels apart (in input space) is preserved as ~5 pixels
apart in the feature map.

**TD-MPC2 encoder** (Hansen et al. 2023, arXiv:2310.16828):

```
Layer 1: Conv2d(in, C, kernel=7, stride=2) -> ReLU     64x64 -> 29x29
Layer 2: Conv2d(C, C, kernel=5, stride=2) -> ReLU      29x29 -> 13x13
Layer 3: Conv2d(C, C, kernel=3, stride=2) -> ReLU      13x13 -> 6x6
Layer 4: Conv2d(C, C, kernel=3, stride=1) -> ReLU      6x6   -> 6x6
Optional: SimNorm (simplex normalization)
```

Key: decreasing kernel sizes (7->5->3->3) capture large context first,
then refine spatially. Stride 2 everywhere (not 4).

**DreamerV3 encoder** (Hafner et al. 2023, arXiv:2301.04104):

```
Channels: [128, 192, 256, 256] (64 * multipliers [2, 3, 4, 4])
Kernel: 5x5 throughout
Activation: SiLU (not ReLU)
Normalization: RMSNorm (not BatchNorm or LayerNorm)
Input preprocessing: symlog
Output: 1024-dim
```

The deepest and widest encoder in the literature. Uses modern activation
(SiLU) and normalization (RMSNorm). Designed for 64x64 input.

### Quantitative comparison

| Encoder | Input | After layer 1 | # params (approx) | Output dim |
|---------|-------|---------------|-------|-----------|
| NatureCNN | 84x84 | 20x20 (4x reduction) | ~30K | 256 (default) |
| DrQ-v2 | 84x84 | 42x42 (2x reduction) | ~35K | 50 |
| TD-MPC2 | 64x64 | 29x29 (2x reduction) | ~50K | configurable |
| DreamerV3 | 64x64 | varies | ~500K | 1024 |

---

## 3. The Spatial Information Problem

### What the CNN must extract for FetchPush

The minimum state information for Push:

| Quantity | Dimensions | Observable from pixels? |
|----------|-----------|------------------------|
| Gripper position | 3D (x, y, z) | Yes -- visible in image |
| Object position | 3D (x, y, z) | Yes -- visible in image |
| Goal position | 3D (x, y, z) | Provided as vector (not in image) |
| Gripper-object relative pos | 3D (derived) | Yes -- spatial relationship |
| Gripper velocity | 3D | No -- needs frame stacking |
| Object velocity | 3D | No -- needs frame stacking |
| Angular velocity | 3D | No -- needs frame stacking |
| Gripper state | 2D | Marginally visible |

The critical quantities are positions and their spatial relationships.
A CNN must extract: "WHERE is the gripper?" and "WHERE is the object?"
This is fundamentally a **spatial coordinate extraction** problem.

### Why flattening destroys spatial structure

NatureCNN produces a 7x7x64 feature map and flattens it to 3,136 dimensions.
This destroys the 2D spatial layout. The MLP that receives this vector must
re-learn spatial relationships from scratch -- a much harder problem than
preserving them through the architecture.

Example: if the gripper activates feature map position (3,4) and the object
activates position (5,4), flattening places these at indices 220 and 284 in a
3,136-dim vector. The MLP must discover that "index 220 and 284 being active
means the gripper is 2 units left of the object." With different random
initialization, these indices would be different. The MLP cannot easily
generalize.

### Three architectural solutions

#### Solution A: Spatial Softmax (Levine et al. 2016, arXiv:1504.00702)

Instead of flattening, apply softmax over spatial dimensions per channel,
then compute expected (x,y) coordinates:

```python
# Feature map: (B, C, H, W)
# 1. Reshape to (B*C, H*W)
# 2. Softmax over spatial dims: attention = softmax(features / temperature)
# 3. Expected position:
#    expected_x = sum(pos_x * attention)  where pos_x = linspace(-1, 1, W)
#    expected_y = sum(pos_y * attention)  where pos_y = linspace(-1, 1, H)
# 4. Output: (B, 2*C) -- two spatial coordinates per channel

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, num_channels, temperature=1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = nn.Parameter(
            torch.ones(1) * temperature, requires_grad=True
        )
        # Create coordinate grids
        pos_x = torch.linspace(-1.0, 1.0, width)
        pos_y = torch.linspace(-1.0, 1.0, height)
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, -1, 1))

    def forward(self, features):
        # features: (B, C, H, W)
        B, C, H, W = features.shape
        # Softmax over spatial dims
        softmax_attention = F.softmax(
            features.reshape(B, C, -1) / self.temperature, dim=-1
        ).reshape(B, C, H, W)
        # Expected coordinates
        expected_x = (softmax_attention.sum(dim=2) * self.pos_x).sum(dim=-1)
        expected_y = (softmax_attention.sum(dim=3) * self.pos_y).sum(dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)  # (B, 2C)
```

**Why this helps for Push:** The output is literally (x,y) coordinates per
feature channel. If channel 5 detects "red puck" and channel 12 detects
"gripper finger," the output contains the puck's (x,y) and the gripper's
(x,y) as explicit numbers. The MLP can directly compute distance, direction,
and relative position from these coordinates.

**Output dimensionality:** 2C values. With 32 channels: 64 dimensions.
Compare to NatureCNN's 3,136-dim flattened output or the 50-dim linear
projection. Spatial softmax is compact and semantically meaningful.

**Papers using spatial softmax for manipulation:**
- Levine et al. 2016 (arXiv:1504.00702): end-to-end visuomotor policies,
  PR2 robot screwing/insertion/hanging
- Finn et al. 2016 (arXiv:1509.06113): deep spatial autoencoders for
  visuomotor learning, PR2 pushing/spatula/rope
- Widely used in Berkeley/Google manipulation pipelines (GPS, guided policy
  search line of work)

#### Solution B: CoordConv (Liu et al. 2018, arXiv:1807.03247)

Add 2 extra input channels containing coordinate grids:

```python
class CoordConv2d(nn.Module):
    """Conv2d with concatenated (x, y) coordinate channels."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)

    def forward(self, x):
        B, C, H, W = x.shape
        # Create coordinate grids normalized to [-1, 1]
        xx = torch.linspace(-1, 1, W, device=x.device).reshape(1, 1, 1, W)
        yy = torch.linspace(-1, 1, H, device=x.device).reshape(1, 1, H, 1)
        xx = xx.expand(B, 1, H, W)
        yy = yy.expand(B, 1, H, W)
        x = torch.cat([x, xx, yy], dim=1)
        return self.conv(x)
```

**Why this helps:** Standard convolutions are translation-equivariant -- they
produce the same local features regardless of where an object is in the image.
For manipulation, absolute position matters: "gripper at (0.3, 0.7)" is
different from "gripper at (0.7, 0.2)." CoordConv gives the network access to
absolute spatial coordinates at zero extra cost.

**Results from the paper:**
- Standard CNN fails at coordinate regression (rendering a pixel at given (x,y))
- CoordConv solves it 150x faster with 10-100x fewer parameters
- Improves object detection by 24% IOU on MNIST
- Benefits Atari RL agents

#### Solution C: DrQ-v2 style encoder (gentler downsampling)

Replace NatureCNN's aggressive first layer with smaller kernels:

```python
class ManipulationEncoder(nn.Module):
    """CNN encoder with gentle spatial downsampling.

    Based on DrQ-v2 (Yarats et al. 2021) with modifications for
    manipulation: smaller kernels preserve spatial detail for
    small objects (gripper, puck).
    """

    def __init__(self, obs_shape, feature_dim=50):
        super().__init__()
        C, H, W = obs_shape  # e.g., (12, 84, 84) with frame_stack=4
        self.convnet = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1),  # 84->42
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 42->42
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 42->42
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 42->21
            nn.ReLU(),
        )
        # Compute flattened size dynamically
        dummy = torch.zeros(1, C, H, W)
        flat_size = self.convnet(dummy).reshape(1, -1).shape[1]
        self.trunk = nn.Sequential(
            nn.Linear(flat_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        h = self.convnet(obs / 255.0)  # normalize pixels
        return self.trunk(h.reshape(h.shape[0], -1))
```

**Spatial preservation comparison:**

```
NatureCNN:  84 -> 20 -> 9 -> 7     (first layer: 4x reduction)
DrQ-v2:     84 -> 42 -> 42 -> 42   (first layer: 2x reduction)
Proposed:   84 -> 42 -> 42 -> 42 -> 21  (gradual reduction)
```

A 4-pixel object in the input becomes:
- NatureCNN layer 1: ~1 pixel (destroyed)
- DrQ-v2 layer 1: ~2 pixels (preserved)
- Proposed layer 1: ~2 pixels (preserved)

---

## 4. Resolution: 84x84 is Probably Fine

### Information content analysis

At 84x84, the Fetch workspace (~50cm x 50cm visible area) maps to:
- 1 pixel ~= 6mm
- Gripper (~4cm across) ~= 6-7 pixels
- Puck (~3cm across) ~= 5 pixels
- Goal marker (~2cm) ~= 3 pixels
- Gripper-puck distance when nearby (~5cm) ~= 8 pixels

This is marginal but sufficient. The issue is not pixel count but how the
CNN processes these pixels.

### Evidence that 84x84 works for manipulation

- DrQ-v2 solves DMControl's finger_spin and reacher_hard at 84x84
  (arXiv:2107.09645) -- these are manipulation tasks requiring precise control
- RIG solves Sawyer pushing at 84x84 (arXiv:1807.04742)
- CURL achieves strong results on DMControl manipulation at 84x84
  (arXiv:2004.04136)

### When higher resolution helps

- Pre-trained encoders (R3M, VIP, VC-1) use 224x224 because they leverage
  ImageNet-pretrained backbones designed for that resolution
- Real-world systems (RT-1) use 256-300 because real camera sensors provide
  higher resolution and downsampling wastes information
- For simulation, the render cost scales with resolution. At 84x84 with 8
  parallel envs, rendering is already the bottleneck (~60% of wall time)

### Recommendation

Stay at 84x84. Higher resolution would increase rendering cost significantly
without addressing the core architectural problem. If we fix the encoder and
still see issues, resolution increase is a second-order optimization.

---

## 5. What Actually Solves Pixel Manipulation

### Published approaches

**RIG -- Reinforcement with Imagined Goals**
(Nair et al. 2018, arXiv:1807.04742)

- Task: Sawyer pushing from 84x84 pixels
- Approach: train convolutional VAE (beta-VAE, 4D latent), then RL (SAC)
  in latent space. Goals are images encoded to latent vectors.
- Key insight: the VAE latent space provides a compact, smooth representation
  where Euclidean distance is meaningful. RL operates on 4D vectors, not
  84x84 images.
- Result: successfully learns pushing from pixels

**Skew-Fit** (Pong et al. 2019, arXiv:1903.03698)

- Extension of RIG with improved goal sampling (skew the distribution toward
  under-explored goals)
- 48x48 or 84x84, VAE + SAC
- Result: improved exploration, solves visual pushing on real Sawyer

**DrQ-v2** (Yarats et al. 2021, arXiv:2107.09645)

- Task: DMControl visual control (including manipulation: finger_spin,
  reacher_hard)
- Approach: custom encoder (3x3 kernels, 32ch, 4 layers) + LayerNorm + Tanh +
  random shift augmentation + n-step returns + exploration schedule
- Key: NO goal conditioning, NO HER. These are dense-reward tasks.
- Result: state-of-the-art sample efficiency on visual DMControl

**DayDreamer** (Hafner et al. 2023, arXiv:2206.14176)

- Applied Dreamer world model to real robots (pick-and-place from 64x64)
- Key: world model learns visual dynamics, policy trained in "imagination"
- Result: learns manipulation from pixels on real hardware with ~1 hour of data

### Key observation: nobody does pixel HER on manipulation

There is a gap in the literature. The approaches that solve pixel manipulation
either:
1. Use a VAE to compress images to low-dim latent space (RIG, Skew-Fit)
2. Use dense rewards without goal conditioning (DrQ-v2)
3. Use a world model (DreamerV3, DayDreamer)
4. Use pre-trained visual encoders (R3M, VIP)

**No published work solves goal-conditioned manipulation from pixels using
end-to-end CNN + HER + SAC.** This is what we are attempting.

This is either a publishing gap (nobody tried it), or an indication that
the approach is fundamentally harder than alternatives. The VAE approach
(RIG) works because it separates representation learning from policy learning.
Our approach asks the CNN to learn good features while simultaneously learning
the policy -- a harder optimization problem.

---

## 6. The Proprioception Insight

### The problem: asking the CNN to do two jobs at once

Our current `goal_mode="both"` setup provides:
- **Pixels** (84x84x12 with frame stacking) -- the CNN input
- **achieved_goal** (3D) -- object position for HER relabeling
- **desired_goal** (3D) -- target position for HER relabeling

But we are NOT passing the robot's **own state**. The Fetch observation dict
contains a `observation` vector with rich proprioceptive data:

```
End-effector position   (3D)  -- where the gripper is in 3D space
End-effector velocity   (3D)  -- how fast the gripper is moving
Gripper finger widths   (2D)  -- how open the gripper is
Gripper finger vel      (2D)  -- how fast fingers are moving
--- For Push, additionally: ---
Object position         (3D)  -- where the puck is
Object rotation         (3D)  -- puck orientation
Object velocity         (3D)  -- puck linear velocity
Object angular velocity (3D)  -- puck angular velocity
Relative position       (3D)  -- gripper-to-object vector
```

Without proprioception, the CNN must solve TWO visual problems simultaneously:
1. "Where is the gripper?" -- solvable from joint encoders, no camera needed
2. "Where is the object?" -- requires vision

This doubles the CNN's burden unnecessarily.

### How proprioception changes the architecture

With proprioception included:

```
Pixels (84x84x12) -> CNN -> visual features (e.g., 64D via spatial softmax)
Robot proprioception (10D) -> identity (or small MLP)
Goal vectors (6D) -> identity
                              -> concatenate -> 80D total
```

The CNN no longer needs to figure out WHERE THE GRIPPER IS (the proprioception
tells it directly with millimeter precision). The CNN only needs to detect the
object -- a much simpler visual task. This is also how real robots work: they
have joint encoders for self-state and cameras for world perception.

### Information asymmetry principle

This is an application of the information asymmetry principle: **use the
cheapest sensor for each piece of information.**

| Information | Best sensor | Current setup | With proprioception |
|-------------|------------|--------------|---------------------|
| Gripper position | Joint encoder (exact) | CNN must infer from pixels | Direct from state vector |
| Gripper velocity | Joint encoder (exact) | Frame stacking (approximate) | Direct from state vector |
| Gripper state | Joint encoder (exact) | CNN must infer from pixels | Direct from state vector |
| Object position | Camera (only option) | CNN must infer | CNN must infer |
| Object velocity | Camera + temporal | Frame stacking | Frame stacking |
| Goal position | Given by env | Goal vector | Goal vector |

With proprioception, the CNN's task simplifies from "understand the entire
scene" to "find the object." This is a much easier representation learning
problem.

### What to pass as proprioception

For FetchPush, the most useful proprioceptive signals are:

**Full proprioception (10D):**
```
observation[:3]   -- end-effector position (x, y, z)
observation[3:6]  -- end-effector velocity (dx, dy, dz)
observation[6:8]  -- gripper finger widths
observation[8:10] -- gripper finger velocities
```

**Minimal proprioception (5D):**
```
observation[:3]   -- end-effector position (x, y, z)
observation[6:8]  -- gripper finger widths
```

The minimal version gives the agent precise knowledge of its own position
and gripper state, while velocity can still come from frame stacking.

### Note on what NOT to pass

We should NOT pass the object-related state dimensions (object position,
velocity, rotation) from the observation vector. Those are exactly what
the CNN should learn to extract from pixels. Passing them would defeat
the purpose of pixel learning. Similarly, the relative position
(gripper-to-object) should be computed by the network, not given.

### Implementation in SB3's Dict observation space

Currently our PixelObservationWrapper produces:

```python
obs_dict = {
    "pixels": np.array (12, 84, 84),       # stacked frames
    "achieved_goal": np.array (3,),          # object position
    "desired_goal": np.array (3,),           # target position
}
```

With proprioception, it would become:

```python
obs_dict = {
    "pixels": np.array (12, 84, 84),        # stacked frames
    "proprioception": np.array (10,),        # robot state
    "achieved_goal": np.array (3,),          # object position
    "desired_goal": np.array (3,),           # target position
}
```

SB3's `CombinedExtractor` (and our `NormalizedCombinedExtractor`) already
handles mixed dict spaces: image keys get the CNN, vector keys get Flatten.
The proprioception vector would be concatenated alongside goal vectors after
the CNN features. Minimal code change in the wrapper.

### Impact on feature concatenation balance

With our current NormalizedCombinedExtractor (cnn_dim=50):

```
Without proprioception:
  50 CNN + 3 achieved_goal + 3 desired_goal = 56 total (goals = 10.7%)

With proprioception (10D):
  50 CNN + 10 proprio + 3 achieved_goal + 3 desired_goal = 66 total
  Known vectors (16D) = 24.2% of input  (vs 10.7% without)
```

With spatial softmax (2 * 32 = 64D):

```
  64 spatial + 10 proprio + 3 achieved_goal + 3 desired_goal = 80 total
  Known vectors (16D) = 20.0% of input
```

In both cases, the clean, known-scale vectors become a larger fraction
of the input, further reducing the CNN noise dominance problem.

### Book narrative consideration

Adding proprioception shifts the story from "pure pixel learning" to
"vision-augmented control" -- but this is actually MORE realistic. Real
robotic systems always combine joint encoders with cameras. No production
robot relies solely on pixels for its own state. Framing this honestly
in the book is pedagogically valuable:

> "A real robot always knows where its arm is -- joint encoders provide
> millimeter-precision state feedback. The camera's job is to tell the
> robot about the WORLD: where the object is, where obstacles are, what
> the scene looks like. Asking a CNN to also figure out the robot's own
> state from pixels is asking it to solve a problem that's already solved
> by cheaper, more precise sensors."

---

## 7. Grounded Suggestions: What to Try Next

### Revised priority ranking

Based on all research findings and the proprioception insight, here is the
revised intervention ranking, ordered by expected impact:

### Tier 1: Replace NatureCNN with DrQ-v2 encoder (highest confidence)

**Replace NatureCNN with a manipulation-appropriate encoder.**

This is the single highest-impact change we haven't tried. The evidence is
strong: every modern visual RL paper uses gentler downsampling than NatureCNN.

Concrete implementation:

```
Option A: DrQ-v2 encoder (proven, minimal code)
  - 4 layers, 3x3 kernels, 32 channels, stride 2 only in first layer
  - LayerNorm + Tanh on output (we already have this)
  - Feature dim 50 (we already have this)
  - Change: first layer kernel and stride ONLY

Option B: DrQ-v2 encoder + spatial softmax (higher risk, higher potential)
  - Same conv layers as Option A
  - Replace flatten + linear with spatial softmax
  - Output: 2 * 32 = 64 values (explicit spatial coordinates)
  - This directly addresses the "where is the object?" problem
```

**Implementation effort:** ~50 lines of new code in visual_encoder.py.
We can make this a drop-in replacement for NatureCNN by following SB3's
BaseFeaturesExtractor interface.

**Expected impact:** The DrQ-v2 encoder is proven on DMControl manipulation.
Combined with our existing normalization and frame stacking, this addresses
the spatial information preservation problem. However, the goal-conditioning
+ HER component is still uncharted territory.

### Tier 2: Add spatial softmax output (medium-high confidence)

Replace flatten with spatial softmax:

```
Conv layers (DrQ-v2 style) -> feature map (32, H, W)
-> SpatialSoftmax -> (64,) explicit (x,y) coordinates
-> concatenate with goal vectors (6,) [+ proprioception (10D)]
-> total: 80 features (known vectors are 20% of input)
```

This is the architecture Levine et al. (2016) used for real-robot manipulation
from pixels. The output is semantically meaningful: literal spatial coordinates
of detected features. This directly addresses the "WHERE is the object?"
problem that manipulation needs to solve.

**Risk:** Spatial softmax assumes each channel detects ONE thing with ONE
location. If a channel responds to multiple locations (e.g., both gripper
fingers), the expected position averages them -- which might actually be fine
(it gives the center between the fingers).

### Tier 3: Add proprioception (medium confidence, low cost)

Pass the robot's own state (end-effector position, velocity, gripper state)
alongside pixel features. This halves the CNN's burden: it only needs to
detect the object, not the robot itself.

```
Pixels -> CNN -> spatial features
Robot state (10D) -> identity
Goal vectors (6D) -> identity
-> concatenate -> MLP policy/critic
```

**Implementation effort:** ~15 lines in pixel_wrapper.py (add
"proprioception" key to obs dict, extract from underlying env observation).

**Expected impact:** Significant -- the CNN no longer needs to solve the
"where is the gripper?" problem, which is arguably harder than "where is the
object?" (the gripper has complex geometry with moving fingers, while the
puck is a simple colored cylinder).

**Caveat for the book:** This makes the setup "pixels + proprioception"
rather than "pure pixels." We should be honest about this distinction.
However, it's the realistic setup for actual robotic systems.

### Tier 4: CoordConv first layer (low cost, uncertain impact)

Add coordinate channels to the first conv layer:

```
Input: (12, 84, 84) with frame_stack=4
-> CoordConv: concat (x_grid, y_grid) -> (14, 84, 84)
-> Conv2d(14, 32, kernel=3, stride=2) -> standard from here
```

**Cost:** 2 extra input channels to the first layer. ~0.1% more parameters.
**Benefit:** gives the network access to absolute spatial position from the
start, without needing to infer it from image content.

### Tier 5: Separate representation learning (if all else fails)

If end-to-end CNN+HER+SAC cannot learn Push from pixels, the fallback is
to separate representation learning from policy learning:

**Option A: VAE approach (RIG-style)**
- Train a convolutional VAE on randomly collected pixel observations
- Encode observations and goals to latent vectors
- Run SAC+HER entirely in latent space
- Pros: proven to work for pushing (Nair et al. 2018)
- Cons: adds a pre-training phase, two-stage pipeline

**Option B: Frozen pre-trained encoder**
- Use R3M or VIP (224x224, ResNet-50) as a frozen feature extractor
- Fine-tune only the policy/critic on top
- Pros: high-quality features immediately
- Cons: requires downloading pre-trained weights, changes the narrative
  from "learning from scratch"

For the book's narrative, Tier 5 would change the story from "we solved
Push from pixels with engineering fixes" to "we need specialized
representation learning" -- which is still pedagogically valuable but
less satisfying.

---

## 8. Implementation Plan for Tier 1 + 2 + 3

### Step 1: Add ManipulationEncoder to visual_encoder.py (~80 lines)

```python
class ManipulationEncoder(BaseFeaturesExtractor):
    """DrQ-v2 style encoder with optional spatial softmax.

    Differences from NatureCNN:
    - 3x3 kernels (vs 8/4/3): preserves spatial detail
    - Stride 2 only in layers 1 and 4 (vs stride 4 in layer 1)
    - 32 channels throughout (vs escalating 32/64/64)
    - Optional spatial softmax output (vs flatten)
    - LayerNorm + Tanh on output (DrQ-v2 trunk pattern)
    """
```

Three variants behind a single class with a `spatial_softmax` flag:
- `ManipulationEncoder(spatial_softmax=False)`: DrQ-v2 encoder + flatten + LN + Tanh
- `ManipulationEncoder(spatial_softmax=True)`: DrQ-v2 conv layers + spatial softmax + LN + Tanh

### Step 2: Add SpatialSoftmax module to visual_encoder.py (~30 lines)

Standalone `SpatialSoftmax` module as documented in Section 3, Solution A.
Learnable temperature parameter. Output: (B, 2C) explicit coordinates.

### Step 3: Add proprioception to PixelObservationWrapper (~15 lines)

New parameter `include_proprioception: bool = False`:
- When True, adds `"proprioception"` key to the observation dict
- Extracts robot-only dimensions from the underlying env's `observation` vector
- For FetchPush: first 10 dimensions (EE pos, EE vel, gripper widths, gripper vel)
- Does NOT include object state dimensions (that's the CNN's job)

```python
# In pixel_wrapper.py, observation() method:
if self._include_proprioception:
    obs_dict["proprioception"] = state_obs[:10]  # robot-only dims
```

### Step 4: Add CLI flags to ch10_visual_reach.py

New arguments:
- `--encoder {nature,drqv2,drqv2-spatial}`: encoder architecture choice
  - `nature`: current NatureCNN (default, backward compatible)
  - `drqv2`: DrQ-v2 style 4-layer encoder + flatten + LN + Tanh
  - `drqv2-spatial`: DrQ-v2 conv layers + spatial softmax + LN + Tanh
- `--proprio`: include proprioceptive state alongside pixels

Wiring:
- `--encoder drqv2` or `drqv2-spatial` -> `ManipulationEncoder` in policy_kwargs
- `--proprio` -> `include_proprioception=True` in PixelObservationWrapper

### Step 5: Smoke tests (2K steps each)

| Test | Config | Verify |
|------|--------|--------|
| drqv2 encoder | `--encoder drqv2 --norm` | Feature dims correct, no crash |
| drqv2-spatial | `--encoder drqv2-spatial --norm` | Spatial softmax output 2C dims |
| drqv2-spatial + proprio | `--encoder drqv2-spatial --norm --proprio` | Proprio vector flows through |
| All with frame stack | Add `--frame-stack 4` to above | Channel dims correct (12ch input) |

### Step 6: Training runs (priority order)

**Primary runs** (launch in parallel, 8M steps each):

| # | Config | Rationale |
|---|--------|-----------|
| 1 | `--encoder drqv2-spatial --norm --proprio --frame-stack 4 --learning-rate 1e-4` | Full pipeline: best encoder + spatial softmax + proprioception. Maximizes chances of success. |
| 2 | `--encoder drqv2-spatial --norm --frame-stack 4 --learning-rate 1e-4` | Same but without proprioception. Tests whether spatial softmax alone is sufficient. |

Both with DrQ augmentation (train-push-pixel-her-drq subcommand).

**Secondary runs** (if primary succeed, for ablation):

| # | Config | What it isolates |
|---|--------|-----------------|
| 3 | `--encoder drqv2 --norm --proprio --frame-stack 4 --learning-rate 1e-4` | DrQ-v2 encoder + proprio but no spatial softmax |
| 4 | `--encoder drqv2 --norm --frame-stack 4 --learning-rate 1e-4` | DrQ-v2 encoder alone (minimal change from current) |

### Step 7: Ablation (if primary runs work)

Isolate which interventions mattered most:
- Encoder alone (drqv2 vs nature) -- spatial preservation
- Spatial softmax (drqv2 vs drqv2-spatial) -- explicit coordinate extraction
- Proprioception (with vs without --proprio) -- halving CNN burden
- Frame stacking (fs=4 vs fs=1) -- velocity information
- Normalization (--norm vs default) -- feature scale balance

---

## 9. Comparison with State+HER Diagnostics

To understand what "working" looks like in the training logs, here are the
state+HER diagnostics for reference:

### State+HER on FetchPush (the gold standard)

```
200K:  success=3%,   reward=-48.3, critic_loss=0.27, actor_loss=0.57
500K:  success=11%,  reward=-44.5, critic_loss=0.03, actor_loss=1.45
1000K: success=4%,   reward=-47.8, critic_loss=0.08, actor_loss=1.39
1500K: success=46%,  reward=-35.8, critic_loss=0.46, actor_loss=1.73  <- inflection
2000K: success=89%,  reward=-20.1, critic_loss=0.42, actor_loss=2.05
```

Key signatures of the hockey-stick inflection:
1. **Critic loss INCREASES** from 0.08 to 0.46 between 1000K and 1500K.
   This means the critic is facing harder value estimation problems as the
   policy discovers new states (object moving, approaching goal).
2. **Actor loss increases steadily** from 0.57 to 2.05. The critic provides
   increasingly strong gradient signals as it learns better Q-values.
3. **Episode reward drops sharply** from -48 to -20. The policy is achieving
   goals and collecting less penalty.

### All pixel runs (baseline and fix)

```
200K-3000K: success=3-13%, reward=-44 to -49, critic_loss=0.10-0.28,
            actor_loss=0.14-2.8 (varies by config but all converge to ~1.6)
```

The critic loss never increases. The policy never discovers new states. The
training dynamics are fundamentally different from state+HER.

### What to look for in Tier 1 runs

If the encoder fix works:
1. Critic loss should INCREASE around the inflection point (policy discovers
   new states, critic must fit harder value functions)
2. Episode reward should start dropping toward -30, -20 (policy achieving goals)
3. Success rate should show sustained upward movement above 15%, not just
   transient spikes

If the critic loss stays flat/decreasing and success rate oscillates in the
5-13% band, the encoder change alone is insufficient.

---

## 10. Book Narrative Implications

### If the encoder + proprioception fix works

The Ch9 narrative becomes a five-act engineering story:

```
Act 1: Measure the pixel penalty (Reach: state vs pixel vs DrQ)
Act 2: The real test -- Push fails from dense, HER solves from state
Act 3: Naive pixel+HER fails -- WHY? (iterative diagnosis)
  -> Hypothesis 1: scale mismatch (CNN noise drowns goal vectors)
     Fix: LayerNorm + Tanh (DrQ-v2 trunk pattern) -> no improvement
  -> Hypothesis 2: missing velocity (single frame)
     Fix: frame stacking (4 frames) -> no improvement
  -> Hypothesis 3: NatureCNN destroys spatial information
     8x8 stride-4 first layer reduces 4-pixel objects to 1 pixel
  -> Hypothesis 4: CNN is solving two problems at once
     "Where is my gripper?" + "Where is the object?"
     Fix: give it proprioception for self-state
Act 4: The right architecture
  -> DrQ-v2 encoder (3x3 kernels, gentle downsampling)
  -> Spatial softmax (explicit coordinate extraction)
  -> Proprioception (robot knows itself, camera sees the world)
  -> Push from pixels works: [PENDING]% success
Act 5: What we learned (principles, not tricks)
  -> Architecture matters more than hyperparameters
  -> Preserve spatial resolution in early CNN layers
  -> For manipulation: extract WHERE things are, not WHAT they look like
  -> Use the cheapest sensor for each piece of information
  -> Real robots always combine joint encoders with cameras
```

This is an honest engineering story: hypothesis -> test -> revise -> test
again -> revise again. The reader learns that diagnosis is iterative and
that the first fix you try is rarely the right one.

### If the encoder fix doesn't work (all tiers exhausted)

The chapter resolves on Reach (pixels work for easy tasks) and documents
Push as requiring specialized representation learning. The Further Reading
section (already drafted in tasks/draft_ch9_further_reading.md) points to
VAEs (RIG), pre-trained encoders (R3M, VIP), and world models (DreamerV3)
as the frontier approaches.

This is still valuable: the reader learns the LIMITS of end-to-end visual RL,
which is arguably more important than a success story that only works under
specific conditions.

### The pedagogical value of iterative failure

Whether or not the fix works, the iterative diagnosis process itself is
highly pedagogical. The reader sees:

1. A clear problem (pixel Push fails)
2. A principled hypothesis (scale mismatch)
3. An evidence-based fix (normalization)
4. Honest evaluation (didn't work)
5. A revised hypothesis informed by deeper analysis (architecture + proprioception)
6. A second fix attempt

This teaches the reader that real engineering is NOT "find the right
hyperparameter" but "understand the problem deeply enough that the solution
becomes obvious." Each failed hypothesis narrows the space and teaches
something real about how vision and RL interact.

---

## 11. Tutorial Notes (for future chapter writing)

### The Two Kinds of "Visual HER"

This is a subtle distinction that the tutorial should make explicit, because
it's easy for readers to conflate them:

**Our approach: HER on vectors, policy on pixels (goal_mode="both")**

```
Policy sees:     pixels (84x84x12)  -- honest visual learning
HER operates on: achieved_goal (3D) + desired_goal (3D) -- vector relabeling
```

HER's relabeling is trivial: swap the 3D desired_goal vector with a 3D
achieved_goal from a future timestep. This is identical to standard state-based
HER. The pixel observations are irrelevant to the relabeling process.

**Full visual HER (Nair et al. 2018, RIG): HER on images, policy on images**

```
Policy sees:     pixels (84x84x3)
Goal is:         a goal IMAGE (84x84x3)
HER operates on: image-space relabeling
```

Here the goal itself is an image, and HER must replace the goal image with
the achieved image from a future timestep. This is harder because:
1. The reward must be computed in image space (or latent space via a VAE)
2. Goal images must be semantically meaningful (not just any frame)
3. The policy must understand "what does the goal image want me to achieve?"

**Why this matters pedagogically:**

Our `goal_mode="both"` setup is an elegant middle ground that the reader should
understand as a deliberate design choice:
- The POLICY learns from pixels (no cheating on the perception side)
- HER uses vectors (efficient, exact goal relabeling)
- The INFORMATION ASYMMETRY is intentional: the policy is harder (pixels)
  while the goal-conditioning mechanism is easy (vectors)

This should be framed in the tutorial as: "We are testing whether a CNN can
learn to CONTROL from pixels, not whether it can learn to SPECIFY GOALS
from pixels. Goal specification from images is a separate, harder problem
(Nair et al. 2018) that we discuss in Further Reading."

### Proprioception as Honest Engineering

The tutorial should frame proprioception not as "cheating" but as honest
engineering practice:

> "A robot that ignores its joint encoders and tries to infer its own arm
> position from a camera image is not being more 'pure' -- it is being
> wasteful. Joint encoders provide millimeter-precision state at microsecond
> latency. Cameras provide rich scene understanding at millisecond latency.
> Using both is not a compromise; it is the right engineering decision.
>
> In our experiments, adding proprioception halves the CNN's burden: it no
> longer needs to solve 'where is my arm?' (answered by joint encoders) and
> can focus entirely on 'where is the object?' (only answerable by the camera).
> This mirrors how every real robotic system operates."

### The Iterative Diagnosis Story

The tutorial should present the failure-diagnosis-revision cycle honestly:

1. "We hypothesized scale mismatch. We fixed it. It didn't help."
2. "We hypothesized missing velocity. We added frame stacking. It didn't help."
3. "We looked deeper: NatureCNN's architecture destroys spatial information."
4. "We realized the CNN was solving two problems it didn't need to solve both."

This teaches the reader that debugging RL systems is ITERATIVE. The first
hypothesis is rarely correct. Each failed fix narrows the search space and
teaches something real about how vision and RL interact. This is more
valuable than a recipe that "just works."

### The Sensor Separation Principle (key tutorial concept)

This deserves a formal definition block in the tutorial (following our
5-step definition template from CLAUDE.md):

**1. Motivating problem:** When we give a robot only camera images, the CNN
must learn to answer two fundamentally different questions: "where is MY arm?"
and "where is the OBJECT?" The first question has a much easier answer
available -- joint encoders already know the arm's position with millimeter
precision.

**2. Intuitive description:** Use the cheapest, most precise sensor for each
piece of information. Joint encoders for self-state, cameras for world-state.
Don't make the CNN solve problems that cheaper sensors already solve.

**3. Formal statement:** Given an observation space partitioned into
self-state $s_{\text{self}}$ (robot proprioception, directly measurable) and
world-state $s_{\text{world}}$ (object positions, only observable via sensors
like cameras), the optimal architecture provides $s_{\text{self}}$ as a
direct vector input and uses the CNN only to extract $s_{\text{world}}$
from pixels. The CNN's learning problem reduces from
$f: \text{pixels} \to (s_{\text{self}}, s_{\text{world}})$ to
$g: \text{pixels} \to s_{\text{world}}$, which is strictly simpler.

**4. Grounding example (the architecture diagram):**

```
Pixels (84x84x12) -> CNN -> spatial features (64D)    [world-state]
Robot proprioception (10D) -> identity                  [self-state]
Goal vectors (6D) -> identity                           [task spec]
                              -> concatenate -> 80D total
```

The CNN no longer needs to figure out where the gripper is (proprioception
tells it directly). The CNN only needs to learn where the object is -- a
much simpler visual task. This is also how real robots work: they have
joint encoders for their own state and cameras for the world.

**5. Non-example:** Passing object-state information (object position, velocity)
alongside pixels would defeat the purpose of pixel learning entirely. The
principle is: give the agent what it ALREADY KNOWS through non-visual sensors
(its own body), but make it LEARN what it can only know through vision
(the world around it).

This principle generalizes beyond our specific setup:
- Self-driving cars: IMU + wheel encoders for ego-state, cameras for road/traffic
- Drones: IMU + barometer for attitude/altitude, cameras for navigation
- Industrial robots: joint encoders for tool position, cameras for part detection

The tutorial should present this as a general engineering principle, not a
task-specific hack.

---

## 12. Summary of Key Decisions

### What we now know

1. **84x84 resolution is sufficient** -- the information is there, the
   encoder just can't extract it (Section 4)
2. **NatureCNN's 8x8 stride-4 first layer destroys spatial detail** --
   small objects (3-5 pixels) become 1 pixel after layer 1 (Section 2)
3. **Normalization fixes gradient scale but not feature quality** --
   clean, bounded garbage is still garbage (Section 1)
4. **Frame stacking alone doesn't help** -- velocity from frame differences
   is useless if the CNN can't extract positions in the first place
5. **The CNN is solving two problems at once** -- finding the gripper AND
   finding the object, when it only needs to do the latter (Section 6)
6. **Nobody has published pixel+HER on manipulation** -- we're in
   uncharted territory, which explains why there's no recipe (Section 5)
7. **Spatial softmax is the manipulation standard** -- Levine, Finn, and the
   Berkeley/Google manipulation pipelines all use it (Section 3)

### The recommended full pipeline

```
Input:
  Pixels (84x84 x 4 frames = 12 channels)
  Proprioception (10D: EE pos/vel, gripper state)
  Goals (6D: achieved + desired)

Encoder (DrQ-v2 style):
  Conv2d(12, 32, 3x3, stride=2)  -> 42x42    [gentle downsampling]
  Conv2d(32, 32, 3x3, stride=1)  -> 42x42
  Conv2d(32, 32, 3x3, stride=1)  -> 42x42
  Conv2d(32, 32, 3x3, stride=2)  -> 21x21

Output layer:
  SpatialSoftmax(32, 21, 21) -> 64D (explicit x,y coords per channel)
  LayerNorm(64) -> Tanh -> bounded [-1, 1]

Concatenation:
  64D spatial + 10D proprio + 3D achieved + 3D desired = 80D total
  Known vectors (16D) = 20% of input

Algorithm:
  SAC + HER (off-policy, goal relabeling on vectors)
  DrQ random shift augmentation on pixels
  lr=1e-4, ent_coef=0.05, gradient_steps=1
  Buffer: 300K (frame stacking memory constraint)
```

---

## 13. References

| Paper | arXiv | Key contribution |
|-------|-------|-----------------|
| NatureCNN (Mnih et al. 2015) | 1312.5602 | 3-layer CNN for Atari, 8x8 stride-4 first layer |
| DrQ (Kostrikov et al. 2020) | 2004.13649 | Random shift augmentation for visual RL |
| DrQ-v2 (Yarats et al. 2021) | 2107.09645 | Custom encoder + LN + Tanh + n-step returns |
| TD-MPC2 (Hansen et al. 2023) | 2310.16828 | Decreasing kernel encoder + SimNorm + world model |
| DreamerV3 (Hafner et al. 2023) | 2301.04104 | Deep CNN encoder + RMSNorm + SiLU |
| SAC-AE (Yarats et al. 2019) | 1910.01741 | Autoencoder regularization for visual SAC |
| CURL (Srinivas et al. 2020) | 2004.04136 | Contrastive learning for RL representations |
| RAD (Laskin et al. 2020) | 2004.14990 | Random augmentation drives visual RL |
| Spatial softmax (Levine et al. 2016) | 1504.00702 | Explicit (x,y) extraction for visuomotor control |
| Spatial autoencoders (Finn et al. 2016) | 1509.06113 | Spatial softmax for PR2 manipulation |
| CoordConv (Liu et al. 2018) | 1807.03247 | Coordinate channels break translation equivariance |
| RIG (Nair et al. 2018) | 1807.04742 | VAE + RL for visual goal-conditioned pushing |
| Skew-Fit (Pong et al. 2019) | 1903.03698 | Improved visual goal sampling via distribution skewing |
| R3M (Nair et al. 2022) | 2203.12601 | ResNet pre-trained on Ego4D for manipulation |
| VIP (Ma et al. 2023) | 2210.00030 | Value-implicit pre-training for visual RL |
| VC-1 (Majumdar et al. 2023) | 2303.18240 | ViT-L pre-trained for manipulation |
| DayDreamer (Hafner et al. 2023) | 2206.14176 | World model for real robot manipulation |
| RT-1 (Brohan et al. 2022) | 2212.06817 | EfficientNet + Transformer for real-world manipulation |
| Fetch envs (Plappert et al. 2018) | 1802.09464 | State-only goal-conditioned manipulation benchmark |

---

## 14. Implementation -- What We Built (2026-02-22)

**Status:** Implementation complete, training runs show same failure pattern.

### Files created/modified

| File | Action | Lines | Purpose |
|------|--------|-------|---------|
| `scripts/labs/manipulation_encoder.py` | **Created** | ~350 | SpatialSoftmax, ManipulationCNN, ManipulationExtractor, verification |
| `scripts/ch09_pixel_push.py` | **Created** | ~990 | Training script: smoke-test, train, eval, compare, feature-stats |
| `scripts/labs/pixel_wrapper.py` | **Extended** | +30 | `proprio_indices` parameter for proprioception passthrough |

### Architecture as implemented

```
Pixels (84x84 x 4 frames = 12ch)
  -> ManipulationCNN (4x Conv2d 3x3, stride 2/1/1/2) -> (32, 21, 21) feature map
  -> SpatialSoftmax (per-channel expected x,y)         -> 64D
  -> LayerNorm -> Tanh                                 -> 64D bounded [-1, 1]

Proprioception (10D: grip_pos + gripper_state + grip_velp + gripper_vel)
  -> Flatten                                           -> 10D

Goals (achieved_goal + desired_goal)
  -> Flatten                                           -> 6D

-> concatenate -> 80D -> SAC MLP heads (256, 256) -> actions
```

Implementation verified via:
- `manipulation_encoder.py --verify`: all unit tests pass (SpatialSoftmax peak
  tests, ManipulationCNN shape, ManipulationExtractor features_dim=80, SB3
  integration with 50-step training)
- `ch09_pixel_push.py smoke-test`: 10K-step pipeline test passes
- `ch09_pixel_push.py feature-stats`: pixel features bounded [-0.40, 0.79],
  no saturation, SpatialSoftmax temperature=1.0

### Bugs fixed during implementation

1. **SpatialSoftmax broadcasting** (`manipulation_encoder.py:110`):
   `pos_y` buffer was shaped `(1, H, 1)` but needed `(1, 1, H)` for
   broadcasting against `softmax_attention.sum(dim=3)` which gives `(B, C, H)`.
   Fixed by changing `pos_y.reshape(1, -1, 1)` to `pos_y.reshape(1, 1, -1)`.

2. **SB3 SAC features_extractor is None** (`ch09_pixel_push.py` smoke test):
   With `share_features_extractor=False` (SB3 default), `model.policy.features_extractor`
   is None. Actor and critic have separate extractors. Fixed by checking
   `model.policy.actor.features_extractor` instead.

### Diagnostic additions (user-authored, 2026-02-22)

Five changes to enable clean ablations and diagnostics:

| Flag / Feature | File | Purpose |
|---------------|------|---------|
| `--no-pixels` | `ch09_pixel_push.py:87, 230-266` | Vector-only baseline via `ProprioGoalWrapper` |
| `--native-render` | `ch09_pixel_push.py:88, 216-218` | Render at 84x84 directly (skip PIL resize) |
| `--share-encoder` | `ch09_pixel_push.py:103, 365, 497` | Set `share_features_extractor=True` in SB3 |
| `feature-stats` | `ch09_pixel_push.py:817-890` | Per-key post-encoder stats (mean/std/min/max) |
| `include_images` | `manipulation_encoder.py:240, 251-252` | Skip image keys for vector-only mode |

Config tag encoding (`_config_tag`, line 171): `noPix`, `native`, `noSS`,
`noPro`, `noDrQ`, `shareEnc`, `fs{N}`. The `noDrQ` flag only appears when
pixels are enabled but DrQ is explicitly disabled (true ablation, not
implied by `--no-pixels`).

---

## 15. Training Results -- ManipulationCNN (2026-02-22)

### Primary run: full pipeline (ManipulationCNN + SpatialSoftmax + Proprio + DrQ + HER)

```
Container: ch09_pixel_push (Docker, --gpus all)
Command: python scripts/ch09_pixel_push.py train --seed 0
Config: pixels=True, spatial_softmax=True, proprio=True, drq=True,
        frame_stack=4, lr=1e-4, ent_coef=0.05, gamma=0.95,
        buffer_size=200K, batch_size=256, n_envs=4
```

| Steps | success_rate | ep_rew_mean | critic_loss | actor_loss | fps |
|-------|-------------|-------------|-------------|------------|-----|
| 162K  | 0.14        | -44.3       | 0.293       | 1.095      | 97  |
| 556K  | 0.02        | -48.5       | 0.090       | 1.645      | 23  |
| 1.66M | 0.06        | -47.7       | 0.066       | 1.706      | 23  |
| 1.92M | 0.03        | -48.2       | 0.070       | 1.610      | 23  |
| 2.0M  | 0.03        | -48.4       | 0.069       | --         | 23  |

**Verdict: FAIL.** Same flat 2-14% success pattern as NatureCNN.
Critic loss declining monotonically (0.29 -> 0.07), no hockey-stick
inflection. The "too easy" critic pattern indicates the policy is not
discovering new states.

Note: fps dropped from ~97 to ~23 at `learning_starts=1000` because each
step now includes ~7 CNN forward/backward passes (critic + actor + target).

### Feature diagnostics (pre-training)

```
feature-stats --seed 0 --n-samples 32:
  pixels:         dim=64  mean=0.0419 std=0.2001 min=-0.4030 max=0.7939
  proprioception: dim=10  mean=0.2476 std=0.4208 min=-0.0264 max=1.3581
  achieved_goal:  dim= 3  mean=0.7456 std=0.3349 min=0.4249 max=1.2078
  desired_goal:   dim= 3  mean=0.9124 std=0.4152 min=0.4249 max=1.4395
  Concatenated:   dim=80  mean=0.0948 std=0.3560 min=-0.4030 max=1.4395

SpatialSoftmax temperature: [1.0] (learnable, starts at 1.0)
```

Features are bounded, non-saturated, no NaN/Inf. The representation is
well-initialized -- the problem is not feature quality at init but
learning dynamics.

### Comparison with state+HER reference

| Config | @500K | @1M | @1.5M | @2M | Hockey stick? |
|--------|-------|-----|-------|-----|--------------|
| State+HER (Ch4) | 11% | 4%* | 46% | 89% | YES @ ~1.3M |
| NatureCNN+HER (Ch9 round 1-4) | 5-8% | 5-8% | 5-8% | 5-8% | No |
| ManipulationCNN+SS+Proprio+DrQ+HER | 2% | 6% | 6% | 3% | No |

*State+HER dips at 1M before the inflection -- this is normal HER
exploration dynamics, not failure.

---

## 16. Ablation Protocol (2026-02-22)

### 3-run ablation plan

| Run | Config | Container | Purpose | Status |
|-----|--------|-----------|---------|--------|
| 1 | `--no-pixels --n-envs 8 --buffer-size 1M --lr 3e-4 --total-steps 2M` | `ch09_run1_vector` | Prove HER+SAC works in this script | **RUNNING** (0% @ 1.67M) |
| 2 | `--share-encoder --total-steps 2M` | (pending Run 1) | Critic gradients through CNN | Pending |
| 3 | `--share-encoder --native-render --total-steps 2M` | (gated on Run 2) | Speed + invariance check | Pending |

### Run 1 result: vector-only POMDP

```
Container: ch09_run1_vector (Docker, --gpus all)
Command: python scripts/ch09_pixel_push.py train --seed 0 --no-pixels
         --n-envs 8 --buffer-size 1000000 --learning-rate 3e-4 --total-steps 2000000
Observation: {proprioception (10D), achieved_goal (3D), desired_goal (3D)} = 16D
```

| Steps | success_rate | ep_rew_mean | critic_loss | fps |
|-------|-------------|-------------|-------------|-----|
| 12K   | 0.09        | -45.3       | --          | 613 |
| 432K  | 0.04        | -47.9       | 0.023       | 571 |
| 1.67M | 0.00        | -48.6       | --          | 593 |

**Verdict: FAIL.** The 16D observation (proprio + goals, no object
dynamics) is not sufficient for Push. This is a POMDP -- the agent can see
where the object IS (achieved_goal) but not how it's MOVING (no object_velp,
object_velr). See Lesson 6 in `tasks/lessons.md`.

**CRITICAL CAVEAT:** This run does NOT answer "is the Ch9 pipeline correct?"
because it changed the observation simultaneously. Ch4's reference (99% @
1.86M) used the full 25D+6D=31D observation. A fair control would use the
same 31D observation through Ch9's script. The correct interpretation of
Run 1 is: "16D proprio+goals is insufficient for Push" -- not "the pipeline
is broken."

### Stop rules (from ablation protocol)

| Milestone | Run 1 target | Run 1 actual | Verdict |
|-----------|-------------|--------------|---------|
| 200K | success >= 3% | ~9% (12K early) then declining | Briefly met |
| 500K | success >= 10%, reward toward -45 | 4%, -47.9 | **MISSED** |
| 1.0M | stop if <= 10%, reward -47 to -50 | 0%, -48.6 | **STOP RULE HIT** |
| 1.5M | success >= 30% (hockey stick) | 0% | **FAIL** |

### Pre-flight diagnostics

Feature-stats for each config were verified before launch:
- Vector-only: 16D, all sane ranges, no NaN/Inf
- Pixel+share-encoder: 80D, pixel features bounded [-0.40, 0.79]

---

## 17. Current State and Next Steps

### What we know (updated)

| # | Finding | Evidence |
|---|---------|----------|
| 1 | 84x84 resolution sufficient | Literature + Section 4 analysis |
| 2 | NatureCNN stride-4 destroys spatial detail | 5 runs @ 8M steps, all flat |
| 3 | Normalization fixes gradients, not features | Lesson 5 |
| 4 | Frame stacking alone doesn't help | Run 3-4 in Section 1 |
| 5 | ManipulationCNN + SpatialSoftmax: necessary but not sufficient | Section 15 primary run |
| 6 | Proprioception alone doesn't help | Section 15 primary run includes proprio |
| 7 | 16D proprio+goals is POMDP, fails for Push | Section 16 Run 1 |
| 8 | No published pixel+HER on manipulation exists | Section 5 literature gap |
| 9 | Architecture change alone doesn't produce hockey-stick | Section 15 vs Section 9 comparison |

### Hypotheses for why ManipulationCNN still fails

**H1: Gradient flow (most likely).** With `share_features_extractor=False`
(SB3 default), actor and critic have separate CNN copies. The critic's CNN
learns from Q-value targets but its learned features never reach the actor.
The actor's CNN only gets policy gradient signal, which is noisy and
uninformative in the 5% success regime. DrQ-v2 shares the encoder --
critic gradients teach the CNN what to see.

**H2: Optimization difficulty.** End-to-end training from sparse rewards
asks the CNN to learn features AND the policy to learn actions
simultaneously. Published approaches separate these (VAE pre-training in
RIG, frozen encoders in R3M). Even DrQ-v2 uses dense rewards, not sparse.

**H3: Exploration.** At 5% success, HER relabeling generates some positive
signal, but the policy may not explore enough to discover the pushing
behavior. The critic's declining loss suggests it's "settling" into a
bad equilibrium.

### Immediate next steps

1. **Run 2: `--share-encoder`** -- test H1. If critic gradients flowing
   through the CNN help, we should see the hockey-stick by 1M steps.
   Stop rule: abort if <= 10% at 1M.

2. **Full-state control** -- run Ch9 script with the full 25D observation
   (either modify `ProprioGoalWrapper` or run `ch04_her_sparse_reach_push.py`
   directly as a control). This confirms pipeline correctness.

3. **If `--share-encoder` fails** -- consider separated representation
   learning (Tier 5 in Section 7): VAE pre-training or frozen encoder.

### Files referenced in this document

| File | Role |
|------|------|
| `scripts/labs/manipulation_encoder.py` | ManipulationCNN, SpatialSoftmax, ManipulationExtractor |
| `scripts/ch09_pixel_push.py` | Training script with ablation flags |
| `scripts/labs/pixel_wrapper.py` | PixelObservationWrapper with proprio support |
| `scripts/labs/visual_encoder.py` | NatureCNN (the "before" reference) |
| `scripts/labs/image_augmentation.py` | HerDrQDictReplayBuffer, RandomShiftAug |
| `scripts/ch04_her_sparse_reach_push.py` | State+HER reference (99% @ 1.86M) |
| `tasks/lessons.md` | Lessons 1-7 (Lessons 6-7 added this session) |
| `CLAUDE.md` | Concept registry, Ch9 entries |
