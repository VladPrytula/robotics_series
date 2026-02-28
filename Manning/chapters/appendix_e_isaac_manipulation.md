# Appendix E: Isaac Lab Manipulation

**This appendix covers:**

- Transferring the SAC methodology from MuJoCo Fetch to Isaac Lab's Lift-Cube task, demonstrating that the same algorithm and diagnostic skills work across simulators
- Understanding how Isaac Lab's observation conventions (`{policy}` vs `{observation, achieved_goal, desired_goal}`) affect pipeline design -- and why porting a method does not mean drop-in compatibility
- Diagnosing and fixing a catastrophic curriculum-induced reward collapse that reveals a fundamental incompatibility between off-policy replay buffers and non-stationary reward functions
- Running pixel-based manipulation on Isaac Lab using the native TiledCamera sensor, achieving 24x faster throughput than MuJoCo pixel training
- Comparing wall-clock performance: 14 minutes (Isaac, 8M steps) vs 40 hours (MuJoCo pixel, 8M steps) -- a 170x speedup from GPU-parallel physics

## Chapter Bridge

Through Chapters 1-6, you built a complete workflow on Gymnasium-Robotics Fetch tasks: environment contracts, PPO for dense rewards, SAC for off-policy efficiency, HER for sparse rewards, and curriculum learning for multi-phase manipulation. Every experiment used MuJoCo physics running on CPU, with throughput around 500-600 frames per second for state-based training. That depth was intentional -- one environment family let us isolate algorithmic effects without confounding variables from the simulator itself.

The natural next question is whether the methodology transfers. If you switch from MuJoCo to a different physics engine, do the same algorithms, the same diagnostic patterns, and the same experiment contracts still work? And can GPU-parallel physics make the same experiments run fast enough to iterate in minutes rather than hours?

This appendix provides portability evidence. We apply the same SAC methodology to Isaac Lab's `Isaac-Lift-Cube-Franka-v0` -- a GPU-parallel manipulation environment with different observation conventions, a different action space, and engineering constraints we have not encountered before. The same diagnostic skills (observation inspection, dense-first debugging, curriculum crash analysis) prove essential in the new simulator. What we gain is speed: 8 million training steps complete in 14 minutes on a single GPU, compared to roughly 40 hours for MuJoCo pixel training at the same scale.

Two constraints shape scope. Isaac Lab requires Linux with an NVIDIA GPU, so this entire appendix is GPU-only -- no Mac, no CPU fallback. And we practice honest method-benchmark matching: the primary demonstration uses dense reward shaping, which is substantially easier from an exploration standpoint than the sparse-reward tasks in Chapters 4-5. We are explicit about what this proves and what it does not.

## WHY: Lifting Is Different From Reaching

### The Task

Picture a seven-joint Franka Emika arm -- matte gray links tapering from a heavy shoulder to a slender wrist, with a parallel-jaw gripper at the end whose fingers can open about 8 cm. The arm sits behind a flat table in an Isaac Lab simulation scene, lit by neutral overhead lights that cast soft shadows. On the table rests a small red cube, roughly 5 cm on a side, sitting at a random position within the arm's reachable workspace. Somewhere above the table, a green translucent marker indicates the target pose -- the position and orientation where the agent must deliver the cube. Success means picking the cube up off the table and holding it at the target location, which can be anywhere in a volume roughly 30 cm on a side above the table surface. When the trained policy runs, the arm sweeps toward the cube with a smooth arc, closes its fingers around it, lifts it deliberately, and tracks it to the target -- a four-phase motion that takes about 250 simulation steps.

Reaching, the task we started with in Chapter 2, requires only one of those phases. The gripper moves toward a target point, and the episode succeeds when the fingertips arrive within 5 cm. There is no object to manipulate, no grasping, no lifting. Lift-Cube requires all four phases executed in sequence, and each phase demands qualitatively different control:

1. **Approach** -- move the gripper toward the cube, requiring large coordinated arm movements.
2. **Grasp** -- close the gripper around the cube at the right moment, requiring precise gripper timing relative to arm position.
3. **Lift** -- raise the cube off the table, requiring coordinated upward motion while maintaining grip force.
4. **Track** -- move the held cube to the target pose, requiring fine position control with the added inertia and contact dynamics of the grasped object.

A Reach policy only needs phase 1. A Lift-Cube policy must execute all four in the right order, and a failure at any phase -- approaching the wrong side of the cube, closing the gripper too early, losing grip during the lift -- means the entire episode fails. This is the same multi-phase structure we saw in Chapter 6's PickAndPlace capstone, translated to a different simulator.

### Staged Dense Reward

Isaac Lab's Lift-Cube environment uses a staged dense reward structure that mirrors Chapter 5's curriculum pattern -- except here the curriculum is baked into the reward function rather than requiring a separate wrapper:

| Reward term | Weight | What it rewards |
|-------------|--------|----------------|
| `reaching_object` | 1.0 | Distance from gripper to cube |
| `lifting_object` | 15.0 | Height of cube above table |
| `object_goal_tracking` | 16.0 | Distance from cube to target pose |
| `object_goal_tracking_fine_grained` | 5.0 | Precise tracking bonus (tighter threshold) |
| `action_rate` | -0.0001 | Penalizes jerky actions |
| `joint_vel` | -0.0001 | Penalizes fast joint movement |

The weight structure creates a natural gradient for learning. Early in training, the agent can only maximize the reaching bonus (weight 1.0), because it has not yet learned to grasp. Once it discovers that grasping enables lifting, it unlocks a much larger reward signal (weight 15.0 -- fifteen times the reaching bonus). And once lifting is reliable, the agent discovers that tracking the target to its precise position yields the largest reward component of all (weight 16.0 + 5.0 for fine-grained tracking). The agent does not need to be told to learn these phases in order; the reward magnitudes ensure it.

This is the same idea as Chapter 5's goal stratification, where we structured the goal distribution so that easy goals appeared first and harder goals appeared as the agent improved. The difference is that Isaac Lab encodes the curriculum in reward weights rather than in goal sampling. Dense shaping like this is a well-understood technique, and it makes Lift-Cube substantially easier from an exploration standpoint than sparse-reward tasks (which is why we call out the difficulty comparison later in this appendix).

### The Hidden Curriculum (and Why SAC Must Disable It)

Isaac Lab's Lift-Cube env contains a second curriculum mechanism that is less obvious but critically important. A **CurriculumManager** scales the `action_rate` and `joint_vel` penalty weights during training:

| Term | Initial weight | Final weight | Scaling |
|------|---------------|-------------|---------|
| `action_rate` | -0.0001 | -0.1 | 1000x |
| `joint_vel` | -0.0001 | -0.1 | 1000x |

The design intent is sound: start with tiny penalties so the agent can explore freely, thrashing around and making big movements to discover how to reach and grasp. Once it learns the basics, ramp up the penalties to encourage smooth, controlled motion. First learn *what* to do, then learn to do it *smoothly*. NVIDIA designed this curriculum for PPO, and it works well there because PPO is on-policy -- it collects a batch of experience, updates the policy, and discards all old data. When the penalties increase, PPO's next batch is collected entirely under the new reward scale, so there is no conflict between old and new reward magnitudes.

SAC cannot tolerate this. SAC stores hundreds of thousands of transitions in a replay buffer and reuses them repeatedly for gradient updates. When the CurriculumManager scales penalties by 1000x, those old transitions still have rewards of magnitude ~5, but new transitions have rewards of magnitude ~5000. The Q-function trains on a mix of both and must somehow reconcile two incompatible scales -- which is like training a regression model where half the labels are in meters and the other half are in kilometers, with no indicator of which is which.

In our experiments, this caused a catastrophic reward collapse at ~4.6M steps. The reward went from -4.14 (the best value the agent had achieved, indicating successful lifting and tracking) to -3,050 in a single 64K-step interval. The critic loss exploded 540x and the training never recovered. We will show the full diagnostic in the Run It section; here, the key lesson is the general principle.

> **Warning:** Any system that changes the reward function mid-training is incompatible with off-policy replay buffers. The buffer stores transitions under the reward scale that was active when they were collected, and it has no mechanism to distinguish "old scale" from "new scale" transitions. This is a specific instance of a broader assumption: replay buffers assume the MDP is **stationary** -- that the transition dynamics and reward function do not change over time. Curriculum learning that modifies reward weights violates this stationarity assumption. PPO is immune because it discards experience after each update.

Our fix is straightforward: the pipeline automatically disables the CurriculumManager for SAC by setting all curriculum terms to `None` at environment creation time. The penalty signals still exist at their initial constant values (-0.0001) -- they provide a small incentive for smooth motion without poisoning the replay buffer with incompatible reward scales.

### Observation Space and Action Space

Before we can apply SAC to Lift-Cube, we need to understand what the agent sees and what it can do -- the observation and action spaces that define the MDP interface.

**Observations.** Lift-Cube provides a 36-dimensional flat state vector:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| `joint_pos` | 9 | Franka arm joint positions (7) + gripper (2) |
| `joint_vel` | 9 | Joint velocities |
| `object_position` | 3 | Cube position in world frame |
| `target_object_position` | 7 | Target pose (position + quaternion) |
| `actions` | 8 | Previous action (feedback for smoothness) |

This is a fully observable MDP: a single observation frame contains everything the agent needs to decide the optimal action. Joint positions and velocities give proprioception, object position tells the agent where the cube is, the target pose tells it where the cube needs to go, and the previous action enables smooth motion planning. No temporal reasoning is needed, so a feedforward MLP suffices -- no CNN, no LSTM, no frame stacking.

Isaac Lab wraps this observation in a dict with a single `policy` key (`{"policy": (num_envs, 36)}`), which is structurally different from the Gymnasium-Robotics convention of separate `observation`, `achieved_goal`, and `desired_goal` keys. This matters for two reasons. First, it means most Isaac Lab envs are not goal-conditioned in the Gymnasium-Robotics sense -- there is no built-in mechanism for HER relabeling. Second, it means SB3 uses `MlpPolicy` (which expects a flat vector) rather than `MultiInputPolicy` (which expects a dict with multiple keys). Our pipeline detects this automatically and selects the right policy class.

**Actions.** The action space is 8-dimensional: 7 joint position targets for the Franka arm plus 1 gripper command. This is more complex than Fetch's 4D Cartesian deltas (`dx, dy, dz, gripper`) -- the agent controls individual joints rather than end-effector velocity, so it must learn the kinematic relationships that an inverse-kinematics controller would handle in the Fetch environments. But SAC handles both interfaces equally well: continuous actions are continuous actions, regardless of whether they represent end-effector velocities or joint positions.

One engineering detail worth noting: Isaac Lab declares its action space as `Box(-inf, inf)` because the environment's internal action manager handles clipping and scaling. SB3's SAC assumes bounded actions for its tanh squashing (the policy outputs `tanh(z)`, which maps to `[-1, 1]`, then rescales to action bounds). Isaac Lab's official `Sb3VecEnvWrapper` addresses this by clipping the declared bounds to `[-100, 100]`. If you write a custom wrapper that passes infinite bounds through to SB3, you will get NaN in the actor loss.

### Why SAC Works on This Task

The problem structure tells us SAC is the right algorithm, following the same derivation-from-constraints reasoning we used in Chapters 3-4. The action space is continuous (8D joint targets), which rules out DQN and points toward actor-critic methods. The reward is dense, so off-policy learning works well without HER. The 36D state vector is fully observable, which means an MLP suffices. And SAC's replay buffer reuses experience, providing superior sample efficiency compared to PPO's on-policy approach -- NVIDIA's reference configuration for Lift-Cube uses PPO with 16.3M steps, while our SAC budget is 8M steps, a 2x reduction, and we see strong learning at just 2M steps.

The entropy bonus in SAC also helps here in a specific way. During the early training phases, the agent needs to explore a large joint-space to discover that approaching the cube from above (rather than the side) enables grasping. SAC's maximum-entropy objective encourages the policy to maintain uncertainty across these approach angles, which means it is more likely to stumble upon successful grasps than a purely greedy policy would be. Once grasping is discovered, the entropy naturally decreases as the policy commits to the successful strategy, and the automatic temperature tuning (from Chapter 3) manages this transition without manual intervention.

> **Getting started with Isaac Lab:** This appendix requires Linux with an NVIDIA GPU. Build the Isaac container with `bash docker/build.sh isaac`, then use `bash docker/dev-isaac.sh` as the entry point for all commands (replacing the `docker/dev.sh` used in earlier chapters). The NGC base image is roughly 15 GB; initial download takes time but is cached for subsequent builds. On first launch, Isaac Sim compiles Vulkan shaders (30-90 seconds of apparent silence) -- this is normal and does not repeat on subsequent runs.

## HOW: Build It Components

The Build It track for this appendix contains the same SAC math you implemented in Chapters 3-4, adapted to handle two observation conventions: the Gymnasium-Robotics goal-conditioned layout (`{observation, achieved_goal, desired_goal}`) and the Isaac Lab flat-dict layout (`{policy}`). The components run on CPU with synthetic data -- no Isaac Sim installation needed -- so you can verify the math even if you do not have a Linux GPU.

We present the SAC core first (components 1-5), then the goal relabeling mechanics (components 6-9). Lift-Cube itself uses SAC without HER (it is dense-reward and not goal-conditioned), but the relabeling code demonstrates how HER would work on a goal-conditioned Isaac Lab task, and it reuses the same invariants from Chapter 5.

### E.1 Dict Observation Encoder

The first component handles the structural difference between Gymnasium-Robotics and Isaac Lab observations. Both expose dict observations, but with different keys. We need an encoder that flattens whatever keys are present into a single feature vector for the actor and critic networks.

The operation is a concatenation: given a dict with $K$ keys, each mapping to a tensor of shape $(B, d_k)$ where $B$ is the batch size and $d_k$ is the dimension of key $k$, the encoder produces a flat tensor of shape $(B, \sum_k d_k)$.

<!-- Listing E.1: from scripts/labs/isaac_sac_minimal.py:dict_flatten_encoder -->

```python
# Listing E.1: Dict observation encoder
# (scripts/labs/isaac_sac_minimal.py:dict_flatten_encoder)

class DictFlattenEncoder(nn.Module):
    """Flatten selected dict-observation keys into one feature vector.

    Supports both observation conventions:
    - Gymnasium-Robotics: keys=["observation", "achieved_goal", "desired_goal"]
    - Isaac Lab flat-dict: keys=["policy"]
    """

    def __init__(self, keys: list[str]):
        super().__init__()
        if not keys:
            raise ValueError("keys must be non-empty")
        self.keys = keys

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for key in self.keys:
            x = obs[key]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            parts.append(x.float())
        return torch.cat(parts, dim=-1)
```

The `keys` parameter makes this encoder agnostic to the observation layout. For Gymnasium-Robotics, you pass `["observation", "achieved_goal", "desired_goal"]` and the encoder concatenates all three into a (batch, 25+3+3) = (batch, 31) vector. For Isaac Lab, you pass `["policy"]` and it passes the 36D vector through unchanged. The same downstream networks work in both cases -- they just see a flat feature vector of the appropriate dimension.

> **Checkpoint:** Instantiate `DictFlattenEncoder(["observation", "achieved_goal", "desired_goal"])` with a batch of synthetic goal-conditioned observations (obs_dim=18, goal_dim=3). The output should have shape `(batch, 24)` and contain all finite values. Then instantiate `DictFlattenEncoder(["policy"])` with Isaac-style observations (obs_dim=32). The output should have shape `(batch, 32)`.

### E.2 Squashed Gaussian Actor

SAC's policy outputs continuous actions through a squashed Gaussian. The actor network maps the flat observation to a mean $\mu$ and log-standard-deviation $\log\sigma$ for each action dimension, samples $u \sim \mathcal{N}(\mu, \sigma^2)$ via the reparameterization trick, and applies $a = \tanh(u)$ to bound actions to $(-1, 1)$.

The log-probability must account for the tanh squashing:

$$\log\pi(a|s) = \log\pi(u|s) - \sum_{i=1}^{d} \log(1 - \tanh^2(u_i))$$

where $d$ is the action dimension and the second term is the Jacobian correction for the change of variables from $u$ to $a$.

<!-- Listing E.2: from scripts/labs/isaac_sac_minimal.py:squashed_gaussian_actor -->

```python
# Listing E.2: Squashed Gaussian actor
# (scripts/labs/isaac_sac_minimal.py:squashed_gaussian_actor)

class SquashedGaussianActor(nn.Module):
    """SAC actor with tanh squashing and log-prob correction."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        hidden = hidden_dims or [256, 256]
        self.backbone = _mlp(obs_dim, hidden[-1], hidden[:-1])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs_flat: torch.Tensor):
        h = self.backbone(obs_flat)
        mu = self.mu(h)
        log_std = torch.clamp(
            self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        u = dist.rsample()          # reparameterization trick
        action = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(
            1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob
```

The `LOG_STD_MIN` and `LOG_STD_MAX` clamps prevent numerical instability -- without them, the log-standard-deviation can drift to extreme values early in training, causing either deterministic collapse (very negative log_std) or entropy explosion (very positive log_std). The `1e-6` in the Jacobian correction prevents `log(0)` when `tanh(u)` saturates near +/-1.

> **Checkpoint:** Feed a batch of flat observations through the actor. Actions should have shape `(batch, act_dim)` with all values in `(-1, 1)`. Log-probabilities should have shape `(batch,)` with all finite values.

### E.3 Twin Q Critic

SAC uses twin Q-networks to combat overestimation bias (the same clipped double-Q technique from Chapter 3). Each critic takes a concatenated `[observation, action]` input and outputs a scalar Q-value. The minimum of the two Q-values is used for the Bellman target, which prevents the optimistic errors that plague single-critic methods:

$$Q_{\text{target}} = \min(Q_1(s', a'), Q_2(s', a'))$$

<!-- Listing E.3: from scripts/labs/isaac_sac_minimal.py:twin_q_critic -->

```python
# Listing E.3: Twin Q critic
# (scripts/labs/isaac_sac_minimal.py:twin_q_critic)

class TwinQCritic(nn.Module):
    """Twin Q critics to reduce overestimation bias."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        hidden = hidden_dims or [256, 256]
        self.q1 = _mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = _mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs_flat: torch.Tensor,
                act: torch.Tensor):
        x = torch.cat([obs_flat, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
```

Each Q-network is a `[256, 256]` MLP that takes `obs_dim + act_dim` inputs and outputs a single scalar. For Lift-Cube, that is 36 + 8 = 44 inputs. For Gymnasium-Robotics Fetch tasks, it would be 31 + 4 = 35 inputs. The architecture is identical in both cases -- only the input dimension changes.

> **Checkpoint:** Feed a batch of `(observation, action)` pairs. Both `q1` and `q2` should return shape `(batch,)` with all finite values. At initialization, Q-values will be near zero (random weights, no training signal yet).

### E.4 SAC Losses

SAC has three loss functions that work together. The critic loss minimizes the soft Bellman error, the actor loss maximizes expected Q-values minus an entropy penalty, and the temperature loss tunes the entropy coefficient automatically.

**Critic loss:** For each critic $Q_i$ (where $i \in \{1, 2\}$), we minimize the MSE against the soft Bellman target:

$$L_{\text{critic}} = \sum_{i=1}^{2} \text{MSE}\big(Q_i(s, a),\; r + \gamma(1 - d)(\min_j Q_{\bar{\theta}_j}(s', a') - \alpha \log\pi(a'|s'))\big)$$

where $\bar{\theta}$ denotes the target network parameters, $\gamma = 0.99$ is the discount factor, $d \in \{0, 1\}$ is the done flag, and $\alpha$ is the temperature.

**Actor loss:** The policy maximizes Q-values while maintaining entropy:

$$L_{\text{actor}} = \mathbb{E}_{a \sim \pi}\big[\alpha \log\pi(a|s) - \min_j Q_j(s, a)\big]$$

**Temperature loss:** The dual gradient descent that tunes $\alpha$ automatically:

$$L_{\alpha} = -\alpha\big(\log\pi(a|s) + \bar{H}\big)$$

where $\bar{H} = -\text{dim}(\mathcal{A})$ is the target entropy (negative of the action dimension, so $\bar{H} = -8$ for Lift-Cube and $\bar{H} = -4$ for Fetch tasks).

<!-- Listing E.4: from scripts/labs/isaac_sac_minimal.py:sac_losses -->

```python
# Listing E.4: SAC loss functions
# (scripts/labs/isaac_sac_minimal.py:sac_losses)

def critic_loss(encoder, actor, critic, critic_target,
                batch, gamma, alpha):
    obs_flat = encoder(batch.obs)
    next_flat = encoder(batch.next_obs)
    q1, q2 = critic(obs_flat, batch.actions)
    with torch.no_grad():
        next_a, next_logp = actor(next_flat)
        tq1, tq2 = critic_target(next_flat, next_a)
        tq = torch.min(tq1, tq2)
        target = batch.rewards + gamma * (1.0 - batch.dones) \
            * (tq - alpha * next_logp)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def actor_loss(encoder, actor, critic, batch, alpha):
    obs_flat = encoder(batch.obs)
    act, logp = actor(obs_flat)
    q1, q2 = critic(obs_flat, act)
    q = torch.min(q1, q2)
    loss = (alpha * logp - q).mean()
    return loss, logp


def temperature_loss(log_alpha, logp, target_entropy):
    alpha = log_alpha.exp()
    return -(alpha * (logp.detach() + target_entropy)).mean()
```

Notice that `critic_loss` uses `torch.no_grad()` for the Bellman target -- gradients flow through the current Q-values but not through the target computation, which is the standard semi-gradient approach. The actor loss detaches from the critic gradients by using `torch.min(q1, q2)` on freshly computed Q-values (not the target network). And the temperature loss detaches `logp` because we want to adjust $\alpha$ in response to the current entropy, not to change the policy through $\alpha$'s gradient.

> **Checkpoint:** After 25 update steps on synthetic data, all three losses should be finite. The `alpha` value should remain positive (it is parameterized as `exp(log_alpha)`, which guarantees positivity). Typical values after 25 updates: `critic_loss` in the range 0.01-10, `actor_loss` in -5 to 5, `alpha` near 1.0 (it starts at `exp(0) = 1.0` and adjusts slowly).

### E.5 SAC Update Step

The wiring step assembles the individual losses into a single update cycle. The ordering matters: critic first (so the actor loss uses up-to-date Q-values), then actor (so the temperature loss uses the current policy's entropy), then temperature, and finally a Polyak soft update of the target network ($\tau = 0.005$).

<!-- Listing E.5: from scripts/labs/isaac_sac_minimal.py:sac_update_step -->

```python
# Listing E.5: SAC update step (wiring)
# (scripts/labs/isaac_sac_minimal.py:sac_update_step)

def sac_update_step(encoder, actor, critic, critic_target,
                    log_alpha, batch, *, actor_opt, critic_opt,
                    alpha_opt, gamma=0.99, tau=0.005,
                    target_entropy=None):
    if target_entropy is None:
        target_entropy = -float(batch.actions.shape[-1])
    alpha = log_alpha.exp()

    c_loss = critic_loss(
        encoder, actor, critic, critic_target,
        batch, gamma, alpha)
    critic_opt.zero_grad()
    c_loss.backward()
    critic_opt.step()

    a_loss, logp = actor_loss(
        encoder, actor, critic, batch, alpha)
    actor_opt.zero_grad()
    a_loss.backward()
    actor_opt.step()

    t_loss = temperature_loss(log_alpha, logp, target_entropy)
    alpha_opt.zero_grad()
    t_loss.backward()
    alpha_opt.step()

    _polyak_update(critic, critic_target, tau)

    return {"critic_loss": float(c_loss.item()),
            "actor_loss": float(a_loss.item()),
            "alpha_loss": float(t_loss.item()),
            "alpha": float(log_alpha.exp().item()),
            "entropy": float((-logp.mean()).item())}
```

The Polyak update `_polyak_update(critic, critic_target, tau)` blends the target network toward the online network: $\bar{\theta} \leftarrow \tau\theta + (1 - \tau)\bar{\theta}$ with $\tau = 0.005$. This slow-moving target provides stable Bellman targets while the online network learns rapidly.

> **Checkpoint:** Run `python scripts/labs/isaac_sac_minimal.py --verify`. After 25 updates on synthetic data, the returned dict should contain all finite values and `alpha > 0`. The verification tests both the Gymnasium-Robotics observation layout and the Isaac Lab layout to confirm the encoder handles both correctly.

### E.6 Goal Transition Structures

The remaining four components form the HER relabeling track. While Lift-Cube itself does not use HER (it has dense rewards and no goal-conditioned observation structure), these components demonstrate how the relabeling mechanics from Chapter 5 work in a simulator-agnostic way. They would apply to any Isaac Lab task that exposes goal-conditioned observations.

We start with the data structures: a `GoalTransition` NamedTuple that stores everything needed for relabeling, and a `GoalStrategy` enum for the three sampling strategies from Chapter 5.

<!-- Listing E.6: from scripts/labs/isaac_goal_relabeler.py:goal_transition_structs -->

```python
# Listing E.6: Goal transition structures
# (scripts/labs/isaac_goal_relabeler.py:goal_transition_structs)

class GoalTransition(NamedTuple):
    """Goal-conditioned transition for HER-style relabeling."""
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    achieved_goal: np.ndarray
    desired_goal: np.ndarray


class GoalEpisode(NamedTuple):
    transitions: list[GoalTransition]
    def __len__(self) -> int:
        return len(self.transitions)


class GoalStrategy(Enum):
    FINAL = "final"
    FUTURE = "future"
    EPISODE = "episode"
```

The key design choice is keeping `achieved_goal` and `desired_goal` as explicit fields rather than embedding them in `obs`. HER relabeling replaces `desired_goal` while keeping `achieved_goal` unchanged, so having them as separate fields makes the relabeling operation clean and auditable.

> **Checkpoint:** Create a `GoalTransition` with synthetic data. Verify that `GoalStrategy` has exactly three members: `FINAL`, `FUTURE`, and `EPISODE`.

### E.7 Goal Sampling

Given an episode and a timestep $t$, the goal sampler selects $k$ alternative desired goals from the achieved goals at other timesteps in the same episode. The three strategies differ in which timesteps they draw from:

- **Future** ($t' > t$): Goals from later in the same episode -- the most common strategy, because later achieved goals tend to be closer to what the agent was trying to do.
- **Final** ($t' = T$): Always the last achieved goal in the episode.
- **Episode** ($t' \in [0, T]$): Any achieved goal in the episode, uniformly sampled.

<!-- Listing E.7: from scripts/labs/isaac_goal_relabeler.py:isaac_goal_sampling -->

```python
# Listing E.7: Goal sampling strategies
# (scripts/labs/isaac_goal_relabeler.py:isaac_goal_sampling)

def sample_goals(episode, t, strategy=GoalStrategy.FUTURE, k=4):
    """Sample k alternative desired goals from same episode."""
    n = len(episode)
    if n == 0:
        return []
    if strategy == GoalStrategy.FINAL:
        g = episode.transitions[-1].achieved_goal
        return [g.copy() for _ in range(k)]
    if strategy == GoalStrategy.FUTURE:
        future = list(range(t + 1, n))
        if not future:
            return [episode.transitions[-1].achieved_goal.copy()
                    for _ in range(k)]
        picked = np.random.choice(
            future, size=min(k, len(future)), replace=False).tolist()
        while len(picked) < k:
            picked.append(future[-1])
        return [episode.transitions[i].achieved_goal.copy()
                for i in picked]
    if strategy == GoalStrategy.EPISODE:
        picked = np.random.choice(n, size=k, replace=True).tolist()
        return [episode.transitions[i].achieved_goal.copy()
                for i in picked]
    raise ValueError(f"Unknown strategy: {strategy}")
```

> **Checkpoint:** Sample 4 goals from an episode with the `FUTURE` strategy at timestep $t = 10$. You should get exactly 4 goals, each with the same shape as `achieved_goal`. All sampled indices should be greater than 10.

### E.8 Transition Relabeling

This is the core HER operation. Given a transition and a new desired goal, we keep the observation, action, and achieved goal unchanged, replace the desired goal, and recompute the reward using the environment's reward function. The critical invariant from Chapter 5 holds: the achieved goal is a deterministic function of the state, so swapping the desired goal does not change what the agent actually did -- it changes what the agent was trying to do.

<!-- Listing E.8: from scripts/labs/isaac_goal_relabeler.py:isaac_relabel_transition -->

```python
# Listing E.8: Transition relabeling
# (scripts/labs/isaac_goal_relabeler.py:isaac_relabel_transition)

def relabel_transition(transition, new_goal, reward_fn):
    """HER core: replace desired_goal, recompute reward."""
    new_reward = reward_fn(transition.achieved_goal, new_goal)
    return GoalTransition(
        obs=transition.obs,
        action=transition.action,
        reward=float(new_reward),
        next_obs=transition.next_obs,
        done=transition.done,
        achieved_goal=transition.achieved_goal,
        desired_goal=new_goal,
    )
```

The function is five lines of logic, but it encodes the key insight of HER: failed trajectories contain implicit information about what they *did* achieve, and relabeling converts that information into positive training signal.

> **Checkpoint:** Relabel a transition with its own `achieved_goal` as the new desired goal. The reward should be 0.0 (success) because the agent achieved exactly the goal we are now claiming it was trying to reach. The `achieved_goal` field should be unchanged.

### E.9 Episode HER Processing

The final component wires sampling and relabeling together at the episode level. For each transition in the episode, it keeps the original transition and (with probability `her_ratio`) adds $k$ relabeled copies with alternative goals. The `her_ratio` parameter (typically 0.8) controls what fraction of transitions receive relabeled augmentation.

<!-- Listing E.9: from scripts/labs/isaac_goal_relabeler.py:isaac_her_episode_processing -->

```python
# Listing E.9: Episode HER processing
# (scripts/labs/isaac_goal_relabeler.py:isaac_her_episode_processing)

def process_episode_with_her(episode, reward_fn, *,
                             strategy=GoalStrategy.FUTURE,
                             k=4, her_ratio=0.8):
    """Original + relabeled transitions for one episode."""
    out = []
    for t, tr in enumerate(episode.transitions):
        out.append(tr)
        if np.random.random() >= her_ratio:
            continue
        goals = sample_goals(episode, t=t,
                             strategy=strategy, k=k)
        for g in goals:
            out.append(relabel_transition(tr, g, reward_fn))
    return out


def success_fraction(transitions):
    if not transitions:
        return 0.0
    return sum(1 for t in transitions
               if t.reward >= 0.0) / len(transitions)
```

With `k=4` and `her_ratio=0.8`, an episode of 60 transitions produces roughly 60 + (60 * 0.8 * 4) = 252 transitions -- a 4.2x data amplification. More importantly, the relabeled transitions contain a much higher fraction of successes than the originals, because many of them relabel with goals the agent actually achieved.

> **Checkpoint:** Run `python scripts/labs/isaac_goal_relabeler.py --verify`. The output should show that HER increases both the transition count (from 60 to ~250) and the success fraction (from near-zero to 40%+). The relabeling invariant -- self-achieved goals yield success -- should pass.

### Verification Summary

Both lab files run on CPU without Isaac Sim, which means you can verify the math on any machine:

```bash
# SAC math -- tests both observation conventions (~10-20 seconds)
python scripts/labs/isaac_sac_minimal.py --verify

# Goal relabeling invariants and data amplification (~1-5 seconds)
python scripts/labs/isaac_goal_relabeler.py --verify
```

The SAC verification tests two observation conventions (Gymnasium-Robotics goal-conditioned and Isaac Lab flat-dict), confirming that the same encoder-actor-critic pipeline handles both layouts. The goal relabeler verification confirms the HER invariants: self-achieved relabeling yields success, data amplification produces the expected transition counts, and the success fraction increases after relabeling.

For a live training demonstration of the from-scratch SAC implementation, the `--demo` modes in Chapters 3 and 4 apply the same math on MuJoCo Fetch tasks.

With the math verified, we can move to the production pipeline -- where SB3 implements the same computations at scale, with vectorized rollouts across 256 parallel environments on a GPU.

## Bridge: From Scratch to SB3

The components you built in Listings E.1-E.9 map directly to SB3's internal structure. The correspondence is worth making explicit, both to demystify SB3's internals and to confirm that the from-scratch code implements the same math that powers the production runs.

| Build It component | SB3 equivalent | Notes |
|--------------------|---------------|-------|
| `DictFlattenEncoder` (E.1) | `MultiInputPolicy` feature extractor (goal-conditioned) or `MlpPolicy` `FlattenExtractor` (Isaac flat-dict) | SB3 auto-detects dict vs flat obs and selects the right policy class. Our encoder demonstrates the same dispatch logic explicitly. |
| `SquashedGaussianActor` (E.2) | `SAC.actor` (`SquashedDiagGaussianDistribution`) | SB3 wraps the same tanh-squashed normal with the same log-prob correction. The `LOG_STD_MIN` / `LOG_STD_MAX` clamps match SB3's defaults. |
| `TwinQCritic` (E.3) | `SAC.critic` + `SAC.critic_target` | SB3 maintains two Q-networks plus a target copy with Polyak averaging, identical to our `critic` / `critic_target` pair. |
| `critic_loss` / `actor_loss` / `temperature_loss` (E.4) | `SAC.train()` internal computation | SB3 computes the same three losses in the same order: critic first, actor second, temperature third. The soft Bellman target, actor entropy term, and dual gradient for alpha are mathematically identical. |
| `sac_update_step` (E.5) | One iteration of `SAC.train()` | SB3's training loop calls critic, actor, and temperature updates sequentially, then runs the Polyak update with the same $\tau = 0.005$ default. |
| `GoalTransition` / `GoalStrategy` / relabeling (E.6-E.9) | `HerReplayBuffer` in `sb3_contrib` | SB3's HER implementation uses the same future/final/episode strategies and the same relabel-then-recompute-reward pattern. The `n_sampled_goal` parameter corresponds to our $k$. |

You can verify this mapping by running the bridge mode:

```bash
python scripts/labs/isaac_sac_minimal.py --bridge
```

The bridge is structural rather than numerical (the from-scratch and SB3 networks have different random initializations, so exact weight matches are not expected). What it confirms is that both pipelines accept the same observation layouts, produce actions in the same range, and compute losses with the same functional form. When you see SB3's `ent_coef` logged during training, you are watching the same automatic temperature tuning you implemented in Listing E.4. When TensorBoard shows `train/critic_loss`, that is the same soft Bellman error from `critic_loss()`. The production pipeline adds vectorized rollout collection, efficient GPU-based replay sampling, and proper checkpointing -- engineering that does not change the underlying math.

## Run It

### Experiment Card

```
---------------------------------------------------------
EXPERIMENT CARD: Appendix E -- SAC on Isaac Lift-Cube
---------------------------------------------------------
Algorithm:    SAC (MlpPolicy [256, 256], auto-temperature)
Environment:  Isaac-Lift-Cube-Franka-v0
Fast path:    2,000,000 steps, seed 0, 256 envs
Time:         ~3.6 min (A100 GPU) -- includes Isaac Sim boot
Full run:     8,000,000 steps, seed 0, 256 envs, ~14 min

Run command (fast path):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 2000000

Run command (full):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99

Checkpoint track (skip training):
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip

Expected artifacts:
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.meta.json
  results/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_eval.json
  results/appendix_e_isaac_env_catalog.json

Success criteria (fast path, 2M steps):
  return_mean > -22 (approaching grasping phase)
  41% positive-return episodes (bimodal distribution)

Success criteria (full run, 8M steps):
  return_mean = +0.54 +/- 0.05
  100/100 positive-return episodes
  Throughput >= 9,000 fps at 256 envs

Pixel variant (4M steps, 64 envs, ~56 min):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 --pixel \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 64 --total-steps 4000000
  Success: return_mean ~ -1.07 at convergence
  Throughput: ~1,181 fps (TiledCamera, 64 envs)

Full multi-seed results: see REPRODUCE IT at end of appendix.
---------------------------------------------------------
```

### State-Based Training (8M Steps)

The full training run uses 256 parallel environments on a single GPU. At this parallelism level, PhysX batches the simulation efficiently and SB3 collects 256 transitions per environment step, which keeps the replay buffer growing rapidly and the GPU utilization reasonable.

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99
```

The training progression tells a clear story of phase discovery. Watch `ep_rew_mean` in TensorBoard (or in the console output) as the agent moves through the four control phases:

| Timesteps | ep_rew_mean | Phase |
|-----------|------------|-------|
| 64K | -27.2 | Reaching only -- arm moves but cannot find the cube reliably |
| 1M | -24.8 | Reaching improving -- arm orients toward the cube more consistently |
| 2M | -21.8 | Grasping beginning -- first successful grasps appear |
| 3M | -15.4 | Grasping reliable -- reward jumps as the lifting bonus activates |
| 4M | -6.4 | Lifting -- the agent lifts the cube off the table regularly |
| 5M | -1.7 | Goal tracking -- the agent carries the cube toward the target |
| 8M | -1.4 | Converged -- consistent reach-grasp-lift-track across goal configurations |

The transitions between phases are not gradual -- they show as distinct jumps in the reward curve. The jump from -21.8 to -15.4 between 2M and 3M steps marks the moment grasping becomes reliable enough to consistently activate the lifting bonus (weight 15.0). The jump from -6.4 to -1.7 between 4M and 5M steps marks the transition from lifting to goal tracking (weight 16.0 + 5.0). These jumps are the staged reward design working as intended: each new phase unlocks a much larger reward signal than the last. As Figure E.1 shows, the phase transitions appear as distinct jumps in the learning curve rather than a smooth ascent.

![Learning curve for SAC on Isaac-Lift-Cube showing ep_rew_mean over 8M steps with phase annotations: reaching (-27), grasping (-15), lifting (-6), tracking (-1.4)](figures/appendix_e_state_curve.png)

Figure E.1: State-based learning curve for SAC on Isaac-Lift-Cube-Franka-v0 over 8M steps. Phase annotations mark the four control stages: reaching (-27), grasping (-15), lifting (-6), and tracking (-1.4). The reward jumps between phases reflect the staged dense reward weights unlocking progressively larger signals. (Generated by `python scripts/plot_appendix_e_figures.py state-curve --log-dir runs/appendix_e/`.)

Throughput averages **9,377 fps** at 256 envs, which means the full 8M-step run completes in **853 seconds (~14 minutes)**. Compare this to Chapter 4's state-based MuJoCo training at 500-600 fps -- Isaac Lab is roughly 15x faster per timestep, and the gap compounds over millions of steps.

### Video Progression

To see the learning progression rather than just read about it, we recorded deterministic episodes from checkpoints at three training stages. The state-based videos show the four-phase control sequence emerging:

| Stage | Video | Behavior |
|-------|-------|----------|
| Grasping (3M) | `videos/appendix_e_state_3M_grasping.mp4` | Arm reaches cube, attempts grasp, partial lifts |
| Lifting (5M) | `videos/appendix_e_state_5M_lifting.mp4` | Reliable grasp and lift, beginning target tracking |
| Converged (8M) | `videos/appendix_e_state_8M_converged.mp4` | Full reach-grasp-lift-track, smooth and purposeful |

The pixel pipeline produces a matching progression from the TiledCamera-based training:

| Stage | Video | Behavior |
|-------|-------|----------|
| Pre-takeoff (500K) | `videos/appendix_e_pixel_500K_random.mp4` | Exploring while CNN learns visual features |
| Post-takeoff (1.5M) | `videos/appendix_e_pixel_1500K_reaching.mp4` | Reaching and grasping from 84x84 pixels |
| Converged (4M) | `videos/appendix_e_pixel_4M_converged.mp4` | Full manipulation from pixels |

Generate all six with `bash scripts/record_appendix_e_videos.sh`, or record individual checkpoints with `python3 scripts/appendix_e_isaac_manipulation.py record --headless --ckpt <path>` (add `--pixel` for pixel-trained checkpoints).

### The Curriculum Crash

Our first attempt at the full 8M run included the CurriculumManager (we had not yet understood its interaction with SAC's replay buffer). The first 4.6M steps looked promising -- the agent reached, grasped, and began lifting, with reward climbing steadily to -4.14. Then this happened:

```
ts=4,624,128  rew=-4.14   <-- best ever
ts=4,688,128  rew=-3,050  <-- 740x worse in one interval
```

The critic loss exploded from 0.05 to 26.2 (a 540x increase), and the entropy coefficient loss flipped sign. The reward never recovered. Figure E.4 overlays the clean run and the crashed run, making the divergence point unmistakable.

![Overlay of two training curves: clean run converging smoothly to -1.4, and curriculum-enabled run collapsing from -4.14 to -3050 at 4.6M steps](figures/appendix_e_crash_diagnostic.png)

Figure E.4: Curriculum crash diagnostic. The clean run (CurriculumManager disabled) converges smoothly to -1.4, while the curriculum-enabled run collapses from -4.14 to -3,050 at ~4.6M steps when the CurriculumManager scales penalty weights by 1000x. The reward never recovers because the replay buffer contains transitions under two incompatible reward scales. (Generated by `python scripts/plot_appendix_e_figures.py crash-diagnostic --log-dir runs/appendix_e_crash/`.)

The root cause -- CurriculumManager scaling reward penalty weights mid-training, which poisons the replay buffer -- is explained in the "Hidden Curriculum" section above. The fix is to disable the CurriculumManager entirely for off-policy algorithms, which our pipeline now does automatically. The clean 8M run (shown in the progression table above) converges smoothly to +0.54 return with 100% positive episodes.

> **Tip:** If you encounter a similar mid-training collapse in a new Isaac Lab environment, check whether it has curriculum terms that modify reward weights. A quick diagnostic: `bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py smoke --headless --dense-env-id <your-env> --smoke-steps 100 2>&1 | grep -i curriculum`. If curriculum terms appear, the pipeline should disable them automatically for SAC/TD3.

### Honest Difficulty Comparison

The wall-clock speedup from GPU-parallel physics is dramatic, and it can create a misleading impression that we have solved a harder problem than we actually have. We want to be clear about where Lift-Cube sits on the difficulty ladder established in the main text.

| Property | Isaac-Lift-Cube-Franka-v0 | FetchPickAndPlace-v4 |
|----------|---------------------------|----------------------|
| Reward type | Dense (staged shaping) | Sparse ($-1$ / $0$) |
| Goal conditioning | No (`policy` key only) | Yes (achieved/desired goal) |
| HER needed? | No | Yes (essential) |
| Control phases | 4 (approach, grasp, lift, track) | 4 (approach, grasp, lift, place) |
| Exploration difficulty | Low (dense reward guides agent) | High (needle-in-haystack without HER) |
| Closest Fetch analogue | FetchReachDense + FetchPush-dense | FetchPickAndPlace (sparse) |

Both tasks require four-phase manipulation, but the exploration difficulty is fundamentally different. Isaac Lab's staged dense reward functions as an implicit curriculum: the weight structure (reaching=1.0, lifting=15.0, tracking=21.0) guides the agent through the phases without requiring it to discover the sequence through random exploration. FetchPickAndPlace with sparse rewards provides none of this guidance -- the agent receives -1 for every failed step and 0 only upon full success, which means it must stumble upon a complete grasp-lift-place sequence (or rely on HER relabeling) to generate any positive training signal at all.

**What this appendix proves:** SAC transfers across simulators without algorithm changes. GPU-parallel physics provides 15-170x wall-clock speedup. The same diagnostic skills (observation space inspection, dense-first debugging, curriculum crash analysis) apply in Isaac Lab. The experiment contract (checkpoint + metadata + eval JSON) ports cleanly.

**What this appendix does not prove:** That we can solve sparse-reward manipulation on Isaac Lab (Lift-Cube uses dense reward). That SAC handles contact-rich POMDP tasks (tasks requiring force feedback and recurrence are outside our MLP pipeline). That the pixel results match state-based results in quality (they do not -- pixel training is slower and less sample-efficient, as expected from Chapter 9).

### Pixels on Isaac: TiledCamera Results

Chapter 9 showed that SAC can learn from pixels on MuJoCo FetchPush. Isaac Lab offers the same capability through its native `TiledCamera` sensor, which renders all parallel environments into a single GPU-tiled image and slices per-environment frames -- scaling to 64+ parallel envs on a single GPU without the bottleneck of a single viewport.

The `--pixel` flag handles the setup: it injects a `TiledCamera` sensor at `{ENV_REGEX_NS}/Camera` (84x84 RGB per env), adds the image as an observation term with `clip=(0, 255)` so `Sb3VecEnvWrapper` recognizes it as uint8 image data, and sets `concatenate_terms=False` to produce a Dict observation space. SB3's `MultiInputPolicy` then routes the image through NatureCNN while passing state vectors (joint angles, velocities, object position) through a flat extractor -- the same sensor separation principle from Chapter 9, where the CNN learns world-state from pixels and proprioception comes directly from state vectors.

```bash
bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 --pixel \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 64 --total-steps 4000000
```

The throughput comparison tells the story of what native camera support buys you:

| Mode | num_envs | fps | Wall time (4M steps) |
|------|----------|-----|---------------------|
| State-based | 256 | ~7,010 | ~9.5 min |
| **Pixel (TiledCamera)** | **64** | **~1,181** | **~56 min** |
| Old viewport hack | 1 | ~16 | days (impractical) |
| Ch9 MuJoCo pixel Push | 4 | ~30-50 | ~40 hours |

TiledCamera at 64 envs is ~24x faster than MuJoCo pixel training and ~74x faster than the single-viewport approach. The learning curve shows the same hockey-stick shape from Chapter 9: flat for the first 800K steps while the CNN learns useful features, then rapid takeoff once visual representations become informative:

| Timesteps | ep_rew_mean | Phase |
|-----------|------------|-------|
| 64K | -27.2 | Random exploration |
| 800K | -26.5 | Hockey-stick flat regime |
| 1.0M | -22.0 | **Takeoff** -- reaching discovered via pixels |
| 1.4M | -8.2 | Grasping beginning |
| 2.0M | -3.5 | Lifting |
| 4.0M | -1.07 | Near convergence |

Figure E.3 plots this progression, and the hockey-stick shape is unmistakable -- a long flat regime where the CNN is learning useful visual features, followed by rapid takeoff once those features become informative enough to guide manipulation.

![Pixel training learning curve showing hockey-stick shape: flat regime 0-800K steps, takeoff at 1M, convergence at 4M steps with final reward -1.07](figures/appendix_e_pixel_curve.png)

Figure E.3: Pixel-based learning curve for SAC on Isaac-Lift-Cube using TiledCamera at 64 parallel environments. The hockey-stick shape matches the pattern from Chapter 9: a flat regime (0-800K steps) while the CNN learns visual features, takeoff at ~1M steps, and convergence near -1.07 by 4M steps. (Generated by `python scripts/plot_appendix_e_figures.py pixel-curve --log-dir runs/appendix_e_pixel/`.)

Three lessons from the pixel pipeline are worth calling out. First, `clip=(0, 255)` on the image observation term is essential -- without it, `Sb3VecEnvWrapper` treats the image as a regular float vector rather than a uint8 image, which breaks NatureCNN routing. Second, keeping proprioception alongside the image in a Dict observation (via `concatenate_terms=False`) matters for the same reason it mattered in Chapter 9: the CNN should learn world-state from pixels, not self-state that the robot already knows from its encoders. Third, the curriculum must be disabled for the same replay-buffer stationarity reason as the state-based pipeline.

### Wall-Clock Comparison

The following table puts all of the book's training configurations on a single scale:

| Setup | Steps | fps | Wall time |
|-------|-------|-----|-----------|
| Ch9 pixel Push (MuJoCo) | 8M | 30-50 | ~40 hours |
| Ch4 state Push (MuJoCo) | 2M | 500-600 | ~1 hour |
| Isaac state Lift-Cube (2M) | 2M | ~9,363 | ~3.6 min |
| Isaac state Lift-Cube (8M) | 8M | ~9,377 | ~14 min |
| Isaac pixel Lift-Cube (4M) | 4M | ~1,181 | ~56 min |

Figure E.2 makes the wall-clock gap visually concrete -- note that the horizontal axis uses a log scale, since the difference between 3.6 minutes and 40 hours spans nearly three orders of magnitude.

![Bar chart comparing wall-clock training times across five configurations: MuJoCo pixel (40h), MuJoCo state (1h), Isaac state 2M (3.6min), Isaac state 8M (14min), Isaac pixel (56min)](figures/appendix_e_wall_clock.png)

Figure E.2: Wall-clock comparison across five training configurations from the book, on a log-scale horizontal axis. GPU-parallel physics compresses training times from hours to minutes for state-based runs, and from days to under an hour for pixel-based runs. (Generated by `python scripts/plot_appendix_e_figures.py wall-clock`.)

The speedup from GPU-parallel physics ranges from ~15x (state-based, compared to MuJoCo state) to ~170x (compared to MuJoCo pixel). Even pixel-based Isaac training, which adds the overhead of rendering 84x84 RGB for every env at every step, completes 4M steps in under an hour -- a task that takes MuJoCo roughly 40 hours. The implication for iteration speed is significant: a hyperparameter sweep that would take days on MuJoCo takes hours on Isaac Lab, and a single experiment that would require an overnight run completes during a coffee break.

## What Can Go Wrong

Isaac Lab has a different failure surface than MuJoCo. Many issues stem from the Omniverse/PhysX runtime rather than the RL algorithm, so the error messages can be unfamiliar even if you have extensive MuJoCo experience. Here are the eight failure modes we have encountered, in rough order of how likely you are to hit them.

**SimulationContext singleton.** Your script hangs after the first environment closes, or raises `RuntimeError("Simulation context already exists")`. Isaac Lab's `SimulationContext` is a process-level singleton: after `env.close()`, the PhysX scene and USD stage persist in memory, and creating a second environment in the same process triggers the error. The fix is one environment per process -- the `all` subcommand handles this by running each phase as a separate subprocess. For custom orchestration, use `subprocess.call()`.

**Observation space mismatch.** You see `KeyError: 'achieved_goal'` or `KeyError: 'desired_goal'` when creating a model. This happens when you assume Gymnasium-Robotics observation keys on an Isaac env that uses `{'policy': ...}`. Run `discover-envs` first and check the `probed_envs` section of the catalog JSON. The pipeline detects this automatically and selects the right policy class.

**Infinite action bounds.** NaN appears in the actor loss or Q-values explode when using a custom wrapper. Isaac Lab declares `Box(-inf, inf)` action spaces; passing infinite bounds to SB3's tanh squashing produces undefined gradients. Use `Sb3VecEnvWrapper` (which clips to `[-100, 100]` automatically), or clip bounds explicitly in any custom wrapper.

**Missing `--headless` flag.** A segfault on startup, or a `VkResult` / Vulkan error before any Python code runs. Isaac Sim tried to open a Vulkan display on a headless system (DGX, CI). The fix is to always pass `--headless` on headless machines -- every command in this appendix includes it.

**GPU memory exhaustion.** `CUDA out of memory` during environment creation or training. Isaac Sim uses 8-12 GB of GPU memory to boot the simulation runtime before training begins. Monitor with `nvidia-smi` before launching, free other GPU workloads, or reduce `num_envs`. At 256 Lift-Cube envs, expect ~12-15 GB total.

**First-run shader compilation.** Isaac Sim appears to hang for 30-90 seconds after printing startup logs. This is Vulkan shader compilation on the first launch with a given GPU. Subsequent runs reuse cached shaders stored in named Docker volumes. Deleting those volumes (`docker volume rm isaac-cache-glcache`) triggers recompilation.

**Curriculum reward collapse.** Training reward improves steadily for millions of steps, then collapses catastrophically (e.g., -4 to -3,050 in one interval). The critic loss explodes and training never recovers. This is the CurriculumManager scaling reward penalties mid-training, which poisons the replay buffer with incompatible reward scales. The pipeline disables the CurriculumManager automatically for SAC/TD3, but if you encounter this on a new environment, check for curriculum terms in the startup logs.

**Low throughput from low `num_envs`.** Training runs much slower than expected (e.g., 88 fps on a powerful GPU). PhysX is designed for batched execution -- `num_envs=1` wastes over 95% of compute on kernel launch overhead. Use `--num-envs 256` for Lift-Cube. The difference is roughly 100x: ~9,000 fps at 256 envs vs ~88 fps at 1 env.

## Summary

This appendix demonstrated that the SAC methodology from Chapters 1-6 transfers to Isaac Lab without algorithmic changes. You trained SAC on `Isaac-Lift-Cube-Franka-v0`, a four-phase manipulation task (approach, grasp, lift, track) with dense staged reward, achieving +0.54 mean return with 100% positive episodes after 8M steps and approximately 14 minutes of wall time. Along the way, you diagnosed a catastrophic curriculum crash caused by the interaction between Isaac Lab's CurriculumManager and SAC's replay buffer -- a concrete instance of the general principle that off-policy replay assumes a stationary MDP. The pixel variant, using Isaac Lab's native TiledCamera sensor, solved the same task at 1,181 fps and converged in under an hour, which is 24x faster than MuJoCo pixel training.

The numbers you have in hand are concrete: 9,377 fps for state-based training (15x faster than MuJoCo state), 1,181 fps for pixel training (24x faster than MuJoCo pixel), and a 170x end-to-end speedup compared to MuJoCo pixel on similar-scale tasks. These speedups come from GPU-parallel physics, not from algorithmic improvements, which means they apply to any algorithm you run on Isaac Lab.

What this appendix did not cover: sparse-reward manipulation on Isaac Lab (Lift-Cube uses dense reward shaping, so HER was unnecessary), contact-rich POMDP tasks requiring force feedback and recurrence (beyond our MLP pipeline), and multi-GPU scaling (Isaac Lab supports distributed training, but a single GPU was sufficient for our tasks). Each of these is a natural extension for readers who want to push further.

## Reproduce It

```
---------------------------------------------------------
REPRODUCE IT
---------------------------------------------------------
The results in this appendix come from these runs:

State-based (primary):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 256 --total-steps 8000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99

Evaluation:
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py eval \
    --headless \
    --ckpt checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip

Pixel (TiledCamera):
  bash docker/dev-isaac.sh python3 scripts/appendix_e_isaac_manipulation.py train \
    --headless --seed 0 --pixel \
    --env-id Isaac-Lift-Cube-Franka-v0 \
    --num-envs 64 --total-steps 4000000 \
    --checkpoint-freq 500000 \
    --learning-rate 3e-4 --gamma 0.99

Hardware:     NVIDIA A100-SXM4-80GB (any modern GPU works; times will vary)
Time:         ~14 min (state, 8M steps) / ~56 min (pixel, 4M steps)
Seeds:        0 (single seed -- appendix scope)

Artifacts produced:
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.zip
  checkpoints/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0.meta.json
  results/appendix_e_sac_Isaac-Lift-Cube-Franka-v0_seed0_eval.json
  results/appendix_e_isaac_env_catalog.json
  videos/appendix_e_Isaac-Lift-Cube-Franka-v0_*.mp4

Results summary -- State-based (8M steps, seed 0):
  return_mean:  +0.54 +/- 0.05  (100 episodes, deterministic)
  positive_eps: 100/100 (100%)
  min_return:   +0.46
  max_return:   +0.68
  throughput:   9,377 fps (256 envs)
  wall_time:    853 seconds (~14 min)

Results summary -- Pixel (4M steps, seed 0):
  return_mean:  ~-1.07 at convergence
  throughput:   ~1,181 fps (64 envs)
  wall_time:    ~56 min

Wall-clock comparison:
  Ch9 pixel Push (MuJoCo):   ~40 hours (30-50 fps)
  Ch4 state Push (MuJoCo):   ~1 hour (500-600 fps)
  Isaac Lift-Cube (2M):      ~3.6 min (9,363 fps)
  Isaac Lift-Cube (8M):      ~14 min (9,377 fps)
  Isaac Lift-Cube pixel:     ~56 min (1,181 fps)

If your numbers differ by more than ~20%, check the
"What Can Go Wrong" section. Isaac throughput is sensitive
to GPU model, num_envs, and whether other workloads share
the GPU.

The pretrained checkpoints are available in the book's
companion repository for readers using the checkpoint track.
---------------------------------------------------------
```

## Exercises

**E.1 (Warm-up): Throughput scaling curve.** Run the Lift-Cube smoke test at `num_envs` = 1, 4, 16, 64, 128, and 256, recording the fps each time. Plot fps vs `num_envs`. At what point does throughput saturate? How does the curve relate to the GPU utilization you observe in `nvidia-smi`? (Hint: PhysX batching has diminishing returns once the GPU compute units are fully occupied.)

**E.2 (Intermediate): Dense-first validation on a different Isaac task.** Pick a different Isaac Lab environment (e.g., `Isaac-Reach-Franka-v0` or `Isaac-Cartpole-Direct-v0`) and run the same pipeline: `discover-envs` to inspect the observation space, `smoke` to validate the training loop, and `train` for 500K steps. Compare the observation layout to Lift-Cube. Does the pipeline's auto-detection of `MlpPolicy` vs `MultiInputPolicy` work correctly? What throughput do you observe, and how does it compare to Lift-Cube at the same `num_envs`?

**E.3 (Intermediate): Reproduce the curriculum crash.** Modify the training script to *re-enable* the CurriculumManager (by commenting out the curriculum-disable logic) and train for 8M steps. Record the exact timestep where the reward collapses. Does the collapse point depend on the seed? Compare the critic loss trajectory to the clean run. This exercise gives you first-hand experience with the replay-buffer stationarity violation described in the text, which is more memorable than reading about it.

**E.4 (Advanced): Pixel ablation.** Run the pixel training pipeline with `concatenate_terms=True` (which flattens the image into the state vector and forces `MlpPolicy` instead of `MultiInputPolicy` with NatureCNN). How does this affect convergence? The prediction from Chapter 9's sensor separation principle is that the MLP will struggle because it receives raw pixel values mixed with proprioception, losing the spatial structure that CNNs exploit. Verify or refute this prediction with a 4M-step run and compare the learning curves.

**E.5 (Advanced): HER on a goal-conditioned Isaac task.** Isaac Lab's `Isaac-Reach-Franka-v0` uses a flat `policy` observation without goal conditioning. Write a thin wrapper that splits the observation into `observation`, `achieved_goal` (gripper position), and `desired_goal` (target position), exposing a `compute_reward` method based on L2 distance with a 5 cm threshold. Then train with SB3's `HerReplayBuffer` using the future strategy. Does HER improve sample efficiency compared to the dense-reward baseline? This exercise bridges the HER mechanics from Chapter 5 to Isaac Lab's observation convention.

