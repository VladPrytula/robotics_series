# Scaffold: Chapter 2 -- Environment Anatomy

## Classification
Type: Environment
Source tutorial: tutorials/ch01_fetch_env_anatomy.md
Book chapter output: Manning/chapters/ch02_env_anatomy.md
Lab file: scripts/labs/env_anatomy.py (new; needs creation by Lab Engineer)

---

## Experiment Card

```
---------------------------------------------------------
EXPERIMENT CARD: Fetch Environment Inspection
---------------------------------------------------------
Algorithm:    None (inspection and verification only)
Environment:  FetchReachDense-v4, FetchReach-v4, FetchPush-v4

Run command (full inspection):
  bash docker/dev.sh python scripts/ch01_env_anatomy.py all \
    --seed 0

Time:         < 2 min (CPU or GPU)

Checkpoint track:
  N/A (no training; all commands produce results in seconds)

Expected artifacts:
  results/ch01_env_describe.json
  results/ch01_random_metrics.json

Success criteria:
  ch01_env_describe.json exists, contains observation_space
    with keys: observation (shape [10]), achieved_goal (shape [3]),
    desired_goal (shape [3])
  reward-check prints "OK:" (zero mismatches across 500 steps)
  ch01_random_metrics.json exists, success_rate near 0.0-0.1,
    return_mean in [-25, -10] for dense, ep_len_mean == 50
---------------------------------------------------------
```

---

## Verify It Block

(Replaces Reproduce It -- this is an Environment chapter with no training.)

```
---------------------------------------------------------
VERIFY IT
---------------------------------------------------------
This chapter does not train any policies. The verification
commands below confirm that your environment installation
is correct and that all inspection outputs match expected
values.

  bash docker/dev.sh python scripts/ch01_env_anatomy.py all \
    --seed 0

Hardware:     Any machine with Docker (no GPU required)
Time:         < 2 min

Artifacts produced:
  results/ch01_env_describe.json
  results/ch01_random_metrics.json

Expected inspection outputs:
  describe:
    observation_space.observation.shape == [10]
    observation_space.achieved_goal.shape == [3]
    observation_space.desired_goal.shape == [3]
    action_space.shape == [4]
    action_space.low == [-1, -1, -1, -1]
    action_space.high == [1, 1, 1, 1]
    distance_threshold == 0.05
    reward_type == "dense"

  reward-check:
    "OK: reward checks passed" (500 steps, 3 random goals
    per step, atol=1e-6, zero mismatches)

  random-episodes (FetchReachDense-v4, 10 episodes):
    success_rate: 0.0-0.1
    return_mean: approx -15 to -25
    ep_len_mean: 50
    final_distance_mean: approx 0.05-0.15

Lab verification:
  bash docker/dev.sh python scripts/labs/env_anatomy.py --verify
    All checks pass in < 1 min on CPU.

  bash docker/dev.sh python scripts/labs/env_anatomy.py --bridge
    Manual compute_reward matches env.step() reward on 100 steps.
    Dense reward matches -np.linalg.norm(ag - dg) within atol=1e-10.
    Sparse reward matches threshold formula within atol=1e-10.
    Relabeled goals: compute_reward produces correct reward for
    arbitrary goals (simulating HER relabeling).

If any check fails, see "What Can Go Wrong" in the chapter.
---------------------------------------------------------
```

---

## Build It Components

This is an Environment chapter, so Build It is lighter than an Algorithm
chapter. The reader inspects data structures rather than implementing
losses. Each component has a lab region and a verification check.

| # | Component | Concept | Lab file:region | Verify check |
|---|-----------|---------|-----------------|--------------|
| 1 | Observation dictionary inspector | Goal-conditioned observation structure: `obs["observation"]` (10,), `obs["achieved_goal"]` (3,), `obs["desired_goal"]` (3,) | `labs/env_anatomy.py:obs_inspector` | Print keys, shapes, dtypes; assert shapes match (10,), (3,), (3,); assert all values finite; assert goal positions within workspace bounds x:[1.0,1.6], y:[0.4,1.1], z:[0.4,0.6] |
| 2 | Action space explorer | Action semantics: 4D Cartesian deltas dx,dy,dz,gripper in [-1,1]; step the env with axis-aligned actions and observe end-effector displacement | `labs/env_anatomy.py:action_explorer` | Apply action [1,0,0,0] and verify end-effector moves in +x; apply [0,0,1,0] and verify +z movement; assert action_space.shape == (4,); assert bounds [-1,1] |
| 3 | Goal space visualizer | Goal-conditioned MDP goal space G: sample 100 goals via reset, verify 3D Cartesian positions within workspace bounds; show achieved_goal == obs["observation"][:3] for Reach | `labs/env_anatomy.py:goal_space` | Reset 100 times, collect desired_goal and achieved_goal; assert all 3D; assert workspace bounds; for FetchReach, assert np.allclose(achieved_goal, obs["observation"][:3]) confirming phi(s) = grip_pos |
| 4 | Dense reward verifier | Dense reward R = -||ag - dg||; manually compute and compare to env.step() and compute_reward() | `labs/env_anatomy.py:dense_reward_check` | Over 100 steps: compute -np.linalg.norm(ag - dg), compare to step_reward and compute_reward output; all three match within atol=1e-10; assert reward is always <= 0 |
| 5 | Sparse reward verifier | Sparse reward R = 0 if ||ag - dg|| <= epsilon else -1; verify threshold epsilon=0.05 and compute_reward consistency | `labs/env_anatomy.py:sparse_reward_check` | Over 100 steps on FetchReach-v4: verify reward is exactly 0.0 or -1.0; verify threshold matches env.unwrapped.distance_threshold; verify compute_reward matches step reward exactly |
| 6 | Goal relabeling simulator | HER applicability: call compute_reward with arbitrary (non-env) goals to prove relabeling works; the critical invariant | `labs/env_anatomy.py:relabel_check` | Sample 10 random goals from goal space; call compute_reward(ag, random_goal, info) for each; verify result matches -||ag - random_goal|| (dense) or threshold formula (sparse); no errors, no NaN; this proves HER's reward recomputation is valid |
| 7 | Cross-environment comparison | Compare observation dims across FetchReach (10D), FetchPush (25D), FetchPickAndPlace (25D); goal semantics change (grip_pos vs object_pos) | `labs/env_anatomy.py:cross_env_compare` | Create each env, reset, print obs["observation"].shape; assert Reach=10, Push=25, PickAndPlace=25; for Push, assert achieved_goal != obs["observation"][:3] (object pos, not grip pos) |

**Ordering rationale:** Foundation first -- the reader sees what the agent
perceives (1-2), then understands goals (3), then rewards (4-5), then the
HER connection (6), then generalizes across tasks (7). Each builds on the
previous.

---

## Bridging Proof

The bridging proof for an Environment chapter connects the manual ("from
scratch") inspection to the production script (`scripts/ch01_env_anatomy.py`).

- **Inputs (same data fed to both):**
  Env: FetchReachDense-v4, seed=42, 100 random actions.
  For each step: record achieved_goal, desired_goal, step_reward.

- **From-scratch output (lab code):**
  For each step, manually compute:
  - `manual_reward = -np.linalg.norm(ag - dg)`
  - `cr_reward = env.unwrapped.compute_reward(ag, dg, info)`
  Then for 5 randomly sampled alternative goals per step:
  - `relabel_reward = env.unwrapped.compute_reward(ag, random_goal, info)`
  - `expected_relabel = -np.linalg.norm(ag - random_goal)`

- **Production output (ch01_env_anatomy.py reward-check):**
  Same 100 steps, same env/seed, reports OK with zero mismatches.

- **Match criteria:**
  - `step_reward == cr_reward` (exact float match, atol=1e-10)
  - `step_reward == manual_reward` (within atol=1e-10)
  - `relabel_reward == expected_relabel` (within atol=1e-10)
  - Zero mismatches out of 100 steps x 5 random goals = 500 reward checks
  - Both lab `--bridge` and production `reward-check` report OK

- **Lab mode:** `--bridge`
  Runs the side-by-side comparison and prints match/mismatch annotations.

---

## What Can Go Wrong

| Symptom | Likely cause | Diagnostic |
|---------|-------------|------------|
| `obs` is a flat ndarray, not a dict | Old gymnasium-robotics version (< 1.0) or wrong env ID | `print(type(obs))` after reset; check `pip show gymnasium-robotics` version >= 1.0 |
| `obs["observation"].shape` is not (10,) for FetchReach | Wrong env variant (Push/PickAndPlace have 25D) or version mismatch | Print shape and verify env_id; check `env.observation_space` |
| `compute_reward` raises `AttributeError` | Calling on wrapped env instead of `env.unwrapped` | Use `env.unwrapped.compute_reward(ag, dg, info)` |
| Step reward and compute_reward differ | Version mismatch between gymnasium and gymnasium-robotics; or info dict stale | Upgrade both packages; verify with `pip list | grep gymnasium` |
| `desired_goal` is always the same across resets | Not passing different seeds to `env.reset(seed=...)` | Pass unique seed per episode: `env.reset(seed=42+ep)` |
| `achieved_goal` does not match `obs["observation"][:3]` on FetchPush | This is correct for Push -- achieved_goal is object position, not gripper position | Check env variant; for Push/PickAndPlace, achieved_goal tracks the object |
| `is_success` is True immediately after reset | Goal happened to be sampled at gripper position (rare) | Run multiple episodes; this happens < 5% of the time and is normal |
| Workspace bounds look wrong (positions near 0 or very large) | Different MuJoCo model or coordinate frame issue | Check `env.unwrapped.initial_gripper_xpos`; typical values are [1.34, 0.75, 0.53] |
| `EnvironmentNameNotFound` for `FetchReachDense-v4` | gymnasium-robotics not installed or wrong version | `pip install gymnasium-robotics`; verify with `python -c "import gymnasium_robotics"` |
| Random baseline success_rate much higher than 0.1 | Distance threshold larger than expected or goal space very small | Check `env.unwrapped.distance_threshold`; should be 0.05 |

---

## Adaptation Notes

### Cut from tutorial

- **Formal Definition blocks for GCMDP:** The tutorial (Part II, Sec 2.1)
  has a heavy formal section with Definition (Goal-Conditioned MDP),
  Definition (Goal-Conditioned Observation), Proposition (Reward
  Recomputation) in full mathematical notation. Cut the formal theorem-style
  presentation. Keep the concepts but present them conversationally: "The
  MDP has seven pieces..." rather than "$\mathcal{M} = (\mathcal{S},
  \mathcal{A}, \mathcal{G}, P, R, \phi, \gamma)$ where..."

- **Part IV "Theoretical Implications" (Sec 4.1-4.3):** The tutorial has
  ~800 words of formal propositions (HER sufficient conditions, corollary
  about non-goal environments). Replace with a shorter, practical "why this
  matters for HER" paragraph in the Build It flow, grounded in the
  relabeling simulation (component 6).

- **Appendix B "Formal Verification of HER Requirements":** Redundant
  with Build It component 6. Cut entirely; the lab code covers this.

- **The Abstract:** Tutorial opens with a formal abstract. Drop for Manning;
  the Opening Promise replaces it.

- **"Part" structure (Part 0, I, II, III, IV, V, VI):** Flatten to Manning
  section numbering (2.1, 2.2, ...). The six-part structure is too heavy
  for a 20-page chapter.

- **Section 0.1 "Where We Are" / 0.2 "What We Are Simulating":**
  The full MuJoCo physics description (~400 words), real robot spec table,
  and Further Reading links are interesting but too long. Compress the
  MuJoCo description to ~2 sentences and the robot description to ~3
  sentences. Move the spec tables to a sidebar or cut them.

- **Section 0.4 "The Four Fetch Tasks":** Already covered in Ch1 Manning
  chapter (section 1.2). Keep only a brief reminder.

### Keep from tutorial

- **The core inspection sequence (Sec 3.1-3.4):** Describe, reward-check,
  random-episodes. This is the "Run It" backbone. Keep the commands and
  expected outputs.

- **Section 0.5 "Why Environment Anatomy Matters":** The three-bullet
  explanation of why dictionary obs + compute_reward + threshold enable
  HER. This is the chapter's central argument. Keep and expand as the
  WHY section spine.

- **Section 3.3 "Reward Consistency Verification":** The critical invariant
  check. Central to Build It components 4-6.

- **Appendix A "Observation Component Details":** The 10D and 25D breakdown
  tables. Essential reference. Keep as a table in Build It component 1 or
  as a sidebar.

- **The action semantics table (Sec 3.2):** Index-to-semantic mapping for
  the 4D action vector. Keep for Build It component 2.

- **Section 4.2 "Sparse vs Dense Reward Trade-off":** Keep the practical
  comparison but trim the formal notation.

- **Deliverables (Part V):** The self-check questions. Adapt as end-of-section
  verification checkpoints.

### Add for Manning

- **Chapter Bridge (from Ch1):** Ch1 established that the environment works
  (proof of life). Gap: we verified the pipeline runs but do not yet
  understand what the robot sees, what actions mean, or how rewards work.
  Ch2 fills this gap: a complete anatomy of observations, actions, goals,
  and rewards. Foreshadow: Ch3 will use this understanding to train PPO
  on dense Reach.

- **Opening Promise:** 3-5 bullet "This chapter covers" block.

- **Build It as narrative spine:** The tutorial presents inspection as
  Part III "Implementation" after the theory. For Manning, interleave:
  introduce concept -> lab code -> verification checkpoint. Build It
  components 1-7 become the chapter's main flow.

- **Cross-environment comparison (component 7):** The tutorial mentions
  FetchPush/PickAndPlace dimensions in passing (Sec 4.3, Appendix A).
  Make this a proper Build It component so the reader sees how the
  interface generalizes before we need it in later chapters.

- **Concrete workspace bounds:** Add a figure description or ASCII diagram
  showing the Fetch workspace: x in [1.0, 1.6], y in [0.4, 1.1],
  z in [0.4, 0.6], with the table surface, robot base, and typical
  goal positions annotated.

- **Explicit "what does the policy network see?" summary:** After all Build
  It components, add a summary table showing exactly what SB3's
  MultiInputPolicy will receive when we use it in Ch3. Map observation
  dict keys to network input dimensions. This bridges Build It to Run It
  in later chapters.

- **Exercises:** 3-5 graduated exercises (see below).

---

## Chapter Bridge

1. **Capability established:** Chapter 1 verified that the entire
   computational stack works -- Docker, MuJoCo, rendering, training loop.
   You have a proof-of-life: the environment runs, checkpoints save, and
   the `compute_reward` invariant holds.

2. **Gap:** You know the environment *works*, but not what it *says*. What
   do the 10 numbers in the observation vector mean? What happens when
   the robot takes action [1, 0, 0, 0]? Why does the reward function
   return -0.073 instead of -1? How does the environment know the robot
   "succeeded"? Without answering these questions, you cannot debug
   training failures or choose appropriate algorithms.

3. **This chapter adds:** A complete anatomy of Fetch environment
   observations, actions, rewards, and goals. You will inspect every
   component by hand, verify reward computation against the distance
   formula, simulate HER-style goal relabeling, and establish the
   random-policy baseline that every trained agent must beat.

4. **Foreshadow:** With the environment understood, Chapter 3 trains a
   real policy (PPO on dense Reach). The observation shapes you document
   here determine the network architecture. The random baseline you
   establish here is the floor that PPO must exceed.

---

## Opening Promise

> **This chapter covers:**
>
> - Inspecting the dictionary observation structure that every Fetch
>   environment returns -- what each number means physically and why
>   observations are dictionaries, not flat vectors
> - Understanding what the 4D action vector controls: Cartesian deltas
>   for the end-effector and a gripper open/close command
> - Verifying reward computation for both dense (distance-based) and sparse
>   (binary success/failure) variants, and proving the critical invariant
>   that makes Hindsight Experience Replay possible
> - Simulating goal relabeling by hand -- calling `compute_reward` with
>   goals the environment never intended -- to see why the Fetch interface
>   enables HER
> - Establishing a random-policy baseline (success rate, mean return, goal
>   distance) that every trained agent in later chapters must beat

---

## Estimated Length

| Section | Words |
|---------|-------|
| Opening promise + chapter bridge | 400 |
| 2.1 WHY: Why environment anatomy matters (the felt problem, three questions checklist, what misunderstandings cost) | 1,200 |
| 2.2 The Fetch task family (brief recap from Ch1, workspace description) | 600 |
| 2.3 Build It: Observation dictionary (component 1 + component 3 goal space) | 1,200 |
| 2.4 Build It: Action semantics (component 2) | 700 |
| 2.5 Build It: Reward computation -- dense and sparse (components 4-5) | 1,200 |
| 2.6 Build It: Goal relabeling simulation (component 6, HER preview) | 800 |
| 2.7 Build It: Cross-environment comparison (component 7) | 600 |
| 2.8 Bridge: Manual inspection meets the production script | 500 |
| 2.9 Run It: The inspection pipeline (experiment card, commands, artifacts) | 800 |
| 2.10 What can go wrong | 800 |
| 2.11 Summary + bridge to Ch3 | 500 |
| Verify It block | 300 |
| Exercises (4 exercises) | 600 |
| **Total** | **~9,200** |

(Target range: 6,000-10,000 words. This estimate leaves room for code
listings, which Manning counts separately from prose but are included in
overall page count.)

---

## Concept Registry Additions

Terms this chapter introduces (to be added to the Concept Registry under Ch1,
which maps to Manning Ch2):

- **goal-conditioned MDP** ($\mathcal{S}, \mathcal{A}, \mathcal{G}, P, R, \phi, \gamma$): the seven-tuple formalism; presented conversationally, not as a formal Definition block
- **goal-conditioned observation**: the dictionary tuple $(\bar{s}, g_a, g_d)$
- **goal-achievement mapping $\phi$**: function extracting achieved goal from state (grip_pos for Reach, object_pos for Push)
- **compute_reward API**: `env.unwrapped.compute_reward(achieved_goal, desired_goal, info)`
- **dense reward**: $R = -\|g_a - g_d\|_2$
- **sparse reward**: $R = 0$ if $\|g_a - g_d\|_2 \leq \epsilon$ else $-1$
- **success threshold $\epsilon$**: 0.05 m (5 cm) for Fetch environments
- **dictionary observation structure**: the three-key dict (observation, achieved_goal, desired_goal)
- **critical invariant (reward recomputation)**: `env.step()` reward == `compute_reward(ag, dg, info)`
- **HER applicability theorem** (informal): explicit achieved_goal + recomputable reward -> HER is applicable

These match the Concept Registry entries for "Ch1" in the root CLAUDE.md.

---

## Dependencies

- **Lab regions needed (for Lab Engineer):**
  New file `scripts/labs/env_anatomy.py` with regions:
  - `obs_inspector` -- inspect obs dict structure, shapes, dtypes, bounds
  - `action_explorer` -- step with axis-aligned actions, measure displacement
  - `goal_space` -- sample goals via reset, check bounds, verify phi
  - `dense_reward_check` -- verify dense reward = -||ag-dg|| = step_reward = compute_reward
  - `sparse_reward_check` -- verify sparse reward = 0/-1, threshold, compute_reward match
  - `relabel_check` -- call compute_reward with arbitrary goals, verify correctness
  - `cross_env_compare` -- compare obs dims across Reach/Push/PickAndPlace
  - Modes: `--verify` (runs all checks, < 1 min), `--bridge` (side-by-side with production script output)
  - No `--demo` mode (no training in this chapter)

- **Pretrained checkpoints needed:** None (Environment chapter, no training).

- **Previous chapter concepts used (from Manning Ch1 / tutorial Ch0):**
  - reproducibility, container, image (defined in Ch0 concept registry)
  - proof of life, rendering backend
  - The three Hadamard diagnostic questions (introduced in Ch1)
  - The experiment contract / "no vibes" rule (introduced in Ch1)
  - `docker/dev.sh` workflow (introduced in Ch1)

- **Production script used:** `scripts/ch01_env_anatomy.py` (already exists;
  no changes needed -- the lab code wraps the same logic in teaching-oriented
  components with snippet-include regions)

---

## Exercises

**1. (Verify) Confirm observation structure across seeds.**

Reset `FetchReachDense-v4` with 5 different seeds. For each reset, verify
that the observation dictionary has the same three keys and same shapes.
Record the range of `desired_goal` values across resets. Expected: goals
span the workspace (x roughly 1.05-1.55, y roughly 0.40-1.10, z roughly
0.42-0.60). If all goals are identical, something is wrong with the seed
handling.

**2. (Tweak) Compare dense and sparse rewards on the same trajectory.**

Create both `FetchReachDense-v4` and `FetchReach-v4` with the same seed.
Take the same sequence of 50 random actions in both. Compare the reward
sequences side by side. Questions: (a) Is the dense return always more
negative than the sparse return? (b) What fraction of steps have sparse
reward = 0? (c) At what distance does the sparse reward switch from -1 to 0?

Expected: sparse reward is 0 only when distance < 0.05. Dense return is
typically -15 to -25; sparse return is typically -50 (all -1s).

**3. (Extend) Observation breakdown for FetchPush.**

Create `FetchPushDense-v4` and inspect the 25D observation vector. Using
Appendix A (the observation component table), identify: (a) which indices
correspond to the object position, (b) what `achieved_goal` represents (hint:
it is the object position, not the gripper position), and (c) what happens
to the object position when you take action [0,0,0,0] for 50 steps (does
the object move?).

Expected: object stays roughly in place (gravity keeps it on the table);
achieved_goal tracks the object, not the gripper.

**4. (Challenge) Estimate success rate as a function of distance threshold.**

Run 100 random episodes on `FetchReachDense-v4`. For each episode, record the
final distance between achieved_goal and desired_goal. Then plot (or tabulate)
what the success rate *would be* at thresholds of 0.01, 0.02, 0.05, 0.10, and
0.20 meters. How does the "difficulty" of the task change with the threshold?

Expected: at threshold 0.01, random success is ~0%; at 0.20, it may be 5-15%.
The default threshold of 0.05 makes random success very rare (0-5%).
