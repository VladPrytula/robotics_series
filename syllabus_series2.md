# Series 2: Robot Shenanigans
## *From Pixels to Reality -- Building Robots That Do Cool Stuff*

> **One policy, ten tasks. One simulation, real robot. One party, impressed friends.**

![Placeholder for Series 2 demo](videos/series2_demo_placeholder.gif)

*Series 1 taught a robot to reach for things using coordinates. Series 2 teaches it to see, drive, stack, and maybe open your beer.*

---

## What You Will Build

By the end of Series 2, you will have trained robots to accomplish tasks that you would actually want to show someone. Along the way, you will:

| Part | Chapters | Milestone | What You Learn |
|------|----------|-----------|----------------|
| I | 0-3 | Visual manipulation | CNNs for RL, augmentation, visual goals |
| II | 4-6 | Racing and walking | Driving from pixels, locomotion gaits |
| III | 7-10 | Complex manipulation | Doors, stacking, tool use |
| IV | 11-13 | Multi-task generalist | One policy for many tasks, curriculum |
| V | 14-16 | Reality-ready policies | Domain randomization, system ID |
| Bonus | B1-B3 | Real robot (optional) | Hardware deployment, the party trick |

Each chapter includes runnable commands and "done when" criteria. We find this structure useful because visual RL and sim-to-real are even more frustrating than Series 1 -- the failure modes are subtler and the debugging is harder. Having concrete checkpoints helps.

## Prerequisites

**Series 1 completion.** You should have:
- Working Docker + MuJoCo + GPU setup
- Understanding of SAC + HER
- Experience with goal-conditioned RL on Fetch tasks
- The reproducibility mindset (seeds, metrics, artifacts)

**Additional for Series 2:**
- Basic familiarity with CNNs (you do not need to be a vision expert)
- Willingness to wait longer for training (visual RL is slower)
- Patience for debugging (visual policies fail in creative ways)

## Our Approach

Series 1 derived SAC + HER from problem constraints. Series 2 extends this thinking:

- State observations are a luxury -> visual observations (cameras are what robots have)
- Single task is limiting -> multi-task learning (real robots do many things)
- Simulation is wrong -> domain randomization and system ID (robustness to reality)
- Perfect is the enemy of deployed -> practical transfer (some sim-to-real success beats none)

The core insight: **the methods we use are responses to new constraints, not arbitrary choices.**

## Who This Is For

You have completed Series 1 or equivalent. You are comfortable with RL fundamentals, Docker workflows, and GPU training. You want to move beyond "it works in simulation on state observations" toward "it works on pixels and might work on a real robot."

**Time investment:** 16 weeks for core chapters, additional time for bonus hardware chapters. Visual RL experiments run longer. Budget accordingly.

**Hardware:** The core 16 chapters require only simulation. The 3 bonus chapters require purchasing a robot (~$150-$500 depending on choice). Hardware is entirely optional.

---

## Quick Links

**Tutorials** (theory + context):
- Part I: Visual RL (Chapters 0-3)
- Part II: Driving and Locomotion (Chapters 4-6)
- Part III: Complex Manipulation (Chapters 7-10)
- Part IV: Multi-Task Learning (Chapters 11-13)
- Part V: Sim-to-Real (Chapters 14-16)
- Bonus: Real Hardware (Chapters B1-B3)

**Scripts** (executable code):
- `scripts/s2_ch00_visual_reach.py` -- visual FetchReach
- `scripts/s2_ch04_car_racing.py` -- CarRacing-v2
- `scripts/s2_ch07_door_opening.py` -- door manipulation
- `scripts/s2_ch11_multitask.py` -- MetaWorld multi-task
- `scripts/s2_ch14_domain_rand.py` -- domain randomization

---

## Tooling Stack

Series 2 expands beyond Gymnasium-Robotics:

| Environment | Source | Chapters | Purpose |
|-------------|--------|----------|---------|
| Gymnasium-Robotics (Fetch) | `gymnasium-robotics` | 0-3 | Visual goal-conditioned RL |
| CarRacing-v2 | `gymnasium[box2d]` | 4-5 | Driving from pixels |
| DMControl | `dm_control` | 6 | Locomotion benchmarks |
| Robosuite | `robosuite` | 7-9 | Contact-rich manipulation |
| MetaWorld | `metaworld` | 11 | Multi-task benchmark |
| Custom MuJoCo | We build | 10, 12 | Tool use, curriculum |

**New dependencies:**
```bash
# Visual RL
pip install opencv-python kornia  # augmentations

# Driving
pip install gymnasium[box2d]

# Locomotion
pip install dm_control

# Manipulation
pip install robosuite metaworld
```

---

## Part I: Eyes Wide Open (Chapters 0-3)
*"Your robot just got cameras. Now what?"*

### Chapter 0: The Pixel Problem

**Goal:** Solve FetchReach using only image observations -- no coordinate cheating.

**WHY this matters:**
Real robots have cameras, not coordinate oracles. Moving from state to pixels is the first step toward deployable policies. This chapter establishes the visual RL foundation we build on throughout Series 2.

**What we build:**
- Image observation wrapper for Fetch environments (84x84x3)
- CNN encoder for SAC (simple 3-layer architecture)
- Frame stacking for temporal information

**The practical CNN recipe:**
```
Conv(32, 3x3, stride=2) -> ReLU ->
Conv(64, 3x3, stride=2) -> ReLU ->
Conv(64, 3x3, stride=2) -> ReLU ->
Flatten -> Linear(256)
```
This is not optimal. This is sufficient. We move on.

**Key concepts:**
- Partial observability -- images do not directly encode velocity
- The curse of dimensionality -- 84x84x3 = 21,168 dimensions vs ~25 for state
- Why visual RL is slower (more parameters, harder credit assignment)

**Steps:**
- [ ] Implement `VisualFetchWrapper` that renders and returns images
- [ ] Implement CNN encoder compatible with SB3
- [ ] Train visual SAC+HER on FetchReach
- [ ] Compare sample efficiency: visual vs state-based

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch00_visual_reach.py train --total-steps 500000
bash docker/dev.sh python scripts/s2_ch00_visual_reach.py eval --ckpt checkpoints/visual_reach.zip
bash docker/dev.sh python scripts/s2_ch00_visual_reach.py compare  # visual vs state
```

**Done when:**
- Visual SAC+HER achieves >80% success rate on FetchReach
- You can quantify the sample efficiency gap (expect 2-5x slower than state-based)
- You understand why frame stacking helps

---

### Chapter 1: Data Augmentation for Free Robustness

**Goal:** Make visual policies robust to visual variations without extra environment samples.

**WHY this matters:**
Visual policies are fragile. Change the lighting, they break. Data augmentation during training provides robustness almost for free -- this is one of the most impactful techniques in visual RL.

**What we build:**
- DrQ-style augmentation (random crop, color jitter)
- Augmentation ablation study
- Robustness evaluation protocol

**Key concepts:**
- Augmentation as implicit domain randomization
- Why random crop works (translation invariance, diverse crops from single image)
- The augmentation-performance tradeoff (too much hurts, too little leaves fragility)

**Augmentations to implement:**
1. Random crop (most important)
2. Color jitter (brightness, contrast, saturation)
3. Random convolution (texture randomization)

**Steps:**
- [ ] Implement augmentation wrappers
- [ ] Train with each augmentation type
- [ ] Ablation: which augmentations help most?
- [ ] Evaluate robustness under visual perturbation

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch01_augmentation.py train --aug random_crop
bash docker/dev.sh python scripts/s2_ch01_augmentation.py train --aug color_jitter
bash docker/dev.sh python scripts/s2_ch01_augmentation.py train --aug all
bash docker/dev.sh python scripts/s2_ch01_augmentation.py robustness-test --ckpt checkpoints/aug_all.zip
```

**Done when:**
- Augmented policy maintains >70% success under visual distractors not seen during training
- You have empirical evidence for which augmentations matter most
- Non-augmented baseline degrades significantly under same distractors

---

### Chapter 2: Representation Matters (But We Will Not Obsess)

**Goal:** Understand when to use pretrained encoders vs end-to-end learning.

**WHY this matters:**
The CNN encoder learns a representation. Should we pretrain it? Use frozen ImageNet weights? This chapter provides practical guidance without diving into representation learning research.

**What we compare:**
1. End-to-end learning (Chapter 0 baseline)
2. Frozen pretrained encoder (ResNet-18 ImageNet weights)
3. Contrastive auxiliary loss (CURL-lite)

**Key concepts:**
- Why pretrained vision helps (or does not) for robotics
- The representation learning landscape (brief overview, not deep dive)
- Practical recommendation: for simple tasks with <1M steps, end-to-end + augmentation usually wins

**Steps:**
- [ ] Implement frozen encoder option
- [ ] Implement simple contrastive auxiliary loss
- [ ] Compare all three approaches on FetchReach and FetchPush
- [ ] Document when each approach is appropriate

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch02_representations.py train --encoder end2end
bash docker/dev.sh python scripts/s2_ch02_representations.py train --encoder frozen_resnet
bash docker/dev.sh python scripts/s2_ch02_representations.py train --encoder contrastive
bash docker/dev.sh python scripts/s2_ch02_representations.py compare
```

**Done when:**
- You have empirical comparison across encoder types
- You can explain when pretrained encoders help (spoiler: mostly for larger tasks)
- You adopt a practical default for remaining chapters

---

### Chapter 3: Visual Goal-Conditioning

**Goal:** Specify goals as images, not coordinates -- "make the scene look like THIS."

**WHY this matters:**
Image goals are natural for robots. Instead of "move block to (0.3, 0.2)", we show a picture of where we want it. This is how humans often communicate tasks.

**What we build:**
- Goal-image conditioned policy
- Visual HER (relabel with achieved images, not coordinates)
- Goal encoder architecture

**Key concepts:**
- Goal images are rich specifications
- The representation alignment problem (goal encoder and observation encoder must produce compatible embeddings)
- Implementation: concatenate goal image to observation (simple but effective)

**Steps:**
- [ ] Implement visual goal wrapper (captures goal images)
- [ ] Adapt HER for image goals
- [ ] Train on visual FetchPush with image goals
- [ ] Evaluate: show novel goal image, measure success

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch03_visual_goals.py train --env FetchPush-v4 --total-steps 1000000
bash docker/dev.sh python scripts/s2_ch03_visual_goals.py eval --ckpt checkpoints/visual_goal_push.zip
bash docker/dev.sh python scripts/s2_ch03_visual_goals.py demo --ckpt checkpoints/visual_goal_push.zip  # interactive goal setting
```

**Done when:**
- Visual goal-conditioned policy solves FetchPush with >70% success
- You can show the robot an image of desired block position, and it pushes accordingly
- Visual HER relabeling works correctly (verify with logging)

---

## Part II: Need for Speed (Chapters 4-6)
*"Everything so far moved at 2 fps. Time to go fast."*

### Chapter 4: Your First Racing Game

**Goal:** Train a driving agent on CarRacing-v2 from pixels.

**WHY this matters:**
Driving is intuitive, visual, and immediately fun. It is also a gateway to real robot deployment -- toy RC cars with cameras are cheap and accessible. This chapter establishes visual control in a dynamic domain.

**The environment:**
CarRacing-v2 (Gymnasium) -- top-down view, continuous steering/gas/brake, procedurally generated tracks.

**What we build:**
- SAC adaptation for CarRacing
- Frame skip and action repeat
- Handling the control asymmetry (steering is symmetric, gas/brake are not)

**Key concepts:**
- Frame skip -- run action for N steps, observe result (typical: N=4)
- Action repeat vs frame stacking (related but different)
- The "do not go backwards" challenge

**Steps:**
- [ ] Implement CarRacing wrapper with frame skip
- [ ] Adapt visual SAC pipeline
- [ ] Train and evaluate on fixed track seed
- [ ] Train and evaluate on random tracks

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch04_car_racing.py train --total-steps 1000000
bash docker/dev.sh python scripts/s2_ch04_car_racing.py eval --ckpt checkpoints/car_racing.zip --n-episodes 20
bash docker/dev.sh python scripts/s2_ch04_car_racing.py leaderboard  # compare lap times
```

**Fun metric:** Lap time leaderboard. Can you beat 40 seconds?

**Done when:**
- Agent consistently completes laps without going backwards
- Average lap time <50 seconds
- You can watch videos of your agent racing (actually satisfying)

---

### Chapter 5: Track Mastery and Generalization

**Goal:** Train a single policy that handles procedurally-generated tracks.

**WHY this matters:**
One track is easy to overfit. CarRacing generates infinite random tracks. Can one policy handle them all? This tests generalization -- critical for real-world deployment.

**What we build:**
- Training on procedural track distribution
- Evaluation protocol across diverse tracks
- Failure mode analysis (tight corners, S-curves)

**Key concepts:**
- In-distribution vs out-of-distribution generalization
- Why procedural environments are RL-friendly (infinite training data)
- Track-specific vs general driving skills

**Steps:**
- [ ] Train on diverse procedural tracks (multiple seeds per episode)
- [ ] Evaluate on held-out track seeds
- [ ] Analyze failure modes by track geometry
- [ ] Attempt recovery training for difficult corners

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch05_track_generalization.py train --track-seeds 0-100
bash docker/dev.sh python scripts/s2_ch05_track_generalization.py eval --track-seeds 100-200  # held out
bash docker/dev.sh python scripts/s2_ch05_track_generalization.py analyze-failures --ckpt checkpoints/car_general.zip
```

**Done when:**
- Single policy completes >80% of random tracks (not seen during training)
- You can identify which track features cause failures
- Generalization gap documented (training tracks vs held-out tracks)

---

### Chapter 6: Creatures That Walk

**Goal:** Train locomotion policies from pixels on DMControl.

**WHY this matters:**
Driving has wheels. Legs are different -- periodic, rhythmic, with complex contact dynamics. This chapter rounds out Part II with a fundamentally different control challenge.

**The environments:**
- Walker-Walk: bipedal walking (medium difficulty)
- Cheetah-Run: quadruped running (harder, faster dynamics)

**What we build:**
- Visual SAC adaptation for DMControl
- Handling proprioceptive + visual observations
- The "blind walking" ablation

**Key concepts:**
- Locomotion as periodic control
- Why visual locomotion is hard (self-occlusion, no obvious visual goal)
- Proprioception matters -- the robot needs to feel its joints

**Steps:**
- [ ] Implement DMControl wrapper with visual observations
- [ ] Train Walker-Walk from pixels
- [ ] Train Cheetah-Run from pixels
- [ ] Ablation: vision-only vs proprio-only vs both

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch06_locomotion.py train --env walker-walk --total-steps 1000000
bash docker/dev.sh python scripts/s2_ch06_locomotion.py train --env cheetah-run --total-steps 2000000
bash docker/dev.sh python scripts/s2_ch06_locomotion.py ablation --env walker-walk
```

**Done when:**
- Walker walks stably for 1000 steps without falling
- Cheetah runs with reasonable gait (no flailing)
- Ablation shows proprioception helps (vision-only struggles)

---

## Part III: Clever Hands (Chapters 7-10)
*"Reaching was the warm-up. Now we manipulate."*

### Chapter 7: The Door Problem

**Goal:** Train a policy to open a door -- harder than it sounds.

**WHY this matters:**
Opening a door requires sequential precision: approach, grasp handle, rotate, pull. It is a gateway to "contact-rich manipulation" where the robot must reason about physical interactions.

**The environment:**
Robosuite Door environment (or custom MuJoCo door)

**What we build:**
- Staged reward shaping (approach -> grasp -> rotate -> pull)
- Comparison: sparse vs shaped rewards
- Handle-grasp precision analysis

**Key concepts:**
- Contact-rich manipulation is hard (friction, grip, leverage)
- Why doors test the limits of goal-conditioned RL
- Gripper orientation matters more than position

**Steps:**
- [ ] Set up Robosuite Door environment
- [ ] Train with dense shaped reward (establishes baseline)
- [ ] Train with sparse reward + HER (the real test)
- [ ] Analyze grasp success vs door open success

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch07_door_opening.py train --reward dense --total-steps 1000000
bash docker/dev.sh python scripts/s2_ch07_door_opening.py train --reward sparse --her --total-steps 2000000
bash docker/dev.sh python scripts/s2_ch07_door_opening.py eval --ckpt checkpoints/door_sparse.zip
```

**Done when:**
- >70% door opening success from varied starting positions
- You understand the failure modes (missed handle, insufficient rotation, etc.)
- Sparse+HER approaches shaped reward performance (may require more steps)

---

### Chapter 8: Picking Things Up (For Real This Time)

**Goal:** Visual pick-and-place with diverse objects.

**WHY this matters:**
Series 1 FetchPickAndPlace used state observations. Real picking requires visual perception of diverse objects. This chapter bridges that gap.

**The environment:**
Robosuite Pick-and-Place or MetaWorld pick tasks

**What we build:**
- Visual policy for pick-and-place
- Implicit object pose estimation (through the policy)
- Object diversity training (shapes, colors)

**Key concepts:**
- Grasp planning vs learned grasping
- Why shape diversity is harder than color diversity (geometry matters)
- The "picked but failed to place" decomposition

**Steps:**
- [ ] Set up visual pick-and-place environment
- [ ] Train on single object type (baseline)
- [ ] Train on diverse object set (5+ shapes)
- [ ] Evaluate generalization to novel objects

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch08_visual_pick.py train --objects cube --total-steps 1000000
bash docker/dev.sh python scripts/s2_ch08_visual_pick.py train --objects diverse --total-steps 2000000
bash docker/dev.sh python scripts/s2_ch08_visual_pick.py eval --ckpt checkpoints/pick_diverse.zip --objects novel
```

**Done when:**
- >80% pick-and-place success on training object distribution
- >60% success on held-out object shapes
- You can decompose failures into pick failures vs place failures

---

### Chapter 9: Stacking Blocks

**Goal:** Stack blocks -- the gateway to long-horizon sequential tasks.

**WHY this matters:**
Stacking requires precision AND sequencing. Each block placement affects the next. Errors compound. This is qualitatively harder than single-step manipulation.

**The environment:**
Robosuite Stack or custom MuJoCo stacking

**What we build:**
- 2-block stacking (harder than you think)
- 3-block stacking (much harder)
- Curriculum from easy to hard configurations

**Key concepts:**
- Compounding errors -- early mistakes propagate
- Tower stability -- physics matters
- Why curriculum helps for sequential tasks

**Steps:**
- [ ] Implement stacking environment with configurable stack height
- [ ] Train 2-block stacking
- [ ] Train 3-block stacking with curriculum (start with blocks close)
- [ ] Compare curriculum vs no-curriculum

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch09_stacking.py train --height 2 --total-steps 1000000
bash docker/dev.sh python scripts/s2_ch09_stacking.py train --height 3 --curriculum --total-steps 3000000
bash docker/dev.sh python scripts/s2_ch09_stacking.py compare --height 3  # curriculum vs flat
```

**Done when:**
- 2-block stacking: >80% success
- 3-block stacking: >50% success (this is legitimately hard)
- Curriculum demonstrably helps for 3-block

---

### Chapter 10: Tool Use -- The Lever

**Goal:** Use a tool to accomplish a task the arm cannot do directly.

**WHY this matters:**
Tool use is a hallmark of intelligence. The robot must discover that a stick extends its reach. No one tells it to use the tool -- it must figure this out from sparse reward.

**The environment:**
Custom MuJoCo: object out of reach, stick available, goal is to push object to target.

**What we build:**
- Tool-use environment (object, tool, goal)
- Reward only for final object position (not for using tool)
- Analysis of emergent tool use

**Key concepts:**
- Credit assignment through the tool (challenging)
- Hierarchical structure emerges (grasp tool -> use tool)
- Emergent vs shaped tool use (we want emergent)

**Steps:**
- [ ] Build custom tool-use environment
- [ ] Train with sparse goal reward only
- [ ] Verify tool use emerges (not hard-coded)
- [ ] Analyze when/how tool use is discovered during training

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch10_tool_use.py build-env  # creates custom environment
bash docker/dev.sh python scripts/s2_ch10_tool_use.py train --total-steps 3000000
bash docker/dev.sh python scripts/s2_ch10_tool_use.py analyze --ckpt checkpoints/tool_use.zip
```

**Fun hook:** Watch the robot discover it needs the stick without being told.

**Done when:**
- >60% success rate on out-of-reach targets
- Video evidence of emergent tool use
- The policy does NOT receive explicit "use tool" reward

---

## Part IV: The Generalist (Chapters 11-13)
*"One task, one policy is nice. Ten tasks, one policy is better."*

### Chapter 11: The Multi-Task Challenge

**Goal:** Train a single policy to solve 10 different manipulation tasks.

**WHY this matters:**
Real robots should do more than one thing. Multi-task learning tests whether representations and skills can be shared across tasks.

**The environment:**
MetaWorld ML10 (10 manipulation tasks: reach, push, pick-place, door-open, drawer-close, etc.)

**What we build:**
- Task-conditioned policy (one-hot task ID input)
- Multi-task SAC with shared replay
- Per-task success tracking

**Key concepts:**
- Positive transfer -- learning reach helps learning push
- Negative transfer -- some tasks interfere
- The multi-task optimization problem (balancing task gradients)

**Steps:**
- [ ] Set up MetaWorld ML10 benchmark
- [ ] Implement task-conditioned policy
- [ ] Train multi-task SAC
- [ ] Evaluate per-task and aggregate success

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch11_multitask.py train --benchmark ml10 --total-steps 5000000
bash docker/dev.sh python scripts/s2_ch11_multitask.py eval --ckpt checkpoints/multitask_ml10.zip
bash docker/dev.sh python scripts/s2_ch11_multitask.py per-task-analysis
```

**Done when:**
- Single policy achieves >50% average success across 10 tasks
- You can identify which tasks benefit from sharing vs which interfere
- Per-task success rates documented (some will be higher than others)

---

### Chapter 12: Curriculum Learning

**Goal:** Learn hard tasks by starting easy and increasing difficulty.

**WHY this matters:**
Some tasks are too hard to learn from scratch. Curriculum learning provides a path: master easy versions first, gradually increase difficulty.

**The task:**
Progressive stacking (1 -> 2 -> 3 -> 4 blocks)

**What we build:**
- Manual curriculum (advance when success > threshold)
- Automatic curriculum (sample tasks proportional to learning progress)
- Reverse curriculum (start near goal, expand initial state distribution)

**Key concepts:**
- Curriculum design space (task difficulty, initial states, goals)
- Catastrophic forgetting during curriculum (getting worse at easy tasks)
- When curriculum helps vs hurts

**Steps:**
- [ ] Implement curriculum scheduler
- [ ] Train with manual curriculum (1 -> 2 -> 3 -> 4 blocks)
- [ ] Compare: curriculum vs training directly on hard task
- [ ] Implement automatic curriculum (learning-progress-based)

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch12_curriculum.py train --curriculum manual --total-steps 5000000
bash docker/dev.sh python scripts/s2_ch12_curriculum.py train --curriculum auto --total-steps 5000000
bash docker/dev.sh python scripts/s2_ch12_curriculum.py train --curriculum none --total-steps 5000000  # baseline
bash docker/dev.sh python scripts/s2_ch12_curriculum.py compare
```

**Done when:**
- Curriculum-trained agent stacks 4 blocks
- Non-curriculum baseline fails at 3+ blocks (or requires much more data)
- You understand when to use curriculum (sequential, hard-exploration tasks)

---

### Chapter 13: Adaptation and Fine-Tuning

**Goal:** Adapt a trained policy to distribution shift without retraining from scratch.

**WHY this matters:**
Deployment conditions will differ from training. Fine-tuning provides a middle ground between "train from scratch" and "hope it generalizes."

**The scenario:**
Train on blue blocks -> deploy on red blocks. Train on smooth table -> deploy on textured table.

**What we build:**
- Fine-tuning protocol (how many samples to adapt?)
- Forgetting analysis (does fine-tuning hurt original performance?)
- Brief exploration of techniques to reduce forgetting

**Key concepts:**
- Distribution shift is inevitable
- Few-shot adaptation vs full retraining
- When to fine-tune vs when to randomize during initial training

**Brief note on DAGGER:**
For small distribution shifts, sometimes a few human corrections are more valuable than thousands of RL samples. DAGGER (Dataset Aggregation) interleaves human demonstrations with policy execution. We mention this approach but do not implement a full pipeline -- the insight is that human guidance can be sample-efficient for fine-tuning.

**Steps:**
- [ ] Train policy on distribution A (blue blocks, smooth table)
- [ ] Evaluate on distribution B (red blocks, textured table) -- expect degradation
- [ ] Fine-tune on small samples from distribution B
- [ ] Measure: how many samples to recover performance?

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch13_adaptation.py train --distribution A --total-steps 1000000
bash docker/dev.sh python scripts/s2_ch13_adaptation.py eval --ckpt checkpoints/dist_a.zip --distribution B
bash docker/dev.sh python scripts/s2_ch13_adaptation.py finetune --ckpt checkpoints/dist_a.zip --distribution B --samples 1000
bash docker/dev.sh python scripts/s2_ch13_adaptation.py forgetting-analysis
```

**Done when:**
- Policy adapts to distribution B with <1000 fine-tuning samples
- Forgetting quantified (performance on A after fine-tuning on B)
- You have a practical recommendation for adaptation vs randomization

---

## Part V: Bridge to Reality (Chapters 14-16)
*"All simulation. But simulation that respects reality."*

### Chapter 14: Domain Randomization -- The Blunt Instrument

**Goal:** Train policies robust to simulation-reality gap through massive randomization.

**WHY this matters:**
If you randomize everything during training, the real world is just another sample from the distribution. This "brute force" approach is surprisingly effective.

**What we randomize:**
- **Visual:** lighting, colors, textures, camera position
- **Dynamics:** friction, mass, damping, actuator strength
- **Initial conditions:** object poses, robot starting configuration

**What we build:**
- Randomization wrappers for environments
- Uniform vs structured randomization
- Ablation: which randomizations matter most

**Key concepts:**
- The robustness-performance tradeoff (more randomization = worse average, better worst-case)
- Automatic Domain Randomization (ADR) -- expand randomization as policy improves
- Mental model: "reality is one point in randomization distribution"

**Steps:**
- [ ] Implement visual randomization wrapper
- [ ] Implement dynamics randomization wrapper
- [ ] Train with light, medium, heavy randomization
- [ ] Evaluate robustness under extreme perturbation

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch14_domain_rand.py train --rand-level none
bash docker/dev.sh python scripts/s2_ch14_domain_rand.py train --rand-level light
bash docker/dev.sh python scripts/s2_ch14_domain_rand.py train --rand-level heavy
bash docker/dev.sh python scripts/s2_ch14_domain_rand.py stress-test --ckpt checkpoints/rand_heavy.zip
```

**Done when:**
- Heavy-randomization policy maintains >60% success under extreme perturbations (2x mass, 0.5x friction, visual distractors)
- Light/no randomization policies fail under same perturbations
- You can recommend appropriate randomization levels for different deployment scenarios

---

### Chapter 15: System Identification -- The Precise Instrument

**Goal:** Make simulation match reality by identifying physical parameters.

**WHY this matters:**
What if we could make simulation accurate instead of robust to inaccuracy? System identification measures real parameters and plugs them into simulation.

**The approach:**
Measure real robot parameters (friction, damping, mass) from trajectory data -> update simulator -> train with accurate dynamics.

**What we explore:**
- Simple least-squares system ID
- The "real2sim" pipeline
- Identifiability -- can we determine parameters from data?

**Key concepts:**
- System ID vs domain randomization (complementary, not competing)
- Which parameters are identifiable from trajectories
- The unknown unknowns -- parameters we forgot to model

**Note:** Without a real robot, we simulate this: treat one MuJoCo config as "real" and another as "sim", then identify the "real" parameters from trajectory data.

**Steps:**
- [ ] Implement trajectory collection in "real" environment
- [ ] Implement least-squares parameter estimation
- [ ] Update "sim" environment with identified parameters
- [ ] Compare: identified sim vs default sim vs real

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch15_sysid.py collect-real-data --n-trajectories 50
bash docker/dev.sh python scripts/s2_ch15_sysid.py identify --data results/real_trajectories.json
bash docker/dev.sh python scripts/s2_ch15_sysid.py compare-trajectories
bash docker/dev.sh python scripts/s2_ch15_sysid.py train-with-identified --total-steps 1000000
```

**Done when:**
- Identified simulation produces trajectories closer to "real" than default simulation
- You understand which parameters are identifiable and which require randomization
- Practical guidelines documented for when to use ID vs randomization

---

### Chapter 16: The Reality-Ready Policy

**Goal:** Combine everything into a policy designed for real deployment.

**WHY this matters:**
This chapter is the synthesis. A policy that passes all our simulated stress tests is our best candidate for real robot deployment.

**The recipe:**
1. Randomize what you cannot measure
2. Identify what you can measure
3. Use visual augmentation
4. Test under perturbation before claiming "ready"

**What we build:**
- Final "reality-ready" visual manipulation policy
- Comprehensive perturbation test suite
- Transfer confidence score -- quantifying expected sim-to-real success

**Key concepts:**
- The pre-deployment checklist
- Common failure modes and diagnosis
- Managing expectations -- 70% sim success does not mean 70% real success

**Steps:**
- [ ] Train policy with combined randomization + identification
- [ ] Run full perturbation test suite
- [ ] Generate "transfer confidence" report
- [ ] Document policy strengths and expected failure modes

**Commands:**
```bash
bash docker/dev.sh python scripts/s2_ch16_reality_ready.py train --total-steps 3000000
bash docker/dev.sh python scripts/s2_ch16_reality_ready.py test-suite --ckpt checkpoints/reality_ready.zip
bash docker/dev.sh python scripts/s2_ch16_reality_ready.py confidence-report --ckpt checkpoints/reality_ready.zip
```

**Done when:**
- Policy passes all simulated stress tests
- Transfer confidence report generated
- You have a concrete artifact ready for hardware deployment (if desired)

---

## Bonus Part: The Real Thing (Chapters B1-B3)
*"For those who want to touch grass (or robots)"*

> **These chapters are entirely optional.** They require purchasing hardware. The main series (Chapters 0-16) is complete without them. If you do not want to buy a robot, stop here -- you have learned visual RL, multi-task learning, and sim-to-real principles, all in simulation.

### Chapter B1: Choosing and Setting Up Your Robot

**Goal:** Select, purchase, and set up hardware for real deployment.

**Budget tiers:**

| Budget | Platform | Best For |
|--------|----------|----------|
| ~$150 | Donkey Car kit / WaveShare JetRacer | Driving (Part II transfer) |
| ~$200 | Low-cost servo arm (SO-ARM100, LewanSoul) | Simple reaching |
| ~$400 | Trossen WidowX 200 | Full manipulation |
| ~$100 | DIY: Raspberry Pi + servos + 3D printed arm | Learning, limited capability |

**What we cover:**
- Evaluating robots for RL (action space, sensors, repeatability)
- ROS2 basics -- just enough to send commands and read sensors
- Camera calibration crash course
- Safety mindset -- the robot WILL do something stupid

**Steps:**
- [ ] Choose and purchase robot based on budget and interests
- [ ] Assemble hardware
- [ ] Verify basic command/control (move joints, read sensors)
- [ ] Calibrate camera
- [ ] Run random actions without breaking anything

**Done when:**
- Robot moves on command
- Camera feed accessible from Python
- No smoke, no broken servos, no injuries

---

### Chapter B2: Your First Real Transfer

**Goal:** Transfer a simulation-trained policy to real hardware.

**The task (choose one):**
- **Driving:** Visual lane following on tape track (if you bought car)
- **Manipulation:** Reaching to a colored target (if you bought arm)

**What we cover:**
- Action space calibration (sim actions -> real actuator commands)
- Observation alignment (sim camera -> real camera)
- Systematic debugging (because first attempt WILL fail)
- Recording and analyzing failure cases

**Key insight:** The first transfer attempt will fail. The value is in the debugging.

**Steps:**
- [ ] Implement real robot interface matching sim action/observation space
- [ ] Attempt direct policy transfer
- [ ] Document failures (video + logs)
- [ ] Iterate: adjust calibration, re-train with better randomization, retry

**Done when:**
- A policy trained 100% in simulation accomplishes the task on real hardware
- Success rate may be lower than sim -- that is expected
- Video evidence of at least one successful episode

---

### Chapter B3: The Party Trick

**Goal:** Accomplish something impressive enough to show friends.

**WHY this matters:**
You have earned the right to show off. Choose a capstone that makes people say "wait, the robot learned that?"

**Choose your capstone:**

| Task | Hardware Needed | Difficulty | Impress Factor |
|------|-----------------|------------|----------------|
| Autonomous lap around room | Donkey Car + tape track | Medium | High |
| Pick up specific object | Arm + camera | Hard | High |
| Sort objects by color | Arm + camera | Hard | Very High |
| Open a bottle (twist-off) | Arm + gripper | Very Hard | Maximum |
| Light switch toggle | Arm | Medium | Medium |

**What we cover:**
- Task decomposition for your chosen capstone
- Iterative improvement (train -> deploy -> analyze -> repeat)
- Recording your demo video
- What to tell people ("I trained this using reinforcement learning...")

**Steps:**
- [ ] Choose capstone task based on hardware and ambition
- [ ] Decompose into subtasks if needed
- [ ] Train/fine-tune for chosen task
- [ ] Record demo video with 3/5 successful attempts
- [ ] Celebrate

**Done when:**
- Video evidence of 3/5 successful task completions
- At least one witness (human, not just camera)
- You can explain what you did without saying "it just works"

---

## Series 2 Summary

| Part | Chapters | Theme | Capstone |
|------|----------|-------|----------|
| I | 0-3 | Visual RL | Visual goal-conditioned FetchPush |
| II | 4-6 | Speed | Racing agent + walking creature |
| III | 7-10 | Manipulation | Tool use |
| IV | 11-13 | Generalization | 10-task multi-task policy |
| V | 14-16 | Sim-to-Real | Reality-ready policy |
| Bonus | B1-B3 | Hardware | Party trick demo |

**Total:** 17 core chapters + 3 optional bonus = 20 chapters

---

## Reproducibility and Artifacts

Series 2 follows Series 1 conventions:

**Checkpoints:** `checkpoints/s2_<algo>_<task>_seed<N>.zip` + `.meta.json`

**Evaluation reports:** `results/s2_chNN_<topic>_eval.json`

**TensorBoard logs:** `runs/s2/<chapter>/<task>/seed<N>/`

**Videos:** `videos/s2_<chapter>_<task>.mp4`

**Seed discipline:** 3-5 seeds per configuration, report mean +/- std

---

## What Comes Next?

After Series 2, possible directions include:
- Model-based RL (Dreamer, MBPO) for sample efficiency
- Language-conditioned policies (instruction following)
- Sim-to-real at scale (domain adaptation, real-world fine-tuning)
- Multi-robot coordination
- Deformable object manipulation (cloth, rope -- frontier research)

But first, finish Series 2. One step at a time.

---

## References

**Visual RL:**
- Yarats et al. (2021). "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels" (DrQ)
- Laskin et al. (2020). "CURL: Contrastive Unsupervised Representations for Reinforcement Learning"

**Sim-to-Real:**
- Tobin et al. (2017). "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- OpenAI (2019). "Solving Rubik's Cube with a Robot Hand" (extreme domain randomization)

**Multi-Task RL:**
- Yu et al. (2020). "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"

**Locomotion:**
- Tassa et al. (2018). "DeepMind Control Suite" (DMControl)

**Environments:**
- [Robosuite](https://robosuite.ai/)
- [MetaWorld](https://meta-world.github.io/)
- [DMControl](https://github.com/deepmind/dm_control)
- [Donkey Car](https://www.donkeycar.com/)
