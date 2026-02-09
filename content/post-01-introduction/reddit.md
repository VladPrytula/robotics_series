# Reddit Post

## For r/reinforcementlearning

---

**Title:** [P] Open-source curriculum: Goal-conditioned RL for robotic manipulation (SAC+HER, Fetch environments)

**Body:**

I've been building an open-source curriculum for learning goal-conditioned RL through robotic manipulation. Sharing in case it's useful to others.

**What it is:**

A 10-week course using Gymnasium-Robotics Fetch environments (Reach, Push, PickAndPlace). The method is SAC + HER, derived from problem constraints rather than presented as a recipe.

**Why I built it:**

Most tutorials show you *what* to type, not *why* it works. When I started with RL, I spent weeks debugging failures I couldn't understand because I didn't have the mental model for what was happening.

This curriculum tries to fix that by:

- Formulating the problem mathematically before introducing algorithms
- Explaining *why* SAC+HER follows from the problem structure (continuous actions → actor-critic, sparse rewards → off-policy, goal conditioning → HER)
- Verifying everything with multi-seed experiments
- Using Docker for reproducibility

**What's included:**

- Chapters 0-1: Environment setup, Fetch anatomy, reward verification
- Chapters 2-10: PPO baseline → SAC → HER → harder tasks (in progress)
- Docker tooling for DGX/GPU clusters
- CLI tools for training and evaluation

**Current status:**

Chapters 0-1 are complete and tested. Chapters 2-10 are being written incrementally. The core training/eval infrastructure works.

**Links:**

- GitHub: https://github.com/VladPrytula/robotics_series
- Docs: https://vladprytula.github.io/robotics_series/

Happy to answer questions or hear feedback. Particularly interested in what trips people up when learning goal-conditioned RL--want to make sure the curriculum addresses real pain points.

---

## For r/MachineLearning

---

**Title:** [P] Teaching RL through robotic manipulation: an open-source curriculum emphasizing problem formulation

**Body:**

I've been frustrated by RL tutorials that show you what to type but not why it works. Built an open-source curriculum that takes a different approach.

**The philosophy:**

Before asking "how do I train a policy?", ask:
- Does a solution exist in my hypothesis class?
- What mathematical properties must the environment have for my algorithm to work?
- Will my results be reproducible across seeds?

**The method:**

SAC + HER on Gymnasium-Robotics Fetch tasks. The choice isn't arbitrary--it's derived from constraints:

| Constraint | Implication | Solution |
|------------|-------------|----------|
| Continuous actions | Can't do argmax | Actor-critic |
| Sparse rewards | Need sample reuse | Off-policy |
| Goal conditioning | Need relabeling | HER |

**Current state:**

- Chapters 0-1 complete (env setup, interface verification)
- Chapters 2-10 in progress (training, HER, harder tasks)
- Docker-first, multi-seed experiments, everything version-controlled

GitHub: https://github.com/VladPrytula/robotics_series

Feedback welcome, especially on what's confusing or missing.

---

## For r/robotics

---

**Title:** Open-source RL curriculum using simulated Fetch robot (MuJoCo)

**Body:**

Built an open-source curriculum for learning RL through robotic manipulation. Uses the Fetch robot in MuJoCo simulation.

**Tasks covered:**
- FetchReach (move end-effector to position)
- FetchPush (push object to target)
- FetchPickAndPlace (pick up and place object)

**Method:** SAC + HER (Soft Actor-Critic with Hindsight Experience Replay)

**Why Fetch?**

The simulated Fetch matches the kinematics of the real Fetch Mobile Manipulator (7-DOF arm, parallel-jaw gripper). Policies trained in simulation can potentially transfer to hardware, though sim-to-real isn't covered in this curriculum.

**What makes it different:**

- Explains the environment interface in detail (observation structure, action semantics, reward computation)
- Derives algorithm choices from problem structure
- Docker-based for reproducibility
- Multi-seed experiments

GitHub: https://github.com/VladPrytula/robotics_series

Happy to answer questions about the Fetch environments or the approach.

---

## Posting Notes

- Post on weekday mornings (US time) for best visibility
- Don't post to multiple subreddits on the same day (looks spammy)
- Schedule: r/reinforcementlearning Monday, r/MachineLearning Wednesday, r/robotics Friday
- Engage genuinely with comments--this builds reputation
- If post gets traction, cross-post to r/learnmachinelearning the following week
