# Part 6: Pushing the Boundaries (The Bleeding Edge)

## Date: 2026-02-25
## Status: Outline -- to be developed after Ch9-10 are complete

This takes the book from "a solid textbook on standard RL" to "a modern manual
that bridges into state-of-the-art research paradigms."

---

## Chapter 11: Empowerment and Unsupervised Skill Discovery

**The Problem:** HER is a magical fix for sparse rewards, but it has a fatal
flaw: it only works if you accidentally succeed or interact meaningfully. If
you never touch the block, HER relabeling is useless.

**The Bleeding-Edge Solution:** Intrinsic motivation and empowerment. We teach
the robot to "play" without any external reward before we ever give it a task.

### Sections

- **11.1 The limits of HER and random exploration:** Why complex manipulation
  (like stacking or tool use) breaks standard exploration.

- **11.2 Intrinsic Curiosity (ICM):** Rewarding the agent for prediction errors
  (going where the forward dynamics model is uncertain).

- **11.3 Empowerment & DIAYN (Diversity is All You Need):** Learning a
  repertoire of distinct skills without a reward function by maximizing mutual
  information between skills and states.

- **11.4 Zero-Shot Downstream Tasks:** How a pre-trained "empowered" policy can
  solve a sparse manipulation task in a fraction of the time by composing
  learned behaviors.

### Script idea
`scripts/ch11_diayn_push.py` -- pre-trains SAC on Push without any goal or
reward, just an intrinsic reward module. Visualize the distinct "skills" it
discovers. Then fine-tune on sparse Push and show the speedup.

---

## Chapter 12: Automated Curriculum and Asymmetric Self-Play

**The Problem:** In Chapter 6 (Pick and Place), we used a manually designed
curriculum (e.g., starting the block in the air, then lowering it).
Hand-designing these schedules is tedious and brittle.

**The Bleeding-Edge Solution:** Algorithms that automatically generate their
own syllabus based on the agent's current competence.

### Sections

- **12.1 Reverse Curriculum Generation:** Instead of starting at the start
  state and hoping to hit the goal, start exactly at the goal state, take
  random actions backwards, and train the agent to recover. Slowly expand
  the radius.

- **12.2 Prioritized Level Replay (PLR):** Sampling starting configurations
  based on where the agent has the highest TD error (training exactly where
  the agent is currently confused).

- **12.3 Asymmetric Self-Play:** The "Alice and Bob" paradigm. A "Teacher"
  policy is rewarded for finding initial states that are just hard enough for
  the "Student" policy to solve, creating a continuous, automated curriculum.

---

## Chapter 13: (Bonus) World Models and Latent Imagination

Since we just solved Push from pixels (Ch9), we know exactly how painful the
rendering bottleneck and sample inefficiency are.

### Sections

- **13.1 The Rendering Bottleneck:** Why model-free RL on pixels is
  fundamentally bounded by FPS.

- **13.2 Learning a World Model:** Training a VAE and a recurrent dynamics
  model to predict the next latent state (Dreamer/DayDreamer architecture).

- **13.3 Dreaming of Manipulation:** Training the SAC actor-critic entirely
  inside the latent model's hallucinated rollouts without hitting the
  environment.

- **13.4 Real-World Implications:** Why this is currently the bleeding edge
  for sample-efficient robot learning.

---

## Integration Notes

- **Start with tutorials:** ch11_empowerment.md, ch12_curriculum.md,
  ch13_world_models.md
- **Publisher pitch:** Takes the book from standard RL textbook to modern
  research manual
- **Prerequisite chain:** Ch11 builds on Ch4 (HER limits), Ch9 (pixel RL).
  Ch12 builds on Ch5 (manual curriculum). Ch13 builds on Ch9 (pixel
  bottleneck).
- **Development order:** After Ch9-10 pixel tutorials are complete and
  validated.
