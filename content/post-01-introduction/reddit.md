# Reddit Post

**Title:** [P] Open-source curriculum: Goal-conditioned RL for robotic manipulation (SAC+HER, Fetch environments)

---

I've been building an open-source curriculum for learning goal-conditioned RL through robotic manipulation. Sharing in case it's useful to others.

**What it is:**

A 10-week course using Gymnasium-Robotics Fetch environments (Reach, Push, PickAndPlace). The method is SAC + HER, derived from problem constraints rather than presented as a recipe.

**Why I built it:**

Most tutorials show you *what* to type, not *why* it works. When I started with RL, I spent weeks debugging failures I couldn't understand because I didn't have the mental model for what was happening.

This curriculum tries to fix that by:

- Formulating the problem mathematically before introducing algorithms
- Explaining *why* SAC+HER follows from the problem structure (continuous actions -> actor-critic, sparse rewards -> off-policy, goal conditioning -> HER)
- Verifying everything with multi-seed experiments
- Using Docker for reproducibility

**What's included:**

- Chapters 0-1: Environment setup, Fetch anatomy, reward verification
- Chapters 2-10: PPO baseline -> SAC -> HER -> harder tasks (in progress)
- Docker tooling for DGX/GPU clusters
- CLI tools for training and evaluation

**Current status:**

Chapters 0-1 are complete and tested. Chapters 2-10 are being written incrementally. The core training/eval infrastructure works.

**Links:**

- GitHub: https://github.com/VladPrytula/robotics_series
- Docs: https://vladprytula.github.io/robotics_series/

Happy to answer questions or hear feedback. Particularly interested in what trips people up when learning goal-conditioned RL--want to make sure the curriculum addresses real pain points.
