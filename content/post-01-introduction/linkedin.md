# LinkedIn Post

---

RL has a reproducibility crisis.

Henderson et al. (2018) showed that published results often fail to replicate--even with the same code.

The problem isn't bad researchers. It's that RL algorithms are sensitive to details tutorials don't mention: reward scaling, observation normalization, random seeds.

When training fails, most people ask: "Why doesn't my code work?"

Better question: "Under what conditions does this algorithm succeed?"

---

At the zooplus SE AI Hub, we think a lot about what makes ML systems reliable at scale. E-commerce logistics--millions of SKUs, dynamic demand, real-time decisions--doesn't forgive brittle models.

That mindset applies to robotics too.

There's a useful habit from applied math and physics: before solving a problem, ask whether it's well-posed.

- Does a solution exist?
- Is it unique?
- Does it depend continuously on the input?

For RL:

- Can a neural network policy actually solve this task?
- Are there multiple qualitatively different solutions?
- Will different random seeds give similar results?

Example: sparse rewards + random exploration.

The probability of randomly reaching a goal in high-dimensional space is essentially zero. No success = no gradient signal = no learning.

This isn't a hyperparameter problem. The problem formulation itself makes learning impossible.

---

I'm working on an open-source curriculum that teaches RL with this lens:

- Problem formulation before solution
- Real robotics simulation (Fetch arm--originally designed for warehouse automation)
- Derive algorithms from constraints, don't just copy-paste
- Multi-seed experiments, not single lucky runs

It's slow. 10 weeks. Chapters build on each other.

But when something breaks, you'll have the mental model to understand why.

---

First two chapters are live:
github.com/VladPrytula/robotics_series

Feedback welcome--especially on what's confusing or missing.

#reinforcementlearning #robotics #machinelearning #opensource #zooplus

---

## Posting Notes

- Upload the robot GIF as native video (better engagement)
- Post Tuesday-Thursday, 8-10am local time
- Respond to every comment in first 2 hours
- Don't edit post after publishing (kills reach)
- Tag zooplus SE company page if possible
