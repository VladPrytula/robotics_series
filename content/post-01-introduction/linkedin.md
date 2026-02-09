# LinkedIn Post

---

RL has a reproducibility crisis.

Henderson et al. (2018) showed that published results often fail to replicate--even with the same code.

The problem isn't bad researchers. It's that RL algorithms are sensitive to details tutorials don't mention: reward scaling, observation normalization, random seeds.

When training fails, most people ask: "Why doesn't my code work?"

Better question: "Under what conditions does this algorithm succeed?"

---

The French mathematical tradition asks three questions before solving anything:

→ Does a solution exist?
→ Is it unique?
→ Does it depend continuously on the data?

Applied to RL, this becomes:

→ Can a neural network policy solve this task?
→ Are there multiple qualitatively different solutions?
→ Will different random seeds give similar results?

Example: sparse rewards + random exploration.

The probability of randomly reaching a goal in high-dimensional space is essentially zero. No success = no gradient signal = no learning.

This isn't a hyperparameter problem. The problem formulation itself makes learning impossible.

---

I'm building an open-source curriculum that teaches RL differently:

✓ Problem formulation BEFORE solution
✓ Real robotics simulation (Fetch arm, MuJoCo physics)
✓ Derive algorithms, don't just copy-paste them
✓ Verify with multi-seed experiments, not single lucky runs

It's not fast. It's 10 weeks. You can't skip chapters.

But when something breaks, you'll know WHY.

---

First two chapters are live:
🔗 github.com/VladPrytula/robotics_series

Who else is tired of tutorials that work until they don't?

#reinforcementlearning #robotics #machinelearning #opensource #curriculum

---

## Posting Notes

- Upload the robot GIF as native video (better engagement)
- Post Tuesday-Thursday, 8-10am local time
- Respond to every comment in first 2 hours
- Don't edit post after publishing (kills reach)
