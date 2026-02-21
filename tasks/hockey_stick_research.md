# The Hockey-Stick Learning Curve in HER: Research Notes

**Date:** 2026-02-19
**Status:** Research complete, ready for book integration
**Target location:** Ch9 section 9.6 (sidebar or boxed explanation)

---

## 1. The Empirical Observation

Push state + HER (FetchPush-v4, SAC, ent_coef=0.05, n_sampled_goal=4, future):

| Steps | Success Rate | Phase |
|------:|------------:|-------|
| 0-1.2M | 3-10% | Flat |
| 1.28M | 14% | Inflection begins |
| 1.36M | 19% | Accelerating |
| 1.40M | 32% | Positive feedback active |
| 1.52M | 52% | Steep climb |
| 1.64M | 77% | |
| 1.84M | 90% | Near-saturation |
| 1.88M | 97% | Peak |
| 2.0M | 89% | Slight regression from peak |

The entire jump from 10% to 90% occurred in ~600K steps (~30% of total budget).
The same config at 1M steps showed ~2% success -- indistinguishable from random.

---

## 2. Three Mechanisms (Our Explanation)

We explain the hockey stick as three interacting mechanisms. Each has
independent formal backing in the literature, though no single paper
assembles the complete picture.

### 2.1 Value Propagation Bottleneck (the flat phase)

**Intuition:** In sparse-reward Push, the critic starts knowing nothing --
Q(s, a | g) is approximately -(T - t) everywhere (maximum penalty). HER
seeds the critic with relabeled near-object successes: transitions where the
achieved goal happened to be close to the object's final position. The
Bellman backup propagates this value information one step outward per update:

$$Q(s_t, a_t \mid g) \leftarrow r_t + \gamma \max_{a'} Q(s_{t+1}, a' \mid g)$$

If the critic already knows Q-values for goals within distance d of the
object's starting position, one backup extends that knowledge to distance
d + delta. This is a **discrete diffusion process** through goal space.

With n_sampled_goal=4, each real transition generates 4 relabeled
transitions with different goals sampled from the trajectory's future. This
effectively seeds the diffusion from 4 different points per transition,
speeding the wavefront expansion.

**Formal backing:** Laidlaw, Zhu, Russell & Dragan (2024) introduce the
**effective horizon** k* -- the minimum number of value iteration steps
needed before greedy actions become near-optimal. They prove sample
complexity is exponential in k*:

> Sample complexity ~ C^{k*} * |function class complexity|

For sparse-reward Push, k* is proportional to the number of Bellman backup
steps needed to propagate reward from goal states to the initial state
distribution. HER reduces k* by injecting synthetic nearby goals (so the
propagation starts closer to the current state), but does not eliminate it.

The exponential dependence on k* creates threshold behavior: below a
critical training budget, essentially zero learning; above it, rapid
improvement. This is the formal version of our "wavefront" argument.

**Citation:** Laidlaw, C., Zhu, B., Russell, S., & Dragan, A. (2024).
"The Effective Horizon Explains Deep RL Performance in Stochastic
Environments." ICLR 2024 (Spotlight). arXiv:2312.08369.

**Supporting evidence from backward-update literature:** Lee, Choi & Chung
(2018) show that standard TD learning propagates value one step per update.
In a 50-step episode with terminal reward, it takes O(50) updates for the
reward to propagate to the initial state. Their Episodic Backward Update
(EBU) propagates through the entire episode in one pass, achieving 10-20x
sample efficiency improvement -- directly demonstrating that one-step
propagation is a primary bottleneck.

**Citation:** Lee, S. Y., Choi, S., & Chung, S. Y. (2018). "Sample-Efficient
Deep Reinforcement Learning via Episodic Backward Update." arXiv:1805.12375.

### 2.2 Geometric Phase Transition (the inflection)

**Intuition:** Test goals are sampled uniformly over the table surface (a 2D
region). The agent's "competence region" -- the set of goals it can
reliably push the object to -- starts as a small region near the object's
initial position and grows as the value wavefront expands.

The fraction of test goals the agent can reach is:

$$\text{success\_rate}(t) \approx \frac{\text{Area}(\mathcal{R}(t) \cap \mathcal{G})}{\text{Area}(\mathcal{G})}$$

where R(t) is the competence region and G is the goal sampling region.

If we simplify R(t) as a ball of radius r(t), this gives pi * r^2 / L^2 in
2D -- quadratic in the competence radius. When r is small relative to L,
the overlap is negligible. When r crosses a critical threshold r*, the
overlap grows rapidly.

**Why the simplification is OK for pedagogy but not for theory:** The real
competence region is not a nice ball -- it is a jagged, irregularly-shaped
set determined by exploration trajectories and function approximation
generalization. The formal version of this argument does not need the
geometric simplification.

**Formal backing:** Wang & Isola (2022) prove that the optimal
goal-conditioned Q-function has a **quasimetric structure** -- it satisfies
a form of the triangle inequality (d(a,c) <= d(a,b) + d(b,c)) but not
symmetry. Once the neural network learns enough correct local distances, the
triangle inequality *forces* global consistency -- analogous to percolation
in statistical physics. Before the critical density of correct local
distances, the policy is locally sensible but globally suboptimal. After
percolation, it can plan globally optimal paths.

This is more rigorous than our pi*r^2/L^2 argument because it does not
assume a nice geometric shape for the competence region. It instead argues
from the structure of the value function itself.

**Citation:** Wang, T. & Isola, P. (2022). "On the Learning and Learnability
of Quasimetrics." ICLR 2022. arXiv:2206.15478.

**Additional support:** Eysenbach, Zhang, Salakhutdinov & Levine (2022) show
that contrastive representation learning produces representations whose
inner products equal goal-conditioned value functions. The phase transition
corresponds to the representation space "crystallizing" into a globally
consistent metric structure.

**Citation:** Eysenbach, B., Zhang, T., Salakhutdinov, R., & Levine, S.
(2022). "Contrastive Learning as Goal-Conditioned Reinforcement Learning."
NeurIPS 2022. arXiv:2206.07568.

### 2.3 Positive Feedback Loop (the steep phase)

**Intuition:** Once some real goals are reached, a self-reinforcing cycle
activates:

1. Successful episodes provide real (non-relabeled) reward signal
2. Real reward is higher-quality than HER-relabeled reward (no distribution
   mismatch from goal substitution)
3. Better Q-values drive the policy to explore further from the current
   competence boundary
4. More distant successes expand the competence region
5. Expanded region covers more test goals -> more real reward -> repeat

This converts the linear wavefront expansion into super-linear growth.
The steep part of the hockey stick is this positive feedback regime.

**Formal backing:** Huang et al. (2025) analyze learning dynamics using
Fourier analysis on difficulty spectra. With a **discontinuous difficulty
spectrum** (gap between easy and hard problems), learning exhibits
"grokking-type phase transitions, producing prolonged plateaus before
progress recurs." With a **smooth difficulty spectrum**, a "relay mechanism"
enables sustained improvement where gradient signals from simpler problems
progressively unlock harder ones.

HER creates a smooth difficulty spectrum by relabeling with achievable
goals, but the relay from "nearby goals" to "actual test goals" still
requires crossing a gap. The positive feedback loop is the relay crossing
that gap: once one actual test goal is reached, the gradient signal from
that success unlocks nearby test goals, which unlock further test goals.

**Citation:** Huang, Y., Wen, Z., Chi, Y., Wei, Y., Singh, A., Liang, Y.,
& Chen, Y. (2025). "On the Learning Dynamics of RLVR at the Edge of
Competence." arXiv:2602.14872.

**Additional support:** Ren, Dong, Zhou, Liu & Peng (2019) show that HER
"exploits previous replays by constructing imaginary goals in a simple
heuristic way, acting like an implicit curriculum" -- but this curriculum is
limited to states the agent has already visited. The positive feedback loop
is the mechanism by which the implicit curriculum extends beyond its initial
reach.

**Citation:** Ren, Z., Dong, K., Zhou, Y., Liu, Q., & Peng, J. (2019).
"Exploration via Hindsight Goal Generation." NeurIPS 2019. arXiv:1906.04279.

---

## 3. The Nucleation Metaphor

The three mechanisms together resemble **nucleation** in condensed matter
physics. A supercooled liquid contains many microscopic crystal seeds, each
growing slowly. Below the critical nucleus size, surface energy dominates and
seeds dissolve. Once a seed reaches critical size, bulk energy dominates and
growth becomes self-sustaining. The transition is sharp -- the system appears
unchanged until suddenly crystallization races through the material.

In our setting:
- Supercooled liquid = the random policy that cannot reach goals
- Microscopic seeds = HER-relabeled near-object value knowledge
- Surface energy penalty = the gap between relabeled goals and test goals
- Critical nucleus = competence radius reaching the test goal distribution
- Crystallization = the positive feedback loop driving rapid improvement

This metaphor is pedagogically valuable because it explains both the long
flat phase (seeds are growing below detection threshold) and the sharp
transition (positive feedback once critical size is reached).

**Note:** This is a metaphor, not a formal mapping. No published paper
rigorously applies nucleation theory to RL learning dynamics. We should
present it as intuition and be honest about that.

---

## 4. What the Literature Does NOT Provide

We should be honest in the book about the gaps:

1. **No closed-form inflection point.** No paper gives t_crit as a function
   of environment parameters, HER hyperparameters, and network architecture.

2. **No phase transition theorem for HER.** The percolation analogy is
   suggestive but has not been rigorously formalized for goal-conditioned RL
   with hindsight relabeling. Tong (2024, arXiv:2511.20503) makes a
   percolation connection for generative models but the RL extension is
   underdeveloped.

3. **No HER-specific convergence rates.** Existing bounds are for general
   off-policy Q-learning (Shah & Xie 2018, Li et al. 2022). Applying them
   to HER requires accounting for the non-stationary goal distribution in
   the replay buffer, which no existing theorem handles.

4. **The positive feedback loop is informal.** Everyone in the field
   understands it intuitively, but it has not been formalized as a dynamical
   system with provable convergence acceleration.

**How to handle this in the book:** "The full picture is assembled from three
independent theoretical results (effective horizon, quasimetric learning,
difficulty spectrum analysis), not derived from a single theorem. We find this
honest framing more useful than pretending the theory is complete."

---

## 5. What Goes in the Book

### Section 9.6: "Why the learning curve looks like this" (sidebar/box)

**Target length:** ~400-500 words + 1 figure (learning curve with annotated
phases) + 1 table (the empirical data above)

**Structure:**

1. **Show the data.** The learning curve table from our experiment. Annotate
   the three phases directly on the figure.

2. **Three mechanisms, one paragraph each:**
   - Value propagation: Bellman backups diffuse value through goal space.
     HER seeds the process. Speed is limited by the effective horizon
     (Laidlaw et al. 2024).
   - Geometric threshold: the competence region must overlap with the test
     goal distribution. In 2D, overlap scales quadratically with the
     competence radius -- creating a sharp threshold between "almost none"
     and "many" goals reachable.
   - Positive feedback: once real goals are reached, the gradient signal
     quality jumps, accelerating further expansion. Huang et al. (2025) call
     this the "relay mechanism."

3. **The nucleation metaphor.** One paragraph connecting the three mechanisms
   to a physical analogy readers may find intuitive. Explicitly mark this
   as metaphor, not formal theory.

4. **The practical lesson.** "If your HER learning curve is flat at 1M steps,
   it may not be broken -- the value wavefront may simply not have reached
   the test goal distribution yet. Our experiment went from 2% at 1M to 89%
   at 2M. The diagnostic: check whether relabeled-goal success (computed
   from the HER buffer) is improving even when test-goal success is flat.
   If relabeled success is climbing, the wavefront is growing -- give it
   more time."

5. **Further reading.** Point to Laidlaw et al. (2024) for the effective
   horizon framework, Wang & Isola (2022) for quasimetric structure, and
   Eysenbach et al. (2022) for the contrastive/metric learning perspective.

### Concept registry additions (CLAUDE.md)

Already added:
- value wavefront (Bellman diffusion through goal space)
- hockey-stick learning curve (geometric phase transition + positive feedback)
- critical competence radius

Should also add:
- effective horizon k* (Laidlaw et al. 2024)

### Canonical references additions (CLAUDE.md)

| Topic | Reference | Specific Section |
|-------|-----------|-----------------|
| Effective horizon | Laidlaw et al. (2024) | arXiv:2312.08369 |
| Quasimetric Q-functions | Wang & Isola (2022) | arXiv:2206.15478 |
| Difficulty spectrum dynamics | Huang et al. (2025) | arXiv:2602.14872 |

---

## 6. Experimental Data for the Book

### Push state (no HER) -- the diagnostic
- Env: FetchPushDense-v4, SAC, auto ent_coef, 1M steps
- Final success: 8%
- Entropy collapsed to 0.00013 by ~200K steps (structural failure)
- Reward plateau: -8.5 (arm near goal, object unmoved)
- Diagnosis: "deceptively dense" -- reward gradient toward goal position,
  not through object

### Push state + HER -- the hockey stick
- Env: FetchPush-v4, SAC+HER, ent_coef=0.05, n_sampled_goal=4, future, 2M steps
- Final success: 89% (peak 97% at 1.88M)
- Hockey-stick inflection: ~1.28M steps
- Flat phase: 0-1.2M steps (3-10%)
- Steep phase: 1.28M-1.84M steps (14% -> 90%)
- Training time: 3760s (532 steps/sec)

### Key comparison
- Same config at 1M steps: ~2% success (pre-inflection)
- Same config at 2M steps: ~89% success (post-inflection)
- The only difference: whether training crosses the phase transition
