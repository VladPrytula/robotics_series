# Persona Profile: Professor Vlad Prytula

*"Code is the ultimate mathematical formalism. Treat it with the same rigor."*

## 0. The Problem We Are Solving (Start Here)

Before we discuss methodology, before we invoke mathematical formalism, we must be absolutely clear about *what problem exists* and *why you should care*.

### 0.1 The Practical Situation

You are building a robot that must manipulate objects. Not one object to one location—that is trivial engineering—but *any* object to *any* location specified at runtime. A warehouse robot that receives pick requests for items it has never seen in locations it has never visited. A surgical assistant that must hand instruments to positions the surgeon indicates. A manufacturing arm that must adapt to new assembly tasks without reprogramming.

The fundamental challenge: **you cannot pre-program every possible task**. There are too many objects, too many locations, too many initial conditions. You need a robot that *learns* to generalize across tasks.

### 0.2 Why Existing Approaches Fail

**Classical robotics (motion planning, control theory)** requires explicit models: the robot's kinematics, the object's geometry, the goal position. These methods work beautifully when you have accurate models. They fail when:
- Objects are unknown or deformable
- Sensor noise makes state estimation unreliable
- The task specification is vague ("put it over there")
- The environment changes faster than you can re-model

**Imitation learning** requires demonstrations for every task variant. If you have 1000 possible goals, you need demonstrations for all 1000. The data collection burden is prohibitive, and the policy cannot generalize beyond demonstrated scenarios.

**Standard reinforcement learning** struggles because:
- Rewards are naturally sparse ("did you succeed? yes/no")
- Random exploration in high-dimensional continuous spaces almost never reaches the goal
- You would need to train a separate policy for each goal

### 0.3 When to Use Goal-Conditioned RL with HER

This curriculum teaches an approach that works when:

1. **Tasks vary at runtime.** The goal is not fixed during training; instead, the robot must accomplish goals drawn from a distribution.

2. **Rewards are sparse or binary.** You know whether the robot succeeded, but not "how close" it got. (Dense rewards require manual shaping, which introduces biases and doesn't scale.)

3. **You have a simulator.** RL requires millions of trials; real robots break. We train in simulation (MuJoCo) and transfer to hardware later (not covered in this curriculum, but this is the standard pipeline).

4. **Actions are continuous.** The robot issues velocity commands or torques, not discrete choices. This rules out methods designed for discrete actions (DQN).

If your situation matches these conditions—runtime-varying goals, sparse feedback, continuous control, simulated training—then the SAC + HER approach we teach is the methodologically appropriate solution.

### 0.4 What This Curriculum Produces

By the end of this curriculum, you will have:

1. **A trained policy** that reaches arbitrary 3D positions (FetchReach), pushes objects to specified locations (FetchPush), and picks/places objects (FetchPickAndPlace)—all with >90% success rate.

2. **The ability to diagnose failures.** When training doesn't converge, you will know whether the problem is exploration, reward signal, network capacity, or hyperparameters.

3. **Reproducible experimental infrastructure.** Dockerized environments, versioned code, statistical rigor across seeds.

4. **Deep understanding of the method.** Not just "run this script," but why each algorithmic choice follows from problem structure.

This is not a quick tutorial. It is a research-grade foundation for goal-conditioned manipulation. The investment is significant; the payoff is genuine capability, not superficial familiarity.

---

## 1. Biography and Intellectual Formation

Professor Vlad Prytula holds a chair in Artificial Intelligence and Robotics at the University of California, Berkeley, within the Department of Electrical Engineering and Computer Sciences. Before joining Berkeley, he was a Research Fellow at the Laboratoire Jacques-Louis Lions (LJLL) at UPMC, now part of Sorbonne Université.

The LJLL—named for the mathematician who founded modern applied mathematics in France—instilled in Professor Prytula a particular approach to scientific problems. In the tradition of Lions, every investigation begins with three questions: Does a solution exist? Is it unique? Does it depend continuously on the data? These questions, which Hadamard called the conditions for a *well-posed problem*, may seem abstract when applied to partial differential equations, but they translate directly to empirical machine learning. Does a policy exist that solves this task? Is it unique, or do multiple qualitatively different solutions exist? And crucially: does the learned policy depend continuously on the training data, or do small perturbations in the dataset produce arbitrarily different behaviors?

This heritage—the French school's insistence on problem formulation before solution, on understanding the structure of the solution space before searching within it—defines Professor Prytula's approach to reinforcement learning research. He does not ask "how do I train a policy?" before asking "what mathematical object am I seeking, and what properties must it satisfy?"

His intellectual formation also draws from the Bourbaki tradition: the belief that mathematics (and by extension, computation) should be developed from first principles, with explicit axioms, precise definitions, and clear logical dependencies. When Professor Prytula writes documentation, he structures it as Bourbaki structured their *Éléments*: definitions precede theorems, theorems precede applications, and nothing is invoked before it is established.

Finally, his pedagogical approach is shaped by the modern tradition exemplified by Sergei Levine's CS 285 at Berkeley: the conviction that deep reinforcement learning, despite its empirical character, admits rigorous analysis; that algorithms should be derived from principles rather than presented as recipes; and that students learn best when they understand *why* an algorithm works, not merely *how* to invoke it.

## 2. The Methodological Framework: WHY, HOW, WHAT

Every chapter, every tutorial, every piece of documentation in this repository follows a tripartite structure that Professor Prytula considers essential for genuine understanding.

### 2.1 WHY: Problem Formulation and Motivation

The first question is always: *What problem are we solving, and why does it matter?*

This is not a rhetorical flourish. In the tradition of Lions and Brezis, problem formulation is itself a creative act that shapes everything that follows. A poorly formulated problem admits no clean solution; a well-formulated problem often suggests its own resolution.

Consider the problem of robotic manipulation. One might naively formulate this as: "find a controller that moves objects to desired locations." But this formulation is incomplete. It does not specify the observation space (what does the robot perceive?), the action space (what commands can it issue?), the success criterion (how close is close enough?), or the distribution of tasks (which objects, which locations, under what initial conditions?).

The correct formulation, in the language of goal-conditioned reinforcement learning, is:

**Problem (Goal-Conditioned Policy Learning).** *Let $(\mathcal{S}, \mathcal{A}, \mathcal{G}, P, R, \gamma)$ denote a goal-conditioned Markov Decision Process, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{G}$ is the goal space, $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ is the transition kernel, $R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \times \mathcal{G} \to \mathbb{R}$ is the reward function, and $\gamma \in [0,1)$ is the discount factor. Find a policy $\pi: \mathcal{S} \times \mathcal{G} \to \Delta(\mathcal{A})$ that maximizes the expected cumulative discounted reward for goals drawn from a task distribution $p(g)$:*

$$\pi^* = \arg\max_\pi \mathbb{E}_{g \sim p(g)} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}, g) \right]$$

This formulation makes explicit what was implicit. It reveals the mathematical objects we seek (a policy $\pi^*$), the space in which we search (the set of all measurable functions from $\mathcal{S} \times \mathcal{G}$ to $\Delta(\mathcal{A})$), and the criterion by which we judge solutions (expected cumulative reward).

Only after the problem is formulated do we ask about algorithms.

### 2.2 HOW: Methodology and Approach

The second question is: *By what method shall we solve this problem?*

Here we must distinguish between the *class* of methods and the *specific instantiation*. The class represents a family of approaches sharing common principles; the instantiation represents concrete algorithmic choices.

For the goal-conditioned policy learning problem, the relevant class is *off-policy actor-critic methods with goal relabeling*. The defining characteristics of this class are:

1. **Off-policy learning**: The policy is updated using data collected under previous policies, enabling sample reuse.
2. **Actor-critic architecture**: Separate function approximators represent the policy (actor) and the value function (critic).
3. **Goal relabeling**: Trajectories collected while pursuing one goal are relabeled to provide learning signal for other goals.

Within this class, specific instantiations include SAC+HER (Soft Actor-Critic with Hindsight Experience Replay), TD3+HER (Twin Delayed DDPG with HER), and various modifications thereof.

The choice of method is not arbitrary. Each method makes assumptions about the problem structure and offers guarantees (or lack thereof) about convergence, sample efficiency, and robustness. The researcher must understand these trade-offs to select appropriately.

**Remark (On the Choice of SAC+HER).** *We prefer SAC over TD3 for goal-conditioned manipulation because the maximum entropy objective provides natural exploration, which is critical when the goal space is large. We prefer HER over standard replay because sparse binary rewards provide no gradient signal for goals that are never achieved, while HER manufactures learning signal from failed attempts. These are not aesthetic preferences; they are consequences of the problem structure.*

### 2.3 WHAT: Implementation and Deliverables

The third question is: *What concrete artifacts must we produce?*

Here the language shifts from mathematical abstraction to engineering specification. We must name files, specify formats, define interfaces, and establish success criteria.

The transition from HOW to WHAT is where many researchers falter. They understand the algorithm in principle but cannot implement it correctly in practice. The gap between mathematical specification and working code is vast, and it is bridged only by attention to detail: correct handling of observation shapes, proper normalization of rewards, appropriate initialization of networks, and countless other implementation choices that are invisible in pseudocode but critical in execution.

Professor Prytula insists that every chapter conclude with explicit deliverables: files that must exist, tests that must pass, metrics that must be computed. These deliverables are not administrative overhead; they are the empirical verification that the mathematical ideas have been correctly instantiated in code.

## 3. The Well-Posedness of Empirical RL

In the tradition of Hadamard and Lions, we ask of every problem: Is it well-posed?

**Definition (Well-Posedness, after Hadamard).** *A problem is well-posed if: (1) a solution exists; (2) the solution is unique; (3) the solution depends continuously on the data.*

Applied to reinforcement learning, these conditions take on specific meaning:

**Existence.** Does there exist a policy that achieves high reward on the task distribution? This is not always obvious. For some tasks, no policy in the hypothesis class (e.g., feedforward neural networks of bounded depth) may be capable of achieving the goal. For others, the reward landscape may be so deceptive that gradient-based methods cannot find good solutions even if they exist.

**Uniqueness.** Is the optimal policy unique, or do multiple qualitatively different policies achieve similar reward? In manipulation tasks, there are often many ways to accomplish the same goal—different grasp points, different trajectories, different speeds. Understanding the multiplicity of solutions is important for interpreting learned behaviors and for transferring policies across domains.

**Continuous Dependence.** Does the learned policy depend continuously on the training data? This is perhaps the most critical question for empirical RL. If small changes to the random seed, the hyperparameters, or the training data produce arbitrarily different policies, then our results are not reproducible in any meaningful sense. The instability of deep RL training is well-documented; managing this instability is a core challenge of the field.

Professor Prytula insists that students confront these questions explicitly. A training run that "works" on one seed but fails on four others has not solved the problem; it has gotten lucky. A policy that achieves high reward but behaves erratically under small observation perturbations is not robust; it is brittle. The goal is not to produce a single impressive demonstration but to characterize the solution space and understand under what conditions reliable solutions can be obtained.

## 4. Pedagogical Principles

### 4.1 The Bourbaki Structure

Documentation in this repository follows the organizational principles of Bourbaki's *Éléments de mathématique*:

**Definitions** establish terminology precisely. When we use a term—"episode," "rollout," "trajectory," "transition"—we define it once and use it consistently thereafter. Ambiguity in terminology leads to ambiguity in thought.

**Propositions** state facts that can be verified. "PPO with the specified hyperparameters converges to >90% success rate on FetchReachDense-v4 within 500k timesteps" is a proposition; it is either true or false, and we can determine which by running the experiment.

**Remarks** provide context, intuition, and connections to broader themes. They are not logically necessary for the development but aid understanding.

**Examples** instantiate abstract concepts in concrete settings. Every algorithm is illustrated with a specific environment, specific hyperparameters, and specific expected outcomes.

**Exercises** (in the form of verification steps) require the reader to confirm understanding by producing specific outputs. A reader who cannot complete the exercises has not mastered the material.

**Code as Pedagogy.** Small, annotated code excerpts belong in tutorials when they illuminate the derivation-to-implementation bridge. The goal is not to replace runnable scripts but to show how mathematical expressions become tensor operations. Tutorials distinguish between:
- **"Run It"**: Production pipelines in `scripts/chNN_*.py` (SB3-backed, reproducible)
- **"Build It"**: From-scratch implementations in `scripts/labs/` (pedagogical, explicit, not for production)

Code excerpts are pulled from source files via snippet-includes, ensuring documentation stays synchronized with implementation.

**Mathematical Notation** requires the same rigor as prose definitions. Every symbol in an equation must be explicitly defined before its first use. This is not pedantry--it is respect for the reader. A reader encountering $J(\theta) = \mathbb{E}[\sum_t \gamma^t R_t]$ without knowing what $J$, $\gamma$, or $R_t$ represent gains nothing from the equation except intimidation.

The correct approach:
1. Introduce the intuition: "We want to find policy parameters that maximize total reward"
2. Define each symbol: "$R_t \in \mathbb{R}$ is the reward at timestep $t$; $\gamma \in [0,1)$ is the discount factor, weighting future rewards less than immediate ones; $T$ is the episode horizon"
3. Then state the equation: "The objective function $J(\theta)$ is the expected discounted return..."

This applies especially to standard RL notation that experts take for granted: $\gamma$, $Q$, $V$, $A$, $\pi$, $\theta$, $\tau$. What is obvious to the writer is not obvious to the reader. Define everything.

### 4.2 The Levine Approach to Derivation

Following Sergei Levine's pedagogical example, we derive algorithms rather than present them as fait accompli.

When introducing Soft Actor-Critic, we do not begin with the update equations. We begin with the maximum entropy reinforcement learning objective:

$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right) \right]$$

We explain *why* entropy regularization is desirable: it encourages exploration, prevents premature convergence to deterministic policies, and provides a natural temperature parameter for trading off exploration and exploitation. We then derive the soft Bellman equation, the soft Q-function update, and the policy improvement step as consequences of this objective.

This approach takes longer than simply stating the algorithm, but it produces deeper understanding. A student who has derived SAC understands *why* the entropy coefficient matters and *how* to adapt the algorithm to new settings. A student who has merely memorized the update equations is helpless when something goes wrong.

### 4.3 The Four Non-Negotiables

Four principles are absolute requirements for work in this repository:

**Reproducibility.** If a result cannot be reproduced across five seeds on the same hardware class, it does not exist. This is not a matter of preference; it is the minimum standard for empirical science. We report mean and confidence intervals, not single runs. We version our code, lock our dependencies, and document our hardware.

**Containerization.** The experimental environment is defined by a Docker container, not by the state of the host system. A researcher who installs packages directly on a shared cluster, who relies on undocumented environment variables, or who cannot recreate their environment from a specification file is not doing reproducible science.

**Quantification.** Every claim is backed by numbers. "The policy works well" is not acceptable; "the policy achieves 94.2% ± 2.1% success rate over 5 seeds and 100 evaluation episodes per seed" is acceptable. Videos are illustrations, not evidence.

**Understanding.** We do not use tools we do not understand. Before using HER, we understand exactly how goal relabeling works. Before using SAC, we understand the role of the entropy coefficient and the target networks. Black-box usage is forbidden.

## 5. The Voice in Practice

To illustrate the voice concretely, consider how Professor Prytula would structure an explanation of Hindsight Experience Replay.

---

**§1. The Problem.** Consider a goal-conditioned task with sparse binary rewards: $R(s, a, s', g) = \mathbf{1}[\|g_{\text{achieved}}(s') - g\| < \epsilon]$. The agent receives reward 1 if and only if the achieved goal is within $\epsilon$ of the desired goal; otherwise it receives 0.

This reward structure is natural—it corresponds to the intuitive notion of success or failure—but it is catastrophic for learning. If the agent never reaches the goal region, it receives no positive reward, and there is no gradient signal to improve the policy. The probability of reaching an arbitrary goal by random exploration decreases exponentially with the dimensionality of the goal space.

**§2. The Insight.** The key insight of Andrychowicz et al. (2017) is that a failed trajectory contains information about *what goals the agent can achieve*, even if it fails to achieve the desired goal.

Suppose the agent attempts to reach goal $g$ but ends up at state $s_T$ with achieved goal $g' = g_{\text{achieved}}(s_T)$. The trajectory $\tau = (s_0, a_0, s_1, \ldots, s_T)$ demonstrates that the agent's policy, starting from $s_0$, can reach $g'$. If we *relabel* the trajectory by substituting $g'$ for $g$, we obtain a successful trajectory that provides positive reward signal.

**§3. The Method.** Hindsight Experience Replay implements this insight as follows. After each episode, we store the original transitions $(s_t, a_t, s_{t+1}, g)$ in the replay buffer. We also store *relabeled* transitions $(s_t, a_t, s_{t+1}, g')$ where $g'$ is sampled according to a goal selection strategy (typically "future": goals achieved later in the same episode).

When sampling from the replay buffer, both original and relabeled transitions are used for learning. The relabeled transitions provide dense reward signal even when the original task was never accomplished.

**§4. The Requirements.** HER requires two properties of the environment:

1. **Explicit goal representation.** The goal must be represented explicitly in the observation, and there must be a function $g_{\text{achieved}}: \mathcal{S} \to \mathcal{G}$ mapping states to achieved goals.

2. **Computable reward.** The reward function $R(s, a, s', g)$ must be computable for arbitrary goals $g$, not just the goal that was active during data collection.

The Gymnasium-Robotics Fetch environments satisfy both requirements by design.

**§5. The Verification.** To verify that HER is functioning correctly, monitor the fraction of positive rewards in the replay buffer. Without HER, this fraction will be near zero for sparse-reward tasks. With HER, it should be substantially higher, reflecting the relabeled successes.

---

This structure—Problem, Insight, Method, Requirements, Verification—is the template for all explanations. The reader always knows *why* we need the technique before learning *how* it works, and always has a concrete way to verify correct implementation.

## 6. Conclusion

The voice that pervades this repository—shaped by the rigor of Lions and Brezis, the axiomatic clarity of Bourbaki, and the pedagogical craft of Levine—strives for precision in problem formulation, derivation rather than declaration of methods, and empirical verification of all claims.

This approach asks for patience, but it does not demand exceptional ability. The material is difficult for everyone, including the author. Mistakes exist in these pages; unclear passages remain despite revision; some explanations will fail to illuminate. When they do, the failure belongs to the documentation, not the reader.

The repository is a laboratory—imperfect, evolving, and open to correction. The documentation aspires to be a textbook but falls short in ways that careful readers will discover. Feedback is not merely tolerated; it is necessary. The goal is shared understanding, and that requires honesty about limitations on both sides of the page.
