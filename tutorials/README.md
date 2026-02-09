# Tutorials

## Prologue: On the Purpose of This Collection

This collection of tutorials constitutes a systematic treatment of goal-conditioned reinforcement learning for robotic manipulation. It is not a quickstart guide, nor a collection of recipes, nor a reference manual. It is, in the tradition of the great mathematical textbooks, a *course of study*: a carefully sequenced development of ideas, each building on what came before, designed to produce genuine understanding rather than superficial familiarity.

The reader who completes this course will not merely know how to train a policy on Fetch tasks. They will understand *why* goal-conditioned formulations enable certain algorithmic techniques; *how* the mathematical structure of the problem constrains the space of effective solutions; and *what* properties a well-designed experimental pipeline must satisfy to produce reproducible results.

This is the standard we set. Anything less is not worth the reader's time.

## The Central Questions

In the tradition of Hadamard and Lions, we begin every investigation with fundamental questions about the problem itself. For reinforcement learning in robotic manipulation, these questions are:

**The Existence Question.** Does there exist, within our hypothesis class of neural network policies, a function that achieves high success rate on the task distribution? This question is non-trivial. The hypothesis class is constrained (feedforward networks of bounded depth and width); the task distribution may include configurations that are kinematically difficult or dynamically unstable; the reward landscape may be deceptive, with local optima that trap gradient-based methods.

**The Learnability Question.** Given that a good policy exists, can our learning algorithm find it? The gap between existence and learnability is vast. A policy may exist in principle yet be unreachable by stochastic gradient descent from random initialization. The learning algorithm may require more samples than are practically available, or may be sensitive to hyperparameters in ways that make reliable training impossible.

**The Stability Question.** Suppose we find a good policy. Does it depend continuously on the training process? If small changes to the random seed, the hyperparameter settings, or the training data produce qualitatively different policies, then our results are not reproducible in any scientifically meaningful sense. The notorious instability of deep RL training makes this question particularly urgent.

These tutorials do not answer these questions in full generality—that would require a treatise, not a tutorial series. But they provide the conceptual framework and empirical tools to investigate these questions for specific tasks and algorithms. The reader who completes this course will know how to *ask* these questions precisely and how to *answer* them through careful experimentation.

## The Methodological Framework

Every chapter in this collection follows a tripartite structure: **WHY**, **HOW**, **WHAT**.

### WHY: Problem Formulation

Before presenting any technique, we formulate the problem it addresses. What mathematical object are we seeking? What properties must it satisfy? What makes this problem difficult?

This is not mere throat-clearing. In the tradition of the French school of applied mathematics—Lions, Brezis, and their intellectual descendants—problem formulation is itself a creative act that shapes everything that follows. A poorly formulated problem admits no clean solution; a well-formulated problem often suggests its own resolution.

When we introduce Hindsight Experience Replay in Chapter 4, we do not begin with the algorithm. We begin with the problem: sparse binary rewards provide no gradient signal when the goal is never reached; the probability of reaching an arbitrary goal by random exploration decreases exponentially with goal-space dimension; therefore, any practical algorithm must manufacture learning signal from failed attempts. The problem formulation makes the solution almost inevitable.

### HOW: Methodology and Derivation

Having established *what* problem we are solving and *why* it is difficult, we turn to *how* we shall solve it.

Here we follow the pedagogical tradition exemplified by Sergei Levine's CS 285: algorithms are *derived*, not *declared*. We do not present update equations as fait accompli; we show where they come from. We do not invoke design choices as received wisdom; we justify them from first principles.

When we introduce Soft Actor-Critic, we begin with the maximum entropy objective and derive the soft Bellman equation, the Q-function update, and the policy improvement step as logical consequences. A student who has followed this derivation understands not just *what* SAC does but *why* it does it—and therefore knows how to adapt it when standard settings fail.

### WHAT: Implementation and Verification

The transition from mathematical abstraction to working code is where many researchers falter. The gap between "I understand the algorithm" and "my implementation produces correct results" is vast, and it is bridged only by meticulous attention to detail.

Every chapter concludes with explicit deliverables: files that must exist, tests that must pass, metrics that must lie within specified ranges. These are not administrative overhead; they are the empirical verification that the mathematical ideas have been correctly instantiated in code.

A reader who cannot produce the specified deliverables has not completed the chapter, regardless of how well they believe they understand the material. Understanding without verification is indistinguishable from misunderstanding.

## The Structure of the Chapters

Each chapter follows a consistent organization, modeled on the structure of mathematical monographs:

**Abstract.** A concise statement of what the chapter accomplishes: what problem is addressed, what methods are introduced, what deliverables are produced.

**Theoretical Foundations.** The mathematical context for the chapter's content. Definitions are stated precisely; connections to the broader literature are established; the reader understands *why* the material matters before seeing *how* it works.

**The Experimental Environment.** The computational context for the chapter's experiments. We specify the container, the hardware assumptions, the environment variables, and any other infrastructure requirements. Reproducibility demands that these details be explicit.

**Implementation Details.** The core of the chapter: code (in the form of versioned scripts), analysis, and explanation. We do not merely show what to type; we explain what it means and why it works.

**Verification and Sanity Checks.** Concrete tests that confirm correct implementation. A reader who passes these tests has correctly instantiated the chapter's ideas; a reader who fails them has made an error that must be diagnosed before proceeding.

**Deliverables.** The artifacts that constitute "done": files, metrics, plots. The reader knows unambiguously what they must produce.

## The Chapter Sequence

The chapters are ordered to build systematically from foundations to applications:

**Chapter 0: A Containerized "Proof of Life" on Spark DGX.** The existence of a working experimental environment is the precondition for all subsequent work. This chapter establishes that precondition: we verify GPU access, enter a reproducible container, validate MuJoCo rendering, and confirm that a training loop executes without error. No learning happens here; we are merely establishing that the laboratory is functional.

*WHY:* Without a reproducible environment, no experimental result can be trusted. Containerization is not a convenience; it is a requirement for scientific validity.

*HOW:* We use Docker with the NVIDIA runtime, mount the repository into the container, manage dependencies via a virtual environment, and configure rendering backends for headless operation.

*WHAT:* A working `docker/dev.sh` entrypoint; a rendered frame proving MuJoCo works; a saved checkpoint proving the training loop completes.

**Chapter 1: The Anatomy of Goal-Conditioned Fetch Environments.** Before training any agent, we must understand precisely what the agent perceives, what actions it can take, and how rewards are computed. This chapter dissects the Fetch environment interface.

*WHY:* Goal-conditioned environments have a specific structure—observation dictionaries with `achieved_goal` and `desired_goal` keys, reward functions that can be queried for arbitrary goals—that enables techniques like Hindsight Experience Replay. Understanding this structure is prerequisite to using those techniques correctly.

*HOW:* We inspect observation and action spaces programmatically, verify reward consistency between `env.step()` and `compute_reward()`, and collect baseline metrics with random policies.

*WHAT:* JSON schemas describing the environment interface; verified reward consistency; random baseline metrics establishing the performance floor.

Subsequent chapters—on PPO baselines, SAC, HER, robustness, and beyond—follow the same structure, each building on the foundations laid by its predecessors.

## Prerequisites

**Mathematical.** The reader should be comfortable with the fundamentals of probability theory (random variables, expectations, conditional distributions), optimization (gradients, stochastic gradient descent), and the basic formalism of Markov Decision Processes (states, actions, transitions, rewards, policies, value functions). Sutton and Barto's *Reinforcement Learning: An Introduction* provides sufficient background; familiarity with Bertsekas and Tsitsiklis's *Neuro-Dynamic Programming* is helpful but not required.

**Computational.** The reader should be comfortable with Python programming, command-line operations, and basic Docker usage. Familiarity with PyTorch is helpful; familiarity with Stable Baselines 3 is not required and will be developed in the tutorials.

**Infrastructural.** The reader should have access to a machine with an NVIDIA GPU and a properly configured Docker installation. The tutorials are written with the Spark DGX cluster in mind but should be adaptable to any similar environment.

## How to Use These Tutorials

**Sequential completion is expected.** The chapters build on each other; skipping ahead will leave gaps in understanding that will manifest as confusion or incorrect implementations later. A reader who cannot complete Chapter 0 has no business attempting Chapter 3.

**Verification is not optional.** Every chapter includes sanity checks and deliverables. Complete them. A reader who skips verification to save time will spend more time later debugging problems that could have been caught early.

**Understanding is the goal, not speed.** These tutorials are designed to be worked through carefully, not skimmed. A reader who spends a week on Chapter 1 and emerges with genuine understanding has accomplished more than one who rushes through all chapters in a day and emerges with recipes they cannot adapt.

**The scripts are the source of truth.** Tutorials reference scripts in the `scripts/` directory; they do not embed code to be copied and pasted. The scripts are versioned, tested, and maintained. The prose explains what the scripts do and why; the scripts are what you actually run.

## On the Voice of These Tutorials

These tutorials are written in the voice of Professor Vlad Prytula, whose pedagogical philosophy is documented in `vlad_prytula_persona.md`. The voice is formal, precise, and demanding. It does not apologize for complexity; it explains complexity. It does not offer shortcuts; it offers understanding.

This voice is a deliberate choice. Reinforcement learning for robotic manipulation is a difficult subject that rewards rigor and punishes sloppiness. A tutorial that pretends otherwise does the reader a disservice.

The reader who finds this voice congenial will thrive. The reader who finds it off-putting should consider whether they are seeking genuine mastery or merely the appearance of competence. These tutorials offer the former; the latter can be found elsewhere.

---

*"The purpose of a tutorial is not to make the subject seem easy. It is to make the subject become understood."*
