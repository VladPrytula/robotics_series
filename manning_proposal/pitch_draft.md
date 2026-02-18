# Manning Book Proposal -- Submission Draft

**Status:** DRAFT -- ready for author review, then submission to proposals@manning.com
**Last updated:** 2026-02-18

---

## Part A: Title Discussion

### Recommended Title

**Robotics Reinforcement Learning in Action**

*Subtitle:* Train, debug, and stress-test goal-conditioned manipulation policies from first principles to pixels

### Why this title works

- **Series fit:** Follows the "[Subject] in Action" pattern (Kubernetes in Action, Spring in Action, Deep Reinforcement Learning in Action). Manning editors expect this.
- **Differentiation from existing Manning title:** "Deep Reinforcement Learning in Action" (Zai & Brown, 2020) covers general deep RL. Ours is scoped to *robotics* -- a distinct domain with its own observation structure (goal-conditioned dicts), action semantics (continuous Cartesian deltas), and evaluation needs (success rate, not just return).
- **Searchability:** Contains all three keywords a practitioner would search: "robotics", "reinforcement learning", "in action" (practical).

### Alternative titles considered

| Title | Pros | Cons |
|-------|------|------|
| **Reinforcement Learning for Robotics in Action** | Reads more naturally in English | Slightly long; "for" weakens the brand impact |
| **Robot Learning in Action** | Short, punchy | Too broad -- could include supervised learning, imitation learning |
| **Goal-Conditioned Robot Learning in Action** | Technically precise | Too niche for a bookstore browser |
| **RL for Robot Manipulation in Action** | Accurate to content | Narrows perceived scope; "manipulation" may intimidate beginners |

**Recommendation:** Go with **"Robotics Reinforcement Learning in Action"** as the primary. Manning editors may adjust during contract discussions -- that is normal. The subtitle does the precision work.

---

## Part B: Gap Analysis -- Manning Template vs. What You Have

Manning's proposal form (`ManningBookProposal.docx`) has 10 sections. Here is where you stand on each:

| # | Manning Section | Status | What Exists | What's Missing |
|---|----------------|--------|-------------|----------------|
| 1 | **Author Information** | **RED** | Nothing written | Bio, credentials, relevant experience, online presence |
| 2 | **Book Topic / Why Now** | GREEN | `overview.md` "Why This Topic Is Important Now" | Minor polish only |
| 3 | **Book Summary** | GREEN | `overview.md` "What the Reader Will Be Able to Do" + differentiators | Condense into 2-paragraph elevator pitch |
| 4 | **Reader Questions** | YELLOW | `overview.md` has audience + prerequisites | Needs "common questions the book answers" list |
| 5 | **Target Audience** | YELLOW | `overview.md` has general audience | Needs specific job titles, company types, use cases |
| 6 | **Competition & Ecosystem** | YELLOW | `overview.md` has 4 Manning titles | Missing: non-Manning books, online courses, YouTube, communities |
| 7 | **Book Specifications** | **RED** | Nothing explicit | Word count, illustration count, chapter count, code repo URL |
| 8 | **Contact Details** | **RED** | Nothing | Author name, email, social links |
| 9 | **Publication Timeline** | **RED** | Nothing | Milestone dates for 1/3, 2/3, full manuscript, MEAP launch |
| 10 | **Table of Contents** | GREEN | `toc.md` -- 13 chapters, 5 parts, section-level detail | Add 1-2 sentence learning objective per chapter |

**Additionally required (not in template, but expected by acquisitions editors):**

| Item | Status | Notes |
|------|--------|-------|
| **Sample chapters (full prose)** | GREEN | Ch1 (732 lines), Ch2 (874 lines), Ch3 (957 lines) -- all with PDF/DOCX |
| **Marketing plan / audience reach** | **RED** | How will you promote? Talks, blog, social, community? |
| **"From scratch" definition** | YELLOW | Exists in persona doc but not as standalone proposal section |

**Summary:** 4 RED gaps (author bio, specs, contact, timeline), 3 YELLOW gaps (reader questions, audience detail, competition breadth), 3 GREEN sections ready.

---

## Part C: The Proposal (Draft for Submission)

*What follows is the actual proposal text, section by section, formatted for Manning's template. Author-specific sections are marked [AUTHOR: fill in] where only you can provide the content.*

---

### Section 1: About the Author

> [AUTHOR: This section needs your voice. Below is a template -- fill in the brackets and adjust tone.]

**Author name:** [Your full name]

**Professional background:**
[AUTHOR: 2-3 sentences. Example structure: "I am a [title] at [company/institution] working on [relevant domain]. My work focuses on [specific area relevant to the book]. I have [N] years of experience with [RL/robotics/ML engineering]."]

**Why I am the right person to write this book:**
[AUTHOR: 2-3 sentences connecting your experience to the book's unique angle. Example: "I built the entire codebase backing this book while training manipulation policies on DGX infrastructure, encountering (and solving) every silent failure mode described in the text. The debugging methodology in this book comes from real engineering practice, not textbook exercises."]

**Relevant publications, talks, or teaching:**
[AUTHOR: List any. If none, that is fine -- Manning accepts first-time authors. Focus on your engineering credentials instead.]

**Online presence:**
- GitHub: [URL -- ideally the book's companion repo]
- [Blog / Twitter / LinkedIn / other -- whatever you have]

**Writing experience:**
[AUTHOR: Even informal writing counts. "I have written technical documentation for [X]" or "I maintain a technical blog at [URL]" or "This would be my first book, but the companion repository contains 5,000+ lines of tutorial prose that demonstrate my technical writing ability."]

---

### Section 2: Tell Us About the Book's Topic

Reinforcement learning for robotics has crossed a practical threshold. MuJoCo -- formerly a \$500/year commercial license -- is now open-source and actively maintained. Gymnasium-Robotics provides goal-conditioned task interfaces with explicit success signals. Stable Baselines3 has made strong actor-critic algorithms (PPO, SAC, TD3) and Hindsight Experience Replay accessible to any Python developer. The tooling exists.

But the tooling is not enough. Practitioners who install these packages and start training robot manipulation agents face a failure mode that has no equivalent in traditional software: *everything runs, nothing learns.* There are no compiler errors, no stack traces, no red text. The training loop finishes, the checkpoint saves, and the robot sits still -- or flails -- or succeeds 3% of the time. Diagnosing whether the problem is a software bug, a misconfigured environment, or a fundamental algorithm limitation requires engineering skills that no existing book teaches.

**Why this topic is important now:**

1. **The tooling barrier has fallen, but the engineering barrier remains.** MuJoCo + SB3 + HER are accessible, but practitioners lack reliable methods to diagnose failures and verify results in goal-conditioned robotics settings.

2. **The field is moving from demos to deployments.** The robotics community has shifted from "one task, one video" toward generalizable goal-conditioned behavior with principled evaluation (multi-seed protocols, confidence intervals, stress testing). Engineers need a bridge from "cool demo" to "deliverable-grade policy."

3. **Sparse rewards are the real problem, and no practical guide covers them honestly.** Most RL books use dense, hand-shaped rewards. Real robotics tasks have binary success/failure signals. Hindsight Experience Replay (HER) solves this -- but using it correctly requires understanding replay buffers, goal relabeling, and reward recomputation at an engineering level.

4. **Reproducibility is now a first-class concern.** Henderson et al. (2018, AAAI) documented a reproducibility crisis in deep RL. This book treats reproducibility as an engineering deliverable, not an aspiration.

---

### Section 3: Book Summary

**Robotics Reinforcement Learning in Action** is a hands-on engineering guide to training, evaluating, and debugging goal-conditioned manipulation policies using MuJoCo, Gymnasium-Robotics, and Stable Baselines3. It is organized around a single task family -- the Fetch robotic arm suite (Reach, Push, PickAndPlace) -- that stays constant while the reader's capabilities evolve from basic policy training to sparse-reward learning to pixel-based control.

**The core promise:** By the end of this book, the reader will not just have trained policies that work -- they will have an *auditable experimental record* (metrics JSON, checksummed checkpoints, evaluation reports) that proves the policies work, quantifies how robust they are, and documents what was tried.

**What makes this book different from existing RL resources:**

- **Build It / Bridge / Run It structure.** Every algorithm chapter has three tracks. *Build It* implements the core mechanics from scratch in PyTorch (losses, updates, buffers, relabeling) with verification checkpoints. *Bridge* proves the from-scratch implementation agrees with SB3 on the same data. *Run It* scales to multi-million-step robotics training using SB3, producing auditable artifacts. Readers choose their depth; all tracks produce verifiable results.

- **Sparse rewards done methodically.** The book does not handwave exploration difficulty. It shows, with controlled experiments, why learning from binary success/failure signals fails -- and why HER changes the data distribution enough to learn. This is the single most important technique for practical goal-conditioned robotics.

- **Five compute tiers.** From a 2-minute CPU sanity check to an overnight multi-seed experiment, every chapter offers access points for readers with different hardware. No GPU? Use provided checkpoints and evaluate. Have a DGX? Run the full reproduction protocol.

- **Debuggability as a first-class goal.** The reader learns to diagnose flat reward curves, entropy collapse, replay buffer pathologies, and reward misconfigurations -- not just copy hyperparameters.

---

### Section 4: Common Reader Questions This Book Answers

1. **"I installed MuJoCo and SB3. I ran training for 8 hours. Success rate is 0%. What went wrong?"** -- Chapter 1 establishes a systematic verification protocol. Chapters 3-4 teach diagnostic skills (reward curves, Q-value tracking, entropy monitoring).

2. **"How do I train a robot to reach arbitrary goals when the only feedback is success or failure?"** -- Chapters 5-6 introduce HER for sparse goal-conditioned learning and verify it works with controlled experiments.

3. **"My policy works in one video but fails when I test it properly. How do I know if it actually learned?"** -- Chapters 2-3 establish a fixed-seed evaluation protocol. Chapter 8 introduces stress testing with degradation curves.

4. **"I can solve Reach, but PickAndPlace is a completely different difficulty level. Where do I even start?"** -- Chapter 6 provides a capstone that builds on every technique from earlier chapters.

5. **"How much of this can I build from scratch vs. needing to use a library?"** -- Every algorithm chapter includes from-scratch implementations of core mechanics (actor-critic updates, replay buffers, HER relabeling) alongside production SB3 pipelines.

6. **"I want to move from state vectors to camera images. What changes?"** -- Chapters 10-11 make this transition within the same Fetch task family, quantifying the sample-efficiency cost.

7. **"How do I know if my policy is robust enough to deploy on hardware?"** -- Chapters 8 and 13 provide stress-test protocols and a deployment-readiness checklist.

---

### Section 5: Target Audience

**Primary readers:**

| Reader Profile | Background | Motivation |
|----------------|-----------|------------|
| **ML engineer at a robotics startup** | Python, PyTorch, has trained classifiers/NLP models | Needs to apply RL to manipulation tasks; wants reproducible evaluation for stakeholder reports |
| **Software engineer exploring robotics** | Strong Python, basic ML, no robotics experience | Wants a self-contained learning path from "hello world" to working manipulation policies |
| **Graduate student in robotics/ML** | Theory from coursework (MDPs, policy gradients) | Needs practical implementation skills and engineering methodology for thesis experiments |
| **Research engineer at an RL lab** | Familiar with SB3/RL algorithms | Wants systematic debugging and evaluation methodology; tired of "it works on my machine" results |

**Minimum prerequisites:**
- Python proficiency (functions, classes, iterators)
- Basic PyTorch (tensors, autograd, nn.Module, optimizers)
- RL vocabulary at the level of one introductory course or Sutton & Barto chapters 1-3 (policies, rewards, value functions)
- Command-line comfort (bash, docker basics helpful but taught in Chapter 1)

**What the reader does NOT need:**
- Robotics background (the book starts from "what is a Fetch arm?")
- Prior MuJoCo experience
- Access to physical robots (all work is in simulation)
- A powerful GPU for the first 6 chapters (CPU paths and checkpoints provided)

---

### Section 6: Competition and Ecosystem

**Direct Manning competitors:**

| Title | Author(s) | Year | How This Book Differs |
|-------|-----------|------|-----------------------|
| *Deep Reinforcement Learning in Action* | Zai & Brown | 2020 | Covers general deep RL (Atari, grid worlds). Not focused on robotics, continuous control, goal-conditioned interfaces, or reproducible evaluation protocols. |
| *Grokking Deep Reinforcement Learning* | Morales | 2019 | Approachable conceptual treatment. Not centered on robotics task structure (continuous actions + sparse goals) or engineering methodology. |
| *Robotics for Programmers* | Bihlmaier | 2024 | Broad robotics programming (ROS, kinematics, perception). Not RL-centric. Complementary rather than competing. |
| *Reinforcement Learning for Business* | Aghazadeh | 2025 | RL for business optimization. Different domain, different evaluation constraints. |

**Non-Manning competitors:**

| Resource | How This Book Differs |
|----------|----------------------|
| Sutton & Barto, *Reinforcement Learning* (2018) | The definitive theoretical foundation. Not an implementation guide, not robotics-focused. We reference it throughout. |
| OpenAI Spinning Up | Algorithm walkthroughs with code. No robotics task structure, no goal conditioning, no evaluation methodology, no stress testing. |
| Coursera/Udacity RL courses | Video-based, assignment-driven. Do not produce a deployable robotics pipeline or auditable experimental record. |
| Packt RL titles | Generally lower editorial bar, less cohesive structure. Our single-spine (Fetch) approach and Build It / Run It pedagogy are distinct. |
| arXiv/blog tutorials | Scattered, version-fragile, no unified evaluation protocol. This book is the integrated, maintained resource. |

**The market gap in one sentence:** There is no book that teaches RL practitioners to engineer, debug, and stress-test goal-conditioned manipulation policies with an auditable experimental record, from dense rewards through sparse rewards through pixels.

---

### Section 7: Book Specifications

| Specification | Estimate |
|---------------|----------|
| **Number of chapters** | 13 (organized in 5 parts) |
| **Estimated word count** | 95,000 -- 115,000 words |
| **Estimated page count** | 380 -- 450 pages (Manning "In Action" format) |
| **Estimated illustrations** | 80 -- 100 figures (learning curves, architecture diagrams, environment screenshots, comparison plots) |
| **Code listings per chapter** | 8 -- 15 (mix of snippet-includes from labs and inline examples) |
| **Companion repository** | Public GitHub repo with all scripts, Docker setup, lab modules, and pretrained checkpoints |
| **Software dependencies** | Python 3.10+, PyTorch, Stable Baselines3, Gymnasium-Robotics, MuJoCo (all open-source) |

**Note on illustrations:** All figures are self-generated from MuJoCo renders and matplotlib plots -- no external assets, no licensing concerns, fully reproducible via documented commands.

---

### Section 8: Contact Details

> [AUTHOR: Fill in]

- **Full name:** [Your name]
- **Email:** [Your email]
- **Location / timezone:** [City, Country, timezone]
- **GitHub:** [URL]
- **LinkedIn / Twitter / Blog:** [URLs]
- **Preferred communication:** [Email / Slack / etc.]

---

### Section 9: Proposed Schedule

> [AUTHOR: Adjust dates based on your realistic availability. The schedule below assumes ~1 chapter per month, which is the standard Manning pace.]

| Milestone | Target Date | Deliverable |
|-----------|-------------|-------------|
| **Proposal submission** | [Month 0] | Proposal + 3 sample chapters (Ch1, Ch2, Ch3) |
| **Contract signed** | [Month 0 + 6-8 weeks] | After editorial board review |
| **Ch4 draft (SAC)** | [Month 1 after contract] | Completes Part 2 |
| **Ch5 draft (HER)** | [Month 2] | Core differentiator chapter |
| **Ch6 draft (PickAndPlace capstone)** | [Month 3] | Completes Part 3; MEAP launch candidate |
| **MEAP launch (Ch1-6)** | [Month 3-4] | First 6 chapters available to early readers |
| **Ch7-9 drafts (Engineering)** | [Months 5-7] | Part 4 complete |
| **Ch10-11 drafts (Pixels)** | [Months 8-9] | Part 5 core |
| **Ch12-13 drafts (Advanced + Playbook)** | [Months 10-11] | Part 5 complete |
| **Full manuscript review** | [Month 12] | All 13 chapters in review |
| **Final manuscript to production** | [Month 14] | Post-review revisions complete |
| **Estimated publication** | [Month 17-20] | After production (copy-edit, layout, index) |

**Why this schedule is credible:**
- 3 chapters already written, reviewed, and produced as PDF/DOCX
- 6 tutorial source documents (10,000+ lines) ready to adapt for Ch1-6
- 9 lab modules with from-scratch implementations already coded and tested
- Complete chapter production protocol (scaffold -> write -> review -> revise -> publish) already proven on 3 chapters
- Docker infrastructure, training scripts, and evaluation pipeline all working

---

### Section 10: Table of Contents (Annotated)

**Part 1 -- Start Running, Start Measuring**

**Chapter 1: Proof of life: a reproducible robotics RL loop**
*The reader sets up a containerized MuJoCo + Gymnasium-Robotics environment and verifies every component -- GPU, physics, rendering, training, evaluation -- with concrete evidence. Introduces the three diagnostic questions and the experiment contract (checkpoints, metadata, eval reports) used throughout the book.*

1.1 Why robotics RL fails silently
1.2 The task family: goal-conditioned Fetch manipulation
1.3 The experiment contract (training, evaluation, artifacts)
1.4 Running the proof-of-life pipeline
1.5 Reading results and debugging first failures
1.6 Summary

**Chapter 2: What the robot actually sees: observations, rewards, and success**
*The reader inspects the Fetch environment by hand -- decoding dictionary observations, manually computing rewards, and verifying success conditions. Establishes the observation/action/reward semantics that every later chapter depends on.*

2.1 The observation dictionary: observation, achieved_goal, desired_goal
2.2 Actions as Cartesian deltas and what that implies
2.3 Reward semantics and compute_reward
2.4 A metrics schema you will reuse (success, distance, smoothness)
2.5 Summary

---

**Part 2 -- Baselines That Debug Your Pipeline**

**Chapter 3: PPO as a lie detector (dense Reach)**
*The reader trains PPO on FetchReachDense -- not because PPO is the final method, but because dense-reward PPO is the fastest way to verify the training pipeline works. Includes a from-scratch PPO implementation (actor-critic, GAE, clipped objective) with bridging proof against SB3.*

3.1 Why dense rewards are diagnostic tools
3.2 PPO at a practical level (what it optimizes, what can go wrong)
3.3 A fixed evaluation protocol (seeds, determinism, JSON reports)
3.4 Debugging a flat curve (common causes and checks)
3.5 Summary

**Chapter 4: Off-policy without mystery (SAC on dense Reach)**
*The reader learns why off-policy methods matter for robotics (data reuse, replay buffers) and trains SAC on the same task. Includes from-scratch SAC (twin critics, entropy tuning, soft updates) and diagnostic interpretation (Q-values, entropy coefficient, reward distributions).*

4.1 Why off-policy methods matter for robotics
4.2 Replay buffers and what they change about learning
4.3 SAC intuition: entropy, critics, and target networks
4.4 Diagnostics: rewards, Q-values, entropy coefficient, goal distances
4.5 Summary

---

**Part 3 -- Sparse Goals, Real Progress**

**Chapter 5: Learning from failure with HER (sparse Reach and Push)**
*The book's differentiator chapter. The reader confronts sparse rewards (binary success/failure), sees SAC-without-HER stall, then implements HER relabeling from scratch and verifies it produces learning signal. Controlled experiments show why HER works and when it does not.*

5.1 Sparse rewards and the exploration barrier
5.2 HER prerequisites: explicit goals and computable rewards
5.3 Baseline: SAC without HER (and why it stalls)
5.4 HER mechanism: relabeling and goal selection strategies
5.5 Verification: evidence that HER is working
5.6 Summary

**Chapter 6: Capstone manipulation: PickAndPlace with an honest report card**
*The reader applies everything from Chapters 1-5 to a harder task: picking up an object and placing it at a target location. This chapter tests whether the methodology scales, introduces curriculum concepts, and produces a publishable-quality evaluation.*

6.1 Why contact-rich manipulation is harder than Reach/Push
6.2 Dense-debugging vs sparse truth (and how not to fool yourself)
6.3 Curriculum and stress-test splits
6.4 Deliverable-grade evaluation and experiment cards
6.5 Summary

---

**Part 4 -- Engineering-Grade Robotics RL**

**Chapter 7: Policies as controllers: stability, smoothness, and action interfaces**
*Beyond return: the reader learns to evaluate policies as controllers, measuring action smoothness, time-to-success, and oscillation. Practical action-space engineering (scaling, clipping, filtering).*

7.1 Controller-centric metrics (beyond return)
7.2 Action scaling, clipping, and optional filtering
7.3 Time-to-success and "oscillation" as measurable quantities
7.4 Summary

**Chapter 8: Robustness curves: quantify brittleness**
*The reader stress-tests trained policies under controlled perturbations (observation noise, action noise, goal perturbation) and produces degradation curves with confidence bands. The core message: videos are not evidence; degradation curves are.*

8.1 Why videos are not evidence
8.2 Observation noise, action noise, and controlled perturbations
8.3 Degradation curves with confidence bands across seeds
8.4 Summary

**Chapter 9: Evidence-driven tuning: ablations and sweeps**
*The reader runs minimal ablations to answer "what mattered?" and reports results using experiment cards and comparable JSON metrics. Covers reproducibility tolerances across hardware classes.*

9.1 What "reproducible" means in practice (tolerances, hardware classes)
9.2 Minimal ablations that answer "what mattered?"
9.3 Reporting: experiment cards and comparable JSON metrics
9.4 Summary

---

**Part 5 -- Pixels and the Reality Gap**

**Chapter 10: Pixels, no cheating: visual Reach in the same task family**
*The reader moves from privileged state vectors to raw pixel observations within the same Fetch Reach task, measuring the sample-efficiency cost and learning to design practical CNN encoders for RL.*

10.1 What changes when you remove privileged state
10.2 Rendering wrappers and observation design
10.3 A practical CNN encoder for RL
10.4 Measuring the sample-efficiency gap
10.5 Summary

**Chapter 11: Visual robustness that matters: augmentation as a tool**
*Image augmentation (random crop, color jitter) as an engineering tool for visual RL robustness, with controlled ablations showing what helps and what hurts.*

11.1 Visual brittleness and why it appears
11.2 Practical augmentations and why they help
11.3 Ablations: what helps, what hurts
11.4 Summary

**Chapter 12: Visual goals (optional advanced): "make it look like this"**
*Goals specified as target images rather than coordinates. Discusses promise, pitfalls, and evaluation protocols for high-dimensional goals.*

12.1 Goals as images: promise and pitfalls
12.2 Evaluation protocols for high-dimensional goals
12.3 Summary

**Chapter 13: A reality-gap playbook: stress tests before hardware**
*Domain randomization, sim-to-sim system identification, and a deployment-readiness checklist. The capstone for the entire book: can the reader's methodology produce policies that are ready for hardware transfer?*

13.1 Domain randomization: what to randomize and how to measure it
13.2 Sim-to-sim system identification as a controlled rehearsal
13.3 A deployment-readiness checklist backed by tests
13.4 Summary

---

## Part D: Submission Package Checklist

### What to email to proposals@manning.com

| # | Item | Format | Status |
|---|------|--------|--------|
| 1 | **Completed proposal form** (Sections 1-10 above) | DOCX (use Manning's template) | DRAFT -- needs author sections filled in |
| 2 | **Sample Chapter 1:** "Proof of Life" | DOCX or PDF | READY (Manning/output/ch01_proof_of_life.docx) |
| 3 | **Sample Chapter 3:** "PPO as a Lie Detector" | DOCX or PDF | READY (Manning/output/ch03_ppo_dense_reach.docx) |
| 4 | **Link to companion repository** | URL | READY (needs public GitHub link) |
| 5 | **(Optional) Sample Chapter 2:** "Environment Anatomy" | DOCX or PDF | READY (Manning/output/ch02_env_anatomy.docx) |

**Sample chapter strategy:** Send Ch1 + Ch3.
- Ch1 shows the reader experience from page one: Docker setup, verification, the experiment contract. It proves the book is immediately practical.
- Ch3 is the first algorithm chapter: from-scratch PPO, bridging proof, SB3 scale-up. It proves the Build It / Bridge / Run It structure works and that the author can teach algorithm internals.
- Ch2 is available as backup if the editor wants to see the full Part 1.

### What NOT to send

- Do not send the production protocol or agent prompts (internal tooling)
- Do not send the persona document (internal voice guide)
- Do not send the review notes (internal self-audit)
- Do not reference AI-assisted production in the proposal

---

## Part E: Cover Letter Template

> [AUTHOR: Customize and send as the body of your email to proposals@manning.com]

Subject: Book Proposal -- Robotics Reinforcement Learning in Action

Dear Manning Acquisitions Team,

I am writing to propose **Robotics Reinforcement Learning in Action**, a hands-on guide to training, debugging, and stress-testing goal-conditioned manipulation policies using MuJoCo, Gymnasium-Robotics, and Stable Baselines3.

**The gap this book fills:** The tools for robotics RL are now open-source and accessible. But practitioners who install them and start training face a failure mode unique to RL: everything runs, nothing learns. There is no book that teaches engineers to systematically build, verify, and stress-test goal-conditioned manipulation policies -- from dense rewards through sparse rewards (HER) through pixel observations -- with an auditable experimental record.

**What is attached:**
- Completed book proposal form (13 chapters across 5 parts, ~400 pages)
- Sample Chapter 1: "Proof of Life" -- environment setup and verification
- Sample Chapter 3: "PPO as a Lie Detector" -- first algorithm chapter with from-scratch implementation

**What makes this proposal ready:**
- 3 chapters fully written, reviewed, and produced as PDF/DOCX
- 6 chapters have tutorial source material ready for adaptation
- A complete companion repository with training scripts, evaluation pipelines, lab modules, and Docker infrastructure
- 9 from-scratch algorithm implementations (PPO, SAC, HER, curriculum, pixel wrappers) already coded and tested

I would welcome the opportunity to discuss this proposal with your editorial team.

Best regards,
[Your name]
[Your email]
[GitHub repo URL]

---

## Part F: Prioritized Action Items

### Must-do before submission (blocking)

- [ ] **1. Write author bio and credentials** (Section 1 above). This is the #1 gap. Manning needs to know who you are and why you should write this book. Even 4-5 sentences suffice.
- [ ] **2. Fill in contact details** (Section 8). Name, email, GitHub URL.
- [ ] **3. Set realistic timeline dates** (Section 9). Pick a proposal submission date, then count forward.
- [ ] **4. Decide on public GitHub repo URL.** Manning reviewers will want to see the companion code. If the repo is currently private, decide when to make it public (can be after contract signing, but mention it in the proposal).
- [ ] **5. Transfer proposal into Manning's DOCX template.** Download `ManningBookProposal.docx` from manning.com/write-a-book/proposal, fill in each section using the draft text above.
- [ ] **6. Write a marketing/promotion paragraph.** 3-5 sentences: How will you reach readers? Conferences you attend, communities you are part of, blog/social presence, employer support for promotion. Manning cares about this.

### Should-do before submission (strengthening)

- [ ] **7. Expand competition section with online courses.** Add 2-3 online courses (Coursera, Udacity, fast.ai) and explain the differentiation. This is already drafted in Section 6 above.
- [ ] **8. Add 1-2 sentence learning objectives to each ToC entry.** This is partially done in Section 10 above for all 13 chapters. Transfer into the final ToC.
- [ ] **9. Review sample chapters for platform-neutral tone.** The review notes flag DGX-specific language. A quick pass replacing "DGX" with "Linux + NVIDIA GPU" would help.
- [ ] **10. Generate key figures.** Run `bash docker/dev.sh python scripts/capture_proposal_figures.py all --out-dir figures` to produce the self-generated figures referenced in the proposal. Embed 2-3 in the sample chapters.

### Nice-to-have (after submission)

- [ ] **11. Draft Ch4 (SAC) and Ch5 (HER)** to demonstrate momentum when the acquisitions editor asks "how fast can you produce chapters?"
- [ ] **12. Prepare a 1-page "from scratch definition" document** for reviewer Q&A. Not submitted initially, but ready if an editor asks "what do you mean by from scratch?"
- [ ] **13. Set up MEAP-ready infrastructure.** Ensure the build pipeline (build_book.py) produces Manning-compatible DOCX that can feed into their LiveBook system.

---

## Part G: Anticipating Editor Questions

Based on Manning's review process and the TuneTheWeb author report, here are likely questions from the acquisitions editor and suggested responses:

**Q: "How is this different from Deep Reinforcement Learning in Action?"**
A: Zai & Brown (2020) is an excellent general deep RL book covering Atari, grid worlds, and multiple algorithm families. Our book is specifically about *robotics manipulation* with goal-conditioned interfaces -- a domain with its own observation structure (dictionary obs with achieved/desired goals), action semantics (continuous Cartesian deltas), and evaluation needs (success rate, not just return). We also cover HER for sparse rewards, which is essential for practical robotics and not covered in their book.

**Q: "Your ToC has 13 chapters -- is that realistic?"**
A: Parts 1-3 (Ch1-6) are fully backed by existing tutorials and code. Parts 4-5 (Ch7-13) are outlined but less developed. We propose launching MEAP with Ch1-6, giving us 6+ months to develop the engineering and pixels chapters. Chapter 12 is explicitly marked as optional/advanced and could move to an online appendix if scope needs to shrink.

**Q: "What do you mean by 'from scratch'?"**
A: We build the core algorithm mechanics from scratch in PyTorch -- loss functions, gradient updates, replay buffers, HER relabeling logic -- and verify each component with concrete numerical checks. We do *not* reimplement Gymnasium, MuJoCo, or the training loop infrastructure. SB3 is used for production-scale runs. A "bridging proof" section in each algorithm chapter demonstrates that our from-scratch code and SB3 compute the same values on the same data.

**Q: "Does the reader need expensive GPU hardware?"**
A: No. Every chapter offers five compute tiers, from a 2-minute CPU sanity check to an overnight multi-seed GPU experiment. Readers without a GPU can use provided pretrained checkpoints for evaluation chapters. The first 6 chapters are feasible on a modern laptop with CPU only.

**Q: "How will you promote the book?"**
A: [AUTHOR: Fill in. Examples: "I will present at [conference], share on [social platform], contribute to [community]. My employer [does/doesn't] support technical writing. The companion GitHub repository will serve as ongoing promotion."]
