# Posting Strategy Notes (PRIVATE - do not commit)

## General Strategy

Post every two weeks, alternating between platforms.

## LinkedIn

- Upload the robot GIF as native video (better engagement)
- Post Tuesday-Thursday, 8-10am local time
- Respond to every comment in first 2 hours
- Don't edit post after publishing (kills reach)
- Tag zooplus SE company page if possible

## Reddit

- Post on weekday mornings (US time) for best visibility
- Don't post to multiple subreddits on the same day (looks spammy)
- Schedule: r/reinforcementlearning Monday, r/MachineLearning Wednesday, r/robotics Friday
- Engage genuinely with comments--this builds reputation
- If post gets traction, cross-post to r/learnmachinelearning the following week

### r/MachineLearning variant

Emphasize problem formulation philosophy. Use this title:

**Title:** [P] Teaching RL through robotic manipulation: an open-source curriculum emphasizing problem formulation

Adjust body to focus on:

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

### r/robotics variant

Emphasize the Fetch robot and simulation. Use this title:

**Title:** Open-source RL curriculum using simulated Fetch robot (MuJoCo)

Adjust body to focus on:

**Tasks covered:**
- FetchReach (move end-effector to position)
- FetchPush (push object to target)
- FetchPickAndPlace (pick up and place object)

**Why Fetch?**

The simulated Fetch matches the kinematics of the real Fetch Mobile Manipulator (7-DOF arm, parallel-jaw gripper). Policies trained in simulation can potentially transfer to hardware, though sim-to-real isn't covered in this curriculum.

## Substack

- Publish on Tuesday mornings (highest open rates)
- Include a clear call-to-action at the end
- Cross-promote to LinkedIn same day
- Enable comments for engagement
