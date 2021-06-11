Title: Fundamentals of Reinforcement Learning.
Date: 06-10-2021
Category: reinforcement learning, note
Tags: reinforcement learning
Slug: ReinforcementLearning
Authors: toohsk
Summary: Rough note of reinforcement learing.

# Rough note of reinforcement learing.

This is a note of the Udemy's [Modern Reinforcement Learning: Deep Q Learning in PyTorch course](https://www.udemy.com/course/deep-q-learning-from-paper-to-code/), chapter 2.<br>
This note is just for helping my understanding so if you want correct information, please take a look above course.<br>
It is hard and also I'm still in the path of this course but I highly recommend to take this!

There are a lot of equations and I hand write them, but I won't upload them.

And records of my studying is uploaded to [GitHub repository](https://github.com/toohsk/udemy-deep-q-learning-from-paper-to-code), so please stay tune.

# Notes

## What's Reinforcement Learning?

Interactions between agent and environment.

- Agent learns and act(decision making)
- Environment gives states and rewards
- Rewards tell the agent what is good and bad

The biggest difference between Reinforcement learning and Supervised learning is, in the Reinforcement learning agent also makes decisions from the action space based on the current state but it affect all future rewards.

Actions cause state transitions


    p(s', r | a, s) != 1

s: state
r: reward
a: action

this equation defines probabilities of given next state (s’) and reward(r) when agent have taken an action (a) in the state (s).

And summation of probabilities get 1.

    Sigma( p(s', r | a, s) = 1

Total rewards are expression as G, and total rewards in time step t is expressed as

    G_{t} = R_{t+1} + R_{t+2} + ... + R_{T}

In reinforcement learning, tasks are expected episodic, which it stands for there is final state in the task.
But in some tasks continues to infinity means as forever. And in this case, rewards towards to infinity, too.<br>
To deal with this problem, rewards are fixed by discounting.
Discount factor is expressed with Gamma, γ.
When gamma equals to 1, the agent values all future rewards equally and interests in play long game.
When gamma equals to 0, the agent is interests in the short games, it place no values for all future rewards.<br>
Gamma needs to be in the range of `0 <= γ <= 1`, and in the practice it is set in range of `0.95 <= γ <= 0.99`.
And the discounted total rewards are expressed as

    G_{t} = γ^0 * R_{t+1} + γ^1 * R_{t+2} + γ^2 * R_{t+3} + ... + γ^{T-1} * R_{T} = Sigma(γ^k * R_{t+k+1})

Policy is mapping of states to actions.
And it represented as Pi, π.

## Value Functions, Action Functions

- Every policy has a v and q, and those obey the Bellman equation.
- Policies can be ranked with v and q.
    - better value is better policy
- Bellman optimality equation is recursive
    - v and q functions has next state, s’, and next action, a’

## About Q Learning

Q Learning is based on model free approach.
It is trying to estimate optimal p.

For estimating optimal p, there are two approaches.

1. model based approach
2. model free approach

The model free approach is the way to learn from trial and error, and figure out the best action to get best reward.

## The Explore-Exploit Dilemma

Then How to learn & get maximize rewards?

Taking the best actions by the agent is known as greed.
And taking sub optimal action is known as exploration.
Balancing the two operations is a dilemma.

For this solution is bring another parameter, called Epsilon Greedy.
The epsilon greedy is a hyper parameter for selecting greedy actions or exploration actions.
Epsilon would be decreased over time step, so that the agent would take random actions for first, but go through the steps greedy actions would be more selected by decreasing it.
Epsilon must stay finite.
For practice, epsilon settle on `0.01 <= ε <= 0.1`, and this means the agent spends from 1% to 10% chance to take exploratory actions.

## Temporal Difference Learning

Estimate of value of the policy and needs to refine it.
Temporal Difference Learning is online learning.
At the boot strap using one estimate to update another.

## Q Learning Algorithm

The Q Learning is a tabular and off policy method.

Basic algorithm is shown as below.

1. Initialize Q for all states s and action a
2. Initialize parameters: alpha=0.001, gamma=0.9, epsilon_max=1.0, epsilon_min=0.01
3. Repeat for n_episodes
    1.   Initialize state s
    2. For each step of episode
        1. Choose action a with epsilon-greedy
        2. Perform action a, get new state s’ and reward r
        3. Apply Q function and update Q(s,a)
        4. update status s to s’ by s=s’

Point to implement Q Learning

- Agent should be a class
    - Class has
        - Q table
        - Learning rate
        - discount factor
        - epsilon
        - initializing function the Q table
        - choosing action function based on current state
    - Represent the Q table as a dictionary.
        - states and actions as the key
        - reward would be value
- Decrease epsilon over time to minimum value.
