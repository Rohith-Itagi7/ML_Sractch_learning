# Artificial Intelligence — Foundations & Practical Understanding

Practical notes focused on understanding the fundamental ideas behind Artificial Intelligence, including agents, rational behavior, search, planning, probability, belief states, utility, and decision-making.

---

## 1. What is Artificial Intelligence?

Artificial Intelligence (AI) is the broad field of building systems that can **perceive, reason, make decisions, and act intelligently** to achieve a goal.

AI is not one algorithm.

AI can use different approaches such as:

- Rules and logic
- Search
- Planning
- Probability
- Decision theory
- Machine Learning
- Deep Learning
- Reinforcement Learning

---

### AI vs ML

**AI** → The broader goal of creating intelligent systems.

**ML** → One approach used to achieve AI by learning patterns from data.

A simple way to remember:

```text
AI
│
├── Classical / Symbolic AI
│   ├── Logic
│   ├── Search
│   ├── Planning
│   └── Knowledge Representation
│
├── Probabilistic / Decision AI
│   ├── Probability
│   ├── Bayes
│   ├── Belief State
│   └── Utility / Decision Theory
│
└── Machine Learning
    └── Deep Learning
```

---

# 2. What is an Agent?

An **agent** is a system that:

1. Perceives its environment
2. Maintains information about the environment
3. Chooses an action
4. Performs the action to achieve a goal

### Examples

- Robot
- Self-driving car
- Trading bot
- Game-playing system
- Software agent

### Basic Agent Loop

```text
Environment
     ↓
 Perception
     ↓
   Agent
     ↓
 Decision
     ↓
  Action
     ↓
Environment changes
     ↓
   Repeat
```

---

# 3. Intelligence

In practical AI, intelligence can be understood as the ability of a system to choose **useful actions to achieve its goals**.

Conceptually:

```python
def choose_action(state):
    # evaluate possible actions
    # choose a useful action
    return best_action
```

AI tries to build systems capable of making such decisions.

---

# 4. Rational Behavior

A **rational agent** chooses the action expected to produce the best outcome according to its goal, performance measure, and available information.

### Example

A robot has two possible paths:

```text
Path A → Short but dangerous
Path B → Slightly longer but safe
```

A rational agent does not blindly choose the shortest path.

It considers what outcome is best according to its objective.

> Rational does not necessarily mean "perfect." It means choosing the best expected action given the available information and objective.

---

# 5. State

A **state** represents the current situation of the environment relevant to the agent.

### Example

```python
state = {
    "agent_position": (2, 2),
    "gold_found": False,
    "possible_pits": [(2, 3), (3, 2)]
}
```

The state tells the agent what it currently knows or represents about the world.

---

# 6. State Space

The **state space** is the collection of possible states that an AI system could encounter.

For a simple game:

```text
State 1
   ↓
State 2 → State 3
   ↓
State 4 → Goal
```

AI algorithms can search through these possible states to find a solution.

---

# 7. Search

**Search** means exploring possible states and actions to find a path toward a goal.

### Example

```text
        Start
          ↓
          A
        ↙   ↘
       B     C
              ↓
             Goal
```

Classical AI uses algorithms such as:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform Cost Search
- Greedy Best-First Search
- A*

---

## A* Search

A* evaluates a state using:

```text
f(n) = g(n) + h(n)
```

Where:

```text
g(n) = cost already travelled

h(n) = estimated cost to the goal

f(n) = estimated total cost
```

Search is useful for:

- Path finding
- Games
- Puzzle solving
- Navigation
- Planning

---

# 8. Optimal Solution

An **optimal solution** is the best solution according to a defined objective.

### Example

Goal: Reach destination

```text
Path A → 20 km
Path B → 15 km
Path C → 18 km
```

If the objective is minimum distance:

```text
Optimal = Path B
```

But "best" depends on the objective.

If safety is more important than distance:

```text
Path A → Safest
Path B → Shortest
```

Then the optimal choice may be Path A.

> Optimality depends on the objective or performance measure.

---

# 9. Planning

**Planning** means determining a sequence of actions that can achieve a goal.

### Example

Goal: Get gold

```text
Plan:

1. Move forward
2. Turn right
3. Move forward
4. Grab gold
5. Return
```

Planning is used in:

- Robotics
- Scheduling
- Navigation
- Autonomous systems
- Game AI

---

# 10. Uncertainty

Real environments are not completely known.

An agent may not know:

- What is actually happening
- What will happen after an action
- Where an object is
- Whether its sensor is correct

Instead of saying:

```text
Cavity = True
```

the system may represent:

```text
P(Cavity | Toothache) = 0.8
```

Meaning:

> Given the available information, the probability/belief of a cavity is 0.8.

Probability therefore gives AI a way to represent uncertainty.

---

# 11. Probability Theory in AI

Probability allows an agent to represent how likely different possibilities are.

### Example

```text
Rain = 0.7
No Rain = 0.3
```

The agent can use these probabilities when deciding what to do.

Probability is especially useful when:

- Information is incomplete
- Sensors are uncertain
- Multiple outcomes are possible
- The agent must make decisions despite uncertainty

---

# 12. Bayes' Rule

Bayes' Rule allows an agent to **update its belief when new evidence arrives**.

### Basic Form

```text
P(A | B) = P(B | A) P(A) / P(B)
```

### Practical Idea

```text
Old belief
    ↓
New evidence
    ↓
Bayesian update
    ↓
New belief
```

### Example

A medical AI initially has one belief about a disease.

Then it receives:

```text
Symptom
   +
Test result
   ↓
Bayesian update
   ↓
Updated belief
```

Bayesian reasoning can therefore be used for belief updating under uncertainty.

---

# 13. Belief State

When an agent cannot know the exact state of the world, it can maintain a **belief about possible states**.

### Example

Possible pit locations:

```text
(1,2) → 50%
(2,1) → 50%
```

The agent does not know exactly where the pit is.

So it maintains:

```text
Belief State
```

A belief state can be thought of as:

> "What does the agent currently believe about the possible states of the world?"

This is important for **partially observable environments**.

---

# 14. Utility

Probability tells us **what might happen**.

Utility tells us **how desirable an outcome is**.

### Example

```text
Action A:
80% chance → ₹100 profit
20% chance → ₹50 loss

Action B:
50% chance → ₹500 profit
50% chance → ₹200 loss
```

The agent needs more than probabilities.

It also needs a way to represent:

- Reward
- Cost
- Risk
- Preference
- Benefit

That value is represented using **utility**.

---

# 15. Decision Theory

Decision Theory combines:

```text
Probability
     +
Utility
     ↓
Decision Theory
```

It helps an agent answer:

> "Given what I believe about the world, and what outcomes I prefer, what should I do?"

This is different from simply predicting something.

---

# 16. Maximum Expected Utility (MEU)

Maximum Expected Utility gives an agent a way to choose between actions when outcomes are uncertain.

### Expected Utility

```text
Expected Utility
=
Σ Probability(outcome) × Utility(outcome)
```

The agent chooses the action with the **highest expected utility**.

### Decision Process

```text
Action A
   ↓
Possible outcomes
   ↓
Probability × Utility
   ↓
Expected Utility


Action B
   ↓
Possible outcomes
   ↓
Probability × Utility
   ↓
Expected Utility
```

Then:

```text
Choose the action with higher expected utility
```

This is a practical decision-making principle for rational agents.

---

# 17. Transition Model

A **transition model** describes how the environment may change after an action.

Conceptually:

```text
Current State
      +
    Action
      ↓
Possible Next States
```

### Example

A robot is at position A.

```text
Move Forward
      ↓
80% → reaches B
20% → slips and reaches C
```

The transition model represents this uncertainty.

This concept becomes especially important in:

- Planning
- Markov Decision Processes (MDPs)
- Reinforcement Learning
- Robotics

---

# AI Foundations — Core Mental Model

The concepts above can be connected into one overall picture:

```text
              ENVIRONMENT
                   ↓
              PERCEPTION
                   ↓
                STATE
                   ↓
        ┌────────────────────┐
        │   What do I know?  │
        │                    │
        │ Probability        │
        │ Bayes              │
        │ Belief State       │
        └────────────────────┘
                   ↓
        ┌────────────────────┐
        │ What can I do?     │
        │                    │
        │ Search             │
        │ Planning           │
        │ Transition Model   │
        └────────────────────┘
                   ↓
        ┌────────────────────┐
        │ What is best?      │
        │                    │
        │ Utility            │
        │ Decision Theory    │
        │ MEU                │
        └────────────────────┘
                   ↓
                 ACTION
                   ↓
              ENVIRONMENT
                   ↓
                 Repeat
```

---

# Key Takeaways

By completing these foundations, you should understand:

- What AI is
- AI vs Machine Learning
- What an agent is
- Intelligence and rational behavior
- States and state spaces
- Search
- Optimal solutions
- Planning
- Uncertainty
- Probability
- Bayes' Rule
- Belief states
- Utility
- Decision theory
- Maximum Expected Utility
- Transition models

The central idea is:

```text
Represent the world
       ↓
Handle uncertainty
       ↓
Update beliefs
       ↓
Evaluate possible outcomes
       ↓
Choose the best action
       ↓
Act
       ↓
Observe again
       ↓
Repeat
```

---

# Next Step

After understanding these AI foundations, the next major area is:

```text
Artificial Intelligence
        ↓
Machine Learning
        ↓
Supervised Learning
        ↓
Unsupervised Learning
        ↓
Deep Learning
        ↓
NLP
        ↓
Transformers
        ↓
LLMs
```

The goal is not only to use AI libraries, but to understand **why these systems work and what happens underneath them**.
