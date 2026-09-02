# Artificial Intelligence — Foundations & Practical Understanding

Practical AI learning notes focused on understanding how intelligent agents perceive, reason, handle uncertainty, make decisions, and act.

---

## 1. What is Artificial Intelligence?

Artificial Intelligence (AI) is the broad field of building systems that can **perceive, reason, make decisions, and act intelligently** to achieve a goal.

AI is **not one algorithm**.

AI can use different approaches such as:

- Rules and logic
- Search
- Planning
- Probability
- Decision theory
- Machine Learning
- Deep Learning
- Reinforcement Learning

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

## 2. What is an Agent?

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

## 3. Intelligence

In practical AI, intelligence can be understood as the ability of a system to **choose useful actions to achieve its goals**.

Conceptually:

```python
def choose_action(state):
    # evaluate possible actions
    # choose a useful action
    return best_action
```

AI tries to build systems capable of making such decisions.

---

## 4. Rational Behavior

A **rational agent** chooses the action expected to produce the best outcome according to its goal/performance measure and available information.

For example, a robot has two possible paths:

```text
Path A → short but dangerous
Path B → slightly longer but safe
```

A rational agent does not blindly choose the shortest path.

It considers what outcome is best according to its objective.

---

## 5. State

A **state** represents the current situation of the environment relevant to the agent.

Example:

```python
state = {
    "agent_position": (2, 2),
    "gold_found": False,
    "possible_pits": [(2, 3), (3, 2)]
}
```

The state tells the agent what it currently knows about the world.

---

## 6. State Space

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

## 7. Search

Search means exploring possible states/actions to find a path toward a goal.

Example:

```text
Start
  ↓
 A
 ↙ ↘
B   C
    ↓
   Goal
```

Classical AI uses algorithms such as:

- BFS
- DFS
- Uniform Cost Search
- Greedy Best-First Search
- A*

### A*

A* evaluates a state using:

```text
f(n) = g(n) + h(n)
```

Where:

- `g(n)` = cost already travelled
- `h(n)` = estimated cost to the goal

Search is useful for:

- Path finding
- Games
- Puzzle solving
- Navigation
- Planning

---

## 8. Optimal Solution

An optimal solution is the **best solution according to a defined objective**.

For example:

```text
Goal: Reach destination

Path A → 20 km
Path B → 15 km
Path C → 18 km

Optimal = Path B
```

But "best" depends on the objective.

If safety is more important than distance:

```text
Path A → safest
Path B → shortest
```

Then the optimal choice may be Path A.

---

## 9. Planning

Planning means determining a **sequence of actions** that can achieve a goal.

Example:

```text
Goal: Get gold

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

## 10. Uncertainty

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

> Given the available information, the belief/probability of a cavity is 0.8.

Probability therefore gives AI a way to represent uncertainty.

---

## 11. Probability Theory in AI

Probability allows an agent to represent **how likely different possibilities are**.

Example:

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

## 12. Bayes' Rule

Bayes' Rule allows an agent to **update its belief when new evidence arrives**.

Basic form:

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

Example:

A medical AI initially has one belief about a disease.

Then it receives:

```text
symptom
+
test result
```

It can update its belief about the disease.

Bayesian reasoning can therefore be used for **belief updating under uncertainty**.

---

## 13. Belief State

When an agent cannot know the exact state of the world, it can maintain a **belief about possible states**.

Example:

```text
Possible pit locations:

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

This is important for partially observable environments.

---

## 14. Utility

Probability tells us **what might happen**.

Utility tells us **how desirable the outcome is**.

Example:

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

## 15. Decision Theory

Decision Theory combines:

```text
Probability
+
Utility
=
Decision Theory
```

It helps an agent answer:

> "Given what I believe about the world, and what outcomes I prefer, what should I do?"

This is different from simply predicting something.

---

## 16. Maximum Expected Utility (MEU)

MEU gives the agent a way to choose between actions when outcomes are uncertain.

Conceptually:

```text
Expected Utility
=
Σ Probability(outcome) × Utility(outcome)
```

The agent chooses the action with the highest expected utility.

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

Choose the action with higher expected utility
```

This is a practical decision-making principle for rational agents.

---

## 17. Transition Model

A transition model describes how the environment may change after an action.

Conceptually:

```text
Current State
     +
   Action
     ↓
Possible Next States
```

For example:

```text
Robot at position A

Move Forward
     ↓
80% → reaches B
20% → slips and reaches C
```

The transition model represents this uncertainty.

This concept becomes especially important in:

- Planning
- Markov Decision Processes
- Reinforcement Learning
- Robotics

---

# 18. Wumpus World — Putting AI Concepts Together

Wumpus World is a small artificial environment used to understand intelligent agents.

The environment contains:

- Wumpus
- Pits
- Gold
- Agent

The agent's goal is:

```text
Find gold
+
Avoid Wumpus
+
Avoid pits
+
Escape safely
```

The important point is:

**The agent cannot see the entire world.**

It receives percepts from the environment.

Possible percepts include:

```text
Stench
Breeze
Glitter
Bump
Scream
```

For example:

```text
[Stench, Breeze, None, None, None]
```

This is the agent's input/perception.

The agent does not receive:

```text
"The Wumpus is at (1,3)"
```

Instead, it receives clues and must reason about the hidden world.

---

## 19. How Does the Agent Get Input?

The environment generates the world.

For example:

```text
World:

[Safe] [Safe] [Wumpus]
[Safe] [Pit ] [Safe]
[Agent] [Safe] [Gold]
```

The agent has sensors.

When it enters a location:

```text
Environment
     ↓
Sensor
     ↓
Percept
     ↓
Agent
```

For example:

```text
Agent enters (1,2)

Sensor detects:
Stench = True
Breeze = False
Glitter = False
```

The agent receives this information.

It then reasons about what could be nearby.

---

## 20. What Does the Agent Do With the Input?

Suppose:

```text
Agent at (1,2)

Percept:
Stench = True
```

The agent knows from the environment's rules that a stench indicates the Wumpus is in an adjacent location.

Therefore:

```text
Possible Wumpus locations
        ↓
(1,1), (1,3), (2,2)
```

After considering what it already knows, it may eliminate some possibilities.

Eventually it can infer:

```text
Wumpus → (1,3)
```

This is **reasoning**.

---

## 21. The Important AI Architecture

All of these concepts fit into an agent loop:

```text
             ENVIRONMENT
                  ↓
              Percepts
                  ↓
        ┌─────────────────┐
        │      AGENT      │
        │                 │
        │  State          │
        │  Belief         │
        │  Reasoning      │
        │  Probability    │
        │  Utility        │
        │  Decision       │
        └────────┬────────┘
                 ↓
               Action
                 ↓
             ENVIRONMENT
```

So AI is not simply:

```text
Input → Model → Output
```

A decision-making AI system can instead look like:

```text
Observe
   ↓
Represent current state
   ↓
Reason about possibilities
   ↓
Handle uncertainty
   ↓
Update beliefs
   ↓
Evaluate possible actions
   ↓
Choose action
   ↓
Act
   ↓
Observe again
```

---

## 22. Where Does Machine Learning Fit?

This was one of the biggest points of confusion.

ML is **one way of providing intelligence inside an AI system**.

For example:

```text
              AI AGENT
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
   Perception   Reasoning  Decision
       │                     │
      ML                 Utility/
   Model                 Planning
```

ML can learn things that would be difficult to manually program.

For example, a camera receives:

```text
Raw image
```

An ML model can identify:

```text
Car
Pedestrian
Road
Traffic light
```

The AI agent can then use that information to decide:

```text
Brake
Accelerate
Turn
Wait
```

So:

**ML can provide predictions/information.**

**AI uses information to reason, decide, and act.**

---

## 23. Classical AI vs Machine Learning

### Classical AI

Programmer explicitly defines:

```text
Rules
Logic
Search
Planning
Probability models
Utility
Decision procedures
```

Example:

```python
if obstacle_ahead:
    stop()
```

### Machine Learning

The programmer provides:

```text
Data
+
Learning algorithm
+
Objective
```

The model learns patterns.

Example:

```text
Images
   ↓
Training
   ↓
Model

New image
   ↓
Model
   ↓
"Pedestrian"
```

The learned model replaces some manually written rules.

---

# 24. The Most Important Mental Model

Do not think:

```text
AI = Bayes
AI = A*
AI = Probability
AI = ML
```

Instead think:

```text
AI = Building a system that can intelligently
     perceive, reason, decide and act.
```

Different tools solve different parts:

```text
State
  ↓
What is happening?

Probability
  ↓
What might be happening?

Bayes
  ↓
How should my belief change after new evidence?

Belief State
  ↓
What do I currently believe?

Utility
  ↓
What outcomes do I prefer?

Decision Theory
  ↓
How should I choose?

MEU
  ↓
Which action has the best expected outcome?

Action
  ↓
Do something in the environment.
```

---

# 25. Current Understanding

You have now learned the foundations of **classical and probabilistic decision-making AI**:

- Intelligent agents
- Rational behavior
- State
- State space
- Search
- Optimal solutions
- Planning
- Uncertainty
- Probability
- Bayes' Rule
- Belief State
- Utility
- Decision Theory
- Maximum Expected Utility
- Transition Models
- Wumpus World
- Agent–Environment interaction
- How ML can fit inside an AI system

### The Key Distinction

```text
ML:
"Given this data, what do I predict?"

AI Agent:
"Given what I know, what should I do?"
```

And a modern AI system can combine both:

```text
Raw Environment
      ↓
      ML
      ↓
Useful information
      ↓
State / Belief
      ↓
Reasoning + Probability
      ↓
Utility / Decision
      ↓
Action
      ↓
Environment
```

---

# Next

## Machine Learning Foundations

The next step is to understand how machines **learn patterns from data** and how those learned models can become components of intelligent AI systems.

```text
Data
  ↓
Features / Representation
  ↓
Learning Algorithm
  ↓
Model
  ↓
Prediction
  ↓
Evaluation
  ↓
Improvement
```
