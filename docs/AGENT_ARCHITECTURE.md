# Agent architecture: what the agent has, what it is missing

The JEPA is the agent's world model: the part that understands the game and imagines what an action
will do. It is the core, but not the whole agent. We follow the six modules of LeCun's agent design
(*A Path Towards Autonomous Machine Intelligence*, 2022).

```
              configurator (sets the task: which subgoal now)
                  |                 \
perception --> WORLD MODEL (JEPA) <--> actor (plans by imagining futures)
                  |            ^          |
               memory        cost  <------+
                       (built-in: pain, hunger + learned critic)
```

## Inputs

The world model gets exactly two inputs: the 63x63 game frame (pixels) and the action (1 of 17).
No reward, no game numbers, no text, no other model's output. Health, food, drink, energy and the
inventory exist only as icons in the frame's bottom rows (the status bar).

Other models sit around it: the PPO player only recorded the training games; the top model, the
action prior and the planner all read the JEPA's latents; the recipe is a hand-written list.

## The six modules, today (2026-10-30)

| Module | Role | In this repo | State |
|---|---|---|---|
| Perception | senses → latent | JEPA encoder (pixels → z) | built |
| World model | predicts the next latent, imagines futures | JEPA predictor + top model (4-step chunks) | built: knows what crafting does 99% of the time; surprise AUC 0.998 |
| Actor | finds actions that lower the cost | categorical CEM + action prior | built |
| Cost | what is good or bad | distance to the goal (picture travel + change goal) | **half: no survival cost** |
| Memory | remembers states and places | none | **missing** |
| Configurator | sets subgoals for the task | the recipe, written by hand from the game code | **given, not learned** |

Missing "body sense" belongs with cost: the agent sees its health only as pixels, it was never checked
whether the latent encodes health, food or drink, and nothing tells the planner that losing health is bad.
Every agent so far dies at a median of about 130 steps (phase 3 round 9), before the chain can finish.

## What to add, in order

1. **Body sense:** small readout heads, latent → health, food, drink, energy (labels from replaying the
   recorded seeds, used only for training the heads).
2. **Survival cost:** add to the planner's cost a penalty on imagined futures whose read-out health
   drops; drink or eat when low. This is LeCun's built-in ("intrinsic") cost.
3. **Memory:** keep latents of trees, the table and stone the agent has seen; plan back to the nearest.
4. **Learned configurator:** learn the recipe's order and counts from the recorded games (the inventory
   just before each unlock) instead of writing it by hand.
5. Later: a fast policy trained to copy the planner (speed).

## Fairness rules

- While playing, the agent acts from pixels only: no game state, no inventory numbers. (Open item: the
  counted recipe stages still read the real inventory to decide when a stage is done; this moves to the
  agent's own readout.)
- Game-state labels only train small readout heads and score results.
- The world model stays reward-free: it is trained only to predict the next latent.
- Every result says what was given (recipe, labels) and what was learned.
- No hand-coded behaviour ("if zombie, run left"): behaviour comes from planning with the world model.
