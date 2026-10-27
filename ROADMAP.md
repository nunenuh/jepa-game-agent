# Roadmap

Each phase ends with something you can run and a number that says whether it
worked, including when it didn't.

## 0. Design note (next)

Pick the exact Craftax version (Classic or full), the observation (pixels or
symbolic), and the success measures (achievements unlocked, steps to a goal,
surprise on impossible events). Check what LeWM / H-JEPA code can be reused
from jepa-playground 09 and stable-worldmodel. Approve before any training.

## 1. Data and a baseline

- Generate Craftax episodes on Colab (random + a simple scripted or PPO policy)
- Baselines: random actions, and a standard RL agent's achievements (from the Craftax paper)

## 2. A flat JEPA world model (LeWM) on Craftax

- Train LeWM on Colab A100 (a few hours)
- Plan short goals ("get wood") with CEM, as in jepa-playground 09
- Measure: success on short goals; surprise on impossible events (teleport, items appearing)

## 3. Hierarchy (H-JEPA)

- 2–3 levels, each predicting farther ahead (about a day of Colab, in resumable chunks)
- Measure: long goals ("make a stone pickaxe") flat vs hierarchical

## 4. Memory and its own goals

- Memory bank of past latents (where was the water, the table)
- Curiosity: go where surprise is high; goals from the user or an LLM turned into goal latents

## 5. Speed and a live window

- A fast policy trained in the JEPA's imagination, so it acts in real time
- Live window on the laptop: watch it play, give it goals, mess with its world

## Not realistic here

Minecraft-scale 3D worlds, V-JEPA 2 / RoboJEPA-scale models: beyond Colab.
