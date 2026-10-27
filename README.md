# jepa-game-agent

A JEPA agent that **lives inside a game world**: it sees the screen, predicts
what its actions will do in latent space, remembers, picks its own goals, and
acts. JEPA as the *player's mind*, not as a game generator.

Spin-off of [jepa-playground](https://github.com/nunenuh/jepa-playground),
where experiment 09 showed what a JEPA world model can and can't do yet:

- It **plans**: LeWorldModel pushes the T to the goal in Push-T (74–78%),
  with no reward.
- It **knows physics**: its surprise catches teleported, frozen and reversed
  motion (AUC 0.93–1.00).
- It **fails on long tasks**: whole episodes 2/21, its own goals 1/20.
- **Hierarchy helps**: H-JEPA in Visual AntMaze, 20% flat → 80% with 3 levels.
- It is **slow** (~1 s per move), has **no memory** and **no goals of its own**.

This repo adds one piece per failure:

| Missing | Piece |
|---|---|
| Long tasks | H-JEPA hierarchy + an LLM that breaks a task into steps |
| Speed | a fast policy trained inside the JEPA's imagination (Dreamer / LeWAM style) |
| Memory | a memory bank / map of past latents |
| Own goals | curiosity from surprise + goals from the user or an LLM |

The world: [Craftax](https://github.com/MichaelTMatthews/Craftax), a 2D
Minecraft-like game written in JAX: fast enough to generate millions of
frames on one GPU, rich enough to need long plans (wood → table → pickaxe →
stone → ...).

Status: **planning**. See [ROADMAP.md](ROADMAP.md).

## Hardware

Training on Google Colab (A100 40 GB at most, in resumable 3–6 h chunks);
playing and inference on a laptop GPU (RTX A2000, 4 GB, Docker).
