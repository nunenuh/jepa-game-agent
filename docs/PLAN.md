# Plan

The goal: a JEPA agent that lives inside a game world. It sees the screen,
predicts what its actions will do in latent space, remembers where things
are, picks goals, and acts, fast enough to watch it play live.

It grows out of [jepa-playground](https://github.com/nunenuh/jepa-playground)
experiment 09. Each phase below fixes one failure seen there, and each ends
with a number that says whether it worked. Failures get reported, not hidden.

## Ground rules

- **Training on Colab** (A100 40 GB at most, sessions of 3–6 h, so every
  training run checkpoints and resumes). **Playing and inference on the
  laptop** (RTX A2000, 4 GB, Docker).
- **Every result has a baseline** (random actions at least) and raw numbers
  in a results file.
- **Code lives in `src/jepa_game_agent`**, is tested (`pytest`, 80% coverage
  minimum), type-checked (`mypy --strict`) and linted (`ruff`) in CI.
  Experiments are thin scripts or notebooks in `experiments/` that call the
  package.
- A phase starts only after the previous one's result is written down.

## The world: Craftax-Classic first

[Craftax](https://github.com/MichaelTMatthews/Craftax) (MIT, JAX) is a 2D
Minecraft-like survival game: collect wood, build a table, make a pickaxe,
mine stone, iron, diamonds; eat, drink, sleep, fight. We start with
**Craftax-Classic**: 63×63 pixel frames, 17 actions, 22 achievements, a
10,000-step episode limit. It is fast on a GPU (millions of frames per
hour), small enough for a laptop, and its achievements form a natural ladder
of short and long goals.

Full Craftax (130×110 frames, 43 actions, dungeons) comes later, once
Classic works.

## Phases

### Phase 0: design note (next)

Decide and write down, then get approval:
- observation: pixels (63×63) vs symbolic; we plan pixels (a JEPA should
  learn its own state);
- how goals are given: a goal frame, a goal latent from memory, or an
  achievement name turned into a latent;
- the evaluation tasks for every later phase (below) and their baselines;
- what to reuse from jepa-playground 09 (LeWM recipe, CEM settings, SIGReg
  weight, surprise probes).

**Done when:** `docs/DESIGN.md` is approved.

### Phase 1: data and baselines

- Collect Craftax-Classic episodes with `jga collect`: random policy plus a
  stronger one (a PPO checkpoint from Craftax_Baselines if available, or a
  short PPO run of our own) so the data shows the deeper parts of the game.
  Target: ~1M transitions (~12 GB as raw uint8 frames, far less compressed).
  The current `CraftaxEnv` steps one game at a time; collecting this much
  needs a vectorised collector (`jax.vmap` over many games) first.
- Baselines on the evaluation tasks: random actions; PPO's achievement
  rates.

**Done when:** a dataset on Hugging Face Hub or Drive, a data card (size,
policy mix, achievement counts), baseline numbers.

### Phase 2: a flat JEPA world model

- Train the world model (`WorldModel`: CNN encoder + action-conditioned
  causal transformer, SIGReg) on Colab A100, a few hours in resumable chunks.
- Plan with categorical CEM toward goal latents (`plan_to_goal`).
- Probe what the latent knows (position, inventory, health), as in 09 Part 3.
- Surprise on impossible events: teleport the player, make items appear or
  vanish.

**Measures:** short-goal success (reach a tree, collect wood: tasks that
need under ~20 steps); probe R²; surprise AUC.
**Done when:** short goals clearly beat random (09's Push-T ratio is the
reference: 74% vs 2%), or we know why not.

### Phase 3: hierarchy for long goals

- Stack 2–3 world models, each predicting farther ahead (H-JEPA, as in 09
  Part 5), and plan top-down: the top level picks subgoal latents, the
  bottom level walks.
- Optionally an LLM turns a goal like "make a stone pickaxe" into a list of
  achievement subgoals.

**Measures:** long-goal success (place table, make wood pickaxe, collect
stone: 50–300 steps), flat vs hierarchical.
**Done when:** hierarchy beats flat on long goals (09's AntMaze reference:
20% → 80%).

### Phase 4: memory and its own goals

- `LatentMemory`: remember latents with what was there (water, table,
  enemies) and find them again; plan back to a remembered place.
- Curiosity: when no goal is given, go where the world model is most
  surprised (it learns most there).
- Goals from the user or an LLM, turned into goal latents through memory.

**Measures:** return-to-place success; achievements unlocked per episode
with curiosity vs random exploration; survival time.

### Phase 5: speed and a live window

- Planning costs ~1 s per move. Train a fast policy inside the world
  model's imagination (Dreamer / LeWAM style) that imitates the planner, so
  the agent acts in real time; keep the planner for hard moments.
- A live window on the laptop: watch it play, type a goal, teleport it,
  see its surprise and memory.

**Measures:** actions per second; achievement rate of the fast policy vs
the planner.

## Compute budget (estimates)

| Phase | Where | Time |
|---|---|---|
| 1 data | Colab (JAX on GPU) | 1–2 h |
| 2 world model | Colab A100 | 3–6 h, one or two sessions |
| 3 hierarchy | Colab A100 | about a day, in 3–6 h chunks |
| 4 memory / curiosity | laptop + short Colab runs | hours |
| 5 fast policy | Colab A100 | 3–6 h |

## Risks

- **Craftax frames are small and busy:** the inventory bar and the map share
  63×63 pixels; the latent may ignore the inventory. Check with probes in
  phase 2 before planning on it.
- **Discrete actions:** CEM over 17 actions is a different search from
  09's continuous Push-T; it may need more samples.
- **Long horizons stay hard even with hierarchy** (09: 2/10 AntMaze tasks
  still failed). Phase 3 may end with a partial result.
- **JAX and PyTorch together** on Colab: Craftax uses JAX, the model uses
  PyTorch. Collect data in one process, train in another.

## Not realistic here

Minecraft-scale 3D worlds and V-JEPA 2 / RoboJEPA-scale models need far
more compute than Colab.
