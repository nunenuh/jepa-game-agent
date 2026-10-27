# Specs

What each part of `src/jepa_game_agent` does, its interface, and how it is
tested. "Built" parts exist and are tested; "planned" parts are specified
here first and built in their phase.

## Data formats

| Thing | Shape / type | Notes |
|---|---|---|
| Frame | `(H, W, C)` uint8, 0–255 | Craftax-Classic 63×63×3; Craftax 130×110×3 |
| Episode | `frames (T+1, H, W, C)` uint8, `actions (T,)` int64, `rewards (T,)` float32 | frame t is before action t; the last frame is after the last action |
| Episode file | one `.npz`: concatenated `frames`, `actions`, `rewards`, plus `lengths` | `save_episodes` / `load_episodes` |
| Training sample | `frames (W+1, C, H, W)` float in [0, 1], `actions (W,)` | `WindowDataset`, every window inside one episode |
| Latent | `(D,)` float, D = 128 by default | LayerNorm-ed encoder output |
| Checkpoint | `torch.save` dict: `step`, `model`, `opt`, `model_cfg`, `history` | written atomically (temp file, then rename) |

## Built (phase 0–2 foundation)

### `config` — validated settings
`ModelConfig(num_actions, latent_dim=128, channels=3, context=16,
predictor_layers=4, predictor_heads=4, sigreg_weight=0.09)`,
`TrainConfig(steps, batch_size, window, lr, weight_decay, checkpoint_every,
seed)`, `PlannerConfig(horizon, samples, elites, iterations, smoothing,
seed)`. Frozen dataclasses; invalid values raise `ValueError` on creation.

### `envs` — game worlds behind one interface
`GameEnv` protocol: `num_actions`, `frame_shape`, `reset(seed) -> Frame`,
`step(action) -> StepResult(frame, reward, done, info)`.
- `GridWorld`: tiny deterministic pixel grid world (agent, one item,
  `set_agent` to teleport). For tests and CPU debugging.
- `CraftaxEnv`: Craftax pixel envs through JAX (optional extra
  `[craftax]`); frames converted to uint8; `info` holds the 22 (Classic)
  achievement flags.
- `make_env(name)`: `"gridworld"` or a Craftax pixel env name.

Tests: determinism, frame shape and dtype, walls, reward, episode end, bad
actions rejected; Craftax wrapper checked when the extra is installed.

### `data` — episodes
`collect_episode(env, policy, max_steps, seed)`, `random_policy(n)`,
`save_episodes`, `load_episodes`, `WindowDataset(episodes, window)`.
Tests: shapes, validation, save/load round trip, windows never cross
episodes.

### `models` — the JEPA world model
- `Encoder(channels, latent_dim)`: 3 strided convs → adaptive pool 4×4 →
  linear → LayerNorm. Works for any frame size.
- `Predictor(num_actions, latent_dim, context, layers, heads)`: causal
  transformer; input `latent_t + action_embedding_t + position_t`, output t
  predicts latent t+1.
- `sigreg(z)`: LeJEPA's Epps–Pulley statistic on random projections (same
  form as jepa-playground 03–05).
- `WorldModel(cfg)`: `encode`, `loss(frames, actions) -> Losses(total,
  prediction, sigreg)` with `total = MSE(pred, next latent) + w · SIGReg`,
  trained end to end (no EMA, no stop-gradient: SIGReg prevents collapse, as
  in LeWM); `rollout(start, actions, history=None) -> (B, K, D)` imagined
  latents; `surprise(frames, actions) -> (B, T)` per-step error.

Tests: encoder on 16², 63², 130×110 frames; **causality** (changing future
inputs never changes earlier outputs); SIGReg separates Gaussian from
collapsed embeddings; gradients reach every parameter; rollout shapes and
context windowing; training lowers the prediction loss.

### `planning` — search over imagined futures
`categorical_cem(cost_fn, num_actions, cfg) -> (best sequence, cost)`: one
categorical distribution per step, sample `samples` sequences, keep
`elites`, move the distribution toward their action frequencies
(`smoothing`), `iterations` rounds; returns the cheapest sequence seen.
`plan_to_goal(model, now, goal, cfg)`: cost = squared distance between the
final imagined latent and the goal latent.

Tests: finds a known optimal sequence; rejects a malformed cost; with an
exact hand-made model, plans a sequence that lands exactly on the goal.

### `memory` — remember latents
`LatentMemory(dim, capacity)`: ring buffer; `add(z, step, info)`;
`nearest(query, k) -> [(cosine similarity, Memory(step, info))]`.
Tests: nearest lookup, metadata kept, oldest overwritten when full, shape
checks.

### `surprise` — "that can't happen" scores
`SurpriseMeter(decay, warmup)`: exponential running mean and variance of
prediction error; `update(error)` returns how many standard deviations
above normal (0 during warm-up). Tests: normal errors stay low, a
teleport-sized error scores > 10, bad input rejected.

### `training` — resumable training
`train(model, data, cfg, checkpoint, device)` trains to `cfg.steps` total,
resuming from `checkpoint` if it exists, saving every `checkpoint_every`
steps and at the end; `load_checkpoint(path)` rebuilds the model from its
saved config. Tests: loss goes down; a 10-step run continued to 15 steps
gives one unbroken history.

### `cli` — `jga`
`jga collect --env NAME --episodes N --max-steps M --out FILE` and
`jga train --data FILE --num-actions A --steps S --checkpoint FILE`.
Tests: collect then train end to end on the grid world.

## Planned

### Phase 1: `policies` and evaluation
- `data.collect_craftax_batch`: many Craftax games stepped together with
  `jax.vmap` (the one-game `CraftaxEnv` is too slow for ~1M transitions),
  writing the same episode format.
- `policies.PPOPolicy`: load a Craftax PPO checkpoint for data collection.
- `evaluation.tasks`: named tasks with a start, a goal and a step budget
  (`reach_tree`, `collect_wood`, `place_table`, `make_wood_pickaxe`,
  `collect_stone`, ...), each judged by Craftax achievement flags or
  positions, not by latent distance.
- `evaluation.run(agent, tasks, episodes, seed) -> results dict` written to
  JSON. Every task reports a random-action baseline.

### Phase 2: probes and impossible events
- `evaluation.probes`: linear / MLP probes from latents to player position,
  inventory counts, health (R² on held-out episodes).
- `evaluation.impossible`: teleport, item appears, item vanishes; surprise
  AUC vs normal steps.

### Phase 3: `hierarchy`
- `HierarchicalWorldModel(levels)`: level k predicts `stride_k` steps
  ahead on latents pooled over `stride_k` steps; planning top-down, each
  level's plan turned into goal latents for the level below.
- Optional `goals.llm`: goal text → list of achievement subgoals (any LLM
  API, behind an interface so tests use a fake).

### Phase 4: memory in the loop and curiosity
- `agent.Agent`: perceive → update memory and surprise → choose goal (user,
  LLM, memory, or curiosity) → plan → act.
- Curiosity goal: the remembered latent with the highest recent surprise
  that hasn't been visited lately.

### Phase 5: fast policy and live window
- `policies.ImaginedPolicy`: small network trained on world-model rollouts
  to imitate the planner's first action.
- `live`: pygame window (as jepa-playground 09 `live.py`): the game, the
  goal, memory hits, a surprise graph; keys to give goals and teleport.

## Quality bar (every phase)

- `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` with ≥ 80%
  coverage pass in CI on every push.
- New code comes with unit tests; slow or GPU-only tests get a marker and
  are skipped in CI.
- Results files are JSON with the config and the code commit that made them.
