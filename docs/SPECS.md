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
  achievement flags, `StepResult.achievements` the unlocked ones. **No
  auto-reset**: Craftax's auto-reset swaps in a new world on the step the
  player dies (found by a test: the game clock read 0 after death), which
  would plant a fake "teleport" at every episode end. `state` / `set_state`
  expose the full game state, to stage impossible events.
- `make_env(name)`: `"gridworld"` or a Craftax pixel env name.

Tests: determinism, frame shape and dtype, walls, reward, episode end, bad
actions rejected; Craftax wrapper checked when the extra is installed.

### `data.craftax_batch` — many games at once (phase 1, built)
`collect_craftax_batch(name, seeds, max_steps, policy=None, chunk=100)`:
one game per seed, each world built by its own reset (Craftax's batched
reset depends on the batch size), then stepped together with `jax.vmap`
inside a jitted `lax.scan`; frames move to the host as uint8 every `chunk` steps. Same key
schedule as `CraftaxEnv`, so every episode replays there from (seed,
actions): the same game, frames equal up to 1/255 (batched rendering can
round a shaded pixel differently). `random_batch_policy(n)`,
`with_epsilon(policy, epsilon, n)` (random action with probability ε).
Tests (craftax marker, own CI job): shapes, replay equivalence, ε rate,
CLI end to end.

### `data` — episodes
`collect_episode(env, policy, max_steps, seed)`, `random_policy(n)`,
`save_episodes`, `load_episodes`, `WindowDataset(episodes, window)`,
`data_card(episodes)` (size, length stats, achievement counts). Each
`Episode` keeps its reset `seed` and the `achievements` it unlocked, in
order.
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

### `evaluation` — tasks, baselines, scoring (phase 1, built)
- `ReachTask(seed, start, prefix, goal_actions)`, `make_reach_tasks(episodes,
  k, n, seed, env=None)`: family A from recorded episodes; with `env`,
  trivial tasks (goal state = start state) are skipped.
- `AchievementTask(achievement, seed, budget)`: family B.
- `GameEnv.state_key()`: what "goal reached" means (Craftax: position,
  facing, 12 inventory counts; GridWorld: position).
- Agents: `Agent` protocol (`reset(goal_frames)`, `act(frame)`),
  `RandomAgent`, `StillAgent`, `ReplayAgent` (recorded actions, open loop).
- `evaluate_reach`, `evaluate_achievement` → JSON-ready dicts (success
  rate, per task successes and steps); `run_baselines` / `jga baselines`
  writes all baselines with the config and git commit.
- Tests: replay reaches 100% (GridWorld and real Craftax), still wins only
  trivial tasks (and none once they're skipped), random is reproducible,
  a scripted walker unlocks the grid achievement.

### `policies.mlp` — the PPO data player (phase 1, built)
Craftax_Baselines' PPO (MIT, pinned commit) trained on the **symbolic**
observation by `experiments/phase1_train_ppo.py`; its actor (an MLP: three
tanh layers + logits) exported to a plain `.npz` with `save_actor_params`.
`load_mlp_policy(path, name)` returns a `BatchPolicy` that reads the
symbolic view of the batched game states (`symbolic_obs_fn`) and samples
an action; the dataset records the pixels of the same states.
`BatchPolicy` now takes `(frames, states, key)`. `jga collect --policy
ppo.npz --epsilon 0.2` collects with it. Tests: only actor layers saved,
logits match a NumPy reference, symbolic obs from pixel states, collection
reproducible, CLI end to end; the driver was smoke-tested with real
(tiny) PPO training on CPU, and its export played through `jga collect`.

### Phase 1 (still to build): batched Craftax evaluator & standard metrics
- `evaluation.craftax_batch_eval`:
  - `batch_reach(name, tasks, agent, budget_factor=2, seed=0)`: replays all
    task prefixes in parallel (each world built with its own reset so the
    batch-size quirk never alters a map, then stepped together with
    `jax.vmap`), checks `(position, facing, inventory)` against the goal at
    every step, and agrees 100% with the one-game evaluator.
  - `nontrivial(name, tasks) -> list[bool]`: batched check that `start_key !=
    goal_key`.
  - `batch_achievement(name, tasks, agent, seed=0)`: batched rollout from
    fresh resets for achievement tasks.
- `evaluation.metrics` (international benchmark standards):
  1. **Crafter / Craftax Shifted Geometric Mean Score** (Hafner, ICLR 2022;
     Matthews et al., ICML 2024; DreamerV3, Nature 2025):
     $$S = \exp\!\left(\frac{1}{N} \sum_{i=1}^{N} \ln(1 + s_i)\right) - 1, \quad s_i \in [0, 100]$$
     over all $N = 22$ Craftax-Classic achievements (percentage of evaluation
     episodes unlocking achievement $i$ at least once).
  2. **Reach Success Rate (`SR@K`) and `SPL` (Success weighted by Path
     Length)** (LeWM 2025; H-JEPA 2026; Anderson et al. 2018):
     $$\text{SPL} = \frac{1}{M} \sum_{j=1}^{M} S_j \frac{K_j}{\max(L_j, K_j)}$$
     where $S_j \in \{0, 1\}$ is whether task $j$ succeeded, $K_j =
     |\text{goal\_actions}|$ is the reference horizon, and $L_j$ is `steps_to_success`.
     Bounded between `StillAgent` (0% on nontrivial tasks) and `ReplayAgent`
     (100%).
  3. **Success–Compute Frontier** (H-JEPA, Zhang et al. 2026): task success
     rate as a function of planner world-model forward passes (tokens / FLOPs)
     and wall-clock latency (ms/step), comparing flat CEM (Phase 2),
     hierarchical CEM (Phase 3), and the amortized fast policy (Phase 5).
  4. **Statistical Reporting (`rliable` protocol)** (Agarwal et al., NeurIPS
     2021): Interquartile Mean (**IQM**) and **95% stratified bootstrap
     confidence intervals** across seeds/episodes, avoiding outlier-sensitive
     plain means.

### Phase 2: representation & world-model diagnostics (LeWM / PLDM / IntPhys 2)
- `evaluation.probes` (I-JEPA / LeWM / PLDM standard):
  - Frozen linear and 2-layer MLP probes trained on held-out episodes to
    predict ground-truth state: 12 inventory counts ($R^2$, Pearson $r$),
    4 vitals (`health`, `food`, `drink`, `energy` — $R^2$), facing direction
    (macro-F1), and light level ($R^2$).
  - **Negative control:** global map `(x, y)` from a single egocentric frame
    must stay near chance ($R^2 \approx 0$) until `LatentMemory` is attached
    in Phase 4.
  - **Inverse-dynamics probe / auxiliary head** (H-JEPA DROID protocol):
    predict action $a_t$ from $(z_t, z_{t+1})$ (top-1 accuracy across 17
    actions) to verify the latent captures fine HUD and tile interactions.
- `evaluation.dynamics` (PLDM / LeJEPA / LeWM standard):
  - **Open-Loop Multi-Step Retrieval (`Recall@1`, `Recall@5`, `MRR`) & Cosine
    Similarity** at horizons $H \in \{1, 5, 10, 20\}$: rank predicted latent
    $\hat{z}_{t+H}$ against true $z_{t+H}$ in a pool of $M = 1{,}000$ held-out
    distractor latents (scale-invariant, cannot be gamed by shrinking norms).
  - **Temporal Latent Straightening** (PLDM, Sobal et al. 2025; LeWM 2025):
    mean cosine similarity between consecutive latent velocities
    $\cos(z_{t+1} - z_t,\, z_t - z_{t-1})$.
  - **Effective Rank (`erank`) & SIGReg Normality Statistic** (Roy & Vetterli
    2007; Balestriero & LeCun, LeJEPA 2025): Shannon entropy of normalized
    singular values $\exp(-\sum_i p_i \ln p_i)$ plus Epps–Pulley test to
    certify zero dimensional collapse.
- `evaluation.impossible` (LeWM / Meta IntPhys 2 / V-JEPA 2 standard):
  - Matched pairs of normal vs. `state.replace`-corrupted transitions (player
    teleported 3–5 tiles, inventory item vanishes/appears, adjacent tree turns
    to grass, instant day-to-night flip).
  - Scored by **ROC-AUC** and **FPR@95%TPR** of per-step prediction error
    $\|\hat{z}_{t+1} - z_{t+1}\|_2^2$ and `SurpriseMeter` z-scores.

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
