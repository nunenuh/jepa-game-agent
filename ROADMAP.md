# Roadmap

Progress tracker. The full plan, with measures and risks: [docs/PLAN.md](docs/PLAN.md).
Interfaces and tests per part: [docs/SPECS.md](docs/SPECS.md).

## Foundation (done)

- [x] Package `src/jepa_game_agent`: envs (GridWorld, Craftax), data, world model, CEM planner, memory, surprise, resumable training, CLI
- [x] 31 unit tests on CPU (~99% coverage), Craftax wrapper tested against the real game
- [x] ruff, mypy --strict, pytest with coverage gate; CI on Python 3.10 and 3.12; pre-commit

## Phase 0: design note (next)

- [ ] `docs/DESIGN.md`: observation, goal format, evaluation tasks and baselines, what to reuse from jepa-playground 09
- [ ] Approved before any training

## Phase 1: data and baselines

- [ ] Vectorised Craftax collector (`jax.vmap`)
- [ ] ~1M Craftax-Classic transitions (random + PPO policy), data card
- [ ] Evaluation tasks and random / PPO baselines

## Phase 2: flat JEPA world model

- [ ] Train on Colab A100 (resumable)
- [ ] Short-goal planning vs random
- [ ] Probes (position, inventory, health) and surprise on impossible events

## Phase 3: hierarchy

- [ ] 2–3 level world model, top-down planning
- [ ] Long goals: flat vs hierarchical
- [ ] Optional: LLM breaks goals into achievement subgoals

## Phase 4: memory and its own goals

- [ ] Memory in the agent loop; return-to-place
- [ ] Curiosity from surprise; goals from user / LLM

## Phase 5: speed and a live window

- [ ] Fast policy trained in imagination
- [ ] Live laptop window: watch, give goals, teleport, see surprise
