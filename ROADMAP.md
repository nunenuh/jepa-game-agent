# Roadmap

Progress tracker. The full plan, with measures and risks: [docs/PLAN.md](docs/PLAN.md).
Interfaces and tests per part: [docs/SPECS.md](docs/SPECS.md).

## Foundation (done)

- [x] Package `src/jepa_game_agent`: envs (GridWorld, Craftax), data, world model, CEM planner, memory, surprise, resumable training, CLI
- [x] 31 unit tests on CPU (~99% coverage), Craftax wrapper tested against the real game
- [x] ruff, mypy --strict, pytest with coverage gate; CI on Python 3.10 and 3.12; pre-commit

## Phase 0: design note (next)

- [x] [`docs/DESIGN.md`](docs/DESIGN.md) drafted: observation, goal format, evaluation tasks and baselines, what to reuse from jepa-playground 09
- [x] Approved (2026-10-27): own PPO data policy, goals K = 5/10/20, status bar kept

## Phase 1: data and baselines

- [x] Vectorised Craftax collector (`jax.vmap`), replay-exact; `jga collect` prints a data card
- [x] Fixed: Craftax auto-reset planted a new world at every episode end
- [ ] ~1M Craftax-Classic transitions (random + PPO policy), data card
- [x] Evaluation tasks (reach K steps, achievements), random / still / replay baselines, `jga baselines`
- [x] Fixed: Craftax's batched reset gave some seeds a different world (replay broke); trivial reach tasks skipped
- [x] PPO player code: Craftax_Baselines driver + `.npz` export + `jga collect --policy --epsilon` (smoke-tested locally)
- [ ] Train PPO on Colab A100, collect ~1M transitions, data card, baselines on PPO data

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
