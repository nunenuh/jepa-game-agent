# Roadmap

Progress tracker. The full plan, with measures and risks: [docs/PLAN.md](docs/PLAN.md).
Interfaces and tests per part: [docs/SPECS.md](docs/SPECS.md).

## Foundation (done)

- [x] Package `src/jepa_game_agent`: envs (GridWorld, Craftax), data, world model, CEM planner, memory, surprise, resumable training, CLI
- [x] 31 unit tests on CPU (~99% coverage), Craftax wrapper tested against the real game
- [x] ruff, mypy --strict, pytest with coverage gate; CI on Python 3.10 and 3.12; pre-commit

## Phase 0: design note (done)

- [x] [`docs/DESIGN.md`](docs/DESIGN.md) drafted: observation, goal format, evaluation tasks and baselines, what to reuse from jepa-playground 09
- [x] Approved (2026-10-27): own PPO data policy, goals K = 5/10/20, status bar kept

## Phase 1: data and baselines (done)

- [x] Vectorised Craftax collector (`jax.vmap`), replay-exact; `jga collect` prints a data card
- [x] Fixed: Craftax auto-reset planted a new world at every episode end
- [x] ~1M Craftax-Classic transitions (PPO 901k + random 147k), data card
- [x] Evaluation tasks (reach K steps, achievements), random / still / replay baselines, `jga baselines`
- [x] Fixed: Craftax's batched reset gave some seeds a different world (replay broke); trivial reach tasks skipped
- [x] PPO player code: Craftax_Baselines driver + `.npz` export + `jga collect --policy --epsilon` (smoke-tested locally)
- [x] Train PPO on Colab A100, collect ~1M transitions, data card, baselines on PPO data ([results](experiments/results/README.md): replay 100%, random 4-5%, still 0%)

## Phase 2: flat JEPA world model & 2026 architecture considerations (next)

- [ ] Train baseline 1-vector LeWM on Colab A100 (resumable)
- [ ] Short-goal planning (`SR@K`, `SPL`) & Crafter geometric mean score vs random
- [ ] Probes (inventory, vitals, facing, negative-control map `xy`), open-loop `Recall@1`, and surprise ROC-AUC on impossible events
- [ ] **Phase 2B considerations (if 1-vector blurs HUD/entities):**
  - [ ] Inverse-dynamics auxiliary loss $g(z_t, z_{t+1}) \to \hat{a}_t$ (`H-JEPA`, arXiv 2610.06805)
  - [ ] Split HUD token + map/object-slot encoder & slot masking (`C-JEPA` / `V-JEPA 2.1`)
  - [ ] Factorized HUD vs. map / UOT planning cost (`Slot-JEPA`)

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
