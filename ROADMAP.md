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

## Phase 2: flat JEPA world model & 2026 architecture considerations (main targets met)

- [x] Train baseline 1-vector LeWM on Colab A100 (resumable): 30k steps, ~35 min
- [x] JEPA planning agent + `jga plan`: reach K=5 33% / K=10 12% / K=20 7% vs random 4/4/5% ([results](experiments/results/README.md))
- [ ] Short-goal planning (`SR@K`, `SPL`) & Crafter geometric mean score vs random
- [x] Linear probes (`jga probe`): facing, inventory, negative-control map `xy`
- [x] Surprise ROC-AUC on impossible jumps: 0.998 (target > 0.9)
- [ ] Deferred (see [docs/DEFERRED.md](docs/DEFERRED.md)): vitals probes, open-loop `Recall@1`, harder surprise events
- [ ] **Phase 2B considerations (if 1-vector blurs HUD/entities):**
  - [x] Inverse-dynamics auxiliary loss (facing R² -0.41 → 0.60; K=10 12% → 25%) $g(z_t, z_{t+1}) \to \hat{a}_t$ (`H-JEPA`, arXiv 2610.06805)
  - [x] Split HUD encoder (inventory R² 0.59 → 0.997; planning worse, see phase 3)
  - [ ] Factorized HUD vs. map / UOT planning cost (`Slot-JEPA`)

## Phase 3: hierarchy (reopened for crafting; see [results](experiments/results/README.md))

- [x] Design note [docs/DESIGN_PHASE3.md](docs/DESIGN_PHASE3.md) approved 2026-10-28
- [x] 2-level world model (top: 4-step chunks), top-down planning, snapped subgoals
- [x] Long goals: flat vs hierarchical (K = 20: 13% vs 9%, n.s.; wood 6% → 32–54%)
- [x] HUD encoder (inventory R² 0.997) and HUD-weighted cost: probes up, planning down
- [x] Subgoal chain (`--chain`, fixed order): no crafting beyond 2% table
- [x] Crafting fixes: action prior, balanced cost, change goals, picture + change combined (`--goal-as both`)
- [x] Correct recipe (given, from the game code: a table costs 2 wood): table 38% with `--travel before`
      (random 14%), first wood pickaxe
- [ ] Agent modules still missing (see [docs/AGENT_ARCHITECTURE.md](docs/AGENT_ARCHITECTURE.md)):
  - [ ] Body sense: readout heads latent → health, food, drink, energy
  - [ ] Survival cost in the planner (every agent dies at ~130 steps today)
  - [ ] Memory of trees, table, stone (moved up from phase 4)
  - [ ] Recipe learned from the recorded games instead of given
- [ ] Optional: LLM subgoals

## Phase 4: memory and its own goals

- [ ] Memory in the agent loop; return-to-place
- [ ] Curiosity from surprise; goals from user / LLM

## Phase 5: speed and a live window

- [ ] Fast policy trained in imagination
- [ ] Live laptop window: watch, give goals, teleport, see surprise

## Phase 6: bridge to a real-world setting

- [ ] Same planner and evaluation on a robot-arm simulator or real-robot data
- [ ] Frozen pretrained V-JEPA 2 encoder + small action-conditioned predictor (V-JEPA 2-AC recipe)
- [ ] Continuous-action CEM; what carries over from Craftax
