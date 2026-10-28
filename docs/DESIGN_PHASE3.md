# Design note: phase 3, hierarchy for long goals

Status: **draft, for approval**. Phase 0's note is [DESIGN.md](DESIGN.md);
results so far in [experiments/results](../experiments/results/README.md).

## The problem

The flat planner imagines one step at a time. It works for short goals and
fades with distance (v2 model, same 100 tasks per K):

- K = 5: 32% (random 4%)
- K = 10: 25% (random 4%)
- K = 20: 9% (random 5%)

Craftax's real goals are longer: a table needs ~5 wood, a wood pickaxe a
table, stone a pickaxe; 50–300 steps. Two things go wrong over long
horizons: imagined rollouts drift (errors compound each step), and the
planner's search space grows as 17^K.

jepa-playground 09 saw the same wall in Push-T and fixed much of it with
hierarchy: H-JEPA ([arXiv 2610.06805](https://huggingface.co/papers/2610.06805))
in Visual AntMaze went 20% flat → 80% with 3 levels.

## The idea in one picture

```
goal ──► TOP planner (coarse model: 1 imagined step = 4 game steps)
           picks the next SUBGOAL latent, ~4 steps away
                     │
                     ▼
         BOTTOM planner (today's flat planner, already 32% at K = 5)
           walks to that subgoal, 1 game step at a time
                     │
                     ▼
         every 4 steps: re-plan the top from where we really are
```

A long goal becomes a chain of short hops, each one the size the flat
planner is good at.

## Decisions to approve

### H1. Two levels, sharing the v2 encoder
- **Bottom:** the v2 model as is (inverse dynamics; facing R² 0.60).
- **Top:** a second predictor that jumps **4 game steps** per imagined step:
  input z_t and the 4 actions taken (an action chunk, embedded and summed),
  output z_{t+4}. Same causal transformer as the bottom, trained on the
  **frozen** v2 encoder's latents.
- **Why frozen:** cheap (no image work, ~10–15 min on the A100), and both
  levels then speak the same latent language, so a top-level prediction is
  directly a goal for the bottom planner.
- A third level (16 steps) only if two levels are not enough.

### H2. Top-level planning
- Categorical CEM over action chunks: horizon 5 chunks = 20 game steps
  (10 chunks for K = 40), cost = distance to the goal summed over the
  imagined path (the fix that worked in phase 2).
- The first chunk's imagined latent becomes the bottom's subgoal. The bottom
  plans to it for up to 4 steps, then the top re-plans from the real frame.

### H3. Achievement goals (the long tasks that matter)
- **Goal latent for an achievement:** the mean latent of training frames
  right after that achievement unlocked (DESIGN D4 already allows several
  goal frames, averaged). "Have a wood pickaxe" then looks like the status
  bar with a pickaxe in it.
- Tasks: collect_wood, place_table, make_wood_pickaxe, collect_stone from 50
  fresh worlds, budget 200 steps (same as the phase 1 baselines: random
  48% / 14% / 0% / 0%).
- The Crafter achievement score (deferred from phase 2) is measured here.

### H4. What we compare (same tasks, same budgets)
- Reach K = 10, 20, 40: flat v2 vs hierarchical.
- Achievements: random vs flat v2 vs hierarchical.
- **Done when:** hierarchical beats flat clearly at K = 20 (target ≥ 2×,
  i.e. ≥ 18%) and on at least one achievement random can't do
  (make_wood_pickaxe or collect_stone > 0%). Each miss reported with a
  reason.

### H5. Not in this phase
- LLM subgoals ("make a stone pickaxe" → wood, table, pickaxe, stone): an
  option once H3 works; behind an interface so tests use a fake.
- Learned latent macro-actions (full H-JEPA style): only if action chunks
  are too crude.
- Search over remembered real latents (a graph of the memory bank): a phase
  4 alternative to the top predictor.

## Build order (test-first, as in phase 2)

1. `ChunkPredictor` + training on frozen latents (`jga train-top`), unit
   tests on GridWorld (a 4-step jump is learnable there).
2. `HierarchicalAgent` (top CEM → subgoal → bottom CEM), tested with an exact
   two-level model on GridWorld: must solve reach tasks the flat agent with a
   short horizon cannot.
3. Achievement goal latents + `jga plan --achievement`.
4. Colab: train the top (~15 min), evaluate flat vs hierarchical (~40 min).

## Risks

- **Subgoals the bottom can't reach:** the top imagines a latent no real
  frame matches. Mitigation: snap the subgoal to the nearest real training
  latent (cheap with `LatentMemory`).
- **Achievement goal latents too vague:** averaging many frames may blur to
  "anything with a pickaxe". Mitigation: weight the status-bar part (the
  Phase 2B slot-cost option in SPECS).
- **Still drifting at K = 40:** add the third level.

## Open questions for approval

1. Two levels with a 4-step top, sharing the frozen v2 encoder (H1): OK?
2. Achievement goals as averaged "just unlocked" frames (H3): OK?
3. Done-when targets in H4: OK, or stricter / looser?
