# Results

## `phase1_baselines.json`: baselines on the PPO test split

Run on Colab (2026-10-28), code at 8c166eb (the `commit` field reads `unknown`
because the Colab copy had no `.git`):

```
jga baselines --data test.npz --env Craftax-Classic-Pixels-v1 --k 5 10 20 --tasks 100 \
  --achievement collect_wood place_table make_wood_pickaxe collect_stone --resets 50 --budget 200
```

`test.npz`: 193 PPO episodes, 51k steps (seed % 20 == 0). Unusable reach tasks
(trivial, or the replayed player dies first) were skipped; 100 tasks per K remain.

Reach (success rate, budget 2K):

- K = 5: random 4%, still 0%, replay 100%
- K = 10: random 4%, still 0%, replay 100%
- K = 20: random 5%, still 0%, replay 100%

Achievements (50 fresh worlds, 200 steps):

- collect_wood: random 48%, still 0%
- place_table: random 14%, still 0%
- make_wood_pickaxe: random 0%, still 0%
- collect_stone: random 0%, still 0%

Phase 2 target: the JEPA planner clearly above random at K = 5 and 10.

## Phase 2, first world model: planning on the same reach tasks

Model: flat LeWM (`latent_dim` 128, window 8), 30k steps × batch 128 on the PPO train split
(941,767 transitions, 3,454 episodes), ~35 min on a Colab A100. Final loss 0.83 (prediction 0.11,
SIGReg 8.0). Planner: categorical CEM, horizon 8, 256 samples, 32 elites, 6 iterations, replanning
every step, cost = distance to the goal latent summed over the imagined path. Code at 07e667f.

```
jga plan --checkpoint wm.pt --data test.npz --env Craftax-Classic-Pixels-v1 --k 5 10 20 --tasks 100
```

Reach success (same 100 tasks per K as the baselines above):

- K = 5: JEPA 33%, random 4%, replay 100%
- K = 10: JEPA 12%, random 4%, replay 100%
- K = 20: JEPA 7%, random 5%, replay 100%

Clearly above random at K = 5 (8×), 3× at K = 10, no real gain at K = 20. The median success takes
only 3–4 steps at every K, so the planner mostly wins tasks whose goal ended up close to the start;
longer routes are not solved yet. Reference from jepa-playground 09 (simpler world): 74% vs 2%.

### Planner sweep (same model, K = 5 / 10)

- horizon 8, 256 samples (default): 33% / 12%
- horizon 16: 29% / 11%
- 1024 samples, 64 elites, 8 iterations: 33% / 13%
- both: 26% / 13%

A bigger planner doesn't help: the world model is the bottleneck.

### Linear probes (`jga probe`, 50 random-play episodes × 300 steps, unseen seeds, 1 in 5 held out)

R² of each state field read linearly off the latent:

- facing direction: -0.41 (not encoded; every Craftax action hits the faced tile, and reach goals need it)
- wood: 0.45
- position x / y: 0.35 / -0.24 (expected low: the view is egocentric; the design's negative control)
- sapling: -0.43 (rarely collected, noisy)
- stone, coal, iron, diamond, tools: no score (never collected in the held-out random games)

Likely cause: the encoder pools the 63×63 frame to 4×4, washing out the small player sprite. Next:
inverse-dynamics loss (turning changes only the facing), retrain, probe and plan again.

## Phase 2B step 1: inverse-dynamics loss (`--inverse-weight 0.5`), same data and settings

Training (30k steps): prediction 0.091 (v1 0.107), SIGReg 6.7 (v1 7.8), inverse-dynamics CE 1.08
(chance over 17 actions: 2.83). Code at 6c9148e.

Probes (same protocol), v1 → v2: facing -0.41 → **0.60**, wood 0.45 → 0.59, sapling -0.43 → 0.53,
position x / y 0.35 / -0.24 → 0.28 / -0.05 (negative control, stays low).

Reach (same 100 tasks per K, default planner), v1 → v2:

- K = 5: 33% → 32% (random 4%)
- K = 10: 12% → **25%** (random 4%)
- K = 20: 7% → 9% (random 5%)

Knowing the facing doubles medium-range success; long goals (K = 20) still need phase 3 (hierarchy).

## Surprise test (`jga surprise`, 1000 pairs from the test split, jumps ≥ 50 steps away, code 759c3d5)

ROC-AUC of last-step surprise, jump vs real next frame (target > 0.9):

- v1: **0.997** (mean surprise: real 0.117, jump 1.76)
- v2 (inverse dynamics): **0.998** (real 0.104, jump 1.78)

The model is ~15× more surprised by a teleport than by what really happens next. Caveat: a jump
of 50+ steps changes most of the view, so this is the easy version; small teleports (a few tiles)
or impossible inventory changes are the harder follow-up.

## Phase 3 round 1: two-level hierarchy (code 33267d2)

Top model: 4-step action chunks on frozen v2 latents, 3 seeds (final MSE 0.317 / 0.313 / 0.322). Top plans
5 chunks ahead every 4 steps; the bottom (v2 flat planner) walks to the first imagined chunk.

Reach (same 100 tasks per K; flat K = 10 / 20 from phase 2, K = 40 run fresh):

| K | flat | hier seed 0 / 1 / 2 |
|---|---|---|
| 10 | 25% | 16 / 17 / 15% |
| 20 | 9% | 7 / 6 / 8% |
| 40 | 9% | 7 / 9 / 9% |

Achievements (50 fresh worlds, 200 steps, goal = mean latent of 64 "just unlocked" training frames):

| | random | flat | hier seed 0 / 1 / 2 |
|---|---|---|---|
| collect_wood | 48% | 6% | 32 / 38 / 38% |
| place_table | 14% | 0% | 0 / 2 / 6% |
| make_wood_pickaxe | 0% | 0% | 0% |
| collect_stone | 0% | 0% | 0% |

Success bar (DESIGN_PHASE3 H4) **missed** on both counts: hierarchy does not beat flat at K = 20 (it is
worse at K = 10), and no pickaxe or stone. The one consistent gain is collect_wood, flat 6% → hier
32–38%, still below random. Likely reasons, both risks named in the design: the top imagines subgoals
no real frame matches, and the averaged goal picture is mostly "some view", not "an inventory with X".
Round 2: snap subgoals to real latents; goal pictures that keep the current view and change only the
inventory bar.

## Phase 3 round 2: snapped subgoals, inventory-bar goals (code 8310df9, top seed 0)

Snap: each top subgoal replaced by its nearest of 50k encoded training frames (`--snap-data`).
HUD goal: every step, goal = the current view with the goal frames' bottom 14 pixel rows (`--hud-rows 14`).

Reach (same 100 tasks; McNemar vs flat on the same tasks):

| K | flat | hier (round 1, s0) | hier + snap | p vs flat |
|---|---|---|---|---|
| 10 | 25% | 16% | 24% | 1.00 |
| 20 | 9% | 7% | **13%** (95% CI 8–21%) | 0.45 |
| 40 | 9% | 7% | 5% | 0.39 |

Achievements (50 worlds, 200 steps):

| | flat | hier (round 1, s0) | flat + HUD goal | hier + snap + HUD goal |
|---|---|---|---|---|
| collect_wood | 6% | 32% | 2% | 4% |
| place_table | 0% | 0% | 0% | 0% |
| pickaxe / stone | 0% | 0% | 0% | 0% |

Snapping undoes round 1's loss at K = 10 and gives the best K = 20 so far, but not significantly above
flat. The HUD goal hurts: with the view held fixed, the goal latent sits almost on top of the current
one (the inventory bar is a small part of a pooled 4×4 latent), so the planner has nothing to chase.
Conclusion: planning on one pooled vector cannot see inventory changes well enough; the next lever is
the encoder (phase 2B: a separate HUD token), not the planner.

## Phase 2B step 2 / phase 3 round 3: HUD encoder (code cf308bc)

v3 world model: the bottom 14 pixel rows (inventory bar) get their own small CNN, written to the last 32
of the 128 latent dims; otherwise as v2 (inverse weight 0.5, 30k steps). Final prediction 0.101,
SIGReg 6.6, inverse CE 1.06. New top model on v3 latents (seed 0, final MSE 0.340).

Probes (same protocol), v2 → v3: wood 0.59 → **0.997**, sapling 0.53 → **0.97**, facing 0.60 → 0.69,
position x / y 0.28 / -0.05 → 0.32 / 0.05 (negative control, stays low).

Achievements (50 worlds, 200 steps), goal cost on the 32 HUD dims only (`--goal-on hud`):

| | random | v2 hier (round 1, s0) | v3 flat, HUD cost | v3 hier + snap, HUD cost |
|---|---|---|---|---|
| collect_wood | 48% | 32% | 0% | 16% |
| place_table | 14% | 0% | 0% | 0% |
| pickaxe / stone | 0% | 0% | 0% | 0% |

The encoder now knows the inventory, but a HUD-only cost does not help the planner: it says nothing
about where the tree is, so the cost stays flat until wood is already in hand (a sparse-reward problem
inside the planner).

Same v3 model, whole-picture goals (`--goal-on all`, 200 steps): flat 0% wood, hier + snap 2% wood
(v2 hier: 32%). **v3 plans worse than v2** although its probes are better: the 32 HUD dims (layer-normed,
so as loud as the 96 map dims together) likely dominate the goal distance, and the view gets less room.

## Phase 3 round 4: subgoal chain (`--chain`, code ad3194f)

One run per world (50 worlds, 400 steps): the agent is handed the goal frames of the next achievement not
yet unlocked, in the order wood → table → wood pickaxe → stone. Share of worlds reaching each stage:

| | v3 flat | v3 hier + snap | v2 hier (round 1 top, s0) |
|---|---|---|---|
| collect_wood | 0% | 4% | **54%** |
| place_table | 0% | 0% | 2% |
| pickaxe / stone | 0% | 0% | 0% |

The order alone does not carry the agent through the chain: of the 27 v2 worlds with wood, one placed a
table. A "has a table" goal picture shows a table nearby; nothing in a latent distance says the way there
is one rare action (place_table), so the planner never tries it. (No random baseline at 400 steps yet;
random gets 48% wood in 200.)

### Phase 3 conclusion

- Hierarchy + snapping: best long reach (K = 20: 13% vs 9% flat, not significant); wood up to 32–54%.
- Crafting (table onwards): 0–6% in every variant. Goal-picture planning cannot express "do the rare
  action that makes X", with or without subgoal order or a better inventory encoder.
- Kept: v2 (inverse dynamics) as the planning model; the chain runner and HUD encoder as tools.
- Not tried (deferred): a planning cost that weights view and inventory separately, and action priors
  learned from data (how often an action follows a state) to make rare crafting actions tryable.

## Phase 3 round 5: action prior and balanced cost (code c2054d3)

Action prior: MLP latent → action, trained on the PPO train split (5k steps; cross-entropy 2.26 for v2,
2.23 for v3, chance 2.83); CEM starts from it, blended with 10% uniform (`--prior`). Balanced cost
(`--goal-on mix`): the view's dims and the inventory's dims each sum to one unit.

Chain (50 worlds, 400 steps), share of worlds reaching each stage:

| | wood | table | pickaxe / stone |
|---|---|---|---|
| random (400 steps, single achievement) | 50% | **14%** | 0% |
| v2 hier (round 4) | 54% | 2% | 0% |
| v2 hier + prior | **66%** | 4% | 0% |
| v2 flat + prior | 34% | 4% | 0% |
| v3 hier + snap + mix | 10% | 0% | 0% |
| v3 hier + snap + mix + prior | 22% | 0% | 0% |

The prior helps what players do often (wood: best yet, above random) and balancing partly rescues v3,
but the table stays far below random. Random presses place_table ~23 times in 400 steps; the planner
picks it only if the world model imagines it helping. Even when the prior proposes it, the planner
declines, which points at the world model's prediction of rare actions. Next: check directly whether the
model predicts a table appearing after place_table (frames just before recorded unlocks).

## Diagnostic: does the model know what crafting does? (`jga action-effects`, code b8888ad)

Val-split transitions where a player unlocked each achievement; the model predicts one step ahead for all
17 actions from z_t, and the action really taken is ranked (1 = best, 17 = worst). Next rank: closeness to
the real z_t+1. Goal rank: closeness to the achievement's goal latent (what the planner chases).

| | n | next rank v2 / v3 (top-1) | goal rank v2 / v3 (top-1) |
|---|---|---|---|
| collect_wood | 165 | 1.01 / 1.02 (99% / 99%) | 6.1 / 9.3 (0% / 0%) |
| place_table | 144 | 1.10 / 1.37 (99% / 95%) | 5.2 / 3.3 (2% / 7%) |
| make_wood_pickaxe | 136 | 1.02 / 1.03 (99% / 97%) | 7.9 / 5.0 (0% / 1%) |
| collect_stone | 133 | 1.00 / 1.00 (100% / 100%) | 5.5 / 2.6 (0% / 20%) |
| control (2,000 ordinary steps) | | 4.2 / 4.2 (54% / 54%) | — |

The world model knows what crafting does: the true action is the best match for what really happened
99% of the time, better than for ordinary moves (54%; many moves change little). But the same action is
almost never the one that gets closest to the goal picture (top-1 0–7%, v3 stone 20%). The model is right;
the goal is wrong. An averaged "just unlocked" picture is mostly "some other place", so walking somewhere
beats pressing the craft button. Next: goals as a change instead of a picture.
