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
