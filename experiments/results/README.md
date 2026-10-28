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
