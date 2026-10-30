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

## Phase 3 round 6: change goals (`--goal-as change`, code 6009cd7)

Goal = the current latent + the mean before → after step of recorded unlocks (64 train-split pairs per
achievement), rebuilt every step: "make the change that unlocking X makes".

Diagnostic (`jga action-effects --goal-as change`, v2, same val transitions): the action really taken is
the one closest to the goal for wood 97%, table **99%** (picture goal: 2%), pickaxe 99%, stone 84%.

Chain (v2, 50 worlds, 400 steps):

| | wood | table | pickaxe / stone |
|---|---|---|---|
| random | 50% | 14% | 0% |
| hier + prior, picture goal (round 5) | 66% | 4% | 0% |
| hier, change goal | 4% | 0% | 0% |
| hier + prior, change goal | 6% | 0% | 0% |
| flat, change goal | 4% | 0% | 0% |
| flat + prior, change goal | 4% | 0% | 0% |

The two goal types fail in mirror image. A change goal picks the right action once the agent stands at
the right spot, but far from a tree "one step more wood-like" points nowhere, so the agent stops
travelling (wood 66% → 4%). A picture goal gets the agent there but is the wrong target for the final
action. Next: use both, the picture goal to travel and the change goal for the action (e.g. a cost that
adds the two, or switching to the change goal once it is reachable within the planning horizon).

## Phase 3 round 7: picture + change goals combined (code e61348b)

`--goal-as both`: each step, first plan for the change; play that plan if it gets within reach ×
|change|² of it, otherwise travel toward the picture. `--goal-as blend`: one goal halfway between the two.
All runs: v2, action prior, chain, 50 worlds, 400 steps.

| | wood | table | pickaxe / stone |
|---|---|---|---|
| random | 50% | 14% | 0% |
| hier, picture goal (round 5) | 66% | 4% | 0% |
| hier, both, reach 0.25 | 66% | 4% | 0% |
| hier, both, reach 0.5 | 62% | 6% | 0% |
| hier, both, reach 0.75 | 56% | 0% | 0% |
| hier, blend | 18% | 0% | 0% |
| flat, both, reach 0.5 | 32% | 4% | 0% |

Switching keeps travel intact but tables do not move. **Cause found in the game rules, not the agent:**
in Craftax-Classic a table needs and costs 2 wood, and a wood pickaxe needs 1 more wood at a table
(`craftax_classic/game_logic.py`: `place_block`, `wood >= 2`, `wood - 2`). The chain moves on to the
table at the first wood (the collect_wood unlock), so in most worlds the table is impossible; the few
tables come from worlds that picked up a second wood on the way. The recorded unlock moments used by the
diagnostic always had enough wood, so the diagnostic could not see this. Random play keeps hitting trees
and so gathers the 2 wood. Next: chain stages that count items (wood × 2 → table → wood × 1 → pickaxe).

## Phase 3 round 8: correct recipe, GIVEN (read from the game code; code 346dd09)

Recipe written by hand from `craftax_classic/game_logic.py`: wood × 2 → table (costs 2 wood) → wood × 1 →
wood pickaxe (needs a table nearby) → stone. Counted stages read the real inventory (`collect_wood:2`).
This is outside knowledge given to the agent, not learned. v2 + action prior, 50 worlds; share reaching
each stage:

| | wood ×2 | table | wood ×1 | pickaxe | stone |
|---|---|---|---|---|---|
| random (single achievement, 400 steps) | — | 14% | — | 0% | 0% |
| hier, picture goal, 400 / 800 steps | 32% | 8% | 6% | 0% | 0% |
| **hier, both (reach 0.5), 400 / 800 steps** | 32% | **32%** | 18% | **2%** | 0% |
| flat, both (reach 0.5), 400 steps | 4% | 4% | 0% | 0% | 0% |

With a correct recipe and picture-for-travel + change-for-action, **every world that gathered 2 wood
placed a table** (32 of 32%), above random's 14% for the first time, and one world made a wood pickaxe
(the first ever). Remaining walls: the second wood (32%: trees vanish when chopped, so it needs another
tree) and returning to the table with the third wood (18% → 2%). 400 and 800 steps give identical
results: most likely the agent dies (hunger, thirst, zombies) before step 400, to be checked by logging
episode lengths. Fairness: this recipe is given; next it gets learned from the recorded games.

## Phase 3 round 9: travel to the "before" picture; episode lengths (code 91d746a)

Same given recipe, v2 hier + prior, `--goal-as both`, 50 worlds, 400 steps. `--travel before`: travel
toward the mean of the frames just before each unlock (facing a tree; next to a table with wood) instead
of just after.

| | wood ×2 | table | wood ×1 | pickaxe | stone | episode length median (min–max) |
|---|---|---|---|---|---|---|
| random (single achievement) | — | 14% | — | 0% | 0% | |
| travel after, reach 0.5 (round 8 rerun) | 32% | 32% | 18% | 2% | 0% | 151 (36–266) |
| **travel before, reach 0.5** | **46%** | **38%** | 10% | 0% | 0% | 124 (37–323) |
| travel before, reach 0.25 | 48% | 2% | 0% | 0% | 0% | 130 (37–323) |
| travel before, reach 0.75 | 24% | 24% | 6% | 0% | 0% | 134 (38–349) |

The round 8 run reproduced exactly (same seeds, deterministic planner). Travelling to the "before" picture
gets more worlds to 2 wood (32% → 46%) and the best table rate so far (38%, random 14%). Reach 0.25 is too
strict: the agent gathers wood but rarely judges the table change reachable.

**Every episode in every run ended with the player dying** (`done`, 200 of 200), at a median of 124–151
steps, never running out of the 400-step budget. That is why 400 and 800 steps gave identical results:
the agent has about 130 steps to craft. Survival, not planning, now caps the chain. Next: log health,
food, drink and energy at death to find the cause (zombies, lava, thirst), then handle it.

## What kills the agent (vitals at death, `CraftaxEnv.vitals`, code 15cba36)

The two round 9 setups (travel before / after, reach 0.5) rerun with the player's state recorded at the
end of each episode: 100 episodes, 100 deaths.

| | zombie adjacent, health 0 | thirst / hunger / fatigue | lava | median length |
|---|---|---|---|---|
| travel before | 50 / 50 | 0 | 0 | 124 |
| travel after | 49 / 50 | 1 | 0 | 151 |

**Zombies kill the agent**: 99 of 100 deaths had health 0 with a zombie next to the player. No lava
deaths. Thirst plays a part late (drink 0 in 11–12 of 50, mostly the longer episodes) but is almost never
the cause. Deaths happen in daylight and at night alike (light < 0.5 in 20–25 of 50). Stage rates matched
round 9 exactly (deterministic). The planner has no reason to avoid zombies: its only cost is the goal
distance. Next: the survival cost (`--readout`, `--survival`, code dbd3f41), which penalizes imagined
futures whose read-out health drops, so avoiding or fighting zombies comes from the world model's own
predictions.

## Survival cost, first try (`--readout`, `--survival`, code 7c2118d)

**Body sense** (`jga train-readout` on v2 latents, 300 recorded games replayed for labels, 60 held out):

| | health | food | drink | energy |
|---|---|---|---|---|
| R² (held-out games) | 0.33 | 0.35 | 0.68 | 0.31 |

The v2 latent barely encodes the status bar, like the inventory before the HUD encoder (wood 0.59 → 0.997
with v3). So the readout's sense of health is weak.

**Chain with the survival cost** (round 9 best setup: v2 hier + prior, `--goal-as both --travel before
--reach 0.5`, given recipe, 50 worlds, 400 steps):

| survival λ | wood ×2 | table | wood ×1 | pickaxe | stone | length median (mean) | deaths, zombie adjacent | drink 0 at death |
|---|---|---|---|---|---|---|---|---|
| 0 (round 9) | 46% | 38% | 10% | 0% | 0% | 124 | 50 / 50 | 11–12 |
| 1 | 42% | 38% | 12% | 0% | 0% | 144 (144) | 49 / 50 | 6 |
| 3 | 38% | 34% | 6% | 0% | 0% | 172 (151) | 48 / 50 | 16 |
| 10 | 28% | 24% | 6% | 0% | 0% | 136.5 (147) | 50 / 50 | 16 |

**No real gain.** Every episode still ends in death (150 of 150), almost always with a zombie next to the
player. Mean length rises only ~124 → 145–151 steps, and a stronger penalty takes steps away from crafting
(table 38% → 24% at λ 10). With R² 0.33 the planner cannot tell an imagined hit from noise. Next: a health
readout the world model can actually support: read the status bar with the HUD encoder (v3 HUD dims or a
HUD stream added to v2), then retry the survival cost.

## Survival cost, round 2: fixed labels (code 95d6f1b)

**Label bug found:** replaying a recorded game drifts from the recording after 72–251 steps (Craftax's random
mob events differ), so most vitals labels in the first try described a different game than the frames.
`replay_vitals` now returns the replayed frames with their labels and stops when that game ends.

| readout R² (60 held-out games) | health | food | drink | energy |
|---|---|---|---|---|
| v2, old labels | 0.33 | 0.35 | 0.68 | 0.31 |
| **v2, fixed labels** | **0.62** | 0.56 | 0.78 | 0.56 |
| v3 (HUD encoder), fixed labels | 0.14 | 0.41 | 0.82 | 0.00 |

v3's HUD dims read the inventory (0.997) but not the health hearts. `--readout-model` lets a second world
model imagine the same actions for the readout while v2 plans (used for the v3 rows below).

| survival (round 9 setup, 50 worlds) | wood ×2 | table | wood ×1 | pickaxe | stone | length median (max) | died, zombie adjacent |
|---|---|---|---|---|---|---|---|
| none (round 9) | 46% | 38% | 10% | 0% | 0% | 124 | 50 / 50 |
| v2 readout, λ 1 | 42% | 32% | 10% | 2% | **2%** | 134 (337) | 49 / 50 (1 full chain) |
| v2 readout, λ 3 | 28% | 24% | 8% | 0% | 0% | 149 (288) | 50 / 50 |
| v3 readout, λ 1 | 44% | **42%** | 10% | 2% | 0% | 143 (393) | 50 / 50 |
| v3 readout, λ 3 | 34% | 26% | 6% | 0% | 0% | 135 (304) | 50 / 50 |

**First full chain** (wood → table → wood → pickaxe → stone) in 1 of 50 worlds, and the best table rate
(42%), but both are within a couple of worlds of round 9: not a clear gain. Every other episode still ends
with a zombie next to the player, at a median of 134–149 steps. A health cost reacts only after the hit
(pain, not danger), and a stronger one costs crafting steps. Next: a learned critic that predicts health
about to be lost (`jga train-readout --danger N`, `--critic`, `--danger`, code 83f04cf).

## Danger critic diagnosis (code 83f04cf, cached replays)

The critic as first trained (MLP, 5000 steps, "health lost in the next N steps") scored held-out R² −0.52
(N 10) and −0.48 (N 20): it memorized the training games. On 600 cached replays (130k frames, 1 in 5
games held out):

| critic variant | N = 5 | N = 10 |
|---|---|---|
| health lost, MLP 2000 steps | R² −0.12 (train 0.26) | R² −0.21 (train 0.41) |
| health lost, MLP 300 steps | R² 0.01 | R² 0.02 |
| health lost, linear | R² 0.04 | R² 0.06 |
| hit soon yes/no, MLP 2000 steps | AUC 0.69 (train 0.92) | AUC 0.68 (train 0.93) |
| **hit soon yes/no, MLP 300 steps** | **AUC 0.72** | **AUC 0.73** |
| hit soon yes/no, linear | AUC 0.70 | AUC 0.72 |
| health now (true value) → danger | R² 0.04 | R² 0.04 |

The v2 latent carries only a hint of danger: even a linear head reaches AUC 0.70 and nothing passes 0.73.
Danger comes from the surroundings (zombies), not the health bar, and the pooled 1-vector latent blurs small
sprites (phase 2: facing). Next: v4, a spatial latent (small grid of latents, as in V-JEPA 2-AC and
DINO-WM), still our own reward-free JEPA trained from scratch on Craftax; then the critic again.

## v4 spatial latent: can it see? (code 68fa6e2, design note docs/DESIGN_V4.md)

Training (30k steps × batch 128, window 4, inverse weight 0.5, same PPO train split as v2; 40 min on A100):
prediction 0.043 (v2 0.091; per tile token, not directly comparable), SIGReg 1.76 (v2 6.7), inverse-dynamics
CE **0.84** (v2 1.08): no collapse, and v4 tells better which action was taken.

**Ceiling check first:** fitting raw pixels → vitals directly (random forest, 80 replayed games, 1 in 5 held out)
gives health, food, drink R² 1.00 (energy 0.92): health is plainly visible and the labels are right, so earlier
low readout scores were the latents dropping it.

Same 80 games (17k frames), random forest per representation, 1 in 5 games held out:

| read off | raw pixels (ceiling) | v2 latent | **v4 latent** |
|---|---|---|---|
| zombie adjacent (AUC) | 0.99 | 0.65 | **0.84** |
| zombie within 3 tiles (AUC) | 0.81 | 0.65 | **0.74** |
| hit in the next 10 steps (AUC) | 0.73 | 0.66 | **0.73** |
| health (R²) | 1.00 | 0.16 | **0.74** |

On the 600 cached replays (MLP / linear heads as in the critic diagnosis): danger AUC v4 0.71–0.75 vs v2
0.70–0.73; vitals R² v4 health 0.62 / food 0.51 / drink 0.85 / energy 0.57 (linear) vs v2 0.67 / 0.62 / 0.82 /
0.67 (best MLP).

**The design note's danger bar (0.85) was unreachable:** "hit in the next 10 steps" is random enough that even
raw pixels reach only 0.73, and v4 reaches that ceiling. v4 does see more: adjacent zombies 0.65 → 0.84 and
health 0.16 → 0.74 (random forest), though below the pixels' 0.99 and 1.00. JEPA keeps less of what it can't
predict (health drops and zombie moves are random). Next: play with v4 (retrain the action prior and top model
on it, then the chain with the survival cost and critic).

## v4 plays: flat reach (code 74df24d)

Same 100 reach tasks per K as phases 2–3 (test split), flat planner, default CEM settings:

| reach | K = 5 | K = 10 |
|---|---|---|
| random | 4% | 4% |
| v2 (inverse dynamics) | 32% | 25% |
| **v4 (spatial latent)** | **51%** | **35%** |

The best flat planning so far: the spatial latent helps planning itself, not only what heads can read off it.
The action prior trained as before (loss 2.11). The top model needed `--width 256`
(`TopConfig.width`, code 74df24d): a transformer as wide as v4's 2592-number latent was too big to train in
reasonable time; the narrow one trained in 7 minutes (final loss 0.23). The v4 health readout as trained by
`jga train-readout` (MLP, 1000 steps) scores R² 0.52 / food 0.52 / drink 0.81 / energy 0.35, below the
random forest's 0.74 on the same latent: the MLP head overfits the 2592 inputs.

## Threat sense on v4 (code 66f719a, fcfb94e)

`jga train-threat`: zombie positions per tile (`CraftaxEnv.zombie_tiles`, placed like the renderer, tested
on the real game) from 300 replayed games train a small head on frozen v4 tile tokens; 1 in 5 games held out.
"Adjacent" = a zombie on one of the 8 tiles around the player (9.6% of frames).

| threat head | adjacent AUC | per-tile AUC |
|---|---|---|
| one token per tile (MLP) | 0.65–0.72 | 0.83–0.87 |
| one token per tile, positives weighted 10× | 0.86 | 0.93 |
| **3×3 neighbourhood of tokens (conv), official** | **0.967** | **0.975** |
| random forest on the whole v4 latent (earlier) | 0.84 | — |
| raw pixels (ceiling, earlier) | 0.99 | — |

The v4 encoder spreads a zombie over neighbouring tile tokens, so each tile must be read with its
neighbours. Read this way, v4 sees adjacent zombies almost as well as the pixels do. In the planner,
`--threat-weight λ` adds λ × (imagined zombie probability on the 8 tiles around the player), summed over each
imagined path. Labels only train the head; behaviour still comes from planning.
