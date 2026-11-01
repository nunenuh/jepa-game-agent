# Design note: memory, a map of what the agent has seen

Status: **draft 2026-11-02, for approval.** Results so far: [experiments/results](../experiments/results/README.md).

## The problem

The planner only knows what is on screen now (7 × 9 tiles). Once a tree or the table leaves the view, the
agent has no way back except stumbling on it. The chain shows it: after the table, the agent must go back for
one more wood, and v2 managed that in only 10% of worlds (round 9). Later stages (stone, iron) are rarer
and further away, so this gets worse.

## The idea in one picture

The v4 latent is already one token per tile. A memory can keep those tokens on a map, the way
hippocampal place cells tie what was seen to where.

```
every step:
  1. where am I?   move action + "did my view shift by one tile?"  ──► my position (x, y), from my own latents
  2. remember      the 7 × 9 view tokens, written onto the map at (x + col, y + row)
  3. recall        goal not in view? find the remembered tile most like the goal's tile ──► target (x*, y*)
  4. plan          planner cost += λ × distance from the imagined position to (x*, y*)
```

- **Where am I (path integration):** a move action either moves the player one tile or only turns it (blocked).
  The agent checks which happened in its own latent: does today's token grid match yesterday's shifted by one
  tile, or unshifted? No game state is read. Position starts at (0, 0) each world.
- **Remember:** the map keeps the latest token seen at each world tile (a dictionary, CPU, tiny: a 64 × 64
  world is at most 4096 tokens of 32 numbers).
- **Recall:** the goal picture (e.g. "next to a tree") has a tile next to the player that matters most: the
  one least like ordinary ground (furthest from the average token). Recall returns the nearest remembered
  tile whose token is closest to it, if it is close enough (a threshold set on recorded games).
- **Plan:** the planner already scores imagined action sequences. With a target out of view, each imagined
  path gets a position (its move actions, assumed to succeed) and pays λ × the Manhattan distance left to the
  target. Once the target is in view, this term turns off and the goal-picture cost does the rest.

## Fairness

- Pixels only at play time: position comes from the agent's own actions and latents, never the game's state.
- The map is a data structure, like the CEM planner: no hand-written behaviour ("if no tree, go left"). The
  agent still picks every action by planning; memory only adds a cost term.
- The game's true position is used only to score how well path integration works, never at play time.

## How we measure it

First, does the memory work? (cheap: recorded games, v4 encoder, CPU or a short Colab run)

| check | bar |
|---|---|
| shift test: moved vs blocked, held-out steps | ≥ 95% right |
| position error after 200 steps (vs the game's true position) | ≤ 2 tiles median |
| recall: the remembered "tree" tile really holds a tree | ≥ 80% |

Then, does it play better? Same chain as the Nov 4 runs, 50 worlds:

| check | v4 without memory (Nov 4) | bar |
|---|---|---|
| wood × 1 after the table | to be measured | clearly higher |
| wood pickaxe | to be measured | higher |

## Risks

- **Shift test fails on plain ground** (grass looks the same shifted or not): fall back to the inverse-dynamics
  head's guess, or count the move as taken; errors there matter little because nothing worth recalling is on
  plain ground. Measured by the first check.
- **Recall picks the wrong thing** (a bush that looks like a tree in latent space): the threshold and the
  third check catch it.
- **Darkness at night** changes tokens: recall compares against tokens seen in daylight too; measure first.
- **Distance cost fights the threat cost** (walk past a zombie to reach the tree): that is the planner's job;
  λ is swept (0.5, 1, 2).

## Steps

1. Tests first: shift test on synthetic token grids, map writes at the right world tiles, recall, the distance
   cost on imagined paths, memory off by default (old runs unchanged).
2. Laptop: build `TileMap` (memory/), the shift test, the planner cost term, `--memory λ` on plan-achievement.
3. Colab (after the Nov 4 runs): the "does it work" table on recorded games, then the chain with memory.

## Decisions to approve

- Memory as a tile map on the v4 tokens, position from the agent's own shift test.
- Recall by "least ordinary tile" of the goal picture.
- Memory adds a cost term only (no new behaviour), off by default.
- The bars above.
