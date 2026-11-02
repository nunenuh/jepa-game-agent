# Design note: a learned recipe, and knowing for itself when a stage is done

Status: **approved 2026-11-02** (as drafted; steps below in progress). Results so far: [experiments/results](../experiments/results/README.md).

## The problem

Two things in the chain are given to the agent, not learned (round 8 onward):

| given today | where | what it should be |
|---|---|---|
| the recipe: wood × 2 → table → wood × 1 → wood pickaxe → stone | read by hand from `craftax_classic/game_logic.py`, typed on the command line | learned from recorded games |
| "this stage is done" | `_stage_done` in `evaluation/runner.py` reads the game's inventory and achievement flags **while playing** | the agent's own reading of its own latent |

The second one breaks the rule "pixels only at play time". `AGENT_ARCHITECTURE.md` lists the counted stages as an
open item, but the plain stages (`place_table`) read the game's achievement flags too, so the gap is wider than
that item says. Every chain result so far says the recipe is given; this note closes both.

## The idea in one picture

```
OFFLINE, once, from recorded games (labels allowed, like any readout head)

  achievements + inventory ──► 1. what comes before what?   (order in the games)
  in each recorded game    ──► 2. how much is needed?       (fewest items ever held at the unlock)
                           ──► 3. what is really needed?    (drop what only happens to come first)
                                        │
                                        ▼
                            recipe: an ordered list of stages, saved to a file

PLAYING, every step, pixels only

  latent ──► small "progress" head ──► wood count, table placed? ...
                                        │ stage done?
                                        ▼
                            next stage's goal pictures go to the planner
```

- **Order:** count how often achievement A comes before B across recorded games, keep A → B when A nearly always
  comes first, then trim to the minimal graph. This is SEA's heuristic.
- **Counts:** for each unlock, the fewest of each item held just before it, over all games. Enough games make this
  the true cost; too few overestimate it, never underestimate it. What drops at the unlock shows the price (a table
  costs wood).
- **Filter:** order alone is fooled by habit: if the PPO player always sleeps early, `wake_up` looks like a
  prerequisite of the table. SEA reports the same false edges. Two filters: keep only what the target's item chain
  reaches (items consumed or tools required), and test the rest against the recorded games with random actions mixed
  in (epsilon), where habits are weaker.
- **Progress head:** the same kind of head as the vitals readout, trained on replayed labels: latent → item counts
  and "table placed". The status bar is on screen, so this is reading, not cheating. v3's status-bar encoder read
  the inventory at R² 0.997; v4's is **not measured yet**, and the check below comes first.
- **Scoring stays with the game:** the runner still reads the true flags to score results. It just stops telling
  the agent.

## Prior work this follows

- **SEA** ([arXiv 2305.00508](https://arxiv.org/abs/2305.00508), Crafter): counts which achievement precedes which,
  trims to a minimal graph, and warns about false edges from correlation. Steps 1 and 3 come from here.
- **MSGI** (Sohn et al., ICLR 2020): infers a subtask precondition graph from experience with inductive logic.
  Heavier than we need now; kept as the fallback if the filter isn't enough.
- **Achievement Distillation** ([arXiv 2307.03486](https://arxiv.org/abs/2307.03486), Crafter): contrastive learning
  of achievement structure inside a PPO agent. Not used: it relies on reward signals, which our world model avoids.
- **Grammar-based skill discovery** ([ICML 2026](https://icml.cc/virtual/2026/poster/63476), on Craftax): segments
  *unlabelled* games into skills. The way to drop the labels in step 1 later.

I found no paper doing exactly this for a planner on Craftax, so the combination is ours to test.

## Fairness

- Achievement and inventory labels are used **offline only**, to build the recipe and train the head. **This widens
  the rule a little**: today labels only train small heads and score results; here they also build the recipe.
  Every result will say so: "recipe learned from labelled games". Learning it with no labels is a later step.
- Nothing from the game reaches the agent while it plays.
- The world model is untouched and stays reward-free.

## How we measure it

First, does it work? (CPU only; no GPU units)

| check | bar |
|---|---|
| recipe from the recorded games vs the hand-written one | same order and counts |
| false stages (`wake_up` and the like) after the filter | 0 |
| progress head on v4, held-out: wood count right | ≥ 95% of frames |
| progress head on v4, held-out: "table placed" AUC | ≥ 0.95 |
| a stage called done too early or too late | ≤ 5% of unlocks, within 2 steps |

Then, does it still play? Same chain as the given-recipe run, same 50 worlds, so the results can be compared world
by world. With 50 worlds, a single rate carries about ±14 points, so small differences won't count as real.

| check | given recipe, real flags | bar |
|---|---|---|
| table, wood × 1, pickaxe | Nov 4 runs | no clear drop |
| wrong stage switches | none (flags are exact) | reported |

Matching the given recipe is the win here: the same play, with nothing handed over.

## Risks

- **Too few games reach deep stages:** counts for rare unlocks get overestimated. Only go as deep as the data
  supports, and report how many games each number rests on.
- **A wrong "done" derails the chain:** one misread moves the agent to the next goal too early. Require the reading
  to hold for a few steps.
- **Imagined vs real:** the head only reads real frames here, never imagined ones, so model drift doesn't touch it.
- **The status bar can blur in v4:** health read poorly at first (R² 0.52 with this head, 0.74 with a random forest).
  If wood counts do too, read them with neighbouring tokens, as the threat head does, or use v3 as a second model
  through `--readout-model`.

## What building it taught

- Stricter than first drafted: a true prerequisite never fails, so one game without it disproves it ("nearly always"
  would keep habits).
- Run on real games, it exposed the unlock flag bug: Craftax's flags only fire when the game ends (see the results).

## Steps

1. Tests first: order, counts and filter on made-up games with a known recipe, including a fake habit the filter must
   drop.
2. `jga learn-recipe`: recorded games → recipe file. Laptop for code; the real shards on a CPU runtime.
3. Progress head and its check on v4 (`jga train-progress`).
4. `plan-achievement --recipe FILE --progress HEAD`; the given recipe and real flags stay the default, so old runs
   still reproduce.
5. Colab: the chain with both, compared to the given recipe.

## Decisions to approve

- Labels may build the recipe offline (the fairness change above), declared in every result.
- Order + counts + filter first; MSGI-style inference only if the filter fails.
- The agent decides "stage done" from its own head; the game only scores.
- The bars above.
