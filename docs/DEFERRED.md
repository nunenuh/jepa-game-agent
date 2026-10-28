# Deferred work

Items consciously postponed, with why and what doing them takes. Pick one up
when it blocks something or when there is spare Colab time.

## From phase 2 (deferred 2026-10-28)

Phase 2's main targets are met (reach K = 5 / 10 clearly above random,
facing encoded after the inverse-dynamics loss, surprise AUC 0.998; see
[experiments/results](../experiments/results/README.md)). These checks were
left for later because none of them blocks phase 3.

### Harder surprise events (~1–2 h)
- **Why:** the current test jumps ≥ 50 steps away, which changes most of the
  view, so AUC 0.998 is the easy version.
- **What:** small teleports (1–5 tiles, same game, same moment) and
  impossible inventory changes (wood appears without chopping). Both need
  the true game state: `CraftaxEnv.state.replace(...)` then `set_state`, run
  on the CPU so replays match the recordings.
- **Done when:** AUC reported per event type; target > 0.9 (DESIGN D7).

### Probes for vitals (~1 h)
- **Why:** health, food, drink and energy decide survival; we haven't checked
  that the latent keeps them.
- **What:** add the vitals to the probe targets (they are in the Craftax
  state but not in `state_key`), re-run `jga probe` on v1 and v2.
- **Done when:** R² per vital reported; target > 0.8 (DESIGN, phase 2 done).

### Open-loop `Recall@1` (~1 h)
- **Why:** measures how far imagined rollouts stay recognisable: roll the
  predictor K steps from a real start, is the nearest real latent the true
  one?
- **What:** a function over test-split windows, K = 1, 5, 10, 20.

### Crafter / Craftax achievement score (folded into phase 3)
- **Why:** the standard benchmark number (shifted geometric mean over the 22
  achievements).
- **What:** needs an agent that pursues achievements, not goal pictures;
  that is phase 3's achievement-goal planner, so it is measured there.
