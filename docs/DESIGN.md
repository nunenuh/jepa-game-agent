# Design note (phase 0)

Status: **approved 2026-10-27** (all three open questions: yes). Decisions for phases 1–2, with the
facts they rest on. Phases 3–5 are designed when they start.

## Facts checked (craftax 1.6.1, Craftax-Classic-Pixels-v1)

- **Frame:** 63×63×3, values in [0, 1] (we store uint8). Top 49 rows: a 9×7
  tile view of the world, 7 px per tile, **centred on the player** (the view
  moves with the player). Bottom 14 rows: status (health, food, drink,
  energy) and inventory icons with counts.
- **Actions:** 17, discrete, one per step (NOOP, 4 moves, DO, SLEEP, 4
  PLACE, 6 MAKE).
- **Achievements:** 22, from COLLECT_WOOD up to COLLECT_DIAMOND and
  MAKE_IRON_SWORD. Reported in `info` and in the state.
- **State:** 64×64 world map, player position and direction, health, food,
  drink, energy, inventory (12 counts), mobs, day/night light level.
  `state.replace(...)` edits it (moving the player 3 tiles shifted the view
  as expected), so impossible events can be staged.
- **Determinism:** replaying the same reset seed and actions through
  `CraftaxEnv` reproduces every frame exactly (checked over a 59-step
  episode). Any start state can be rebuilt from (seed, actions so far);
  datasets don't need to store states.
- **Caveat found in phase 1:** Craftax's reset under `jax.vmap` depends on
  the batch size: in a batch of 32, some seeds (1008, 1021, 1024, ...) get a
  different world than when reset alone. The batch collector therefore
  builds each world with its own reset and only steps the games together;
  every episode of a 32-game batch then replays exactly (tested).
- **No trained policy is released** (Craftax_Baselines ships code only).
- A random policy dies early (the check episode ended after 59 steps) and
  rarely gets past COLLECT_WOOD.

## Decisions

### D1. Observation: the full 63×63 pixel frame
Pixels, not the symbolic observation: the point is a JEPA that learns its
own state from what it sees. Keep the status bar in the frame: inventory and
health are part of the state the agent must know. One frame per step; the
predictor's history (context 16) supplies motion and recent events.

### D2. Actions: one per step, no blocks
LeWM grouped 5 Push-T actions into one block because Push-T runs at 10 Hz
with tiny moves. A Craftax step is already a whole move or action, so the
predictor takes one action per step (an embedding of the 17 actions).

### D3. Data: our own PPO policy plus noise, ~1M transitions
- Train PPO with Craftax_Baselines' `ppo.py` on Colab (estimate: 1 h on an
  A100). Collect with it at ε = 0.2 (20% random actions, for variety), plus
  10% of episodes fully random.
- ~1M transitions; split **by episode** 90 / 5 / 5 (train / val / test).
- Store the reset seed with each episode (`Episode.seed`, done) so any
  step can be replayed.
- Needs the vectorised collector (many games at once with `jax.vmap`)
  first; the one-game `CraftaxEnv` stays for evaluation and play.
- Data card: transitions, episodes, length distribution, achievement
  counts, policy mix.

### D4. Goals: a goal frame, encoded
The planner gets a goal latent from a goal frame, as in LeWM. Because the
view is centred on the player, a goal frame means "the surroundings and the
status bar look like this", not "be at map cell (x, y)". That fits the two
task families below. Position goals ("go back to the water") wait for memory
(phase 4).

### D5. Evaluation tasks

**Family A, reach a state (main measure, LeWM's protocol from 09).**
Start: step t of a held-out test episode (rebuilt by replay). Goal: the
frame at step t + K of the same episode. Budget: 2K steps. K = 5, 10, 20.
Success: the player's position, direction and inventory equal the goal
state's at any step within the budget. 100 starts per K, same starts for
every agent.
**Trivial tasks are skipped** (added in phase 1): a first run on random-play
data found that standing still won 57% of K = 5 tasks, because many random
actions change nothing that counts (walking into a wall, crafting without
materials). Tasks whose goal state equals their start state are dropped;
standing still now scores 0%.
- Baselines: random actions; stand still (NOOP); **replay the episode's own
  K actions** (upper bound, as in 09).

**Family B, unlock an achievement.**
From a fresh reset, budget 200 steps: COLLECT_WOOD, PLACE_TABLE,
MAKE_WOOD_PICKAXE, COLLECT_STONE (later phases add the rest). Goal latent:
the average latent of 16 held-out frames taken right after that achievement
was unlocked (the averaging trick from 09, which washes out the
surroundings and keeps the status bar's new item). Success: the
achievement flag. 50 resets per task.
- Baselines: random actions; our PPO policy's rate on the same resets.
- Expected: flat planning handles COLLECT_WOOD; the longer ones are the
  phase 3 test.

### D6. What the latent should know (probes)
Linear and MLP probes from latents (held-out episodes): inventory counts,
health, food, drink, energy, facing direction, day/night. **Map position
should not be readable** (the view moves with the player), so a high
position R² would be a red flag, not a success.

### D7. Impossible events (surprise)
Staged with `state.replace` at test-episode steps: player teleported 3–5
tiles, an inventory item vanishes, a tree next to the player turns into
grass, the day turns to night in one step. Surprise = the world model's
prediction error on the next frame; AUC against normal steps, as in 09
Part 3.

### D8. Model and training, first version + Phase 2B consideration ladder
The built `WorldModel`: CNN encoder (latent 128) + causal transformer
predictor (4 layers, context 16), SIGReg weight 0.09 (LeWM's), end to end.
Training windows of 8 transitions, batch 128, AdamW 3e-4. Budget: one or two
Colab A100 sessions (3–6 h), checkpoints every 500 steps.

**If the single 128-D vector shows weak inventory/entity probes ($R^2 < 0.8$) or low surprise on small-object glitches (the IntPhys 2 failure mode), we follow this upgrade ladder in Phase 2B:**
1. **Add Inverse-Dynamics Auxiliary Loss (`H-JEPA`, arXiv 2610.06805):** add a lightweight head $g([z_t, z_{t+1}]) \to \hat{a}_t$ trained with cross-entropy alongside `prediction + sigreg`. Because `MAKE_*` and `DO` actions only flip a 7×7 HUD icon or one adjacent tile, predicting $a_t$ forces $z_t$ to keep HUD and foreground state.
2. **Split HUD + Map / Object-Slot Latents (`C-JEPA`, arXiv 2602.11389 / `V-JEPA 2.1`, arXiv 2603.14482):** encode the top 49×63 world view into spatial/entity slots $z_t^{\text{map}} \in \mathbb{R}^{S \times d}$ and the bottom 14×63 status bar into a dedicated HUD token $z_t^{\text{hud}} \in \mathbb{R}^d$, applying object/slot-level masking (`C-JEPA`) so background grass tiles (78% of pixels) cannot dominate the loss.
3. **Slot-Weighted / Optimal-Transport Goal Cost (`Slot-JEPA`, arXiv 2610.28074):** weight $z^{\text{hud}}$ higher for achievement/crafting goals and $z^{\text{map}}$ (or Unbalanced Optimal Transport between entity slots) for spatial reach goals.

### D9. Reused from jepa-playground 09
The evaluation protocol (start, goal K steps later, 2K budget), the
replay-the-expert upper bound, the goal-latent averaging, the probe and
surprise-AUC methods, the SIGReg weight. New here: discrete actions
(categorical CEM instead of Gaussian CEM) and an egocentric view.

## Success for phases 1–2

- **Phase 1 done:** dataset and data card published; PPO reaches at least
  PLACE_TABLE on most episodes (otherwise the data won't show crafting);
  baseline numbers for families A and B.
- **Phase 2 done:** family A at K = 5 and 10 clearly above random (09's
  reference: 74% vs 2%); inventory and status probes R² > 0.8; surprise AUC
  > 0.9 on teleports. Each miss reported with a reason.

## Open questions for approval

1. Train our own PPO on Colab (~1 h A100) for the data policy: OK?
2. Family A goal offsets K = 5, 10, 20 (budget 2K): OK, or start smaller?
3. Keep the status bar in the frame (D1): OK?
