# Design note: v4, a spatial latent (one latent per game tile)

Status: **draft 2026-10-30**, for approval. Results so far: [experiments/results](../experiments/results/README.md).

## The problem

v2 squeezes the whole 63×63 frame into one 128-number vector (CNN, average-pool to 4×4, then a linear layer).
Small things get blurred out:

| what the agent needs to see | read off the v2 latent |
|---|---|
| a zombie about to hit (danger critic, held-out) | AUC 0.73 |
| own health (status bar hearts) | R² 0.62 |
| which way the player faces | R² 0.60 |

Zombies cause 99 of 100 deaths, and a survival cost or critic can't avoid what the latent doesn't hold.

## The idea in one picture

A Craftax-Classic frame is a 9×9 grid of 7×7-pixel tiles: 7 rows of the world view plus 2 rows of status bar.
v4 keeps one small latent per tile, the way V-JEPA 2-AC and DINO-WM keep one latent per image patch.

```
frame 63×63 ──► patch encoder (one 7×7 tile → one 32-number token) ──► 9×9 tokens (81 × 32)
                                                                          │
  actions ───────────────────────────────────────────────► predictor: transformer over the tokens of the
                                                            last few frames (block-causal in time) →
                                                            next frame's 81 tokens
```

- **Encoder:** a small CNN applied per tile (stride 7, like ViT patches), then a mixing layer so each token
  also sees its neighbours, then LayerNorm.
- **Predictor:** the tokens of the last 4 frames (context 4, not 16, to keep planning affordable), with the
  action embedding added to every token of its frame; attention can look at earlier frames, never later ones.
- **Training is unchanged:** predict the next latent (MSE) + SIGReg against collapse (on the per-token
  vectors) + the inverse-dynamics loss that fixed facing in v2. No reward, no labels, no reconstruction.
- **Outside the model nothing changes:** `encode` returns the 81 tokens flattened (81 × 32 = 2592 numbers),
  so the CEM planner, picture and change goals, readouts, critic, action prior and top model all keep working.

## How we measure it

First, can it see? This is cheap: reuse the cached-replay readouts, so v4 and v2 are compared on the same games.

| check | v2 | v4 bar |
|---|---|---|
| danger critic, held-out AUC | 0.73 | ≥ 0.85 |
| health readout, held-out R² | 0.62 | ≥ 0.80 |
| facing probe R² | 0.60 | ≥ 0.80 |
| surprise AUC (impossible jumps) | 0.998 | ≥ 0.99 |

Then, does it play better? Same tasks as before, 50 worlds:

| check | v2 today | v4 bar |
|---|---|---|
| reach K = 5 / 10 (flat) | 32% / 25% | no worse |
| chain (round 9 setup + critic): table | 38–42% | ≥ 38% |
| episode length, median | 124–149 steps | > 200 steps |
| wood pickaxe | 0–2% | clearly above 2% |

If v4 sees better but doesn't play better, the next suspect is the cost (2592 numbers, mostly background),
not the eyes.

## Risks

- **Planning compute:** each imagined step predicts 81 tokens instead of 1 vector. Mitigations: context 4,
  mixed precision on the A100; if still too slow, attend within a frame and across time separately.
- **Collapse per tile:** most tiles are grass or sand. SIGReg on the tokens should keep them spread; we
  watch the SIGReg value and per-token variance during training.
- **The goal distance gets dominated by background tiles:** if so, try weighting tiles by how much they change
  between frames, or use the inverse head's features. Only if the play measurements show it.
- **Top model and action prior** must be retrained on the 2592-number latent (a few minutes each).

## Fairness (unchanged)

The world model sees pixels and actions only and is trained to predict its own next latent. It's our own JEPA,
trained from scratch on the same Craftax play data, with no pretrained DINOv2 or V-JEPA 2 weights.
Game-state labels only train readout heads and score results.

## Steps

1. Tests first: patch encoder shapes, block-causal predictor (no peeking at later frames), rollout shapes,
   loss runs, old v2 checkpoints still load.
2. Train v4 on Colab A100 (same data and steps as v2, ~30k).
3. "Can it see?" table (readouts, critic, probes, surprise).
4. Retrain the action prior and top model on v4; reach tasks and chain runs.
5. Only if long chains still stall once the agent survives: H-JEPA-style levels (arXiv 2610.06805).

## Decisions to approve

- One token per 7×7 tile, 32 numbers each (81 × 32 latent).
- Predictor context 4 frames.
- Keep the inverse-dynamics loss.
- The bars above.
