"""Phase 1: train the PPO data player with Craftax_Baselines (MIT) and export its actor as .npz.

Runs on Colab (A100). Craftax_Baselines must be cloned next to this repo at the pinned commit:
    git clone https://github.com/MichaelTMatthews/Craftax_Baselines && git -C Craftax_Baselines checkout 7ce36fa
    python experiments/phase1_train_ppo.py --timesteps 2e8 --out ppo_classic.npz

The player acts on the symbolic observation (fast to train); the dataset records pixels of the same states.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASELINES_COMMIT = "7ce36fa05b84a2c9e758012f1e6da402e1e3a891"


def ppo_config(env_name: str, timesteps: float, num_envs: int, num_steps: int, seed: int) -> dict[str, object]:
    """Craftax_Baselines ppo.py defaults (argparse block at the pinned commit), with wandb and debug off."""
    return {
        "ENV_NAME": env_name,
        "NUM_ENVS": num_envs,
        "TOTAL_TIMESTEPS": int(timesteps),
        "LR": 2e-4,
        "NUM_STEPS": num_steps,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.8,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": False,
        "JIT": True,
        "SEED": seed,
        "USE_WANDB": False,
        "SAVE_POLICY": False,
        "NUM_REPEATS": 1,
        "LAYER_SIZE": 512,
        "WANDB_PROJECT": None,
        "WANDB_ENTITY": None,
        "USE_OPTIMISTIC_RESETS": True,
        "OPTIMISTIC_RESET_RATIO": 16,
        "EXPLORATION_UPDATE_EPOCHS": 4,
        "ICM_REWARD_COEFF": 1.0,
        "TRAIN_ICM": False,
        "ICM_LR": 3e-4,
        "ICM_FORWARD_LOSS_COEF": 1.0,
        "ICM_INVERSE_LOSS_COEF": 1.0,
        "ICM_LAYER_SIZE": 256,
        "ICM_LATENT_SIZE": 32,
        "E3B_REWARD_COEFF": 1.0,
        "USE_E3B": False,
        "E3B_LAMBDA": 0.1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Craftax-Classic-Symbolic-v1")
    ap.add_argument("--timesteps", type=float, default=2e8)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--num-steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", type=Path, default=Path("Craftax_Baselines"))
    ap.add_argument("--out", type=Path, default=Path("ppo_classic.npz"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.baselines))
    import jax
    import ppo  # Craftax_Baselines/ppo.py

    from jepa_game_agent.policies.mlp import save_actor_params

    cfg = ppo_config(args.env, args.timesteps, args.num_envs, args.num_steps, args.seed)
    t0 = time.time()
    out = jax.jit(ppo.make_train(cfg))(jax.random.PRNGKey(args.seed))
    train_state = out["runner_state"][0]
    jax.block_until_ready(train_state.params)
    save_actor_params(train_state.params, args.out)
    print(
        json.dumps(
            {
                "timesteps": int(args.timesteps),
                "seconds": round(time.time() - t0, 1),
                "out": str(args.out),
                "baselines_commit": BASELINES_COMMIT,
            }
        )
    )


if __name__ == "__main__":
    main()
