"""Command line: `jga collect` (record episodes) and `jga train` (train the world model)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from jepa_game_agent.config import ModelConfig, TrainConfig
from jepa_game_agent.data.episodes import (
    WindowDataset,
    collect_episode,
    data_card,
    load_episodes,
    random_policy,
    save_episodes,
)
from jepa_game_agent.envs import make_env
from jepa_game_agent.models.world_model import WorldModel
from jepa_game_agent.training.trainer import train


def _collect(args: argparse.Namespace) -> None:
    seeds = [args.seed + i for i in range(args.episodes)]
    if args.env.startswith("Craftax"):  # many games at once on the GPU
        from jepa_game_agent.data.craftax_batch import collect_craftax_batch

        episodes, collector = collect_craftax_batch(args.env, seeds, args.max_steps), "batch"
    else:
        env = make_env(args.env, seed=args.seed)
        policy = random_policy(env.num_actions)
        episodes, collector = [collect_episode(env, policy, args.max_steps, seed=s) for s in seeds], "single"
    save_episodes(args.out, episodes)
    print(json.dumps({**data_card(episodes), "env": args.env, "collector": collector, "out": str(args.out)}))


def _train(args: argparse.Namespace) -> None:
    episodes = load_episodes(args.data)
    data = WindowDataset(episodes, args.window)
    model = WorldModel(
        ModelConfig(
            num_actions=args.num_actions,
            latent_dim=args.latent_dim,
            channels=episodes[0].frames.shape[-1],
            context=max(args.window, 2),
        )
    )
    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, window=args.window, checkpoint_every=args.checkpoint_every
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train(model, data, cfg, checkpoint=args.checkpoint, device=device)
    print(json.dumps({"steps": history[-1]["step"] if history else 0, "last": history[-1] if history else None}))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jga", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="record episodes with a random policy")
    c.add_argument("--env", default="gridworld", help="gridworld or a Craftax pixel env name")
    c.add_argument("--episodes", type=int, default=10)
    c.add_argument("--max-steps", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=_collect)
    t = sub.add_parser("train", help="train the world model on recorded episodes")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--num-actions", type=int, required=True)
    t.add_argument("--latent-dim", type=int, default=128)
    t.add_argument("--window", type=int, default=8)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--checkpoint", type=Path, default=None)
    t.add_argument("--checkpoint-every", type=int, default=500)
    t.set_defaults(func=_train)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
