"""Command line: `jga collect` (record episodes), `jga train` (train the world model), `jga baselines`."""

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
    extra: dict[str, object] = {}
    if args.env.startswith("Craftax"):  # many games at once on the GPU
        from jepa_game_agent.data.craftax_batch import collect_craftax_batch, craftax_num_actions, with_epsilon

        batch_policy = None
        if args.policy is not None:
            from jepa_game_agent.policies.mlp import load_mlp_policy

            batch_policy = load_mlp_policy(args.policy, args.env)
            if args.epsilon > 0:
                batch_policy = with_epsilon(batch_policy, args.epsilon, craftax_num_actions(args.env))
            extra = {"policy": str(args.policy), "epsilon": args.epsilon}
        elif args.epsilon:
            raise SystemExit("--epsilon needs --policy")
        episodes = collect_craftax_batch(args.env, seeds, args.max_steps, policy=batch_policy)
        collector = "batch"
    else:
        env = make_env(args.env, seed=args.seed)
        policy = random_policy(env.num_actions)
        episodes, collector = [collect_episode(env, policy, args.max_steps, seed=s) for s in seeds], "single"
    save_episodes(args.out, episodes)
    print(json.dumps({**data_card(episodes), "env": args.env, "collector": collector, **extra, "out": str(args.out)}))


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


def _baselines(args: argparse.Namespace) -> None:
    from jepa_game_agent.evaluation.baselines import run_baselines, summary

    if args.env.startswith("Craftax"):  # every task at once on the GPU
        from jepa_game_agent.evaluation.craftax_batch_eval import run_baselines_batch

        results = run_baselines_batch(
            args.env,
            load_episodes(args.data),
            args.k,
            args.tasks,
            args.achievement,
            args.resets,
            args.budget,
            args.seed,
        )
    else:
        env = make_env(args.env, seed=args.seed)
        results = run_baselines(
            env, load_episodes(args.data), args.k, args.tasks, args.achievement, args.resets, args.budget, args.seed
        )
    results["config"] |= {"env": args.env, "data": str(args.data)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(summary(results)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jga", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="record episodes with a random policy")
    c.add_argument("--env", default="gridworld", help="gridworld or a Craftax pixel env name")
    c.add_argument("--episodes", type=int, default=10)
    c.add_argument("--max-steps", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--policy", type=Path, default=None, help="exported PPO weights (.npz); default: random")
    c.add_argument("--epsilon", type=float, default=0.0, help="share of actions replaced by random ones")
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
    b = sub.add_parser("baselines", help="score random / still / replay baselines on a dataset's tasks")
    b.add_argument("--data", type=Path, required=True, help="episodes to draw reach tasks from (test split)")
    b.add_argument("--env", default="gridworld")
    b.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="goal offsets for reach tasks")
    b.add_argument("--tasks", type=int, default=100, help="reach tasks per k")
    b.add_argument("--achievement", nargs="*", default=[], help="achievement names for family B")
    b.add_argument("--resets", type=int, default=50, help="fresh resets per achievement")
    b.add_argument("--budget", type=int, default=200, help="steps allowed per achievement task")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(func=_baselines)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
