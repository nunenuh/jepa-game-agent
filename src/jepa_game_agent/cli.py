"""Command line: `jga collect` (record episodes), `jga train` (train the world model), `jga baselines`,
`jga plan` (score a trained world model's planner), `jga probe` (what the latent knows),
`jga surprise` (does it notice impossible jumps)."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from jepa_game_agent.config import ModelConfig, PlannerConfig, TrainConfig
from jepa_game_agent.data.episodes import (
    WindowDataset,
    collect_episode,
    data_card,
    load_episodes,
    random_policy,
    save_episodes,
    split_of,
)
from jepa_game_agent.envs import make_env
from jepa_game_agent.models.world_model import WorldModel
from jepa_game_agent.training.trainer import load_checkpoint, train


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
    episodes = [e for path in args.data for e in load_episodes(path)]
    if args.split != "all":
        episodes = [e for e in episodes if split_of(e.seed) == args.split]
    data = WindowDataset(episodes, args.window)
    model = WorldModel(
        ModelConfig(
            num_actions=args.num_actions,
            latent_dim=args.latent_dim,
            channels=episodes[0].frames.shape[-1],
            context=max(args.window, 2),
            inverse_weight=args.inverse_weight,
        )
    )
    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, window=args.window, checkpoint_every=args.checkpoint_every
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train(model, data, cfg, checkpoint=args.checkpoint, device=device)
    print(
        json.dumps(
            {
                "split": args.split,
                "episodes": len(episodes),
                "transitions": sum(len(e) for e in episodes),
                "steps": history[-1]["step"] if history else 0,
                "last": history[-1] if history else None,
            }
        )
    )


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


def _plan(args: argparse.Namespace) -> None:
    from jepa_game_agent.evaluation import JepaAgent, evaluate_reach, make_reach_tasks
    from jepa_game_agent.evaluation.baselines import git_commit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)
    cfg = PlannerConfig(horizon=args.horizon, samples=args.samples, elites=args.elites, iterations=args.iterations)
    episodes, env = load_episodes(args.data), make_env(args.env, seed=args.seed)
    reach: dict[str, dict[str, object]] = {}
    for k in args.k:
        if args.env.startswith("Craftax"):  # the same tasks `jga baselines` scored
            from jepa_game_agent.evaluation.craftax_batch_eval import choose_reach_tasks

            tasks = choose_reach_tasks(args.env, episodes, k, args.tasks, args.seed)
        else:
            tasks = make_reach_tasks(episodes, k=k, n=args.tasks, seed=args.seed, env=env)
        reach[f"k={k}"] = {"jepa": evaluate_reach(lambda: env, tasks, lambda t: JepaAgent(model, cfg, device))}
    config = {"env": args.env, "data": str(args.data), "checkpoint": str(args.checkpoint), "ks": list(args.k)}
    config |= {"tasks": args.tasks, "seed": args.seed, "planner": dataclasses.asdict(cfg)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reach": reach, "config": config, "commit": git_commit()}, indent=1))
    print(json.dumps({k: r["jepa"]["success_rate"] for k, r in reach.items()}))  # type: ignore[index]


def _probe(args: argparse.Namespace) -> None:
    from jepa_game_agent.evaluation.baselines import git_commit
    from jepa_game_agent.evaluation.probes import run_probes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)
    env = make_env(args.env, seed=args.seed)
    seeds = [args.seed + i for i in range(args.episodes)]
    res = run_probes(model, env, random_policy(env.num_actions), seeds, args.max_steps, device)
    config = {"env": args.env, "checkpoint": str(args.checkpoint), "episodes": args.episodes}
    config |= {"max_steps": args.max_steps, "seed": args.seed, "policy": "random"}
    res |= {"config": config, "commit": git_commit()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["r2"]))


def _surprise(args: argparse.Namespace) -> None:
    from jepa_game_agent.evaluation.baselines import git_commit
    from jepa_game_agent.evaluation.surprise_eval import surprise_auc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)
    res = surprise_auc(model, load_episodes(args.data), args.window, args.n, args.min_gap, args.seed, device)
    config = {"checkpoint": str(args.checkpoint), "data": str(args.data), "window": args.window}
    config |= {"n": args.n, "min_gap": args.min_gap, "seed": args.seed}
    res |= {"config": config, "commit": git_commit()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("auc", "mean_normal", "mean_jump")}))


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
    t.add_argument("--data", type=Path, nargs="+", required=True, help="one or more episode files (shards)")
    t.add_argument("--split", choices=("all", "train", "val", "test"), default="all", help="keep one split (by seed)")
    t.add_argument("--num-actions", type=int, required=True)
    t.add_argument("--latent-dim", type=int, default=128)
    t.add_argument("--inverse-weight", type=float, default=0.0, help="> 0 adds the inverse-dynamics loss")
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
    pl = sub.add_parser("plan", help="score a trained world model's planner on the baseline reach tasks")
    pl.add_argument("--checkpoint", type=Path, required=True)
    pl.add_argument("--data", type=Path, required=True, help="episodes to draw reach tasks from (test split)")
    pl.add_argument("--env", default="gridworld")
    pl.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    pl.add_argument("--tasks", type=int, default=100)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--horizon", type=int, default=PlannerConfig.horizon)
    pl.add_argument("--samples", type=int, default=PlannerConfig.samples)
    pl.add_argument("--elites", type=int, default=PlannerConfig.elites)
    pl.add_argument("--iterations", type=int, default=PlannerConfig.iterations)
    pl.add_argument("--out", type=Path, required=True)
    pl.set_defaults(func=_plan)
    pr = sub.add_parser("probe", help="linear probes: R² of each state-key field read off the latent")
    pr.add_argument("--checkpoint", type=Path, required=True)
    pr.add_argument("--env", default="gridworld")
    pr.add_argument("--episodes", type=int, default=50, help="random-play episodes (1 in 5 held out)")
    pr.add_argument("--max-steps", type=int, default=300)
    pr.add_argument("--seed", type=int, default=10_000, help="first reset seed (away from the training seeds)")
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(func=_probe)
    su = sub.add_parser("surprise", help="ROC-AUC of surprise: real next frames vs jumps within the same game")
    su.add_argument("--checkpoint", type=Path, required=True)
    su.add_argument("--data", type=Path, required=True, help="episodes (test split)")
    su.add_argument("--window", type=int, default=8, help="history frames before the scored step")
    su.add_argument("--n", type=int, default=1000, help="normal / jump pairs")
    su.add_argument("--min-gap", type=int, default=50, help="a jump lands at least this many steps away")
    su.add_argument("--seed", type=int, default=0)
    su.add_argument("--out", type=Path, required=True)
    su.set_defaults(func=_surprise)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
