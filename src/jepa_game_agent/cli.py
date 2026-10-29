"""Command line: `jga <command> -h` lists each one. Data (`collect`), training (`train`, `train-top`),
evaluation (`baselines`, `plan`, `plan-achievement`, `probe`, `surprise`)."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from jepa_game_agent.config import ModelConfig, PlannerConfig, TrainConfig
from jepa_game_agent.data.episodes import (
    Episode,
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
            hud_rows=args.hud_rows,
            hud_dim=args.hud_dim,
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


def _snap_bank(args: argparse.Namespace, model: WorldModel, device: str) -> torch.Tensor | None:
    """Real latents to snap subgoals to: up to --snap-size encoded frames from the train split of --snap-data."""
    if args.snap_data is None:
        return None
    from jepa_game_agent.training.top_trainer import encode_episodes

    eps = [e for path in args.snap_data for e in load_episodes(path)]
    eps = [e for e in eps if split_of(e.seed) == "train"] or eps  # tiny test sets may have no train split
    z = torch.cat([torch.from_numpy(a) for a in encode_episodes(model, eps, device)])
    keep = torch.randperm(len(z), generator=torch.Generator().manual_seed(args.seed))[: args.snap_size]
    return z[keep]


def _plan(args: argparse.Namespace) -> None:
    from jepa_game_agent.evaluation import Agent, JepaAgent, evaluate_reach, make_reach_tasks
    from jepa_game_agent.evaluation.baselines import git_commit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)
    cfg = PlannerConfig(horizon=args.horizon, samples=args.samples, elites=args.elites, iterations=args.iterations)
    episodes, env = load_episodes(args.data), make_env(args.env, seed=args.seed)
    name = "jepa"

    def make_agent() -> Agent:
        return JepaAgent(model, cfg, device)

    top_cfg = None
    if args.top is not None:  # hierarchical: the top sets subgoals, the bottom walks
        from jepa_game_agent.evaluation import HierarchicalAgent
        from jepa_game_agent.training.top_trainer import load_top

        top, _ = load_top(args.top, device)
        top_cfg = PlannerConfig(args.top_horizon, args.samples, args.elites, args.iterations)
        bank = _snap_bank(args, model, device)
        name = "hier"

        def make_agent() -> Agent:
            assert top_cfg is not None
            return HierarchicalAgent(model, top, top_cfg, cfg, device, bank=bank)

    reach: dict[str, dict[str, object]] = {}
    for k in args.k:
        if args.env.startswith("Craftax"):  # the same tasks `jga baselines` scored
            from jepa_game_agent.evaluation.craftax_batch_eval import choose_reach_tasks

            tasks = choose_reach_tasks(args.env, episodes, k, args.tasks, args.seed)
        else:
            tasks = make_reach_tasks(episodes, k=k, n=args.tasks, seed=args.seed, env=env)
        reach[f"k={k}"] = {name: evaluate_reach(lambda: env, tasks, lambda t: make_agent())}
    config = {"env": args.env, "data": str(args.data), "checkpoint": str(args.checkpoint), "ks": list(args.k)}
    config |= {"tasks": args.tasks, "seed": args.seed, "planner": dataclasses.asdict(cfg)}
    if top_cfg is not None:
        config |= {"top": str(args.top), "top_planner": dataclasses.asdict(top_cfg)}
        if args.snap_data is not None:
            config |= {"snap": {"data": [str(p) for p in args.snap_data], "size": args.snap_size}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reach": reach, "config": config, "commit": git_commit()}, indent=1))
    print(json.dumps({k: r[name]["success_rate"] for k, r in reach.items()}))  # type: ignore[index]


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


def parse_stage(token: str) -> tuple[str, str | None, int]:
    """'place_table' -> ('place_table', None, 0); 'collect_wood:2' -> ('collect_wood', 'wood', 2): have 2 wood."""
    name, _, n = token.partition(":")
    if not n:
        return name, None, 0
    if not name.startswith("collect_") or int(n) < 1:
        raise ValueError(f"bad stage {token!r}: counts need a collect_<item> achievement and a count >= 1")
    return name, name.removeprefix("collect_"), int(n)


def goal_weights(cfg: ModelConfig, goal_on: str, hud_weight: float = 1.0) -> torch.Tensor | None:
    """Per-latent-dim goal-cost weights. all: None (every dim counts the same). hud: only the inventory dims.
    mix: the view's dims and the inventory's dims each sum to one unit (the inventory's times hud_weight),
    so neither part drowns the other out whatever their sizes."""
    if goal_on == "all":
        return None
    if not cfg.hud_rows:
        raise ValueError(f"--goal-on {goal_on} needs a model trained with --hud-rows")
    n_view, n_hud = cfg.latent_dim - cfg.hud_dim, cfg.hud_dim
    if goal_on == "hud":
        return torch.cat([torch.zeros(n_view), torch.ones(n_hud)])
    if goal_on == "mix":
        return torch.cat([torch.full((n_view,), 1.0 / n_view), torch.full((n_hud,), hud_weight / n_hud)])
    raise ValueError(f"unknown goal_on {goal_on!r}")


def _plan_achievement(args: argparse.Namespace) -> None:
    from jepa_game_agent.evaluation import AchievementTask, Agent, JepaAgent, evaluate_achievement, evaluate_chain
    from jepa_game_agent.evaluation.baselines import git_commit
    from jepa_game_agent.evaluation.goals import achievement_goal_frames, achievement_goal_pairs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)
    cfg = PlannerConfig(horizon=args.horizon, samples=args.samples, elites=args.elites, iterations=args.iterations)
    goal_eps = [e for path in args.goal_data for e in load_episodes(path)]
    if args.goal_split != "all":
        goal_eps = [e for e in goal_eps if split_of(e.seed) == args.goal_split]
    env = make_env(args.env, seed=args.seed)
    name = "jepa"

    weights = goal_weights(model.cfg, args.goal_on, args.hud_weight)
    prior = None
    if args.prior is not None:
        from jepa_game_agent.training.prior_trainer import load_prior

        prior, _ = load_prior(args.prior, device)

    def make_agent() -> Agent:
        return JepaAgent(model, cfg, device, hud_rows=args.hud_rows, weights=weights, prior=prior,
                         prior_mix=args.prior_mix, goal_as=args.goal_as, reach=args.reach,
                         travel=args.travel)  # fmt: skip

    top_cfg = None
    if args.top is not None:
        from jepa_game_agent.evaluation import HierarchicalAgent
        from jepa_game_agent.training.top_trainer import load_top

        top, _ = load_top(args.top, device)
        top_cfg = PlannerConfig(args.top_horizon, args.samples, args.elites, args.iterations)
        bank = _snap_bank(args, model, device)
        name = "hier"

        def make_agent() -> Agent:
            assert top_cfg is not None
            return HierarchicalAgent(model, top, top_cfg, cfg, device, hud_rows=args.hud_rows, bank=bank,
                                     weights=weights, prior=prior, prior_mix=args.prior_mix,
                                     goal_as=args.goal_as, reach=args.reach, travel=args.travel)  # fmt: skip

    results: dict[str, dict[str, object]] = {}
    goal_fn = achievement_goal_frames if args.goal_as == "picture" else achievement_goal_pairs
    parsed = [parse_stage(a) for a in args.achievement]
    if any(item is not None for _, item, _ in parsed) and not args.chain:
        raise ValueError("item counts (collect_x:n) only work with --chain")
    goals = {a: goal_fn(goal_eps, a, n=args.goal_frames, seed=args.seed) for a, _, _ in parsed}
    if args.chain:  # one run per world, subgoals in the given order
        from jepa_game_agent.evaluation.runner import ChainStage

        last = parsed[-1][0]
        tasks = [AchievementTask(last, seed=args.seed + i, budget=args.budget) for i in range(args.resets)]
        chain = [ChainStage(a, goals[a], item, n) for a, item, n in parsed]
        counts = None
        if any(st.item is not None for st in chain):
            if not hasattr(env, "inventory"):
                raise ValueError(f"{args.env} has no inventory to count items in")
            counts = lambda e: e.inventory()  # noqa: E731
        probe = (lambda e: e.vitals()) if hasattr(env, "vitals") else None  # what killed the player
        res = evaluate_chain(lambda: env, tasks, lambda t: make_agent(), chain, counts=counts, probe=probe)
    for ach in [] if args.chain else args.achievement:
        tasks = [AchievementTask(ach, seed=args.seed + i, budget=args.budget) for i in range(args.resets)]
        results[ach] = {name: evaluate_achievement(lambda: env, tasks, lambda t: make_agent(), goal_frames=goals[ach])}
    config = {"env": args.env, "checkpoint": str(args.checkpoint), "achievements": list(args.achievement)}
    config |= {"resets": args.resets, "budget": args.budget, "goal_frames": args.goal_frames, "seed": args.seed}
    config |= {"planner": dataclasses.asdict(cfg), "hud_rows": args.hud_rows, "goal_on": args.goal_on}
    config |= {"chain": args.chain, "hud_weight": args.hud_weight, "goal_as": args.goal_as, "reach": args.reach}
    config |= {"travel": args.travel}
    config |= {"prior": None if args.prior is None else str(args.prior), "prior_mix": args.prior_mix}
    if top_cfg is not None:
        config |= {"top": str(args.top), "top_planner": dataclasses.asdict(top_cfg)}
        if args.snap_data is not None:
            config |= {"snap": {"data": [str(p) for p in args.snap_data], "size": args.snap_size}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.chain:
        out = {"chain": {name: res}, "config": config, "commit": git_commit()}
        args.out.write_text(json.dumps(out, indent=1))
        print(json.dumps(res["stage_rates"]))
        return
    args.out.write_text(json.dumps({"achievement": results, "config": config, "commit": git_commit()}, indent=1))
    print(json.dumps({a: r[name]["success_rate"] for a, r in results.items()}))  # type: ignore[index]


def _train_top(args: argparse.Namespace) -> None:
    from jepa_game_agent.config import TopConfig
    from jepa_game_agent.models.top import TopModel
    from jepa_game_agent.training.top_trainer import TopWindowDataset, encode_episodes, train_top

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bottom, _ = load_checkpoint(args.bottom, device)
    episodes = [e for path in args.data for e in load_episodes(path)]
    if args.split != "all":
        episodes = [e for e in episodes if split_of(e.seed) == args.split]
    latents = encode_episodes(bottom, episodes, device)
    data = TopWindowDataset(latents, [e.actions for e in episodes], args.stride, args.window)
    top_cfg = TopConfig(bottom.cfg.num_actions, bottom.cfg.latent_dim, args.stride, max(args.window, 2))
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, window=args.window, seed=args.seed)
    history = train_top(TopModel(top_cfg), data, cfg, checkpoint=args.checkpoint, device=device)
    summary = {"split": args.split, "episodes": len(episodes), "stride": args.stride, "seed": args.seed}
    print(
        json.dumps(summary | {"steps": history[-1]["step"] if history else 0, "last": history[-1] if history else None})
    )


def _train_prior(args: argparse.Namespace) -> None:
    from jepa_game_agent.models.prior import ActionPrior
    from jepa_game_agent.training.prior_trainer import save_prior, train_prior
    from jepa_game_agent.training.top_trainer import encode_episodes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bottom, _ = load_checkpoint(args.bottom, device)
    episodes = [e for path in args.data for e in load_episodes(path)]
    if args.split != "all":
        episodes = [e for e in episodes if split_of(e.seed) == args.split]
    latents = encode_episodes(bottom, episodes, device)
    prior = ActionPrior(bottom.cfg.latent_dim, bottom.cfg.num_actions)
    history = train_prior(prior, latents, [e.actions for e in episodes], args.steps, args.batch_size,
                          seed=args.seed, device=device)  # fmt: skip
    save_prior(args.checkpoint, prior, history)
    print(json.dumps({"split": args.split, "episodes": len(episodes), "last": history[-1]}))


def _action_effects(args: argparse.Namespace) -> None:
    import numpy as np

    from jepa_game_agent.evaluation.action_effects import rank_actions
    from jepa_game_agent.evaluation.baselines import git_commit
    from jepa_game_agent.evaluation.goals import achievement_goal_frames, achievement_goal_pairs, unlock_steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    def pick(eps: list[Episode], split: str) -> list[Episode]:
        return eps if split == "all" else [e for e in eps if split_of(e.seed) == split]

    eps = pick([e for path in args.data for e in load_episodes(path)], args.split)
    goal_eps = pick([e for path in args.goal_data for e in load_episodes(path)], args.goal_split)

    def enc(frames: list[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            return model.encode(x)

    def ranked(trans: list[tuple[Episode, int]], goal: torch.Tensor | None) -> dict[str, object]:
        z_now = enc([e.frames[t] for e, t in trans])
        z_next = enc([e.frames[t + 1] for e, t in trans])
        acts = torch.tensor([int(e.actions[t]) for e, t in trans], device=device)
        res = rank_actions(model, z_now, z_next, acts, goal)
        counts = np.bincount(acts.cpu().numpy(), minlength=model.cfg.num_actions)
        return res | {"actions": counts.tolist()}

    results: dict[str, dict[str, object]] = {}
    for name in args.achievement:
        trans = []
        for e in eps:
            if name in e.achievements:
                try:
                    trans.append((e, unlock_steps(e)[name]))
                except ValueError:  # unlock rewards don't add up: skip the episode
                    continue
        if not trans:
            raise ValueError(f"no transition unlocks {name!r} in --data")
        if args.goal_as == "change":  # goal = this latent + the mean before -> after step of the goal games
            pairs = achievement_goal_pairs(goal_eps, name, n=args.goal_frames, seed=args.seed)
            delta = (enc([p[1] for p in pairs]) - enc([p[0] for p in pairs])).mean(0)
            goal = enc([e.frames[t] for e, t in trans]) + delta
        else:
            goal = enc(achievement_goal_frames(goal_eps, name, n=args.goal_frames, seed=args.seed)).mean(0)
        results[name] = ranked(trans, goal)
    rng = np.random.default_rng(args.seed)
    pool = [(e, t) for e in eps for t in range(len(e.actions))]
    control = ranked([pool[i] for i in rng.choice(len(pool), size=min(args.control, len(pool)), replace=False)], None)
    config = {"checkpoint": str(args.checkpoint), "data": [str(p) for p in args.data], "split": args.split}
    config |= {"goal_split": args.goal_split, "goal_frames": args.goal_frames, "seed": args.seed}
    config |= {"goal_as": args.goal_as}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {"achievement": results, "control": control, "config": config, "commit": git_commit()}
    args.out.write_text(json.dumps(out, indent=1))
    keys = ("n", "mean_next_rank", "next_top1", "mean_goal_rank", "goal_top1")
    summary = {k: {f: r.get(f) for f in keys} for k, r in results.items()}
    summary["control"] = {f: control.get(f) for f in keys[:3]}
    print(json.dumps(summary))


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
    t.add_argument(
        "--hud-rows", type=int, default=0, help="bottom pixel rows with their own encoder (Craftax-Classic: 14)"
    )
    t.add_argument("--hud-dim", type=int, default=ModelConfig.hud_dim, help="latent dims for those rows")
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
    pl.add_argument("--top", type=Path, default=None, help="top-level checkpoint: run the hierarchical agent")
    pl.add_argument("--top-horizon", type=int, default=5, help="top-level plan length, in chunks")
    pl.add_argument("--snap-data", type=Path, nargs="+", default=None, help="games whose frames subgoals snap to")
    pl.add_argument("--snap-size", type=int, default=50_000, help="latents kept for snapping")
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
    tt = sub.add_parser("train-top", help="train the hierarchy's top level on a frozen bottom model's latents")
    tt.add_argument("--data", type=Path, nargs="+", required=True)
    tt.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    tt.add_argument("--bottom", type=Path, required=True, help="trained world model checkpoint (frozen encoder)")
    tt.add_argument("--stride", type=int, default=4, help="game steps per top-level step")
    tt.add_argument("--window", type=int, default=8, help="top-level steps per training sample")
    tt.add_argument("--steps", type=int, default=10_000)
    tt.add_argument("--batch-size", type=int, default=256)
    tt.add_argument("--seed", type=int, default=0)
    tt.add_argument("--checkpoint", type=Path, required=True)
    tt.set_defaults(func=_train_top)
    tp = sub.add_parser("train-prior", help="learn which actions players press in a state (seeds the planner)")
    tp.add_argument("--data", type=Path, nargs="+", required=True)
    tp.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    tp.add_argument("--bottom", type=Path, required=True, help="trained world model checkpoint (frozen encoder)")
    tp.add_argument("--steps", type=int, default=5_000)
    tp.add_argument("--batch-size", type=int, default=512)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--checkpoint", type=Path, required=True)
    tp.set_defaults(func=_train_prior)
    fx = sub.add_parser("action-effects", help="does the model know what an action does? rank the true action")
    fx.add_argument("--checkpoint", type=Path, required=True)
    fx.add_argument("--data", type=Path, nargs="+", required=True, help="recorded games with the transitions")
    fx.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    fx.add_argument("--goal-data", type=Path, nargs="+", required=True, help="recorded games to take goal frames from")
    fx.add_argument("--goal-split", choices=("all", "train", "val", "test"), default="all")
    fx.add_argument("--goal-frames", type=int, default=64)
    fx.add_argument("--achievement", nargs="+", required=True)
    fx.add_argument("--control", type=int, default=2000, help="random ordinary transitions for comparison")
    fx.add_argument("--goal-as", choices=("picture", "change"), default="picture")
    fx.add_argument("--seed", type=int, default=0)
    fx.add_argument("--out", type=Path, required=True)
    fx.set_defaults(func=_action_effects)
    pa = sub.add_parser("plan-achievement", help="pursue achievements, each given as its 'just unlocked' pictures")
    pa.add_argument("--checkpoint", type=Path, required=True)
    pa.add_argument("--top", type=Path, default=None, help="top-level checkpoint: run the hierarchical agent")
    pa.add_argument("--top-horizon", type=int, default=5)
    pa.add_argument("--goal-data", type=Path, nargs="+", required=True, help="recorded games to take goal frames from")
    pa.add_argument("--goal-split", choices=("all", "train", "val", "test"), default="all")
    pa.add_argument("--goal-frames", type=int, default=64, help="goal frames averaged into one goal latent")
    pa.add_argument("--env", default="gridworld")
    pa.add_argument("--achievement", nargs="+", required=True)
    pa.add_argument("--resets", type=int, default=50, help="fresh worlds per achievement")
    pa.add_argument("--budget", type=int, default=200)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--horizon", type=int, default=PlannerConfig.horizon)
    pa.add_argument("--samples", type=int, default=PlannerConfig.samples)
    pa.add_argument("--elites", type=int, default=PlannerConfig.elites)
    pa.add_argument("--iterations", type=int, default=PlannerConfig.iterations)
    pa.add_argument("--snap-data", type=Path, nargs="+", default=None, help="games whose frames subgoals snap to")
    pa.add_argument("--snap-size", type=int, default=50_000, help="latents kept for snapping")
    pa.add_argument(
        "--hud-rows", type=int, default=0, help="goal = this view + goal frames' bottom rows (Craftax-Classic: 14)"
    )
    pa.add_argument(
        "--goal-on", choices=("all", "hud", "mix"), default="all",
        help="hud: score only the inventory latent; mix: view and inventory weighted equally (see --hud-weight)",
    )  # fmt: skip
    pa.add_argument("--hud-weight", type=float, default=1.0, help="mix: the inventory's weight vs the view's")
    pa.add_argument("--prior", type=Path, default=None, help="action prior checkpoint (jga train-prior)")
    pa.add_argument(
        "--goal-as", choices=("picture", "change", "both", "blend"), default="picture",
        help="change: goal = this latent + what unlocking changes (mean before -> after step in recorded games); "
        "both: travel by picture, do the change once reachable; blend: mean of picture and change goals",
    )  # fmt: skip
    pa.add_argument("--reach", type=float, default=0.5, help="both: change counts as reachable within this share")
    pa.add_argument("--travel", choices=("after", "before"), default="after",
                    help="both/blend: travel to the after-unlock picture or to the before-unlock one")  # fmt: skip
    pa.add_argument("--prior-mix", type=float, default=0.1, help="share of uniform blended into the prior")
    pa.add_argument(
        "--chain", action="store_true", help="one run per world: pursue the achievements in order, as subgoals"
    )
    pa.add_argument("--out", type=Path, required=True)
    pa.set_defaults(func=_plan_achievement)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
