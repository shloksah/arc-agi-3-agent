"""
Local RHAE harness: run an agent across all 25 public games and report
per-game and overall RHAE, mirroring the Kaggle leaderboard metric
(leaderboard score = mean RHAE across games, shown as a percentage).

Usage:
    python harness.py                       # explorer agent, all games
    python harness.py --agent random        # random-action baseline
    python harness.py --games cn04,ft09     # subset (game-id prefix match)
    python harness.py --max-actions 500     # per-game action cap (0 = auto)

Each run is saved to runs/run_<timestamp>.json so score progress can be
tracked across agent versions.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from arcengine import GameAction
from core.game_runner import GameRunner
from core.explorer_agent import ExplorerAgent
from core.frugal_explorer import FrugalExplorer

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


class RandomAgent:
    """Floor baseline: uniform random over available actions."""

    def select_action(self, frame, available_actions, history, levels_completed):
        action_id = random.choice(available_actions)
        action = GameAction.from_name(f"ACTION{action_id}")
        data = None
        if action_id == 6:
            data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
        return action, data

    def on_level_complete(self, level, transitions):
        pass

    def on_game_start(self, game_id, env_info):
        pass


def make_agent(name: str, max_actions: int):
    if name == "random":
        return RandomAgent()
    if name == "explorer":
        return ExplorerAgent(action_budget=max_actions)
    if name == "frugal":
        return FrugalExplorer(action_budget=max_actions)
    raise ValueError(f"Unknown agent: {name}")


def auto_action_cap(baseline_actions: list[int]) -> int:
    # Past 5x baseline on every level, no further credit is possible.
    return 5 * sum(baseline_actions)


def main():
    parser = argparse.ArgumentParser(description="Run agent across public games, score RHAE")
    parser.add_argument("--agent", default="frugal", choices=["frugal", "explorer", "random"])
    parser.add_argument("--games", default="", help="Comma-separated game-id prefixes (default: all)")
    parser.add_argument("--max-actions", type=int, default=0, help="Per-game action cap (0 = 5x baseline sum)")
    parser.add_argument("--no-save", action="store_true", help="Skip writing runs/ JSON")
    args = parser.parse_args()

    runner = GameRunner(verbose=False)
    envs = runner.get_environments()
    if args.games:
        prefixes = [g.strip() for g in args.games.split(",") if g.strip()]
        envs = [e for e in envs if any(e.game_id.startswith(p) for p in prefixes)]
    if not envs:
        print("No matching games found")
        return
    envs.sort(key=lambda e: e.game_id)

    print(f"Agent: {args.agent} | Games: {len(envs)}")
    print(f"{'game':<16} {'levels':>7} {'actions':>8} {'rhae%':>7} {'time':>6}")
    print("-" * 50)

    game_rows = []
    t_start = time.time()

    for env_info in envs:
        baselines = env_info.baseline_actions or []
        cap = args.max_actions or auto_action_cap(baselines)
        agent = make_agent(args.agent, cap)

        t0 = time.time()
        row = {
            "game_id": env_info.game_id,
            "baseline_actions": baselines,
            "max_actions": cap,
            "rhae": 0.0,
            "levels_solved": 0,
            "win_levels": len(baselines),
            "total_actions": 0,
            "error": None,
            "level_detail": [],
        }
        try:
            result = runner.play(env_info.game_id, agent, max_actions=cap)
            row["rhae"] = runner.compute_rhae(result, baselines)
            row["levels_solved"] = result.levels_solved
            row["win_levels"] = result.win_levels or len(baselines)
            row["total_actions"] = result.total_actions
            row["level_detail"] = [
                {"level": l.level, "solved": l.solved, "actions": l.actions_taken}
                for l in result.levels
            ]
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        row["seconds"] = round(time.time() - t0, 1)

        status = f"{row['levels_solved']}/{row['win_levels']}"
        rhae_pct = f"{row['rhae'] * 100:.2f}"
        note = " ERROR" if row["error"] else ""
        print(f"{row['game_id'][:15]:<16} {status:>7} {row['total_actions']:>8} {rhae_pct:>7} {row['seconds']:>5.1f}s{note}")
        game_rows.append(row)

    # Leaderboard metric: mean across ALL games (errors count as 0)
    overall = sum(r["rhae"] for r in game_rows) / len(game_rows)
    elapsed = time.time() - t_start
    print("-" * 50)
    print(f"OVERALL RHAE: {overall * 100:.3f}%  (leaderboard-equivalent score)")
    print(f"Total levels solved: {sum(r['levels_solved'] for r in game_rows)}"
          f"/{sum(r['win_levels'] for r in game_rows)} | elapsed {elapsed:.0f}s")
    errors = [r["game_id"] for r in game_rows if r["error"]]
    if errors:
        print(f"Errors in: {', '.join(errors)}")

    if not args.no_save:
        os.makedirs(RUNS_DIR, exist_ok=True)
        out = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": args.agent,
            "overall_rhae": overall,
            "elapsed_seconds": round(elapsed, 1),
            "games": game_rows,
        }
        path = os.path.join(RUNS_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {os.path.relpath(path, os.path.dirname(__file__))}")


if __name__ == "__main__":
    main()
