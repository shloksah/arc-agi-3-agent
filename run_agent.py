"""
Run an agent against ARC-AGI-3 environments and compute RHAE scores.
Usage: python run_agent.py [game_id] [--max-actions N]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from core.game_runner import GameRunner
from core.explorer_agent import ExplorerAgent


def main():
    game_id = sys.argv[1] if len(sys.argv) > 1 else "sb26-7fbdac44"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    runner = GameRunner(verbose=True)
    agent = ExplorerAgent(action_budget=max_actions)

    envs = runner.get_environments()
    env_info = next((e for e in envs if e.game_id == game_id), None)
    if not env_info:
        print(f"Game {game_id} not found")
        return

    print(f"\nRunning ExplorerAgent on {game_id}")
    print(f"  Levels: {len(env_info.baseline_actions)}")
    print(f"  Baseline actions: {env_info.baseline_actions}")
    print(f"  Tags: {env_info.tags}")
    print(f"  Max actions: {max_actions}")
    print("=" * 60)

    result = runner.play(game_id, agent, max_actions=max_actions)

    print("\n" + "=" * 60)
    print(f"RESULTS: {game_id}")
    print(f"  Levels solved: {result.levels_solved}/{len(env_info.baseline_actions)}")
    print(f"  Total actions: {result.total_actions}")

    if result.levels:
        rhae = runner.compute_rhae(result, env_info.baseline_actions)
        print(f"  RHAE Score: {rhae*100:.2f}%")
        print(f"\n  Per-level breakdown:")
        for i, lvl in enumerate(result.levels):
            baseline = env_info.baseline_actions[i] if i < len(env_info.baseline_actions) else "?"
            status = "SOLVED" if lvl.solved else "FAILED"
            if lvl.solved:
                ratio = baseline / lvl.actions_taken if lvl.actions_taken > 0 else 0
                score = min(1.0, ratio) ** 2
                print(f"    Level {i}: {status} in {lvl.actions_taken} actions (baseline={baseline}, ratio={ratio:.2f}, score={score*100:.1f}%)")
            else:
                print(f"    Level {i}: {status} after {lvl.actions_taken} actions (baseline={baseline})")


if __name__ == "__main__":
    main()
