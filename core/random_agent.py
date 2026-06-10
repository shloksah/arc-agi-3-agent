"""
Minimal random agent to verify the arc-agi SDK loop works.
Plays a game with random actions, logs frames and state transitions.
"""

import random
import numpy as np
import arc_agi
from arcengine import GameAction
from arcengine.enums import GameState

ACTION_MAP = {a.value: a for a in GameAction}


def play_game(game_id: str, max_actions: int = 100, verbose: bool = True):
    arc = arc_agi.Arcade()
    env = arc.make(game_id)

    if env is None:
        print(f"Failed to create environment for {game_id}")
        return None

    frame_data = env.reset()
    if frame_data is None:
        print("Failed to reset environment")
        return None

    results = {
        "game_id": game_id,
        "levels_completed": 0,
        "total_actions": 0,
        "frames": [],
        "transitions": [],
    }

    if verbose:
        print(f"Game: {game_id}")
        print(f"State: {frame_data.state}")
        print(f"Available actions: {frame_data.available_actions}")
        print(f"Win levels: {frame_data.win_levels}")
        print(f"Frame shape: {frame_data._frame[0].shape if frame_data._frame else 'N/A'}")
        print("-" * 50)

    prev_frame = frame_data._frame[0].copy() if frame_data._frame else None
    results["frames"].append(prev_frame)

    for step in range(max_actions):
        if frame_data.state == GameState.WIN:
            if verbose:
                print(f"WIN at step {step}! Levels completed: {frame_data.levels_completed}")
            results["levels_completed"] = frame_data.levels_completed
            break

        if frame_data.state == GameState.GAME_OVER:
            if verbose:
                print(f"GAME OVER at step {step}. Levels completed: {frame_data.levels_completed}")
            results["levels_completed"] = frame_data.levels_completed
            break

        available = frame_data.available_actions
        action_id = random.choice(available)
        action = ACTION_MAP[action_id]

        data = None
        if action == GameAction.ACTION6:
            data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}

        frame_data = env.step(action, data=data)
        if frame_data is None:
            if verbose:
                print(f"Step {step}: env.step returned None")
            break

        results["total_actions"] += 1
        curr_frame = frame_data._frame[0].copy() if frame_data._frame else None

        frame_changed = False
        if prev_frame is not None and curr_frame is not None:
            frame_changed = not np.array_equal(prev_frame, curr_frame)

        transition = {
            "step": step,
            "action": action.name,
            "data": data,
            "state": frame_data.state,
            "levels_completed": frame_data.levels_completed,
            "frame_changed": frame_changed,
        }
        results["transitions"].append(transition)

        if verbose and frame_changed:
            print(f"  Step {step}: {action.name}{f' ({data})' if data else ''} → FRAME CHANGED | levels={frame_data.levels_completed}")

        prev_frame = curr_frame
        results["frames"].append(curr_frame)

    results["total_actions"] = len(results["transitions"])
    if verbose:
        changes = sum(1 for t in results["transitions"] if t["frame_changed"])
        print(f"\nSummary: {results['total_actions']} actions, {changes} frame changes, {results['levels_completed']} levels completed")

    return results


if __name__ == "__main__":
    random.seed(42)
    results = play_game("cn04-2fe56bfb", max_actions=50, verbose=True)
