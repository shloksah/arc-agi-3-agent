"""Verify frame parser works across game types."""

import sys
sys.path.insert(0, ".")

import arc_agi
import numpy as np
from arcengine import GameAction
from core.frame_parser import FrameParser
from core.game_runner import ACTION_MAP


def test_game(game_id: str, parser: FrameParser, arc: arc_agi.Arcade):
    env = arc.make(game_id)
    fd = env.reset()
    frame = fd._frame[0]
    state = parser.parse(frame)

    # Basic checks
    assert state.frame.shape == (64, 64), f"Frame shape wrong: {state.frame.shape}"
    assert state.background_color in range(16), f"Invalid bg color: {state.background_color}"
    assert len(state.frame_hash) == 12, f"Hash length wrong: {state.frame_hash}"
    assert state.status_bar.shape == (1, 64), f"Status bar shape wrong"
    assert state.game_area.shape == (63, 64), f"Game area shape wrong"

    # Try actions until something changes
    available = fd.available_actions
    for action_id in available:
        action = ACTION_MAP[action_id]
        data = None
        if action == GameAction.ACTION6:
            non_bg = np.argwhere(frame != state.background_color)
            if len(non_bg) > 0:
                y, x = non_bg[len(non_bg) // 2]
                data = {"x": int(x), "y": int(y)}
            else:
                data = {"x": 32, "y": 32}

        fd2 = env.step(action, data=data)
        if fd2 is None:
            continue
        frame2 = fd2._frame[0]
        if not np.array_equal(frame, frame2):
            state2 = parser.parse(frame2)
            diff = parser.compute_diff(state, state2)
            assert diff.pixels_changed > 0
            print(f"  {game_id}: {action.name} changed {diff.pixels_changed}px, moved={len(diff.objects_moved)} objs")
            return True

    print(f"  {game_id}: no action changed state (may need targeted clicks)")
    return True


def main():
    arc = arc_agi.Arcade()
    parser = FrameParser(min_object_area=4)
    envs = arc.get_environments()

    print(f"Testing frame parser on {len(envs)} games...\n")

    passed = 0
    for env_info in envs[:10]:
        try:
            test_game(env_info.game_id, parser, arc)
            passed += 1
        except Exception as e:
            print(f"  {env_info.game_id}: FAILED - {e}")

    print(f"\n{passed}/10 games parsed successfully")


if __name__ == "__main__":
    main()
