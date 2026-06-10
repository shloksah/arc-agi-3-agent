"""
GameRunner: Core loop for playing ARC-AGI-3 environments.
Wraps the arc-agi SDK with state tracking, frame logging, and transition history.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol
import hashlib
import numpy as np
import arc_agi
from arcengine import GameAction
from arcengine.enums import GameState

ACTION_MAP = {a.value: a for a in GameAction}


@dataclass
class Transition:
    step: int
    action: GameAction
    action_data: Optional[dict]
    frame_before: np.ndarray
    frame_after: np.ndarray
    frame_changed: bool
    state_after: GameState
    levels_completed: int
    frame_hash: str


@dataclass
class LevelResult:
    level: int
    actions_taken: int
    solved: bool
    transitions: list[Transition] = field(default_factory=list)


@dataclass
class GameResult:
    game_id: str
    levels: list[LevelResult] = field(default_factory=list)
    total_actions: int = 0
    win_levels: int = 0

    @property
    def levels_solved(self) -> int:
        return sum(1 for lvl in self.levels if lvl.solved)


class Agent(Protocol):
    """Interface that all agents must implement."""

    def select_action(
        self,
        frame: np.ndarray,
        available_actions: list[int],
        history: list[Transition],
        levels_completed: int,
    ) -> tuple[GameAction, Optional[dict]]:
        """Return (action, data) where data is needed for ACTION6 clicks."""
        ...

    def on_level_complete(self, level: int, transitions: list[Transition]) -> None:
        """Called when a level is completed."""
        ...

    def on_game_start(self, game_id: str, env_info: arc_agi.EnvironmentInfo) -> None:
        """Called when a new game starts."""
        ...


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.md5(frame.tobytes()).hexdigest()[:12]


class GameRunner:
    def __init__(self, verbose: bool = True):
        self.arc = arc_agi.Arcade()
        self.verbose = verbose

    def get_environments(self) -> list[arc_agi.EnvironmentInfo]:
        return self.arc.get_environments()

    def play(self, game_id: str, agent: Agent, max_actions: int = 5000) -> GameResult:
        envs = self.arc.get_environments()
        env_info = next((e for e in envs if e.game_id == game_id), None)

        env = self.arc.make(game_id)
        if env is None:
            raise RuntimeError(f"Failed to create environment: {game_id}")

        frame_data = env.reset()
        if frame_data is None:
            raise RuntimeError(f"Failed to reset environment: {game_id}")

        if env_info:
            agent.on_game_start(game_id, env_info)

        result = GameResult(game_id=game_id, win_levels=frame_data.win_levels)
        current_level_transitions: list[Transition] = []
        current_level = 0
        step = 0

        if self.verbose:
            print(f"Playing {game_id} | win_levels={frame_data.win_levels} | actions={frame_data.available_actions}")

        prev_frame = frame_data._frame[0].copy()

        while step < max_actions:
            if frame_data.state == GameState.WIN:
                current_level_transitions_final = current_level_transitions
                result.levels.append(LevelResult(
                    level=current_level,
                    actions_taken=len(current_level_transitions_final),
                    solved=True,
                    transitions=current_level_transitions_final,
                ))
                agent.on_level_complete(current_level, current_level_transitions_final)
                if self.verbose:
                    print(f"  WIN! All {frame_data.win_levels} levels complete in {step} total actions")
                break

            if frame_data.state == GameState.GAME_OVER:
                # The real environment allows RESET to retry the level, so a
                # death is not terminal. Let the agent attribute the death to
                # the action that caused it, then reset and keep playing.
                if hasattr(agent, "on_game_over"):
                    agent.on_game_over(current_level, current_level_transitions)
                if step >= max_actions:
                    break
                frame_data = env.step(GameAction.RESET)
                if frame_data is None:
                    break
                step += 1
                curr_frame = frame_data._frame[0].copy()
                current_level_transitions.append(Transition(
                    step=step,
                    action=GameAction.RESET,
                    action_data=None,
                    frame_before=prev_frame,
                    frame_after=curr_frame,
                    frame_changed=True,
                    state_after=frame_data.state,
                    levels_completed=frame_data.levels_completed,
                    frame_hash=frame_hash(curr_frame),
                ))
                prev_frame = curr_frame
                continue

            if frame_data.levels_completed > current_level:
                result.levels.append(LevelResult(
                    level=current_level,
                    actions_taken=len(current_level_transitions),
                    solved=True,
                    transitions=current_level_transitions,
                ))
                agent.on_level_complete(current_level, current_level_transitions)
                if self.verbose:
                    print(f"  Level {current_level} solved in {len(current_level_transitions)} actions")
                current_level = frame_data.levels_completed
                current_level_transitions = []

            action, data = agent.select_action(
                frame=prev_frame,
                available_actions=frame_data.available_actions,
                history=current_level_transitions,
                levels_completed=frame_data.levels_completed,
            )

            frame_data = env.step(action, data=data)
            if frame_data is None:
                break

            step += 1
            curr_frame = frame_data._frame[0].copy()
            changed = not np.array_equal(prev_frame, curr_frame)

            t = Transition(
                step=step,
                action=action,
                action_data=data,
                frame_before=prev_frame,
                frame_after=curr_frame,
                frame_changed=changed,
                state_after=frame_data.state,
                levels_completed=frame_data.levels_completed,
                frame_hash=frame_hash(curr_frame),
            )
            current_level_transitions.append(t)
            prev_frame = curr_frame

        # Budget exhausted mid-level: record the unfinished level
        if frame_data is not None and frame_data.state != GameState.WIN and current_level_transitions:
            result.levels.append(LevelResult(
                level=current_level,
                actions_taken=len(current_level_transitions),
                solved=False,
                transitions=current_level_transitions,
            ))

        result.total_actions = step
        return result

    def compute_rhae(self, result: GameResult, baseline_actions: list[int]) -> float:
        """Official RHAE for one game.

        level_score = min(1, baseline/actions)^2, hard 0 beyond 5x baseline.
        Weighted by 1-indexed level number over ALL levels of the game, so
        unattempted/unsolved levels still count in the denominator.
        """
        n_levels = len(baseline_actions)
        if n_levels == 0:
            return 0.0

        solved_actions = {
            lvl.level: lvl.actions_taken for lvl in result.levels if lvl.solved
        }
        total = 0.0
        for i in range(n_levels):
            actions = solved_actions.get(i, 0)
            if actions > 0 and actions <= 5 * baseline_actions[i]:
                total += (i + 1) * min(1.0, baseline_actions[i] / actions) ** 2
        return total / sum(range(1, n_levels + 1))
