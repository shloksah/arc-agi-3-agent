"""
Explorer Agent: Systematic state-graph exploration.
Builds a directed graph of game states, prioritizes untested actions,
and uses undo (ACTION7) for efficient backtracking.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import arc_agi
from arcengine import GameAction
from arcengine.enums import GameState

from core.game_runner import ACTION_MAP, Agent, Transition, frame_hash


@dataclass
class StateNode:
    frame_hash: str
    frame: np.ndarray
    tested_actions: dict[int, Optional[str]] = field(default_factory=dict)
    visits: int = 0

    @property
    def untested_actions(self) -> list[int]:
        return [a for a in self.all_actions if a not in self.tested_actions]

    all_actions: list[int] = field(default_factory=list)


class ExplorerAgent:
    """
    Explores game states systematically by:
    1. Tracking unique frames via hash
    2. Trying untested actions from current state
    3. Using undo to backtrack when all actions tested
    4. Prioritizing actions that previously caused frame changes
    """

    def __init__(self, action_budget: int = 500, prefer_novel: bool = True):
        self.action_budget = action_budget
        self.prefer_novel = prefer_novel
        self.states: dict[str, StateNode] = {}
        self.current_hash: Optional[str] = None
        self.actions_taken = 0
        self.game_id = ""
        self.available_actions: list[int] = []
        self.undo_available = False
        self.successful_clicks: list[tuple[int, int]] = []
        self.prev_action_data: Optional[dict] = None

    def on_game_start(self, game_id: str, env_info: arc_agi.EnvironmentInfo) -> None:
        self.game_id = game_id
        self.states = {}
        self.current_hash = None
        self.actions_taken = 0

    def on_level_complete(self, level: int, transitions: list[Transition]) -> None:
        self.states = {}
        self.current_hash = None
        self.actions_taken = 0
        self.successful_clicks = []

    def _get_or_create_state(self, frame: np.ndarray, available_actions: list[int]) -> StateNode:
        h = frame_hash(frame)
        if h not in self.states:
            self.states[h] = StateNode(
                frame_hash=h,
                frame=frame.copy(),
                all_actions=[a for a in available_actions if a != 7],
            )
        self.states[h].visits += 1
        return self.states[h]

    def select_action(
        self,
        frame: np.ndarray,
        available_actions: list[int],
        history: list[Transition],
        levels_completed: int,
    ) -> tuple[GameAction, Optional[dict]]:
        self.available_actions = available_actions
        self.undo_available = 7 in available_actions
        self.actions_taken += 1

        if history and history[-1].frame_changed and self.prev_action_data:
            x, y = self.prev_action_data.get("x", 0), self.prev_action_data.get("y", 0)
            self.successful_clicks.append((y, x))

        state = self._get_or_create_state(frame, available_actions)
        self.current_hash = state.frame_hash

        untested = state.untested_actions
        if untested:
            action_id = self._pick_action(untested, state)
            action = ACTION_MAP[action_id]
            data = None
            if action == GameAction.ACTION6:
                data = self._pick_click_target(frame)
            state.tested_actions[action_id] = None
            self.prev_action_data = data
            return action, data

        if self.undo_available:
            self.prev_action_data = None
            return GameAction.ACTION7, None

        action_id = random.choice([a for a in available_actions if a != 7])
        action = ACTION_MAP[action_id]
        data = None
        if action == GameAction.ACTION6:
            data = self._pick_click_target(frame)
        self.prev_action_data = data
        return action, data

    def _pick_action(self, untested: list[int], state: StateNode) -> int:
        priority = []
        for a in untested:
            if a <= 5:
                priority.append((0, a))
            else:
                priority.append((1, a))
        priority.sort()
        return priority[0][1]

    def _pick_click_target(self, frame: np.ndarray) -> dict:
        bg_color = frame[0, 0]
        non_bg = np.argwhere(frame != bg_color)

        if len(non_bg) == 0:
            return {"x": random.randint(0, 63), "y": random.randint(0, 63)}

        if self.successful_clicks and random.random() < 0.3:
            y, x = random.choice(self.successful_clicks)
            dy, dx = random.randint(-3, 3), random.randint(-3, 3)
            return {"x": max(0, min(63, x + dx)), "y": max(0, min(63, y + dy))}

        unique_colors = np.unique(frame[frame != bg_color])
        if len(unique_colors) > 0:
            target_color = random.choice(unique_colors)
            color_pixels = np.argwhere(frame == target_color)
            if len(color_pixels) > 0:
                idx = random.randint(0, len(color_pixels) - 1)
                y, x = color_pixels[idx]
                return {"x": int(x), "y": int(y)}

        idx = random.randint(0, len(non_bg) - 1)
        y, x = non_bg[idx]
        return {"x": int(x), "y": int(y)}
