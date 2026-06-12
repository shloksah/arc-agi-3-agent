"""
FrugalExplorer: scoring-aware Go-Explore graph explorer (no neural network).

RHAE punishes every wasted action quadratically and zeroes any level past
5x the human action count, so the design goal is frugality, not coverage:

  * Stable state hashing — a per-level FROZEN mask removes volatile border
    pixels (step counters, timers) so identical play states hash the same.
    Frozen once so hashes never shift mid-level and corrupt the graph.
  * Candidate clicks snap to connected-component pixels (real objects),
    one candidate per (color, coarse cell). When a node is exhausted with
    nothing ever changing the frame, candidates expand tier by tier
    (object pixels -> coarse grid) instead of flailing randomly.
  * Candidates are ordered by the EffectModel, which persists across levels
    of a game and carries an exploration bonus for untried action types.
  * Return-then-explore: the graph records edges (state, action) -> state;
    when the current node is exhausted, BFS finds the nearest node with
    untested candidates and the agent replays the path to it.
  * Deaths are attributed to the action that caused them (per-level hard
    ban + cross-level soft penalty) before the recovery reset.
"""

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from arcengine import GameAction

from core.effect_model import EffectModel
from core.frame_parser import FrameParser
from core.game_runner import Transition

MASK_FREEZE_AFTER = 5     # transitions observed before the UI mask freezes
VOLATILE_THRESHOLD = 0.8  # pixel changes in >= this fraction of steps -> UI
MAX_CLICKS_PER_NODE = 64
VOLATILE_SENTINEL = 16    # colors are 0-15; masked pixels hash as 16
MAX_TIER = 3


@dataclass
class Node:
    candidates: list = field(default_factory=list)   # ordered action keys
    tested: dict = field(default_factory=dict)        # key -> frame_changed
    edges: dict = field(default_factory=dict)         # key -> resulting hash
    visits: int = 0
    tier: int = 0                                     # candidate expansion level


class FrugalExplorer:
    def __init__(self, action_budget: int = 0):
        self.action_budget = action_budget  # informational; harness enforces
        self.parser = FrameParser(min_object_area=2)

    # ── lifecycle hooks ──────────────────────────────────────────────────

    def on_game_start(self, game_id, env_info) -> None:
        self.game_id = game_id
        self.effect = EffectModel()      # persists across levels, not games
        self._reset_level_state()

    def on_level_complete(self, level, transitions) -> None:
        # Credit the winning action so its color/type is favored next level
        if self._last_key is not None:
            self.effect.update(self._last_key, self._last_frame,
                               changed=True, advanced=True)
        self._reset_level_state()

    def on_game_over(self, level, transitions) -> None:
        # Attribute the death BEFORE the recovery reset so the killing
        # action can never be re-selected forever.
        if self._last_key is not None:
            self.deadly.add(self._last_key)
            self.effect.record_death(self._last_key, self._last_frame)
        self._last_key = None
        self._last_hash = None
        self._plan.clear()

    def _reset_level_state(self):
        self.nodes: dict[str, Node] = {}
        self.deadly: set = set()
        self.mask: Optional[np.ndarray] = None   # True = volatile pixel
        self._change_counts = np.zeros((64, 64), dtype=np.int32)
        self._mask_obs = 0
        self._last_key = None
        self._last_hash = None
        self._last_frame = None
        self._undo_useless = False
        self._plan: deque = deque()   # (expected_hash, key) steps to frontier
        self._replan_cooldown = 0     # suppress planning after plan drift
        self._visited = np.zeros((64, 64), dtype=bool)  # avatar ground coverage
        self._move_plan: deque = deque()   # action keys toward a nav target
        self._move_target = None           # (y, x) we are walking toward
        self._blocked: set = set()         # lattice cells where a move failed
        self._nav_fail = 0                 # consecutive abandoned nav plans
        self._nav_rounds = 0               # completed coverage sweeps

    # ── state hashing with frozen UI mask ───────────────────────────────

    def _hash(self, frame: np.ndarray) -> str:
        # arcengine UI templates draw step counters / level pips along the
        # board edges (col 0 in lp85, row 63 in tu93, row 1 pips). They tick
        # every few actions regardless of play — too slowly for volatility
        # detection — and otherwise make identical play states hash
        # differently (state explosion, broken cycle detection). Always
        # exclude the outer ring plus row 1.
        frame = frame.copy()
        frame[:, 0] = frame[:, 63] = VOLATILE_SENTINEL
        frame[0:2, :] = VOLATILE_SENTINEL
        frame[63, :] = VOLATILE_SENTINEL
        if self.mask is not None:
            frame = np.where(self.mask, VOLATILE_SENTINEL, frame)
        return hashlib.md5(frame.tobytes()).hexdigest()[:12]

    def _observe_for_mask(self, t: Transition):
        if self.mask is not None:
            return
        if t.frame_changed:
            self._change_counts += (t.frame_before != t.frame_after)
            self._mask_obs += 1
        if self._mask_obs >= MASK_FREEZE_AFTER:
            volatile = self._change_counts >= VOLATILE_THRESHOLD * self._mask_obs
            # Only border bands can be UI (step counters / score readouts).
            # A mid-board avatar also changes every step — masking it would
            # collapse genuinely different states into one hash.
            border = np.zeros((64, 64), dtype=bool)
            border[:4, :] = border[-4:, :] = True
            border[:, :4] = border[:, -4:] = True
            self.mask = volatile & border
            # Mask frozen: old hashes are stale, rebuild the graph (cheap —
            # we are only a handful of actions into the level)
            self.nodes = {}
            self._plan.clear()

    # ── learning from the previous transition ───────────────────────────

    def _learn(self, history: list[Transition]):
        if not history or self._last_key is None:
            return
        t = history[-1]
        if t.action == GameAction.RESET:
            return
        self._observe_for_mask(t)
        # Learn from NOVELTY, not pixel change: in many games every click
        # repaints something (selection toggles, cursors), so "changed the
        # frame" carries no information. "Reached a state we had not seen
        # before" separates progress from churn, and no-ops are never novel.
        result_hash = self._hash(t.frame_after)
        novel = t.frame_changed and result_hash not in self.nodes
        self.effect.update(self._last_key, t.frame_before, novel)
        self.effect.learn_move(self._last_key, t.frame_before, t.frame_after,
                               int(t.frame_before[0, 0]))
        node = self.nodes.get(self._last_hash)
        if node is not None:
            node.tested[self._last_key] = t.frame_changed
            node.edges[self._last_key] = result_hash
        if self._last_key == 7 and not t.frame_changed:
            self._undo_useless = True

    # ── candidate generation ─────────────────────────────────────────────

    def _click_candidates(self, frame: np.ndarray, tier: int) -> list:
        """Tier 0: component centroids, deduped by (color, region).
        Tier 1: object corner pixels.
        Tier 2+: real non-background pixels, one per coarse cell, with a
        different pixel choice each tier — interactive elements are colored
        pixels, so precision sampling of them beats grid-center scans that
        mostly land on background."""
        clicks = []
        seen = set()

        def add(cx, cy, dedup_exact=False):
            cx, cy = max(0, min(63, cx)), max(0, min(63, cy))
            key = (cx, cy) if dedup_exact else (int(frame[cy, cx]), cy // 8, cx // 8)
            if key not in seen:
                seen.add(key)
                clicks.append(("click", cx, cy))

        if tier == 0:
            state = self.parser.parse(frame)
            for obj in sorted(state.objects, key=lambda o: o.area):
                cx, cy = int(round(obj.center[0])), int(round(obj.center[1])) + 1
                # Snap to a real object pixel if the centroid falls outside
                # (L-shapes, rings)
                if frame[min(63, cy), min(63, cx)] == state.background_color:
                    ys, xs = np.nonzero(obj.mask)
                    cy = int(ys[len(ys) // 2]) + obj.bbox[1] + 1
                    cx = int(xs[len(xs) // 2]) + obj.bbox[0]
                add(cx, cy)
            return clicks[:MAX_CLICKS_PER_NODE]

        if tier == 1:
            state = self.parser.parse(frame)
            for obj in sorted(state.objects, key=lambda o: o.area):
                x0, y0, x1, y1 = obj.bbox
                for cx, cy in ((x0, y0 + 1), (x1, y1 + 1),
                               (x0, y1 + 1), (x1, y0 + 1)):
                    add(cx, cy)
            return clicks[:24]

        # Tier 2+: one non-bg pixel per 8x8 cell; later tiers pick a
        # different representative so repeated expansions reach new pixels
        bg = frame[0, 0]
        pick = tier - 2  # 0: first pixel in cell, 1: last, ...
        for cy0 in range(0, 64, 8):
            for cx0 in range(0, 64, 8):
                cell = frame[cy0:cy0 + 8, cx0:cx0 + 8]
                ys, xs = np.nonzero(cell != bg)
                if len(ys) == 0:
                    continue
                i = min(pick * (len(ys) // 2 + 1), len(ys) - 1)
                add(cx0 + int(xs[i]), cy0 + int(ys[i]), dedup_exact=True)
        return clicks[:32]

    def _build_candidates(self, frame: np.ndarray, available: list[int],
                          tier: int = 0) -> list:
        keys = [a for a in available if 1 <= a <= 5] if tier == 0 else []
        if 6 in available:
            keys.extend(self._click_candidates(frame, tier))
        return keys

    def _expand(self, node: Node, frame: np.ndarray, available: list[int]) -> bool:
        """Add the next candidate tier. Returns True if new keys appeared."""
        while node.tier < MAX_TIER:
            node.tier += 1
            fresh = [k for k in self._build_candidates(frame, available, node.tier)
                     if k not in node.tested and k not in node.candidates]
            if fresh:
                node.candidates.extend(fresh)
                return True
        return False

    # ── frontier navigation (return-then-explore) ───────────────────────

    def _untested(self, node: Node) -> list:
        return [k for k in node.candidates
                if k not in node.tested and k not in self.deadly]

    def _plan_to_frontier(self, start_hash: str) -> bool:
        """BFS over known edges to the nearest node with untested candidates.
        Fills self._plan with (expected_hash, key) steps. Returns success."""
        parents = {start_hash: None}
        queue = deque([start_hash])
        target = None
        while queue:
            h = queue.popleft()
            node = self.nodes.get(h)
            if node is None:
                continue
            if h != start_hash and self._untested(node):
                target = h
                break
            for key, nxt in node.edges.items():
                if key in self.deadly or nxt in parents:
                    continue
                parents[nxt] = (h, key)
                queue.append(nxt)
        if target is None:
            return False
        steps = []
        h = target
        while parents[h] is not None:
            ph, key = parents[h]
            steps.append((ph, key))
            h = ph
        self._plan = deque(reversed(steps))
        return True

    # ── grid path planner ────────────────────────────────────────────────

    def _nav_targets(self, frame: np.ndarray):
        """Unvisited pixels of the rarest colors (goals/keys/exits are a
        few pixels; walls and floors are thousands)."""
        bg = frame[0, 0]
        interior = frame[2:63, 1:63]
        counts = np.bincount(interior.ravel(), minlength=17)
        # absolute smallness, not relative: goals/keys are a few px;
        # "3 rarest colors" in a maze can include the walls themselves
        rare = [c for c in np.argsort(counts) if 0 < counts[c] <= 60
                and c != bg and c != self.effect.avatar_color][:3]
        sel = np.isin(frame, rare) & ~self._visited
        sel[:, 0] = sel[:, 63] = sel[0:2, :] = sel[63, :] = False
        return np.argwhere(sel)

    def _plan_path(self, frame: np.ndarray) -> bool:
        """BFS over the movement lattice (stride = learned displacement)
        from the avatar to the nearest rare-color target, walking only
        floor-colored cells. Fills _move_plan with action keys."""
        if len(self.effect.moves) < 2 or not self.effect.floor_colors:
            return False
        pos = self._avatar_pos(frame)
        if pos is None:
            return False
        targets = self._nav_targets(frame)
        if len(targets) == 0:
            return False
        tset = {(int(y), int(x)) for y, x in targets}
        passable = self.effect.floor_colors | {self.effect.avatar_color}

        start = (int(round(pos[0])), int(round(pos[1])))
        parents = {start: None}
        queue = deque([start])
        goal = None
        steps = 0
        while queue and steps < 4000:
            cy, cx = queue.popleft()
            steps += 1
            if (cy, cx) in tset or any((cy + oy, cx + ox) in tset
                                       for oy in (-1, 0, 1) for ox in (-1, 0, 1)):
                goal = (cy, cx)
                break
            for key, (dy, dx) in self.effect.moves.items():
                ny, nx = cy + dy, cx + dx
                if not (2 <= ny <= 62 and 1 <= nx <= 62):
                    continue
                if (ny, nx) in parents or (ny, nx) in self._blocked:
                    continue
                if int(frame[ny, nx]) not in passable and (ny, nx) not in tset \
                        and not any((ny + oy, nx + ox) in tset
                                    for oy in (-1, 0, 1) for ox in (-1, 0, 1)):
                    continue
                parents[(ny, nx)] = ((cy, cx), key)
                queue.append((ny, nx))
        if goal is None or goal == start:
            return False
        keys = []
        cur = goal
        while parents[cur] is not None:
            prev, key = parents[cur]
            keys.append(key)
            cur = prev
        self._move_plan = deque(reversed(keys))
        self._move_target = goal
        return True

    def _follow_move_plan(self, frame: np.ndarray, history: list):
        """Pop the next nav step, abandoning the plan if the last step
        failed (blocked) or the avatar vanished."""
        if not self._move_plan:
            return None
        if history and self._last_key in self.effect.moves:
            t = history[-1]
            if t.action != GameAction.RESET and not t.frame_changed:
                # the move was blocked: remember the cell and re-plan
                pos = self._avatar_pos(frame)
                if pos is not None:
                    dy, dx = self.effect.moves[self._last_key]
                    self._blocked.add((int(round(pos[0])) + dy,
                                       int(round(pos[1])) + dx))
                self._move_plan.clear()
                self._move_target = None
                self._nav_fail += 1
                return None
        if self._avatar_pos(frame) is None:
            self._move_plan.clear()
            return None
        return self._move_plan.popleft()

    # ── avatar navigation bias ───────────────────────────────────────────

    def _avatar_pos(self, frame: np.ndarray):
        col = self.effect.avatar_color
        if col is None:
            return None
        ys, xs = np.nonzero(frame == col)
        if len(ys) == 0 or len(ys) > 64:
            return None
        return float(ys.mean()), float(xs.mean())

    def _nav_bonus(self, frame: np.ndarray):
        """Bonus per movement action for stepping toward the nearest
        unvisited non-background pixel. Turns random walks into directed
        sweeps in avatar games while keeping graph fallback intact."""
        if len(self.effect.moves) < 2:
            return {}
        pos = self._avatar_pos(frame)
        if pos is None:
            return {}
        ay, ax = pos
        self._visited[int(ay), int(ax)] = True
        bg = frame[0, 0]
        # Steer toward RARE colors (goals, keys, exits are a few pixels;
        # walls and floors are thousands) — nearest-any-pixel just pulls
        # into adjacent walls forever.
        interior = frame[2:63, 1:63]
        counts = np.bincount(interior.ravel(), minlength=17)
        rare = [c for c in np.argsort(counts) if counts[c] > 0
                and c != bg and c != self.effect.avatar_color][:3]
        sel = np.isin(frame, rare) & ~self._visited
        sel[:, 0] = sel[:, 63] = sel[0:2, :] = sel[63, :] = False
        ys, xs = np.nonzero(sel)
        if len(ys) == 0:
            return {}
        d = np.abs(ys - ay) + np.abs(xs - ax)
        i = int(d.argmin())
        ty, tx = float(ys[i]), float(xs[i])
        base = abs(ty - ay) + abs(tx - ax)
        bonus = {}
        for key, (dy, dx) in self.effect.moves.items():
            new = abs(ty - (ay + dy)) + abs(tx - (ax + dx))
            if new < base:
                bonus[key] = 0.8
        return bonus

    # ── action selection ─────────────────────────────────────────────────

    def select_action(self, frame, available_actions, history, levels_completed):
        self._learn(history)

        h = self._hash(frame)
        node = self.nodes.get(h)
        if node is None:
            node = Node(candidates=self._build_candidates(frame, available_actions))
            self.nodes[h] = node
        node.visits += 1

        key = None

        # Follow an active plan while reality matches it
        if self._plan:
            expected_hash, planned_key = self._plan[0]
            if expected_hash == h:
                self._plan.popleft()
                key = planned_key
            else:
                # Drifted off the known path (animation, nondeterminism).
                # Re-planning every step thrashes on big state spaces, so
                # back off and explore locally for a few actions instead.
                self._plan.clear()
                self._replan_cooldown = 3

        # Grid navigation: in movement games, walk a planned path to the
        # nearest rare-color target instead of testing actions cell by cell
        if key is None:
            nav_key = self._follow_move_plan(frame, history)
            if nav_key is None and self._nav_fail < 12:
                if self._plan_path(frame):
                    nav_key = self._move_plan.popleft()
                elif (self._visited.any() and len(self.effect.moves) >= 2
                      and self._nav_rounds < 6):
                    # All targets visited but not won: carry/deliver games
                    # need REVISITS (picked-up items vanish; drop zones must
                    # be re-entered per item). Start a fresh coverage round.
                    self._visited[:] = False
                    self._nav_rounds += 1
                    if self._plan_path(frame):
                        nav_key = self._move_plan.popleft()
            if nav_key is not None:
                if not self._move_plan and self._move_target is not None:
                    ty, tx = self._move_target
                    self._visited[max(0, ty - 2):ty + 3,
                                  max(0, tx - 2):tx + 3] = True
                    self._move_target = None
                key = nav_key

        if key is None:
            untested = self._untested(node)
            if not untested:
                # Navigate to the nearest node that still has untested keys
                if self._replan_cooldown > 0:
                    self._replan_cooldown -= 1
                elif self._plan_to_frontier(h):
                    _, key = self._plan.popleft()
                if key is None:
                    # Whole known graph exhausted: deepen this node's tiers
                    if self._expand(node, frame, available_actions):
                        untested = self._untested(node)
                    elif (7 in available_actions and not self._undo_useless
                          and node.visits <= 3):
                        key = 7
            if key is None:
                if untested:
                    nav = self._nav_bonus(frame)
                    key = max(untested,
                              key=lambda k: self.effect.priority(k, frame)
                              + nav.get(k, 0.0))
                else:
                    # Fully exhausted: repeat something that changed the
                    # frame before (movement chains need repetition)
                    changed = [k for k in node.candidates
                               if node.tested.get(k) and k not in self.deadly]
                    if changed:
                        key = random.choice(changed)
                    elif 6 in available_actions:
                        # NOTHING here has ever changed the frame: re-clicking
                        # tested-dead pixels is pure waste — probe a fresh
                        # untested non-bg pixel instead
                        bg = frame[0, 0]
                        ys, xs = np.nonzero(frame != bg)
                        if len(ys):
                            i = random.randrange(len(ys))
                            key = ("click", int(xs[i]), int(ys[i]))
                        else:
                            key = ("click", random.randrange(64),
                                   random.randrange(64))
                    else:
                        pool = ([k for k in node.candidates if k not in self.deadly]
                                or node.candidates
                                or [a for a in available_actions if a != 7])
                        key = random.choice(pool)

        self._last_key = key
        self._last_hash = h
        self._last_frame = frame

        if isinstance(key, tuple):
            return GameAction.ACTION6, {"x": key[1], "y": key[2]}
        return GameAction.from_name(f"ACTION{key}"), None
