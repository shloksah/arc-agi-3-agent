# =====================================================================
# FrugalExplorer — scoring-aware Go-Explore graph agent (no neural net)
# Kaggle submission build: self-contained port of core/frugal_explorer.py
# + core/effect_model.py onto the official ARC-AGI-3-Agents interface.
#
# RHAE scoring punishes wasted actions quadratically and zeroes any
# level past 5x the human action count, so the design goal is
# frugality: stable state hashing (engine UI masked), effect-model
# ranked candidates, return-then-explore frontier navigation, and
# death attribution. See github repo for the development history.
# =====================================================================
import hashlib
import random
import time
import traceback
from collections import deque
from typing import Any

import numpy as np
from scipy import ndimage

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

GRID = 8                  # coarse cells for the click heatmap
MASK_FREEZE_AFTER = 5     # transitions observed before the UI mask freezes
VOLATILE_THRESHOLD = 0.8  # pixel changes in >= this fraction of steps -> UI
MAX_CLICKS_PER_NODE = 64
VOLATILE_SENTINEL = 16    # colors are 0-15; masked pixels hash as 16
MAX_TIER = 3

PER_GAME_SECONDS = 7 * 60          # wall-clock budget per game
TOTAL_SECONDS = 8 * 3600 - 5 * 60  # global bail before the 9h limit
RUN_START = time.time()


class EffectModel:
    """Training-free tallies of which actions reach novel states."""

    def __init__(self):
        self.simple = {a: [1.0, 2.0] for a in range(1, 6)}
        self.color = {}
        self.heat = np.ones((GRID, GRID, 2), dtype=np.float64)
        self.heat[:, :, 1] += 1.0
        self.adv_keys = set()
        self.adv_colors = set()
        self.deadly_simple = {}
        self.deadly_color = {}
        self.focus_color = None
        self.move_votes = {}
        self.moves = {}
        self.avatar_color = None

    @staticmethod
    def _rate(pair):
        return pair[0] / pair[1]

    def _cell(self, x, y):
        return min(GRID - 1, y * GRID // 64), min(GRID - 1, x * GRID // 64)

    def _clicked_color(self, key, frame):
        _, x, y = key
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            return int(frame[y, x])
        return -1

    def update(self, key, frame, changed, advanced=False):
        if isinstance(key, tuple):
            col = self._clicked_color(key, frame)
            pair = self.color.setdefault(col, [1.0, 2.0])
            pair[0] += 1.0 if changed else 0.0
            pair[1] += 1.0
            gy, gx = self._cell(key[1], key[2])
            self.heat[gy, gx, 0] += 1.0 if changed else 0.0
            self.heat[gy, gx, 1] += 1.0
            if advanced:
                self.adv_colors.add(col)
            best, best_rate = None, 0.45
            for c, p in self.color.items():
                if p[1] >= 8 and p[0] / p[1] > best_rate:
                    best, best_rate = c, p[0] / p[1]
            self.focus_color = best
        else:
            pair = self.simple.setdefault(key, [1.0, 2.0])
            pair[0] += 1.0 if changed else 0.0
            pair[1] += 1.0
            if advanced:
                self.adv_keys.add(key)

    def record_death(self, key, frame):
        if isinstance(key, tuple):
            col = self._clicked_color(key, frame)
            self.deadly_color[col] = self.deadly_color.get(col, 0) + 1
        else:
            self.deadly_simple[key] = self.deadly_simple.get(key, 0) + 1

    def learn_move(self, key, before, after, bg):
        """Per-color centroid tracking: a small sprite whose centroid
        displaces consistently with an action is the avatar, regardless
        of terrain (trails, colored floors)."""
        if isinstance(key, tuple):
            return
        if not (before != after).any():
            return
        b, a = before[2:63, 1:63], after[2:63, 1:63]
        for color in np.unique(b):
            if color == bg:
                continue
            bys, bxs = np.nonzero(b == color)
            if not (1 <= len(bys) <= 40):
                continue
            ays, axs = np.nonzero(a == color)
            if len(ays) == 0 or abs(len(ays) - len(bys)) > len(bys) // 2 + 2:
                continue
            dy = int(round(ays.mean() - bys.mean()))
            dx = int(round(axs.mean() - bxs.mean()))
            if (dy == 0 and dx == 0) or abs(dy) > 10 or abs(dx) > 10:
                continue
            votes = self.move_votes.setdefault(key, {})
            votes[(dy, dx, int(color))] = votes.get((dy, dx, int(color)), 0) + 1
            if votes[(dy, dx, int(color))] >= 2:
                self.moves[key] = (dy, dx)
                self.avatar_color = int(color)

    def priority(self, key, frame):
        if isinstance(key, tuple):
            col = self._clicked_color(key, frame)
            gy, gx = self._cell(key[1], key[2])
            pair = self.color.get(col, [1.0, 2.0])
            p = 0.5 * self._rate(pair)
            p += 0.5 * (self.heat[gy, gx, 0] / self.heat[gy, gx, 1])
            bonus = 1.5 / (pair[1] ** 0.5)
            if self.focus_color is not None and col != self.focus_color:
                bonus *= 0.25
            p += bonus
            if col in self.adv_colors:
                p += 0.6
            p -= 0.4 * self.deadly_color.get(col, 0)
            return p
        pair = self.simple.get(key, [1.0, 2.0])
        p = self._rate(pair)
        p += 1.5 / (pair[1] ** 0.5)
        if key in self.adv_keys:
            p += 0.6
        p -= 0.4 * self.deadly_simple.get(key, 0)
        return p


class Node:
    __slots__ = ("candidates", "tested", "edges", "visits", "tier")

    def __init__(self, candidates):
        self.candidates = candidates
        self.tested = {}
        self.edges = {}
        self.visits = 0
        self.tier = 0


def find_objects(frame, bg):
    """Connected components PER COLOR (embedded blocks must not vanish
    as holes inside one giant bg-vs-rest component)."""
    objects = []  # (area, color, cx, cy, bbox)
    for color in np.unique(frame):
        if color == bg:
            continue
        labeled, n = ndimage.label(frame == color)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(labeled == i)
            if len(ys) < 2:
                continue
            objects.append((len(ys), int(color),
                            int(round(xs.mean())), int(round(ys.mean())),
                            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                            (ys, xs)))
    objects.sort(key=lambda o: o[0])
    return objects


class MyAgent(Agent):
    """Frugal Go-Explore graph explorer with a training-free effect model."""

    MAX_ACTIONS = 4000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(hash(self.game_id) & 0xFFFFFFFF)
        self.game_start = time.time()
        self.effect = EffectModel()
        self._level = -1
        self._reset_level_state()
        print(f"[FrugalExplorer] start game {self.game_id}")

    # ── per-level state ──────────────────────────────────────────────

    def _reset_level_state(self):
        self.nodes = {}
        self.deadly = set()
        self.mask = None
        self._change_counts = np.zeros((64, 64), dtype=np.int32)
        self._mask_obs = 0
        self._last_key = None
        self._last_hash = None
        self._last_frame = None
        self._undo_useless = False
        self._plan = deque()
        self._replan_cooldown = 0
        self._visited = np.zeros((64, 64), dtype=bool)

    # ── frame helpers ────────────────────────────────────────────────

    @staticmethod
    def _raw(latest_frame):
        arr = np.array(latest_frame.frame, dtype=np.int64)
        return arr[-1] if arr.ndim == 3 else arr

    @staticmethod
    def _available(latest_frame):
        out = []
        for a in getattr(latest_frame, "available_actions", []) or []:
            v = getattr(a, "value", a)
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                pass
        return out or [1, 2, 3, 4, 5, 6]

    def _hash(self, frame):
        # arcengine UI templates draw step counters / level pips along the
        # board edges (col 0, row 63, row 1 — varies by game). They tick
        # regardless of play and must never reach the state hash (state
        # explosion / broken cycle detection otherwise). Mask the full ring.
        f = frame.copy()
        f[:, 0] = f[:, 63] = VOLATILE_SENTINEL
        f[0:2, :] = VOLATILE_SENTINEL
        f[63, :] = VOLATILE_SENTINEL
        if self.mask is not None:
            f = np.where(self.mask, VOLATILE_SENTINEL, f)
        return hashlib.md5(f.tobytes()).hexdigest()[:12]

    def _observe_for_mask(self, before, after):
        if self.mask is not None:
            return
        diff = before != after
        if diff.any():
            self._change_counts += diff
            self._mask_obs += 1
        if self._mask_obs >= MASK_FREEZE_AFTER:
            volatile = self._change_counts >= VOLATILE_THRESHOLD * self._mask_obs
            border = np.zeros((64, 64), dtype=bool)
            border[:4, :] = border[-4:, :] = True
            border[:, :4] = border[:, -4:] = True
            self.mask = volatile & border
            self.nodes = {}
            self._plan.clear()

    # ── candidate generation ─────────────────────────────────────────

    def _click_candidates(self, frame, tier):
        clicks = []
        seen = set()
        bg = int(frame[0, 0])

        def add(cx, cy, dedup_exact=False):
            cx, cy = max(0, min(63, cx)), max(0, min(63, cy))
            key = (cx, cy) if dedup_exact else (int(frame[cy, cx]), cy // 8, cx // 8)
            if key not in seen:
                seen.add(key)
                clicks.append(("click", cx, cy))

        if tier == 0:
            for area, color, cx, cy, bbox, (ys, xs) in find_objects(frame, bg):
                if frame[min(63, cy), min(63, cx)] != color:
                    m = len(ys) // 2
                    cy, cx = int(ys[m]), int(xs[m])
                add(cx, cy)
            return clicks[:MAX_CLICKS_PER_NODE]

        if tier == 1:
            for area, color, cx, cy, (x0, y0, x1, y1), _ in find_objects(frame, bg):
                for px, py in ((x0, y0), (x1, y1), (x0, y1), (x1, y0)):
                    add(px, py)
            return clicks[:24]

        pick = tier - 2
        for cy0 in range(0, 64, 8):
            for cx0 in range(0, 64, 8):
                cell = frame[cy0:cy0 + 8, cx0:cx0 + 8]
                ys, xs = np.nonzero(cell != bg)
                if len(ys) == 0:
                    continue
                i = min(pick * (len(ys) // 2 + 1), len(ys) - 1)
                add(cx0 + int(xs[i]), cy0 + int(ys[i]), dedup_exact=True)
        return clicks[:32]

    def _build_candidates(self, frame, available, tier=0):
        keys = [a for a in available if 1 <= a <= 5] if tier == 0 else []
        if 6 in available:
            keys.extend(self._click_candidates(frame, tier))
        return keys

    def _expand(self, node, frame, available):
        while node.tier < MAX_TIER:
            node.tier += 1
            fresh = [k for k in self._build_candidates(frame, available, node.tier)
                     if k not in node.tested and k not in node.candidates]
            if fresh:
                node.candidates.extend(fresh)
                return True
        return False

    # ── frontier navigation ──────────────────────────────────────────

    def _untested(self, node):
        return [k for k in node.candidates
                if k not in node.tested and k not in self.deadly]

    def _plan_to_frontier(self, start_hash):
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

    # ── avatar navigation bias ───────────────────────────────────────

    def _avatar_pos(self, frame):
        col = self.effect.avatar_color
        if col is None:
            return None
        ys, xs = np.nonzero(frame == col)
        if len(ys) == 0 or len(ys) > 64:
            return None
        return float(ys.mean()), float(xs.mean())

    def _nav_bonus(self, frame):
        if len(self.effect.moves) < 2:
            return {}
        pos = self._avatar_pos(frame)
        if pos is None:
            return {}
        ay, ax = pos
        self._visited[int(ay), int(ax)] = True
        bg = frame[0, 0]
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

    # ── learning from the previous transition ────────────────────────

    def _learn(self, current_frame):
        if self._last_key is None or self._last_frame is None:
            return
        before, after = self._last_frame, current_frame
        self._observe_for_mask(before, after)
        changed = not np.array_equal(before, after)
        result_hash = self._hash(after)
        novel = changed and result_hash not in self.nodes
        self.effect.update(self._last_key, before, novel)
        self.effect.learn_move(self._last_key, before, after, int(before[0, 0]))
        node = self.nodes.get(self._last_hash)
        if node is not None:
            node.tested[self._last_key] = changed
            node.edges[self._last_key] = result_hash
        if self._last_key == 7 and not changed:
            self._undo_useless = True

    # ── official interface ───────────────────────────────────────────

    def is_done(self, frames, latest_frame):
        try:
            return any([
                latest_frame.state is GameState.WIN,
                (time.time() - self.game_start) >= PER_GAME_SECONDS,
                (time.time() - RUN_START) >= TOTAL_SECONDS,
            ])
        except Exception:
            traceback.print_exc()
            return True

    def choose_action(self, frames, latest_frame):
        try:
            return self._choose(latest_frame)
        except Exception:
            traceback.print_exc()
            self._last_key = None
            return random.choice([GameAction.ACTION1, GameAction.ACTION2,
                                  GameAction.ACTION3, GameAction.ACTION4])

    def _choose(self, latest_frame):
        state = latest_frame.state

        if state is GameState.NOT_PLAYED:
            self._last_key = None
            return GameAction.RESET

        if state is GameState.GAME_OVER:
            # Attribute the death BEFORE the recovery reset
            if self._last_key is not None:
                self.deadly.add(self._last_key)
                self.effect.record_death(self._last_key, self._last_frame)
            self._last_key = None
            self._last_hash = None
            self._plan.clear()
            return GameAction.RESET

        level = latest_frame.levels_completed
        if level != self._level:
            if self._level >= 0 and level > self._level and self._last_key is not None:
                # credit the winning action's type/color for the next level
                self.effect.update(self._last_key, self._last_frame,
                                   changed=True, advanced=True)
                print(f"[FrugalExplorer] {self.game_id} level {self._level} "
                      f"solved at action {self.action_counter}")
            self._level = level
            self._reset_level_state()

        frame = self._raw(latest_frame)
        available = self._available(latest_frame)

        self._learn(frame)

        h = self._hash(frame)
        node = self.nodes.get(h)
        if node is None:
            node = Node(self._build_candidates(frame, available))
            self.nodes[h] = node
        node.visits += 1

        key = None

        if self._plan:
            expected_hash, planned_key = self._plan[0]
            if expected_hash == h:
                self._plan.popleft()
                key = planned_key
            else:
                self._plan.clear()
                self._replan_cooldown = 3

        if key is None:
            untested = self._untested(node)
            if not untested:
                if self._replan_cooldown > 0:
                    self._replan_cooldown -= 1
                elif self._plan_to_frontier(h):
                    _, key = self._plan.popleft()
                if key is None:
                    if self._expand(node, frame, available):
                        untested = self._untested(node)
                    elif (7 in available and not self._undo_useless
                          and node.visits <= 3):
                        key = 7
            if key is None:
                if untested:
                    nav = self._nav_bonus(frame)
                    key = max(untested,
                              key=lambda k: self.effect.priority(k, frame)
                              + nav.get(k, 0.0))
                else:
                    changed = [k for k in node.candidates
                               if node.tested.get(k) and k not in self.deadly]
                    if changed:
                        key = random.choice(changed)
                    elif 6 in available:
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
                                or node.candidates or [1])
                        key = random.choice(pool)

        self._last_key = key
        self._last_hash = h
        self._last_frame = frame

        if isinstance(key, tuple):
            action = GameAction.ACTION6
            action.set_data({"x": int(key[1]), "y": int(key[2])})
            return action
        return GameAction.from_name(f"ACTION{key}")
