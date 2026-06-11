"""
EffectModel: training-free running estimate of which actions tend to change
the frame. The cheap stand-in for StochasticGoose's CNN — updated after every
real transition, no torch, no warm-up cost.

Persists across levels of one game: mechanics usually carry over, and the
scoring overweights later levels, so transferred knowledge compounds exactly
where the points are.

Action keys take two forms:
    int 1..5            — simple actions (ACTION1-ACTION5)
    ("click", x, y)     — ACTION6 at frame coordinates
"""

import numpy as np

GRID = 8  # coarse cells for the click heatmap (64/8 = 8px per cell)


class EffectModel:
    def __init__(self):
        # key -> [changes, tries], Laplace-smoothed so untried keys keep an
        # optimistic prior of 0.5
        self.simple = {a: [1.0, 2.0] for a in range(1, 6)}
        self.color = {}
        self.heat = np.ones((GRID, GRID, 2), dtype=np.float64)
        self.heat[:, :, 1] += 1.0

        self.adv_keys = set()      # simple actions that ever advanced a level
        self.adv_colors = set()    # clicked colors that ever advanced a level
        self.deadly_simple = {}    # action -> death count (soft, cross-level)
        self.deadly_color = {}     # clicked color -> death count
        self.focus_color = None    # a color with proven novelty rate, if any

        self.move_votes = {}       # action -> {(dy, dx): count}
        self.moves = {}            # action -> confirmed (dy, dx)
        self.avatar_color = None   # color of the sprite that movement shifts
        self.floor_votes = {}      # color -> times revealed under the avatar
        self.floor_colors = set()  # colors the avatar can stand on

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
        """Record the outcome of an executed action."""
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
            # Exploit-on-success: once a color has a proven track record of
            # reaching novel states, it becomes the focus and breadth-seeking
            # on other colors is damped (see priority()).
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

    def learn_move(self, key, before, after, bg):
        """If a simple action translated a small sprite, vote for its
        displacement. Tracks per-color centroids, so it works regardless of
        what terrain the sprite moves over (trail-painting, colored floors).
        Two consistent votes confirm the action as movement and identify
        the avatar color. Purely advisory — it only shapes priorities, so a
        wrong guess is never fatal."""
        if isinstance(key, tuple):
            return
        if not (before != after).any():
            return
        # interior only: border rings are engine UI (counters/pips)
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
                # Floor = what gets revealed where the avatar just was
                vacated = (b == color) & (a != color)
                for fc in np.unique(a[vacated]):
                    self.floor_votes[int(fc)] = self.floor_votes.get(int(fc), 0) + 1
                    if self.floor_votes[int(fc)] >= 2:
                        self.floor_colors.add(int(fc))

    def record_death(self, key, frame):
        if isinstance(key, tuple):
            col = self._clicked_color(key, frame)
            self.deadly_color[col] = self.deadly_color.get(col, 0) + 1
        else:
            self.deadly_simple[key] = self.deadly_simple.get(key, 0) + 1

    def priority(self, key, frame):
        """Expected usefulness of a candidate action; higher is better.

        Includes an exploration bonus that decays with global tries, so a
        never-tried action TYPE outranks a well-known "changes the frame"
        action (guards against tunnel vision on e.g. selection-toggle
        clicks that always change pixels but never progress the game).
        """
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
