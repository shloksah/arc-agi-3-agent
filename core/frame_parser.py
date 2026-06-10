"""
Frame Parser: Converts raw 64x64 frames into structured game state.
Identifies objects, tracks changes, and provides compact state representations.
"""

from dataclasses import dataclass, field
import hashlib
import numpy as np
from scipy import ndimage


@dataclass
class GameObject:
    id: int
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    center: tuple[float, float]
    colors: list[int]
    dominant_color: int
    area: int
    mask: np.ndarray

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1


@dataclass
class FrameDiff:
    pixels_changed: int
    regions_changed: list[tuple[int, int, int, int]]  # bboxes of changed regions
    new_colors: list[int]
    removed_colors: list[int]
    objects_moved: list[tuple[int, tuple[float, float], tuple[float, float]]]  # id, old_center, new_center


@dataclass
class GameState:
    frame: np.ndarray
    frame_hash: str
    background_color: int
    objects: list[GameObject]
    status_bar: np.ndarray
    game_area: np.ndarray
    color_counts: dict[int, int]

    def describe_compact(self) -> str:
        """Compact text representation for LLM consumption."""
        lines = []
        lines.append(f"bg={self.background_color} objects={len(self.objects)}")
        for obj in self.objects:
            x, y = int(obj.center[0]), int(obj.center[1])
            lines.append(f"  obj{obj.id}: pos=({x},{y}) size={obj.width}x{obj.height} color={obj.dominant_color} area={obj.area}")
        return "\n".join(lines)

    def describe_grid(self, downsample: int = 4) -> str:
        """Downsampled grid representation for LLM."""
        h, w = self.game_area.shape
        small = self.game_area[::downsample, ::downsample]
        lines = []
        for row in small:
            lines.append("".join(f"{c:x}" for c in row))
        return "\n".join(lines)


class FrameParser:
    def __init__(self, min_object_area: int = 4):
        self.min_object_area = min_object_area
        self.prev_state: GameState | None = None

    def parse(self, frame: np.ndarray) -> GameState:
        bg_color = self._detect_background(frame)
        status_bar = frame[0:1, :]
        game_area = frame[1:, :]

        objects = self._find_objects(game_area, bg_color)
        color_counts = {}
        for c in np.unique(frame):
            color_counts[int(c)] = int(np.sum(frame == c))

        state = GameState(
            frame=frame.copy(),
            frame_hash=hashlib.md5(frame.tobytes()).hexdigest()[:12],
            background_color=int(bg_color),
            objects=objects,
            status_bar=status_bar,
            game_area=game_area,
            color_counts=color_counts,
        )
        self.prev_state = state
        return state

    def compute_diff(self, state_a: GameState, state_b: GameState) -> FrameDiff:
        diff_mask = state_a.frame != state_b.frame
        pixels_changed = int(np.sum(diff_mask))

        regions = []
        if pixels_changed > 0:
            labeled, num = ndimage.label(diff_mask)
            for i in range(1, num + 1):
                pixels = np.argwhere(labeled == i)
                y_min, x_min = pixels.min(axis=0)
                y_max, x_max = pixels.max(axis=0)
                regions.append((int(x_min), int(y_min), int(x_max), int(y_max)))

        colors_a = set(np.unique(state_a.frame))
        colors_b = set(np.unique(state_b.frame))
        new_colors = sorted(colors_b - colors_a)
        removed_colors = sorted(colors_a - colors_b)

        objects_moved = self._match_moved_objects(state_a.objects, state_b.objects)

        return FrameDiff(
            pixels_changed=pixels_changed,
            regions_changed=regions,
            new_colors=[int(c) for c in new_colors],
            removed_colors=[int(c) for c in removed_colors],
            objects_moved=objects_moved,
        )

    def _detect_background(self, frame: np.ndarray) -> int:
        corners = [frame[0, 0], frame[0, -1], frame[-1, 0], frame[-1, -1]]
        edges = np.concatenate([frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]])
        values, counts = np.unique(edges, return_counts=True)
        return int(values[np.argmax(counts)])

    def _find_objects(self, game_area: np.ndarray, bg_color: int) -> list[GameObject]:
        # Segment PER COLOR, not bg-vs-rest: a board region with embedded
        # colored blocks would otherwise merge into one giant component and
        # the blocks (often the interactive elements) would vanish as holes.
        objects = []
        obj_id = 0
        for color in np.unique(game_area):
            if color == bg_color:
                continue
            labeled, num_features = ndimage.label(game_area == color)
            for i in range(1, num_features + 1):
                component = (labeled == i)
                pixels = np.argwhere(component)

                if len(pixels) < self.min_object_area:
                    continue

                y_min, x_min = pixels.min(axis=0)
                y_max, x_max = pixels.max(axis=0)
                center_y = float(pixels[:, 0].mean())
                center_x = float(pixels[:, 1].mean())

                obj_id += 1
                objects.append(GameObject(
                    id=obj_id,
                    bbox=(int(x_min), int(y_min), int(x_max), int(y_max)),
                    center=(center_x, center_y),
                    colors=[int(color)],
                    dominant_color=int(color),
                    area=len(pixels),
                    mask=component[y_min:y_max+1, x_min:x_max+1],
                ))

        objects.sort(key=lambda o: (o.bbox[1], o.bbox[0]))
        return objects

    def _match_moved_objects(
        self, objects_a: list[GameObject], objects_b: list[GameObject]
    ) -> list[tuple[int, tuple[float, float], tuple[float, float]]]:
        moved = []
        for obj_a in objects_a:
            best_match = None
            best_dist = float("inf")
            for obj_b in objects_b:
                if obj_b.area != obj_a.area or obj_b.dominant_color != obj_a.dominant_color:
                    continue
                dx = obj_a.center[0] - obj_b.center[0]
                dy = obj_a.center[1] - obj_b.center[1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < best_dist and dist > 0.5:
                    best_dist = dist
                    best_match = obj_b
            if best_match and best_dist < 20:
                moved.append((obj_a.id, obj_a.center, best_match.center))
        return moved
