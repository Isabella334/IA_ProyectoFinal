import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathfinding.astar import Vector2i, find_path


# ── State colours ─────────────────────────────────────────────────────────────
STATE_COLORS = {
    "patrol":      "#378ADD",
    "alert":       "#EF9F27",
    "investigate": "#D85A30",
    "chase":       "#E24B4A",
}
STATE_LABELS = {
    "patrol":      "PATROL",
    "alert":       "ALERT",
    "investigate": "INVESTIGATE",
    "chase":       "CHASE",
}


# ── JSON Random Forest inference ──────────────────────────────────────────────
class JSONForest:
    """Runs inference on the serialised npc_model.json without scikit-learn."""

    def __init__(self, path: str = "models/npc_model.json"):
        with open(path, "r") as f:
            data = json.load(f)
        self._classes  = data["classes"]
        self._trees    = data["trees"]
        self._features = data["feature_names"]

    def predict(self, features: dict) -> str:
        """
        Parameters
        ----------
        features : dict  keys must match FEATURE_COLS
        Returns the predicted class label (e.g. 'chase').
        """
        x = [features[k] for k in self._features]
        votes = [0] * len(self._classes)
        for tree in self._trees:
            idx = self._predict_tree(tree, x)
            votes[idx] += 1
        return self._classes[int(np.argmax(votes))]

    def _predict_tree(self, node: dict, x: list) -> int:
        if node["leaf"]:
            return node["class_idx"]
        if x[node["feature_index"]] <= node["threshold"]:
            return self._predict_tree(node["left"], x)
        return self._predict_tree(node["right"], x)


# ── Grid manager ──────────────────────────────────────────────────────────────
class GridManager:
    def __init__(self, width: int, height: int, walls: set):
        self.width  = width
        self.height = height
        self._walls = walls

    def is_walkable(self, pos: Vector2i) -> bool:
        if not (0 <= pos.x < self.width and 0 <= pos.y < self.height):
            return False
        return (pos.x, pos.y) not in self._walls

    def wall_array(self) -> np.ndarray:
        arr = np.zeros((self.height, self.width), dtype=bool)
        for (x, y) in self._walls:
            if 0 <= y < self.height and 0 <= x < self.width:
                arr[y, x] = True
        return arr


# ── Pre-built house layout ────────────────────────────────────────────────────
class HouseMap:
    WIDTH  = 20
    HEIGHT = 16

    PATROL_WAYPOINTS = [
        Vector2i(4,  3),
        Vector2i(14, 3),
        Vector2i(9,  8),
        Vector2i(4,  12),
        Vector2i(14, 12),
    ]

    INVESTIGATION_SPOTS = [
        Vector2i(9, 6),
        Vector2i(9, 9),
        Vector2i(9, 1),
    ]

    THIEF_POSITION = Vector2i(14, 12)

    @classmethod
    def default(cls) -> GridManager:
        walls = set()
        for x in range(cls.WIDTH):
            walls.add((x, 0)); walls.add((x, cls.HEIGHT - 1))
        for y in range(cls.HEIGHT):
            walls.add((0, y)); walls.add((cls.WIDTH - 1, y))
        for y in range(1, 7):
            walls.add((9, y))
        for x in range(1, 9):
            walls.add((x, 7))
        for x in range(10, 19):
            walls.add((x, 7))
        for x in range(1, 9):
            walls.add((x, 9))
        for x in range(10, 19):
            walls.add((x, 9))
        for y in range(10, 15):
            walls.add((9, y))
        for d in [(9,3),(4,7),(14,7),(4,9),(14,9)]:
            walls.discard(d)
        return GridManager(cls.WIDTH, cls.HEIGHT, walls)

    @classmethod
    def room_labels(cls):
        return [
            (4,  3,  "Living Room"),
            (14, 3,  "Kitchen"),
            (9,  8,  "Hallway"),
            (4,  12, "Bedroom 1"),
            (14, 12, "Bedroom 2"),
        ]


# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(npc: Vector2i, thief: Vector2i, paranoia: float,
                   noise: float, has_target: bool) -> dict:
    """
    Builds the 6-feature vector the RF model expects from game state.

    player_visible     : 1 if thief is in the same room / adjacent tile
    visibility_score   : decays with distance (0-1)
    player_distance    : Euclidean pixel-equivalent distance
    noise_level        : passed in directly (increases when thief moves)
    paranoia_level     : accumulates over time
    has_investigation_target : 1 if NPC has a waypoint to check
    """
    dist = math.sqrt((npc.x - thief.x)**2 + (npc.y - thief.y)**2)
    visible = 1.0 if dist <= 3.0 else 0.0
    vis_score = max(0.0, 1.0 - dist / 20.0)
    return {
        "player_visible":          visible,
        "visibility_score":        round(vis_score, 4),
        "player_distance":         round(dist * 10, 4),   # scale to match training data
        "noise_level":             round(noise, 4),
        "paranoia_level":          round(paranoia, 4),
        "has_investigation_target": 1.0 if has_target else 0.0,
    }


# ── Static plotter ────────────────────────────────────────────────────────────
class AStarPlot:
    def __init__(self, grid: GridManager, cell_size: int = 40):
        self._grid = grid
        self._cell_size = cell_size

    def find_path(self, start: Vector2i, goal: Vector2i) -> list:
        return find_path(start, goal, self._grid)

    def show(self, path, *, npc_state="patrol", start=None, goal=None,
             title="NPC Pathfinding — A*"):
        fig = self._build_figure(path, npc_state=npc_state, start=start,
                                 goal=goal, title=title)
        plt.show()
        plt.close(fig)

    def save(self, path, *, npc_state="patrol", start=None, goal=None,
             title="NPC Pathfinding — A*", output_path="models/astar_plot.png",
             dpi=120):
        fig = self._build_figure(path, npc_state=npc_state, start=start,
                                 goal=goal, title=title)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  A* plot saved → {out.resolve()}")

    def _build_figure(self, path, *, npc_state, start, goal, title):
        W, H   = self._grid.width, self._grid.height
        walls  = self._grid.wall_array()
        color  = STATE_COLORS.get(npc_state, "#888780")
        label  = STATE_LABELS.get(npc_state, npc_state.upper())

        fig, ax = plt.subplots(figsize=(W * 0.6, H * 0.6))
        fig.patch.set_facecolor("#1A1A2E")
        ax.set_facecolor("#16213E")

        path_set = set((p.x, p.y) for p in path)

        for y in range(H):
            for x in range(W):
                is_wall = walls[y, x]
                if is_wall:
                    fc = "#0F3460"; ec = "#0A2040"
                elif (x, y) in path_set:
                    fc = color + "28"; ec = "#1E2F50"
                else:
                    fc = "#1A2744"; ec = "#1E2F50"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x, H-1-y), 1, 1, boxstyle="square,pad=0",
                    facecolor=fc, edgecolor=ec, linewidth=0.3))

        for (rx, ry, lbl) in HouseMap.room_labels():
            ax.text(rx+0.5, H-1-ry+0.5, lbl, ha="center", va="center",
                    fontsize=6.5, color="#8899AA", alpha=0.7, zorder=3,
                    fontfamily="monospace")

        for wp in HouseMap.PATROL_WAYPOINTS:
            ax.plot(wp.x+0.5, H-1-wp.y+0.5, "o", color="#378ADD",
                    markersize=5, zorder=5, alpha=0.5)

        if path and len(path) > 1:
            xs = [p.x+0.5 for p in path]
            ys = [H-1-p.y+0.5 for p in path]
            ax.plot(xs, ys, color=color, linewidth=2.5, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round", alpha=0.85)

        if start:
            ax.plot(start.x+0.5, H-1-start.y+0.5, "s", color="#5DCAA5",
                    markersize=10, zorder=6,
                    markeredgecolor="#1A1A2E", markeredgewidth=1.2)
            ax.text(start.x+0.5, H-1-start.y+0.5, "NPC", ha="center",
                    va="center", fontsize=5.5, color="#1A1A2E",
                    fontweight="bold", zorder=7)

        t = goal or HouseMap.THIEF_POSITION
        ax.plot(t.x+0.5, H-1-t.y+0.5, "*", color="#E24B4A", markersize=14,
                zorder=6, markeredgecolor="#1A1A2E", markeredgewidth=0.8)
        ax.text(t.x+0.5, H-1-t.y-0.1, "THIEF", ha="center", va="top",
                fontsize=5, color="#E24B4A", zorder=7, fontfamily="monospace")

        ax.text(W*0.98, H*0.98, f"● {label}", ha="right", va="top",
                fontsize=8, color=color, fontweight="bold",
                fontfamily="monospace", zorder=8)

        legend_elements = [
            mpatches.Patch(facecolor="#0F3460", edgecolor="#0A2040", label="Wall"),
            mpatches.Patch(facecolor="#1A2744", edgecolor="#1E2F50", label="Walkable"),
            plt.Line2D([0],[0], color=color, linewidth=2, label=f"Path ({label})"),
            plt.Line2D([0],[0], marker="o", color="none",
                       markerfacecolor="#378ADD", markersize=6, label="Patrol waypoint"),
            plt.Line2D([0],[0], marker="s", color="none",
                       markerfacecolor="#5DCAA5", markersize=8, label="NPC (homeowner)"),
            plt.Line2D([0],[0], marker="*", color="none",
                       markerfacecolor="#E24B4A", markersize=10, label="Thief"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
                  facecolor="#0F3460", edgecolor="#1E2F50",
                  labelcolor="white", framealpha=0.9)

        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_aspect("equal"); ax.axis("off")
        fig.suptitle(title, fontsize=11, color="white",
                     fontfamily="monospace", y=0.97)
        return fig


# ── Interactive game mode ─────────────────────────────────────────────────────
class InteractiveGame:
    """
    Arrow keys move the thief.
    The homeowner follows via A* and the RF model decides its state each step.

    Parameters
    ----------
    model_path : str   path to models/npc_model.json
    """

    NOISE_ON_MOVE   = 0.25
    NOISE_DECAY     = 0.05
    PARANOIA_GAIN   = 0.03
    PARANOIA_DECAY  = 0.005
    NPC_STEP_EVERY  = 2    # NPC moves 1 tile every N thief moves

    def __init__(self, model_path: str = "models/npc_model.json"):
        self._grid    = HouseMap.default()
        self._forest  = JSONForest(model_path)

        # Positions
        self._thief   = Vector2i(HouseMap.THIEF_POSITION.x,
                                 HouseMap.THIEF_POSITION.y)
        self._npc     = Vector2i(HouseMap.PATROL_WAYPOINTS[0].x,
                                 HouseMap.PATROL_WAYPOINTS[0].y)

        # Simulation state
        self._paranoia     = 0.0
        self._noise        = 0.0
        self._has_target   = False
        self._npc_state    = "patrol"
        self._path: list   = []
        self._step_counter = 0
        self._move_log: list[str] = []

        self._build_canvas()
        self._update()

    # ── Canvas setup ──────────────────────────────────────────────────────────
    def _build_canvas(self):
        W, H = self._grid.width, self._grid.height
        self._fig, self._ax = plt.subplots(figsize=(W * 0.65, H * 0.65 + 1))
        self._fig.patch.set_facecolor("#1A1A2E")
        self._ax.set_facecolor("#16213E")
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.suptitle(
            "NPC AI  —  Arrow keys: move thief  |  Q: quit",
            fontsize=9, color="#8899AA", fontfamily="monospace", y=0.99,
        )

    # ── Input ─────────────────────────────────────────────────────────────────
    def _on_key(self, event):
        DIRS = {
            "up":    Vector2i(0, -1),
            "down":  Vector2i(0,  1),
            "left":  Vector2i(-1, 0),
            "right": Vector2i(1,  0),
        }
        if event.key == "q":
            plt.close(self._fig)
            return
        if event.key not in DIRS:
            return

        d   = DIRS[event.key]
        nxt = Vector2i(self._thief.x + d.x, self._thief.y + d.y)
        if not self._grid.is_walkable(nxt):
            return

        self._thief = nxt
        self._noise = min(1.0, self._noise + self.NOISE_ON_MOVE)
        self._step_counter += 1

        # Advance NPC one step along path every N thief moves
        if self._step_counter % self.NPC_STEP_EVERY == 0 and self._path:
            next_npc = self._path[1] if len(self._path) > 1 else self._path[0]
            self._npc = next_npc

        self._update()

    # ── Simulation step ───────────────────────────────────────────────────────
    def _update(self):
        # Decay noise & paranoia
        self._noise    = max(0.0, self._noise    - self.NOISE_DECAY)
        self._paranoia = max(0.0, self._paranoia - self.PARANOIA_DECAY)

        # Build features & predict state
        feats = build_features(
            self._npc, self._thief,
            self._paranoia, self._noise, self._has_target,
        )
        self._npc_state = self._forest.predict(feats)

        # Paranoia grows when alerted or chasing
        if self._npc_state in ("alert", "investigate", "chase"):
            self._paranoia = min(1.0, self._paranoia + self.PARANOIA_GAIN)
            self._has_target = True
        else:
            self._has_target = False

        # Recompute A* path NPC → Thief
        self._path = find_path(self._npc, self._thief, self._grid)

        self._redraw(feats)

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _redraw(self, feats: dict):
        ax  = self._ax
        ax.cla()
        ax.set_facecolor("#16213E")
        W, H   = self._grid.width, self._grid.height
        walls  = self._grid.wall_array()
        color  = STATE_COLORS[self._npc_state]
        label  = STATE_LABELS[self._npc_state]
        path_set = set((p.x, p.y) for p in self._path)

        # Grid cells
        for y in range(H):
            for x in range(W):
                is_wall = walls[y, x]
                if is_wall:
                    fc = "#0F3460"; ec = "#0A2040"
                elif (x, y) in path_set:
                    fc = color + "28"; ec = "#1E2F50"
                else:
                    fc = "#1A2744"; ec = "#1E2F50"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x, H-1-y), 1, 1, boxstyle="square,pad=0",
                    facecolor=fc, edgecolor=ec, linewidth=0.3))

        # Room labels
        for (rx, ry, lbl) in HouseMap.room_labels():
            ax.text(rx+0.5, H-1-ry+0.5, lbl, ha="center", va="center",
                    fontsize=6, color="#8899AA", alpha=0.55, zorder=3,
                    fontfamily="monospace")

        # Patrol waypoints
        for wp in HouseMap.PATROL_WAYPOINTS:
            ax.plot(wp.x+0.5, H-1-wp.y+0.5, "o", color="#378ADD",
                    markersize=4, zorder=4, alpha=0.35)

        # Path line
        if self._path and len(self._path) > 1:
            xs = [p.x+0.5 for p in self._path]
            ys = [H-1-p.y+0.5 for p in self._path]
            ax.plot(xs, ys, color=color, linewidth=2, zorder=5,
                    solid_capstyle="round", solid_joinstyle="round", alpha=0.75)

        # NPC marker
        nx, ny = self._npc.x, self._npc.y
        ax.add_patch(mpatches.FancyBboxPatch(
            (nx+0.15, H-1-ny+0.15), 0.7, 0.7,
            boxstyle="round,pad=0.05",
            facecolor="#5DCAA5", edgecolor="#1A1A2E",
            linewidth=1.2, zorder=7))
        ax.text(nx+0.5, H-1-ny+0.5, "NPC", ha="center", va="center",
                fontsize=5.5, color="#1A1A2E", fontweight="bold", zorder=8)

        # Thief marker
        tx, ty = self._thief.x, self._thief.y
        ax.text(tx+0.5, H-1-ty+0.5, "★", ha="center", va="center",
                fontsize=14, color="#E24B4A", zorder=7)

        # State badge (top-right)
        ax.text(W-0.2, H-0.3, f"● {label}", ha="right", va="top",
                fontsize=9, color=color, fontweight="bold",
                fontfamily="monospace", zorder=9)

        # HUD — feature values (bottom panel)
        hud_lines = [
            f"distance     {feats['player_distance']:6.1f}",
            f"visibility   {feats['visibility_score']:6.3f}",
            f"noise        {feats['noise_level']:6.3f}",
            f"paranoia     {feats['paranoia_level']:6.3f}",
            f"visible      {'YES' if feats['player_visible'] else 'NO ':>3}",
            f"has_target   {'YES' if feats['has_investigation_target'] else 'NO ':>3}",
        ]
        hud_x = [0.5, 4.0, 8.0, 11.5, 15.0, 17.5]
        for i, line in enumerate(hud_lines):
            ax.text(hud_x[i], -0.6, line, ha="left", va="top",
                    fontsize=6.5, color="#8899AA", fontfamily="monospace", zorder=9)

        ax.set_xlim(0, W); ax.set_ylim(-1.2, H)
        ax.set_aspect("equal"); ax.axis("off")
        self._fig.canvas.draw_idle()

    def run(self):
        plt.tight_layout()
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    MODEL = "models/npc_model.json"

    if not Path(MODEL).exists():
        print(f"  ⚠  Model not found at '{MODEL}'.")
        print("  Run 'python train.py' first to generate it.\n")
        print("  Falling back to static export demo...\n")
        grid  = HouseMap.default()
        plot  = AStarPlot(grid)
        start = HouseMap.PATROL_WAYPOINTS[0]
        goal  = HouseMap.THIEF_POSITION
        path  = plot.find_path(start, goal)
        for state in ["patrol", "alert", "investigate", "chase"]:
            plot.save(path, npc_state=state, start=start, goal=goal,
                      title=f"NPC A* Path — {state.capitalize()}",
                      output_path=f"models/astar_{state}.png")
        plot.show(path, npc_state="chase", start=start, goal=goal)
    else:
        print("  Starting interactive game mode...")
        print("  Arrow keys → move thief  |  Q → quit\n")
        game = InteractiveGame(MODEL)
        game.run()