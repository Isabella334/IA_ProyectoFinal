import json, math, time, threading
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathfinding.astar import Vector2i, find_path


# Colores por estado
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


class JSONForest:
    def __init__(self, path: str = "models/npc_model.json"):
        with open(path) as f:
            data = json.load(f)
        self._classes  = data["classes"]
        self._trees    = data["trees"]
        self._features = data["feature_names"]

    def predict(self, features: dict) -> str:
        x     = [features[k] for k in self._features]
        votes = [0] * len(self._classes)
        for tree in self._trees:
            votes[self._walk(tree, x)] += 1
        return self._classes[int(np.argmax(votes))]

    def _walk(self, node: dict, x: list) -> int:
        if node["leaf"]:
            return node["class_idx"]
        if x[node["feature_index"]] <= node["threshold"]:
            return self._walk(node["left"], x)
        return self._walk(node["right"], x)

class GridManager:
    def __init__(self, width, height, walls):
        self.width = width; self.height = height; self._walls = walls

    def is_walkable(self, pos: Vector2i) -> bool:
        return (0 <= pos.x < self.width and 0 <= pos.y < self.height
                and (pos.x, pos.y) not in self._walls)

    def wall_array(self):
        arr = np.zeros((self.height, self.width), dtype=bool)
        for (x, y) in self._walls:
            if 0 <= y < self.height and 0 <= x < self.width:
                arr[y, x] = True
        return arr


class HouseMap:
    WIDTH = 20; HEIGHT = 16
    PATROL_WAYPOINTS = [
        Vector2i(4,3), Vector2i(14,3), Vector2i(9,8),
        Vector2i(4,12), Vector2i(14,12),
    ]
    THIEF_POSITION = Vector2i(14, 12)

    HIDING_SPOTS = [
        Vector2i(2,  2),  
        Vector2i(16, 2),  
        Vector2i(2,  13), 
        Vector2i(16, 13), 
    ]

    @classmethod
    def default(cls) -> GridManager:
        walls = set()
        for x in range(cls.WIDTH):
            walls.add((x,0)); walls.add((x, cls.HEIGHT-1))
        for y in range(cls.HEIGHT):
            walls.add((0,y)); walls.add((cls.WIDTH-1, y))
        for y in range(1,7):  walls.add((9,y))
        for x in range(1,9):  walls.add((x,7))
        for x in range(10,19): walls.add((x,7))
        for x in range(1,9):  walls.add((x,9))
        for x in range(10,19): walls.add((x,9))
        for y in range(10,15): walls.add((9,y))
        for d in [(9,3),(4,7),(14,7),(4,9),(14,9)]:
            walls.discard(d)
        return GridManager(cls.WIDTH, cls.HEIGHT, walls)

    @classmethod
    def room_labels(cls):
        return [(4,3,"Living Room"),(14,3,"Kitchen"),(9,8,"Hallway"),
                (4,12,"Bedroom 1"),(14,12,"Bedroom 2")]


def build_features(npc: Vector2i, thief: Vector2i,
                   paranoia: float, noise: float, has_target: bool) -> dict:
    dist     = math.sqrt((npc.x-thief.x)**2 + (npc.y-thief.y)**2)
    visible  = 1.0 if dist <= 3.0 else 0.0
    vis_score = max(0.0, 1.0 - dist / 20.0)
    return {
        "player_visible":           visible,
        "visibility_score":         round(vis_score, 4),
        "player_distance":          round(dist * 10, 4),
        "noise_level":              round(noise, 4),
        "paranoia_level":           round(paranoia, 4),
        "has_investigation_target": 1.0 if has_target else 0.0,
    }


class AStarPlot:
    def __init__(self, grid: GridManager, cell_size: int = 40):
        self._grid = grid

    def find_path(self, start, goal):
        return find_path(start, goal, self._grid)

    def show(self, path, *, npc_state="patrol", start=None, goal=None,
             title="NPC Pathfinding — A*"):
        fig = self._build(path, npc_state, start, goal, title)
        plt.show(); plt.close(fig)

    def save(self, path, *, npc_state="patrol", start=None, goal=None,
             title="NPC Pathfinding — A*", output_path="models/astar_plot.png", dpi=120):
        fig = self._build(path, npc_state, start, goal, title)
        out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        print(f"  A* plot saved → {out.resolve()}")

    def _build(self, path, npc_state, start, goal, title):
        W, H  = self._grid.width, self._grid.height
        walls = self._grid.wall_array()
        color = STATE_COLORS.get(npc_state, "#888780")
        label = STATE_LABELS.get(npc_state, npc_state.upper())
        path_set = {(p.x,p.y) for p in path}

        fig, ax = plt.subplots(figsize=(W*.6, H*.6))
        fig.patch.set_facecolor("#1A1A2E"); ax.set_facecolor("#16213E")

        for y in range(H):
            for x in range(W):
                iw = walls[y,x]
                fc = "#0F3460" if iw else (color+"28" if (x,y) in path_set else "#1A2744")
                ec = "#0A2040" if iw else "#1E2F50"
                ax.add_patch(mpatches.FancyBboxPatch((x,H-1-y),1,1,
                    boxstyle="square,pad=0",facecolor=fc,edgecolor=ec,linewidth=0.3))

        for rx,ry,lbl in HouseMap.room_labels():
            ax.text(rx+.5,H-1-ry+.5,lbl,ha="center",va="center",
                    fontsize=6.5,color="#8899AA",alpha=.7,zorder=3,fontfamily="monospace")

        for wp in HouseMap.PATROL_WAYPOINTS:
            ax.plot(wp.x+.5,H-1-wp.y+.5,"o",color="#378ADD",markersize=5,zorder=5,alpha=.5)

        if len(path)>1:
            ax.plot([p.x+.5 for p in path],[H-1-p.y+.5 for p in path],
                    color=color,linewidth=2.5,zorder=4,
                    solid_capstyle="round",solid_joinstyle="round",alpha=.85)

        if start:
            ax.add_patch(mpatches.FancyBboxPatch((start.x+.15,H-1-start.y+.15),.7,.7,
                boxstyle="round,pad=0.05",facecolor="#5DCAA5",edgecolor="#1A1A2E",linewidth=1.2,zorder=7))
            ax.text(start.x+.5,H-1-start.y+.5,"NPC",ha="center",va="center",
                    fontsize=5.5,color="#1A1A2E",fontweight="bold",zorder=8)

        t = goal or HouseMap.THIEF_POSITION
        ax.text(t.x+.5,H-1-t.y+.5,"★",ha="center",va="center",
                fontsize=14,color="#E24B4A",zorder=7)

        ax.text(W-.2,H-.3,f"● {label}",ha="right",va="top",fontsize=9,
                color=color,fontweight="bold",fontfamily="monospace",zorder=9)

        legend_els = [
            mpatches.Patch(facecolor="#0F3460",edgecolor="#0A2040",label="Wall"),
            mpatches.Patch(facecolor="#1A2744",edgecolor="#1E2F50",label="Walkable"),
            plt.Line2D([0],[0],color=color,linewidth=2,label=f"Path ({label})"),
            plt.Line2D([0],[0],marker="o",color="none",markerfacecolor="#378ADD",markersize=6,label="Patrol wp"),
            plt.Line2D([0],[0],marker="s",color="none",markerfacecolor="#5DCAA5",markersize=8,label="NPC"),
            plt.Line2D([0],[0],marker="*",color="none",markerfacecolor="#E24B4A",markersize=10,label="Thief"),
        ]
        ax.legend(handles=legend_els,loc="lower right",fontsize=6.5,
                  facecolor="#0F3460",edgecolor="#1E2F50",labelcolor="white",framealpha=.9)
        ax.set_xlim(0,W); ax.set_ylim(0,H); ax.set_aspect("equal"); ax.axis("off")
        fig.suptitle(title,fontsize=11,color="white",fontfamily="monospace",y=.97)
        return fig


class InteractiveGame:
    FPS            = 4         # ticks por segundo del NPC
    NPC_STEPS_TICK = 1         # tiles que avanza el NPC por tick
    NOISE_ON_MOVE    = 0.20    # ruido que genera cada paso del ladrón
    NOISE_DECAY      = 0.03    # decae por tick sin estímulo
    PARANOIA_GAIN    = 0.04    # sube cuando hay alerta/investigación/chase
    PARANOIA_DECAY   = 0.008   # decae por tick sin estímulo (cooldown Godot)
    PARANOIA_FLOOR   = 0.0     # mínimo absoluto
    PARANOIA_CEIL    = 1.0
    PARANOIA_DECAY_HIDDEN = 0.05   # decae mucho más rápido al esconderse
    HIDE_RANGE            = 1.5    # distancia máxima (tiles) para activar E

    def __init__(self, model_path: str = "models/npc_model.json"):
        self._grid   = HouseMap.default()
        self._forest = JSONForest(model_path)

        self._thief = Vector2i(HouseMap.THIEF_POSITION.x, HouseMap.THIEF_POSITION.y)
        self._npc   = Vector2i(HouseMap.PATROL_WAYPOINTS[0].x,
                               HouseMap.PATROL_WAYPOINTS[0].y)

        self._paranoia   = 0.0
        self._noise      = 0.0
        self._has_target = False
        self._npc_state  = "patrol"
        self._path: list = []
        self._hidden     = False 

        # Patrol waypoint cycling cuando está en patrol
        self._wp_idx     = 0

        self._keys_pressed: set = set()
        self._lock = threading.Lock()
        self._running = True

        self._fig, self._ax = plt.subplots(figsize=(14, 10))
        self._fig.patch.set_facecolor("#1A1A2E")
        self._ax.set_facecolor("#16213E")
        self._fig.canvas.mpl_connect("key_press_event",  self._on_key_press)
        self._fig.canvas.mpl_connect("close_event",      self._on_close)
        self._fig.suptitle(
            "NPC AI Minijuego  ·  Flechas: mover ladrón  ·  E: esconderse  ·  Q: salir",
            fontsize=9, color="#8899AA", fontfamily="monospace", y=.995)

        self._update_path()
        self._redraw()

    def _on_key_press(self, event):
        DIRS = {"up":Vector2i(0,-1),"down":Vector2i(0,1),
                "left":Vector2i(-1,0),"right":Vector2i(1,0)}
        if event.key == "q":
            self._running = False; plt.close(self._fig); return
        if event.key == "e":
            with self._lock:
                if self._hidden:
                    self._hidden = False
                else:
                    near = self._nearest_hiding_spot()
                    if near is not None:
                        self._hidden = True
                        self._noise  = 0.0
            return
        if self._hidden:
            return

        if event.key in DIRS:
            d   = DIRS[event.key]
            nxt = Vector2i(self._thief.x+d.x, self._thief.y+d.y)
            if self._grid.is_walkable(nxt):
                with self._lock:
                    self._thief = nxt
                    self._noise = min(self.PARANOIA_CEIL,
                                     self._noise + self.NOISE_ON_MOVE)

    def _nearest_hiding_spot(self):
        tx, ty = self._thief.x, self._thief.y
        for spot in HouseMap.HIDING_SPOTS:
            dist = math.sqrt((tx - spot.x)**2 + (ty - spot.y)**2)
            if dist <= self.HIDE_RANGE:
                return spot
        return None

    def _on_close(self, event):
        self._running = False
    def _game_loop(self):
        interval = 1.0 / self.FPS
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                self._tick()
            try:
                self._fig.canvas.draw_idle()
                self._fig.canvas.flush_events()
            except Exception:
                break
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _tick(self):
        decay = self.PARANOIA_DECAY_HIDDEN if self._hidden else self.PARANOIA_DECAY
        self._noise    = max(self.PARANOIA_FLOOR,
                             self._noise - self.NOISE_DECAY)
        self._paranoia = max(self.PARANOIA_FLOOR,
                             self._paranoia - decay)
        feats = build_features(
            self._npc, self._thief,
            self._paranoia,
            0.0 if self._hidden else self._noise,
            self._has_target,
        )
        if self._hidden:
            feats["player_visible"]   = 0.0
            feats["visibility_score"] = 0.0
        self._npc_state = self._forest.predict(feats)

        if self._npc_state in ("alert", "investigate", "chase"):
            self._paranoia   = min(self.PARANOIA_CEIL,
                                   self._paranoia + self.PARANOIA_GAIN)
            self._has_target = True
        else:
            if self._paranoia < 0.3:
                self._has_target = False

        self._update_path()
        if self._path and len(self._path) > 1:
            self._npc = self._path[1]

        self._redraw()

    def _update_path(self):
        if self._npc_state == "patrol":
            goal = HouseMap.PATROL_WAYPOINTS[self._wp_idx]
            if self._npc.x == goal.x and self._npc.y == goal.y:
                self._wp_idx = (self._wp_idx + 1) % len(HouseMap.PATROL_WAYPOINTS)
                goal = HouseMap.PATROL_WAYPOINTS[self._wp_idx]
        else:
            goal = self._thief

        self._path = find_path(self._npc, goal, self._grid)

    def _redraw(self):
        ax = self._ax; ax.cla(); ax.set_facecolor("#16213E")
        W, H   = self._grid.width, self._grid.height
        walls  = self._grid.wall_array()
        color  = STATE_COLORS[self._npc_state]
        label  = STATE_LABELS[self._npc_state]
        path_set = {(p.x,p.y) for p in self._path}

        for y in range(H):
            for x in range(W):
                iw = walls[y,x]
                fc = ("#0F3460" if iw
                      else color+"22" if (x,y) in path_set
                      else "#1A2744")
                ec = "#0A2040" if iw else "#1E2F50"
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x,H-1-y),1,1,boxstyle="square,pad=0",
                    facecolor=fc,edgecolor=ec,linewidth=0.3))

        for rx,ry,lbl in HouseMap.room_labels():
            ax.text(rx+.5,H-1-ry+.5,lbl,ha="center",va="center",
                    fontsize=6,color="#8899AA",alpha=.5,zorder=3,
                    fontfamily="monospace")

        for i,wp in enumerate(HouseMap.PATROL_WAYPOINTS):
            alpha = .9 if (self._npc_state=="patrol" and
                           i==self._wp_idx) else .3
            ax.plot(wp.x+.5,H-1-wp.y+.5,"o",color="#378ADD",
                    markersize=5,zorder=4,alpha=alpha)

        if len(self._path) > 1:
            ax.plot([p.x+.5 for p in self._path],
                    [H-1-p.y+.5 for p in self._path],
                    color=color,linewidth=2,zorder=5,
                    solid_capstyle="round",solid_joinstyle="round",alpha=.7)

        nx,ny = self._npc.x, self._npc.y
        ax.add_patch(mpatches.FancyBboxPatch(
            (nx+.1,H-1-ny+.1),.8,.8,boxstyle="round,pad=0.06",
            facecolor="#5DCAA5",edgecolor=color,linewidth=2,zorder=7))
        ax.text(nx+.5,H-1-ny+.5,"NPC",ha="center",va="center",
                fontsize=6,color="#1A1A2E",fontweight="bold",zorder=8)

        near_spot = self._nearest_hiding_spot()
        for spot in HouseMap.HIDING_SPOTS:
            is_active  = self._hidden and (spot.x == self._thief.x and spot.y == self._thief.y or
                         math.sqrt((self._thief.x-spot.x)**2+(self._thief.y-spot.y)**2) <= self.HIDE_RANGE and self._hidden)
            is_nearby  = (not self._hidden and near_spot is not None and
                          spot.x == near_spot.x and spot.y == near_spot.y)
            box_color  = "#5DCAA5" if is_active else ("#EF9F27" if is_nearby else "#334466")
            box_edge   = "#FFFFFF" if is_active else ("#EF9F27" if is_nearby else "#445577")
            ax.add_patch(mpatches.FancyBboxPatch(
                (spot.x+.1, H-1-spot.y+.1), .8, .8,
                boxstyle="round,pad=0.05",
                facecolor=box_color, edgecolor=box_edge,
                linewidth=1.5, zorder=6, alpha=0.85))
            ax.text(spot.x+.5, H-1-spot.y+.5, "📦", ha="center", va="center",
                    fontsize=9, zorder=7)
            if is_nearby and not self._hidden:
                ax.text(spot.x+.5, H-1-spot.y+1.1, "E",
                        ha="center", va="bottom", fontsize=7,
                        color="#EF9F27", fontweight="bold",
                        fontfamily="monospace", zorder=8)

        tx, ty = self._thief.x, self._thief.y
        thief_alpha = 0.25 if self._hidden else 1.0
        ax.text(tx+.5, H-1-ty+.5, "★", ha="center", va="center",
                fontsize=15, color="#E24B4A", zorder=7, alpha=thief_alpha)
        lbl = "escondido" if self._hidden else "ladrón"
        lbl_color = "#5DCAA5" if self._hidden else "#E24B4A"
        ax.text(tx+.5, H-1-ty-.05, lbl, ha="center", va="top",
                fontsize=5, color=lbl_color, fontfamily="monospace", zorder=8)

        ax.text(W-.2, H-.25, f"● {label}", ha="right", va="top",
                fontsize=10, color=color, fontweight="bold",
                fontfamily="monospace", zorder=9)
        if self._hidden:
            ax.text(W-.2, H-.25-0.7, "🙈 ESCONDIDO", ha="right", va="top",
                    fontsize=8, color="#5DCAA5", fontweight="bold",
                    fontfamily="monospace", zorder=9)

        feats = build_features(self._npc,self._thief,
                               self._paranoia,self._noise,self._has_target)

        hud = [
            ("dist",      f"{feats['player_distance']:5.1f}"),
            ("visib",     f"{feats['visibility_score']:.3f}"),
            ("noise",     f"{feats['noise_level']:.3f}"),
            ("paranoia",  f"{feats['paranoia_level']:.3f}"),
            ("visible",   "SÍ " if feats['player_visible'] else "NO "),
            ("has_tgt",   "SÍ " if feats['has_investigation_target'] else "NO "),
        ]

        def bar(val, w=10):
            filled = int(val * w)
            return "█"*filled + "░"*(w-filled)

        for i,(k,v) in enumerate(hud):
            x_pos = 0.6 + i*3.3
            ax.text(x_pos,-0.45,k,ha="left",va="top",fontsize=6,
                    color="#556677",fontfamily="monospace",zorder=9)
            ax.text(x_pos,-0.85,v,ha="left",va="top",fontsize=7.5,
                    color="#AABBCC",fontfamily="monospace",fontweight="bold",zorder=9)

        ax.text(0.6,-1.3,
                f"paranoia  {bar(self._paranoia)}  "
                f"noise  {bar(self._noise)}",
                ha="left",va="top",fontsize=6.5,
                color=color,fontfamily="monospace",zorder=9)

        if self._hidden:
            ax.text(W/2, -1.3, "🙈 escondido — paranoia bajando rápido",
                    ha="center", va="top", fontsize=7, color="#5DCAA5",
                    fontfamily="monospace", alpha=.9, zorder=9)
        elif self._npc_state in ("alert","investigate") and self._paranoia < 0.35:
            ax.text(W/2, -1.3, "← perdiendo rastro...", ha="center", va="top",
                    fontsize=7, color="#EF9F27", fontfamily="monospace",
                    alpha=.8, zorder=9)
        elif self._npc_state == "patrol" and self._paranoia < 0.05:
            ax.text(W/2, -1.3, "patrol normal", ha="center", va="top",
                    fontsize=7, color="#378ADD", fontfamily="monospace",
                    alpha=.6, zorder=9)

        ax.set_xlim(0,W); ax.set_ylim(-1.7,H)
        ax.set_aspect("equal"); ax.axis("off")

    def run(self):
        t = threading.Thread(target=self._game_loop, daemon=True)
        t.start()
        plt.show()
        self._running = False

if __name__ == "__main__":
    MODEL = "models/npc_model.json"
    if not Path(MODEL).exists():
        print(f"  ⚠  Modelo no encontrado en '{MODEL}'.")
        print("  Corre 'python train.py' primero.\n")
        print("  Mostrando demo estático...\n")
        grid  = HouseMap.default()
        plot  = AStarPlot(grid)
        start = HouseMap.PATROL_WAYPOINTS[0]
        goal  = HouseMap.THIEF_POSITION
        path  = plot.find_path(start, goal)
        for state in ["patrol","alert","investigate","chase"]:
            plot.save(path, npc_state=state, start=start, goal=goal,
                      title=f"NPC A* Path — {state.capitalize()}",
                      output_path=f"models/astar_{state}.png")
        plot.show(path, npc_state="chase", start=start, goal=goal)
    else:
        print("  Iniciando minijuego interactivo...")
        print("  Flechas → mover ladrón  |  E → esconderse  |  Q → salir\n")
        game = InteractiveGame(MODEL)
        game.run()