import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle as MplRectangle

from . import world as wmod
from .mapping import RES
from .robot import RADIUS

FRONTIER = "#56B4E9"


def draw_world(ax, world):
    for s in world.shapes:
        if isinstance(s, wmod.Rect):
            ax.add_patch(MplRectangle(s.c - s.half, *(2 * s.half), fc="#3f3f3f",
                                      ec="black", lw=1.0, zorder=2))
        elif isinstance(s, wmod.Circle):
            ax.add_patch(MplCircle(s.c, s.r, fc="#3f3f3f", ec="black", lw=1.0, zorder=2))
        else:
            ax.add_patch(MplPolygon(s.v, fc="#3f3f3f", ec="black", lw=1.0, zorder=2))
    ax.add_patch(MplRectangle((0, 0), world.size, world.size, fill=False,
                              ec="black", lw=1.4, zorder=3))


def style_map_axis(ax, size, title):
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, size + 0.15)
    ax.set_ylim(-0.15, size + 0.15)
    ax.set_xticks(range(0, int(size) + 1, 2))
    ax.set_yticks(range(0, int(size) + 1, 2))
    ax.tick_params(labelsize=8)
    ax.grid(ls=":", lw=0.5, alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10.5, pad=6)


class LiveView:
    def __init__(self, fig, world, map_, robots, name, max_time):
        self.fig = fig
        self.world = world
        self.map = map_
        self.robots = robots
        gs = fig.add_gridspec(1, 3, width_ratios=[1.04, 1.04, 0.92], left=0.05,
                              right=0.985, top=0.84, bottom=0.30, wspace=0.26)
        self.ax_w = fig.add_subplot(gs[0])
        self.ax_m = fig.add_subplot(gs[1])
        self.ax_c = fig.add_subplot(gs[2])
        fig.suptitle(f"AMRN — Dual-Robot Cooperative Exploration · {name}",
                     fontsize=13.5, fontfamily="serif", y=0.965)

        style_map_axis(self.ax_w, world.size, "ground truth and trajectories")
        draw_world(self.ax_w, world)
        self.trails, self.plans, self.bodies, self.heads, self.targets = [], [], [], [], []
        for r in robots:
            self.trails.append(self.ax_w.plot([], [], color=r.color, lw=1.5, zorder=4)[0])
            self.plans.append(self.ax_w.plot([], [], color=r.color, lw=1.0, ls="--",
                                             alpha=0.75, zorder=4)[0])
            body = MplCircle((r.x, r.y), RADIUS, fc=r.color, ec="black", lw=1.0, zorder=6)
            self.ax_w.add_patch(body)
            self.bodies.append(body)
            self.heads.append(self.ax_w.plot([], [], color="black", lw=1.2, zorder=7)[0])
            self.targets.append(self.ax_w.plot([], [], ls="", marker="x", ms=9,
                                               mew=2.0, color=r.color, zorder=5)[0])
            self.ax_w.plot(r.x, r.y, marker="^", ms=6, mfc="white", mec=r.color,
                           mew=1.4, ls="", zorder=5)
        handles = [Line2D([], [], color=r.color, lw=2.2,
                          label=f"{r.name} · {r.strategy.label}") for r in robots]
        handles.append(Line2D([], [], ls="", marker="x", ms=8, mew=2, color="#555555",
                              label="claimed frontier target"))
        self.ax_w.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                         ncol=1, fontsize=8, frameon=False)

        style_map_axis(self.ax_m, world.size, "shared occupancy grid — live SLAM")
        self.img = self.ax_m.imshow(np.full((map_.n, map_.n), 0.55), cmap="gray",
                                    vmin=0.0, vmax=1.0, origin="lower",
                                    extent=(0, world.size, 0, world.size), zorder=1,
                                    interpolation="nearest")
        self.frontier_dots = self.ax_m.plot([], [], ls="", marker="s", ms=2.2,
                                            color=FRONTIER, zorder=3)[0]
        self.map_bots = [self.ax_m.plot([], [], ls="", marker="o", ms=7, mfc=r.color,
                                        mec="black", mew=0.9, zorder=5)[0] for r in robots]
        self.map_tgts = [self.ax_m.plot([], [], ls="", marker="x", ms=8, mew=1.8,
                                        color=r.color, zorder=4)[0] for r in robots]
        handles = [Patch(fc="#ffffff", ec="#888888", label="free"),
                   Patch(fc="#8c8c8c", label="unknown"),
                   Patch(fc="#000000", label="occupied"),
                   Line2D([], [], ls="", marker="s", ms=5, color=FRONTIER, label="frontier")]
        self.ax_m.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                         ncol=4, fontsize=8, frameon=False)

        self.ax_c.set_title("exploration progress and attribution", fontsize=10.5, pad=6)
        self.ax_c.set_xlim(0, 60)
        self.ax_c.set_ylim(0, 102)
        self.ax_c.set_xlabel("time [s]", fontsize=9)
        self.ax_c.set_ylabel("% of explorable area", fontsize=9)
        self.ax_c.tick_params(labelsize=8)
        self.ax_c.grid(alpha=0.2)
        self.ax_c.set_axisbelow(True)
        self.ax_c.axhline(95, color="#999999", lw=0.9, ls="--")
        self.line_total = self.ax_c.plot([], [], color="#1a1a1a", lw=1.9,
                                         label="total known")[0]
        self.line_r = [self.ax_c.plot([], [], color=r.color, lw=1.6,
                                      label=f"{r.name} · {r.strategy.abbr}")[0]
                       for r in robots]
        self.ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
                         fontsize=8, frameon=False)
        self.status = fig.text(0.5, 0.045, "", ha="center", va="center", fontsize=8.6,
                               family="monospace", color="#222222")

    def update(self, sim):
        for r in self.robots:
            tr = np.array(r.trail)
            self.trails[r.id].set_data(tr[:, 0], tr[:, 1])
            if r.path is not None:
                self.plans[r.id].set_data(r.path[:, 0], r.path[:, 1])
            else:
                self.plans[r.id].set_data([], [])
            self.bodies[r.id].center = (r.x, r.y)
            hx = [r.x, r.x + RADIUS * 1.6 * np.cos(r.th)]
            hy = [r.y, r.y + RADIUS * 1.6 * np.sin(r.th)]
            self.heads[r.id].set_data(hx, hy)
            if r.claim is not None:
                self.targets[r.id].set_data([r.claim[0]], [r.claim[1]])
                self.map_tgts[r.id].set_data([r.claim[0]], [r.claim[1]])
            else:
                self.targets[r.id].set_data([], [])
                self.map_tgts[r.id].set_data([], [])
            self.map_bots[r.id].set_data([r.x], [r.y])
        img = np.full((self.map.n, self.map.n), 0.55)
        img[self.map.free_mask] = 1.0
        img[self.map.occ_mask] = 0.0
        self.img.set_data(img)
        if sim.frontier_cache:
            cells = np.concatenate([c["cells"] for c in sim.frontier_cache])
            self.frontier_dots.set_data((cells[:, 1] + 0.5) * RES, (cells[:, 0] + 0.5) * RES)
        else:
            self.frontier_dots.set_data([], [])
        m = sim.metrics
        self.line_total.set_data(m.t, m.cov)
        for i in (0, 1):
            self.line_r[i].set_data(m.t, m.disc[i])
        if sim.t > self.ax_c.get_xlim()[1] * 0.95:
            self.ax_c.set_xlim(0, sim.t * 1.4)
        gap = np.linalg.norm(self.robots[0].pos - self.robots[1].pos)
        head = (f"t={sim.t:6.1f}s   coverage={sim.map.coverage() * 100:5.1f}%   "
                f"gap R1-R2={gap:4.2f}m")
        parts = []
        for r in self.robots:
            tgt = np.linalg.norm(r.path[-1] - r.pos) if r.path is not None else 0.0
            parts.append(f"{r.name}·{r.strategy.abbr} {r.state:<7} tgt {tgt:4.1f}m  "
                         f"disc {sim.map.discovered[r.id]:6d}  path {r.traveled:5.1f}m")
        self.status.set_text(head + "\n" + "   |   ".join(parts))
