import argparse
import itertools
import os

import numpy as np

from drce import explore, lidar, mapping, metrics, planner, world
from drce.mapping import RES
from drce.robot import RADIUS, Robot

STAMP = np.array(mapping.disc_offsets(11))


def build(scenario):
    wd, starts, name = world.make_world(scenario)
    mp = mapping.SharedMap(wd, starts)
    r1 = Robot(0, "R1", "#0072B2", starts[0], 0)
    r2 = Robot(1, "R2", "#D55E00", starts[1], 1)
    r1.strategy = explore.NearestFrontier()
    r2.strategy = explore.InfoGain()
    for r in (r1, r2):
        r.algo_label = r.strategy.label
    return wd, mp, [r1, r2], name


class Sim:
    DT = 0.05

    def __init__(self, wd, mp, robots, max_time=240.0):
        self.world = wd
        self.map = mp
        self.robots = robots
        self.max_time = max_time
        self.metrics = metrics.Metrics()
        self.t = 0.0
        self.step_i = 0
        self.done = None
        self.frontier_cache = []
        self.last_active = 0.0

    def other(self, r):
        return self.robots[1 - r.id]

    def scan_all(self):
        for r in self.robots:
            o = self.other(r)
            angles, ranges, hits, dyn = lidar.scan(self.world, (r.x, r.y, r.th),
                                                   [(o.pos, RADIUS)])
            self.map.integrate(r.id, r.pos, angles, ranges, hits, dyn)

    def plan_to(self, r, goal_cell):
        trav = self.map.traversable()
        o = self.other(r)
        if np.linalg.norm(o.pos - r.pos) < 3.0:
            trav = trav.copy()
            cells = np.array(self.map.cell(o.pos)) + STAMP
            ok = ((cells >= 0) & (cells < self.map.n)).all(1)
            cells = cells[ok]
            far = ((cells - np.array(self.map.cell(r.pos))) ** 2).sum(1) > 36
            cells = cells[far]
            trav[cells[:, 0], cells[:, 1]] = False
        return planner.plan(self.map, trav, r.pos, goal_cell)

    def decide(self, r):
        if self.t < r.cooldown:
            return
        r.cooldown = self.t + 0.4
        sel = explore.select_target(r.strategy, self.map, r, self.other(r), self.t)
        if sel is None:
            r.clear_task("standby")
            return
        path = self.plan_to(r, sel["goal"])
        if path is None or len(path) < 2:
            r.blacklist[explore.blacklist_key(sel["claim"])] = self.t + 8.0
            r.clear_task("standby")
            return
        r.set_path(path, sel["claim"])
        r.goal_cell = sel["goal"]
        r.claims += 1

    def check_path(self, r):
        trav = self.map.traversable()
        ahead = np.asarray(r.path[r.progress:r.progress + 15])
        cells = np.clip((ahead[:, ::-1] / RES).astype(int), 0, self.map.n - 1)
        if not trav[cells[:, 0], cells[:, 1]].all():
            r.replans += 1
            path = self.plan_to(r, r.goal_cell)
            if path is None or len(path) < 2:
                if r.claim is not None:
                    r.blacklist[explore.blacklist_key(r.claim)] = self.t + 8.0
                r.clear_task("standby")
            else:
                claim, goal = r.claim, r.goal_cell
                r.set_path(path, claim, r.state)
                r.goal_cell = goal
            return
        if r.state == "drive" and r.scale > 0:
            prog = r.progress * 0.1
            if prog - r.last_progress < 0.08:
                r.stuck_t += 0.5
            else:
                r.last_progress = prog
                r.stuck_t = 0.0
            if r.stuck_t > 3.5:
                if r.claim is not None:
                    r.blacklist[explore.blacklist_key(r.claim)] = self.t + 8.0
                r.clear_task("standby")

    def resolve_conflict(self):
        r1, r2 = self.robots
        for r in self.robots:
            r.scale = 1.0
        lo = r2 if r2.priority > r1.priority else r1
        hi = self.other(lo)
        d = float(np.linalg.norm(r1.pos - r2.pos))
        if d < 1.1:
            hi.scale = 0.55 if d > 0.6 else 0.25
            lo.scale = 0.6 if lo.state == "retreat" else 0.0
            if lo.state == "drive":
                lo.state = "yield"
            lo.yield_t += self.DT
            if lo.yield_t > 4.0 and lo.state == "yield":
                self.retreat(lo, hi)
        else:
            if lo.state == "yield":
                lo.state = "drive"
            lo.yield_t = 0.0

    def retreat(self, lo, hi):
        trav = self.map.traversable()
        away = lo.pos - hi.pos
        base = np.arctan2(away[1], away[0])
        for da in (0.0, 0.6, -0.6, 1.2, -1.2):
            g = lo.pos + 1.7 * np.array([np.cos(base + da), np.sin(base + da)])
            g = np.clip(g, 0.3, self.map.size - 0.3)
            gc = planner.nearest_traversable(trav, self.map.cell(g), 8)
            if gc is None:
                continue
            path = planner.plan(self.map, trav, lo.pos, gc)
            if path is not None and len(path) > 2:
                lo.set_path(path, None, "retreat")
                lo.goal_cell = gc
                lo.yield_t = 0.0
                return
        lo.yield_t = 0.0

    def step(self):
        if self.step_i % 2 == 0:
            self.scan_all()
        if self.step_i % 12 == 0:
            self.frontier_cache = self.map.frontier_clusters()
        for r in self.robots:
            if r.state == "warmup":
                if self.step_i >= 10:
                    self.decide(r)
            elif r.state == "standby":
                if self.step_i % 12 == r.id:
                    self.decide(r)
            elif r.state in ("drive", "retreat") and self.step_i % 10 == r.id:
                self.check_path(r)
        self.resolve_conflict()
        for r in self.robots:
            if r.step(self.DT):
                was = r.state
                if r.claim is not None:
                    r.blacklist[explore.blacklist_key(r.claim)] = self.t + 6.0
                r.clear_task("standby")
                if was == "drive":
                    self.decide(r)
        if any(r.state in ("drive", "retreat") for r in self.robots):
            self.last_active = self.t
        self.metrics.sample(self.t, self.world, self.map, self.robots,
                            self.step_i % 6 == 0)
        self.t += self.DT
        self.step_i += 1
        if self.done is None:
            cov = self.map.coverage()
            if cov >= 0.985:
                self.done = "coverage target reached"
            elif self.t > 6.0 and not self.frontier_cache and \
                    all(r.state == "standby" for r in self.robots):
                self.done = "frontiers exhausted"
            elif self.t - self.last_active > 12.0:
                self.done = "stalled (residual frontiers unreachable)"
            elif self.t >= self.max_time:
                self.done = "time limit"


def finish(sim, robots, scenario):
    s = sim.metrics.summary(sim.map, robots, sim.t, sim.done)
    metrics.print_report(s)
    os.makedirs("media", exist_ok=True)
    out = f"media/comparison{scenario}.png"
    metrics.comparison_figure(sim.metrics, s, robots, out)
    print(f"comparison figure -> {out}")
    return s


def run_interactive(sim, robots, name, scenario):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from drce.viz import LiveView
    fig = plt.figure(figsize=(13.4, 5.6))
    view = LiveView(fig, sim.world, sim.map, robots, name, sim.max_time)
    state = {"reported": False}

    def update(_):
        for _ in range(4):
            if sim.done is None:
                sim.step()
        view.update(sim)
        if sim.done is not None and not state["reported"]:
            state["reported"] = True
            anim.event_source.stop()
            finish(sim, robots, scenario)
        return []

    anim = FuncAnimation(fig, update, frames=itertools.count(), interval=30,
                         blit=False, cache_frame_data=False)
    plt.show()
    if not state["reported"]:
        finish(sim, robots, scenario)


def run_offline(sim, robots, name, scenario, save):
    view = None
    frames = []
    if save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        from drce.viz import LiveView
        fig = plt.figure(figsize=(13.4, 5.6))
        view = LiveView(fig, sim.world, sim.map, robots, name, sim.max_time)
    while sim.done is None:
        sim.step()
        if view is not None and sim.step_i % 15 == 0:
            view.update(sim)
            view.fig.canvas.draw()
            buf = np.asarray(view.fig.canvas.buffer_rgba())
            frames.append(Image.fromarray(buf[..., :3]))
    finish(sim, robots, scenario)
    if view is not None and frames:
        view.update(sim)
        view.fig.canvas.draw()
        buf = np.asarray(view.fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf[..., :3]))
        frames = frames[:: max(len(frames) // 380, 1)]
        out = f"media/explore{scenario}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=55, loop=0)
        print(f"animation -> {out}")


def main():
    ap = argparse.ArgumentParser(description="AMRN dual-robot cooperative exploration")
    ap.add_argument("--scenario", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-time", type=float, default=240.0)
    args = ap.parse_args()
    wd, mp, robots, name = build(args.scenario)
    sim = Sim(wd, mp, robots, args.max_time)
    print(f"scenario {args.scenario} ({name}): explorable area "
          f"{mp.total * RES * RES:.1f} m^2, robots R1=NF R2=IG")
    if args.headless or args.save:
        run_offline(sim, robots, name, args.scenario, args.save)
    else:
        os.environ.setdefault("DISPLAY", ":1")
        run_interactive(sim, robots, name, args.scenario)


if __name__ == "__main__":
    main()
