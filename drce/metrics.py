import numpy as np

from .mapping import NB8, dilate
from .robot import RADIUS


class Metrics:
    def __init__(self):
        self.t = []
        self.cov = []
        self.disc = [[], []]
        self.min_clear = [np.inf, np.inf]
        self.min_inter = np.inf

    def sample(self, t, world, map_, robots, record):
        for r in robots:
            c = world.clearance(r.pos) - RADIUS
            self.min_clear[r.id] = min(self.min_clear[r.id], c)
        gap = float(np.linalg.norm(robots[0].pos - robots[1].pos)) - 2 * RADIUS
        self.min_inter = min(self.min_inter, gap)
        if record:
            self.t.append(t)
            self.cov.append(map_.coverage() * 100)
            for i in (0, 1):
                self.disc[i].append(map_.discovered[i] / map_.total * 100)

    def map_quality(self, map_):
        pred = map_.occ_mask
        near_true = map_.true_occ | dilate(map_.true_occ, NB8)
        prec = float((pred & near_true).sum() / max(pred.sum(), 1))
        tgt = map_.true_occ & map_.explorable
        near_pred = pred | dilate(pred, NB8)
        rec = float((tgt & near_pred).sum() / max(tgt.sum(), 1))
        return prec, rec

    def summary(self, map_, robots, t_end, done):
        prec, rec = self.map_quality(map_)
        union = (map_.seen[0] | map_.seen[1]).sum()
        overlap = float((map_.seen[0] & map_.seen[1]).sum() / max(union, 1)) * 100
        rows = []
        for r in robots:
            cells = map_.discovered[r.id]
            rows.append({"name": r.name, "algo": r.algo_label, "dist": r.traveled,
                         "cells": cells,
                         "share": cells / max(sum(map_.discovered), 1) * 100,
                         "eff": cells / max(r.traveled, 1e-9),
                         "replans": r.replans, "claims": r.claims,
                         "min_clear": self.min_clear[r.id]})
        return {"rows": rows, "t": t_end, "cov": map_.coverage() * 100, "done": done,
                "min_inter": self.min_inter, "precision": prec, "recall": rec,
                "overlap": overlap}


def print_report(s):
    print(f"\n=== exploration finished: {s['done']}   t = {s['t']:.1f} s   "
          f"coverage = {s['cov']:.1f}% ===")
    print(f"map precision = {s['precision'] * 100:.1f}%   recall = {s['recall'] * 100:.1f}%   "
          f"scan overlap = {s['overlap']:.1f}%   min robot-robot gap = {s['min_inter']:.2f} m")
    hdr = (f"{'robot':<7}{'algorithm':<34}{'dist[m]':>8}{'cells':>8}{'share':>8}"
           f"{'cells/m':>9}{'replans':>9}{'claims':>8}{'clear[m]':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in s["rows"]:
        print(f"{r['name']:<7}{r['algo']:<34}{r['dist']:>8.1f}{r['cells']:>8d}"
              f"{r['share']:>7.1f}%{r['eff']:>9.0f}{r['replans']:>9d}{r['claims']:>8d}"
              f"{r['min_clear']:>10.2f}")


def comparison_figure(metrics, s, robots, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
    fig.suptitle("Strategy comparison — Nearest-Frontier vs Info-Gain",
                 fontsize=13, fontfamily="serif")
    ax = axes[0, 0]
    ax.plot(metrics.t, metrics.cov, color="#1a1a1a", lw=1.9, label="total known")
    for r in robots:
        ax.plot(metrics.t, metrics.disc[r.id], color=r.color, lw=1.6,
                label=f"{r.name} · {r.strategy.abbr} discovered")
    ax.axhline(95, color="#999999", lw=0.9, ls="--")
    ax.set_xlabel("time [s]", fontsize=9)
    ax.set_ylabel("% of explorable area", fontsize=9)
    ax.set_ylim(0, 102)
    ax.set_title("coverage and attribution", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    names = [f"{r.name} · {r.strategy.abbr}" for r in robots]
    colors = [r.color for r in robots]

    def bars(ax, vals, title, fmt):
        b = ax.bar(names, vals, width=0.45, color=colors)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.margins(y=0.15)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    bars(axes[0, 1], [r["dist"] for r in s["rows"]], "distance traveled [m]", "{:.1f}")
    bars(axes[1, 0], [r["cells"] for r in s["rows"]], "map cells discovered", "{:d}")
    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"finished: {s['done']}  ({s['t']:.1f} s, coverage {s['cov']:.1f}%)", ""]
    for r in s["rows"]:
        lines.append(f"{r['name']} · {r['algo']}")
        lines.append(f"    efficiency {r['eff']:.0f} cells/m   share {r['share']:.1f}%")
        lines.append(f"    replans {r['replans']}   claims {r['claims']}   "
                     f"min clearance {r['min_clear']:.2f} m")
        lines.append("")
    lines.append(f"min robot-robot gap  {s['min_inter']:.2f} m")
    lines.append(f"map precision {s['precision'] * 100:.1f}%   recall {s['recall'] * 100:.1f}%")
    lines.append(f"scan overlap {s['overlap']:.1f}%")
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=9.2,
            family="monospace", transform=ax.transAxes)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
