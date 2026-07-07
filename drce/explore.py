import numpy as np

from . import mapping, planner
from .mapping import RES

CLAIM_RADIUS = 1.8
GAIN_RADIUS = 2.0
_GAIN_OFF = None


class NearestFrontier:
    abbr = "NF"
    label = "Nearest-Frontier (Yamauchi)"

    def utility(self, d_m, gain, penalized):
        return -(d_m + (20.0 if penalized else 0.0))


class InfoGain:
    abbr = "IG"
    label = "Info-Gain Utility (Gonzalez-Banos)"

    def utility(self, d_m, gain, penalized):
        return gain * np.exp(-0.35 * d_m) * (0.15 if penalized else 1.0)


def _gain(map_, unknown, cluster):
    global _GAIN_OFF
    if _GAIN_OFF is None:
        _GAIN_OFF = np.array(mapping.disc_offsets(int(GAIN_RADIUS / RES)))
    cy, cx = np.round(cluster["cells"].mean(0)).astype(int)
    ys = np.clip(_GAIN_OFF[:, 0] + cy, 0, map_.n - 1)
    xs = np.clip(_GAIN_OFF[:, 1] + cx, 0, map_.n - 1)
    return int(unknown[ys, xs].sum())


def blacklist_key(xy):
    return (round(xy[0] * 2) / 2, round(xy[1] * 2) / 2)


def select_target(strategy, map_, robot, other, now):
    trav = map_.traversable()
    start = planner.nearest_traversable(trav, map_.cell(robot.pos), 10)
    if start is None:
        return None
    clusters = map_.frontier_clusters()
    if not clusters:
        return None
    field = mapping.hop_field(trav, start)
    unknown = map_.unknown_mask
    robot.blacklist = {k: t for k, t in robot.blacklist.items() if t > now}
    rc = np.array(start)
    best = None
    for cl in clusters:
        if blacklist_key(cl["centroid"]) in robot.blacklist:
            continue
        cells = cl["cells"]
        anchor = tuple(cells[int(np.argmin(((cells - rc) ** 2).sum(1)))])
        approach = planner.nearest_traversable(trav, anchor, 14)
        if approach is None or not np.isfinite(field[approach]):
            continue
        d_m = float(field[approach]) * RES
        gain = _gain(map_, unknown, cl)
        pen = other.claim is not None and np.linalg.norm(cl["centroid"] - other.claim) < CLAIM_RADIUS
        pen = pen or np.linalg.norm(cl["centroid"] - other.pos) < 1.2
        u = strategy.utility(d_m, gain, pen)
        if best is None or u > best["u"]:
            best = {"u": u, "goal": approach, "claim": cl["centroid"],
                    "gain": gain, "dist": d_m}
    return best
