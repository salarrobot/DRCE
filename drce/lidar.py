import numpy as np

N_BEAMS = 72
MAX_RANGE = 3.5


def scan(world, pose, others):
    ox, oy, th = pose
    angles = th + np.linspace(-np.pi, np.pi, N_BEAMS, endpoint=False)
    origin = np.array([ox, oy])
    ranges, hits = world.raycast(origin, angles, MAX_RANGE)
    dyn = np.zeros(N_BEAMS, bool)
    dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
    for c, r in others:
        rel = np.asarray(c, float) - origin
        b = dirs @ rel
        disc = b * b - (rel @ rel - r * r)
        ok = disc > 0
        tt = b - np.sqrt(np.where(ok, disc, 0.0))
        ok &= (tt > 0) & (tt < ranges)
        ranges = np.where(ok, tt, ranges)
        hits = hits | ok
        dyn = dyn | ok
    return angles, ranges, hits, dyn
