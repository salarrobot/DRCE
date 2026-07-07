import heapq

import numpy as np

from .mapping import NB8, RES, dilate, disc_offsets

SQ2 = 2 ** 0.5


def nearest_traversable(trav, c, max_r=16):
    if trav[c]:
        return c
    n = trav.shape[0]
    best = None
    bd = 1e18
    for r in range(1, max_r + 1):
        y0, y1 = max(c[0] - r, 0), min(c[0] + r, n - 1)
        x0, x1 = max(c[1] - r, 0), min(c[1] + r, n - 1)
        sub = trav[y0:y1 + 1, x0:x1 + 1]
        if sub.any():
            ys, xs = np.nonzero(sub)
            dd = (ys + y0 - c[0]) ** 2 + (xs + x0 - c[1]) ** 2
            i = int(np.argmin(dd))
            if dd[i] < bd:
                best = (int(ys[i] + y0), int(xs[i] + x0))
                bd = dd[i]
            if bd <= r * r:
                return best
    return best


def astar(trav, start, goal, limit=90000):
    if start == goal:
        return [start]
    n = trav.shape[0]

    def h(y, x):
        dy, dx = abs(y - goal[0]), abs(x - goal[1])
        return max(dy, dx) + (SQ2 - 1) * min(dy, dx)

    g = {start: 0.0}
    came = {}
    pq = [(h(*start), 0.0, start)]
    seen = set()
    while pq and len(seen) < limit:
        _, gc, c = heapq.heappop(pq)
        if c in seen:
            continue
        seen.add(c)
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            return path[::-1]
        y, x = c
        for dy, dx in NB8:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < n and 0 <= nx < n) or not trav[ny, nx]:
                continue
            if dy and dx and not (trav[y, nx] and trav[ny, x]):
                continue
            ng = gc + (SQ2 if dy and dx else 1.0)
            nc = (ny, nx)
            if ng < g.get(nc, 1e18):
                g[nc] = ng
                came[nc] = c
                heapq.heappush(pq, (ng + h(ny, nx), ng, nc))
    return None


def line_free(trav, a, b):
    d = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    if d == 0:
        return True
    m = int(d * 2.5) + 2
    ts = np.linspace(0.0, 1.0, m)
    ys = np.round(a[0] + (b[0] - a[0]) * ts).astype(int)
    xs = np.round(a[1] + (b[1] - a[1]) * ts).astype(int)
    return bool(trav[ys, xs].all())


def smooth(trav, path):
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(trav, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def densify(cells, step=0.1):
    pts = [np.array([(c[1] + 0.5) * RES, (c[0] + 0.5) * RES]) for c in cells]
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        seg = b - a
        length = np.hypot(*seg)
        k = max(int(length / step), 1)
        for i in range(1, k + 1):
            out.append(a + seg * (i / k))
    return np.array(out)


def plan(map_, trav, start_xy, goal_cell):
    if goal_cell is None:
        return None
    s = nearest_traversable(trav, map_.cell(start_xy), 10)
    if s is None:
        return None
    gcell = nearest_traversable(trav, goal_cell, 14)
    if gcell is None:
        return None
    raw = astar(trav, s, gcell)
    if raw is None:
        return None
    core = ~dilate(~trav, disc_offsets(1))
    return densify(smooth(core, raw))
