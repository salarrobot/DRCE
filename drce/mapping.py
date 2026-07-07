import numpy as np

RES = 0.05
L_FREE = -0.35
L_OCC = 1.0
L_CLAMP = 4.0
TH_FREE = -0.7
TH_OCC = 0.85
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def disc_offsets(r):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r + 0.1]


def dilate(mask, offsets):
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy, dx in offsets:
        y0, y1 = max(dy, 0), h + min(dy, 0)
        x0, x1 = max(dx, 0), w + min(dx, 0)
        out[y0:y1, x0:x1] |= mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def hop_field(trav, start):
    dist = np.full(trav.shape, np.inf)
    if start is None or not trav[start]:
        return dist
    h, w = trav.shape
    visited = np.zeros(trav.shape, bool)
    cur = np.zeros(trav.shape, bool)
    buf = np.zeros(trav.shape, bool)
    cur[start] = True
    visited[start] = True
    dist[start] = 0.0
    d = 0
    while cur.any():
        d += 1
        buf[:] = False
        for dy, dx in NB8:
            y0, y1 = max(dy, 0), h + min(dy, 0)
            x0, x1 = max(dx, 0), w + min(dx, 0)
            buf[y0:y1, x0:x1] |= cur[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        nxt = buf & trav & ~visited
        dist[nxt] = d
        visited |= nxt
        cur = nxt
    return dist


class SharedMap:
    def __init__(self, world, starts):
        self.size = world.size
        self.n = int(round(world.size / RES))
        self.l = np.zeros((self.n, self.n), np.float32)
        self.seen = [np.zeros((self.n, self.n), bool) for _ in range(2)]
        self.discovered = [0, 0]
        self.true_occ = world.true_occupancy(RES)
        free = ~self.true_occ
        reach = np.zeros_like(free)
        for x, y, _ in starts:
            comp = hop_field(free, self.cell((x, y)))
            reach |= np.isfinite(comp)
        self.explorable = reach | (self.true_occ & dilate(reach, NB8))
        self.total = int(self.explorable.sum())
        self.infl = disc_offsets(6)
        self._trav = None

    def cell(self, xy):
        ix = min(max(int(xy[0] / RES), 0), self.n - 1)
        iy = min(max(int(xy[1] / RES), 0), self.n - 1)
        return iy, ix

    def center(self, c):
        return np.array([(c[1] + 0.5) * RES, (c[0] + 0.5) * RES])

    def flat_ids(self, pts):
        c = np.clip((np.asarray(pts).reshape(-1, 2) / RES).astype(int), 0, self.n - 1)
        return c[:, 1] * self.n + c[:, 0]

    def known_flat(self, ids):
        v = self.l.flat[ids]
        return (v <= TH_FREE) | (v >= TH_OCC)

    def integrate(self, rid, origin, angles, ranges, hits, dyn):
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
        step = RES * 0.6
        nt = int(3.6 / step) + 1
        ts = np.arange(nt) * step
        lim = ranges - np.where(dyn, 0.15, RES * 0.8)
        pts = origin + dirs[:, None, :] * ts[None, :, None]
        valid = ts[None, :] < lim[:, None]
        ids = self.flat_ids(pts[valid]) if valid.any() else np.empty(0, np.int64)
        free_ids = np.unique(ids) if ids.size else np.empty(0, np.int64)
        solid = hits & ~dyn
        if solid.any():
            ep1 = origin + dirs[solid] * (ranges[solid] + RES * 0.5)[:, None]
            ep2 = origin + dirs[solid] * (ranges[solid] + RES * 1.5)[:, None]
            occ_ids = np.unique(np.concatenate([self.flat_ids(ep1), self.flat_ids(ep2)]))
        else:
            occ_ids = np.empty(0, np.int64)
        free_ids = np.setdiff1d(free_ids, occ_ids, assume_unique=True)
        touched = np.concatenate([free_ids, occ_ids])
        if not touched.size:
            return
        before = self.known_flat(touched)
        self.l.flat[free_ids] = np.clip(self.l.flat[free_ids] + L_FREE, -L_CLAMP, L_CLAMP)
        self.l.flat[occ_ids] = np.clip(self.l.flat[occ_ids] + L_OCC, -L_CLAMP, L_CLAMP)
        after = self.known_flat(touched)
        self.discovered[rid] += int((after & ~before).sum())
        self.seen[rid].flat[touched] = True
        self._trav = None

    @property
    def occ_mask(self):
        return self.l >= TH_OCC

    @property
    def free_mask(self):
        return self.l <= TH_FREE

    @property
    def unknown_mask(self):
        return (self.l > TH_FREE) & (self.l < TH_OCC)

    def coverage(self):
        known = (self.free_mask | self.occ_mask) & self.explorable
        return known.sum() / self.total

    def traversable(self):
        if self._trav is None:
            self._trav = ~dilate(~self.free_mask, self.infl)
        return self._trav

    def frontier_clusters(self, min_size=8):
        fr = self.free_mask & dilate(self.unknown_mask, NB8)
        cells = np.argwhere(fr)
        clusters = []
        if not len(cells):
            return clusters
        visited = np.zeros(fr.shape, bool)
        for cy, cx in cells:
            if visited[cy, cx]:
                continue
            stack = [(int(cy), int(cx))]
            visited[cy, cx] = True
            group = []
            while stack:
                y, x = stack.pop()
                group.append((y, x))
                for dy, dx in NB8:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.n and 0 <= nx < self.n and fr[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            g = np.array(group)
            if len(g) >= min_size:
                clusters.append({"cells": g, "centroid": self.center(g.mean(0)),
                                 "size": len(g)})
        return clusters
