import numpy as np


class Rect:
    def __init__(self, cx, cy, w, h):
        self.c = np.array([cx, cy], float)
        self.half = np.array([w / 2, h / 2], float)

    def sdf(self, pts):
        q = np.abs(pts - self.c) - self.half
        outer = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inner = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
        return outer + inner


class Circle:
    def __init__(self, cx, cy, r):
        self.c = np.array([cx, cy], float)
        self.r = r

    def sdf(self, pts):
        return np.linalg.norm(pts - self.c, axis=-1) - self.r


class ConvexPolygon:
    def __init__(self, verts):
        self.v = np.asarray(verts, float)

    def sdf(self, pts):
        pts = np.asarray(pts, float)
        p = pts.reshape(-1, 2)
        d = np.full(len(p), np.inf)
        s = np.full(len(p), -np.inf)
        n = len(self.v)
        for i in range(n):
            a, b = self.v[i], self.v[(i + 1) % n]
            e = b - a
            pa = p - a
            t = np.clip(pa @ e / (e @ e), 0.0, 1.0)
            d = np.minimum(d, np.linalg.norm(pa - t[:, None] * e, axis=1))
            s = np.maximum(s, pa @ np.array([e[1], -e[0]]) / np.hypot(*e))
        out = np.where(s > 0, d, s)
        return out.reshape(pts.shape[:-1])


def regular_polygon(cx, cy, r, n, rot=0.0):
    a = rot + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return ConvexPolygon(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))


class World:
    def __init__(self, size, shapes, res=0.02):
        self.size = size
        self.shapes = shapes
        self.res = res
        self.n = int(round(size / res)) + 1
        ax = np.linspace(0.0, size, self.n)
        gx, gy = np.meshgrid(ax, ax)
        pts = np.stack([gx, gy], axis=-1)
        d = np.minimum(np.minimum(gx, size - gx), np.minimum(gy, size - gy))
        for s in shapes:
            d = np.minimum(d, s.sdf(pts))
        self.grid = d.astype(np.float64)

    def sample(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2) / self.res
        x = np.clip(p[:, 0], 0.0, self.n - 1.001)
        y = np.clip(p[:, 1], 0.0, self.n - 1.001)
        x0 = x.astype(int)
        y0 = y.astype(int)
        fx = x - x0
        fy = y - y0
        g = self.grid
        return (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x0 + 1] * fx * (1 - fy)
                + g[y0 + 1, x0] * (1 - fx) * fy + g[y0 + 1, x0 + 1] * fx * fy)

    def clearance(self, xy):
        return float(self.sample(np.asarray(xy, float))[0])

    def raycast(self, origin, angles, max_range):
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
        t = np.zeros(len(angles))
        alive = np.ones(len(angles), bool)
        for _ in range(180):
            p = origin + dirs * t[:, None]
            d = self.sample(p)
            hit = d < 0.035
            t = np.where(alive & ~hit,
                         np.minimum(t + np.maximum(d - 0.02, 0.012), max_range), t)
            alive &= ~hit & (t < max_range - 1e-9)
            if not alive.any():
                break
        d = self.sample(origin + dirs * t[:, None])
        return t, (d < 0.05) & (t < max_range - 1e-9)

    def true_occupancy(self, res):
        n = int(round(self.size / res))
        ax = (np.arange(n) + 0.5) * res
        gx, gy = np.meshgrid(ax, ax)
        pts = np.stack([gx, gy], axis=-1).reshape(-1, 2)
        return (self.sample(pts) < res * 0.55).reshape(n, n)


def make_world(scenario):
    if scenario == 1:
        shapes = [Rect(1.95, 8.05, 1.7, 1.7), Rect(5.0, 8.05, 2.0, 1.7),
                  Rect(8.05, 8.05, 1.7, 1.7), Rect(1.95, 5.0, 1.7, 2.0),
                  Rect(8.05, 5.0, 1.7, 2.0), Rect(1.95, 1.95, 1.7, 1.7),
                  Rect(5.0, 1.95, 2.0, 1.7),
                  regular_polygon(8.05, 1.95, 0.85, 6, 0.0),
                  Circle(5.0, 5.0, 0.38)]
        starts = [(0.55, 0.55, 0.8), (9.45, 9.45, -2.4)]
        name = "Midtbyen"
    elif scenario == 2:
        shapes = [Rect(1.1, 9.0, 2.2, 2.0), Rect(2.45, 9.75, 0.5, 0.5),
                  Rect(0.3, 7.0, 0.6, 2.0), Rect(0.25, 3.5, 0.5, 1.8),
                  Rect(0.7, 0.3, 1.4, 0.6), Circle(0.9, 0.75, 0.28),
                  Rect(3.2, 3.0, 1.8, 0.7), Rect(3.2, 4.6, 1.0, 0.5),
                  Circle(6.2, 4.6, 0.6), Circle(7.0, 4.6, 0.25),
                  Circle(5.8, 5.29, 0.25), Circle(5.8, 3.91, 0.25),
                  Rect(9.7, 8.4, 0.6, 3.2), Rect(8.1, 7.5, 0.6, 1.8),
                  Rect(8.25, 0.25, 3.5, 0.5), Circle(5.2, 7.9, 0.3)]
        starts = [(1.6, 1.8, 0.8), (8.6, 2.8, 2.4)]
        name = "Studio"
    else:
        shapes = [Rect(1.15, 8.6, 2.3, 0.16), Rect(6.4, 8.6, 4.4, 0.16),
                  Rect(8.6, 6.25, 0.16, 4.7), Rect(2.15, 1.6, 4.3, 0.16),
                  Rect(4.3, 2.95, 0.16, 2.7), Rect(4.3, 7.15, 0.16, 2.9),
                  Rect(1.35, 6.4, 2.7, 0.16), Rect(7.25, 6.4, 2.7, 0.16),
                  Rect(1.25, 3.9, 2.5, 0.16),
                  Rect(4.65, 3.9, 0.7, 0.16), Rect(7.5, 3.9, 2.2, 0.16),
                  Rect(1.8, 7.9, 0.16, 1.4), Rect(0.2, 7.2, 0.4, 0.16),
                  Rect(7.9, 4.4, 1.4, 1.0), Rect(8.2, 5.65, 0.8, 1.5),
                  Rect(8.0, 8.2, 1.2, 0.8),
                  Rect(5.09, 8.01, 1.42, 1.02), Rect(6.05, 8.32, 0.5, 0.4),
                  Rect(0.25, 6.8, 0.5, 0.64), Rect(2.09, 8.32, 0.42, 0.4),
                  Circle(0.25, 8.25, 0.35),
                  Rect(0.2, 5.95, 0.4, 1.0), Rect(1.05, 6.0, 1.3, 0.9),
                  Rect(0.2, 2.2, 0.4, 1.2), Rect(1.1, 1.85, 1.4, 0.5),
                  Rect(0.7, 2.3, 0.6, 0.4), Rect(7.5, 5.7, 0.6, 1.6),
                  Rect(7.5, 3.32, 1.6, 1.0)]
        starts = [(2.9, 4.7, 3.1), (9.2, 0.8, 2.8)]
        name = "Ground Floor"
    return World(10.0, shapes), starts, name
