import numpy as np

RADIUS = 0.15
V_MAX = 0.7
W_MAX = 2.6


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Robot:
    def __init__(self, rid, name, color, pose, priority):
        self.id = rid
        self.name = name
        self.color = color
        self.x, self.y, self.th = pose
        self.priority = priority
        self.strategy = None
        self.algo_label = ""
        self.path = None
        self.progress = 0
        self.goal_cell = None
        self.target = None
        self.claim = None
        self.state = "warmup"
        self.scale = 1.0
        self.traveled = 0.0
        self.trail = [(self.x, self.y)]
        self.replans = 0
        self.claims = 0
        self.yield_t = 0.0
        self.stuck_t = 0.0
        self.last_progress = 0.0
        self.cooldown = 0.0
        self.blacklist = {}

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def set_path(self, path, claim, state="drive"):
        self.path = path
        self.progress = 0
        self.last_progress = 0.0
        self.stuck_t = 0.0
        self.claim = claim
        self.state = state

    def clear_task(self, state="standby"):
        self.path = None
        self.goal_cell = None
        self.claim = None
        self.state = state

    def step(self, dt):
        v = w = 0.0
        arrived = False
        if self.path is not None and self.state in ("drive", "retreat") and self.scale > 0:
            j0 = self.progress
            seg = self.path[j0:j0 + 60]
            d = np.linalg.norm(seg - self.pos, axis=1)
            self.progress = j0 + int(np.argmin(d))
            ci = min(self.progress + 3, len(self.path) - 1)
            carrot = self.path[ci]
            err = wrap(np.arctan2(carrot[1] - self.y, carrot[0] - self.x) - self.th)
            w = float(np.clip(3.0 * err, -W_MAX, W_MAX))
            v = 0.0 if abs(err) > 1.25 else V_MAX * max(np.cos(err), 0.0) * self.scale
            if self.progress >= len(self.path) - 2 and np.linalg.norm(self.path[-1] - self.pos) < 0.2:
                arrived = True
                v = w = 0.0
        self.x += v * np.cos(self.th) * dt
        self.y += v * np.sin(self.th) * dt
        self.th = wrap(self.th + w * dt)
        self.traveled += v * dt
        if np.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 0.06:
            self.trail.append((self.x, self.y))
        return arrived
