import time

import numpy as np

from drce import mapping, planner
from main import Sim, build


def check_connectivity(mp, starts):
    trav = ~mapping.dilate(mp.true_occ, mp.infl)
    c1 = planner.nearest_traversable(trav, mp.cell(starts[0][:2]), 8)
    c2 = planner.nearest_traversable(trav, mp.cell(starts[1][:2]), 8)
    assert c1 is not None and c2 is not None, "start pose blocked after inflation"
    field = mapping.hop_field(trav, c1)
    assert np.isfinite(field[c2]), "robot starts not connected after inflation"
    frac = mp.total / mp.n ** 2
    assert frac > 0.45, f"explorable fraction too small: {frac:.2f}"


def run_scenario(sc):
    wd, mp, robots, name = build(sc)
    check_connectivity(mp, [(r.x, r.y) for r in robots])
    sim = Sim(wd, mp, robots, 240.0)
    t0 = time.time()
    while sim.done is None:
        sim.step()
    wall = time.time() - t0
    s = sim.metrics.summary(mp, robots, sim.t, sim.done)
    assert s["cov"] >= 95.0, f"coverage {s['cov']:.1f}% < 95%"
    for r in s["rows"]:
        assert r["min_clear"] > 0.02, f"{r['name']} obstacle clearance {r['min_clear']:.3f} m"
        assert r["share"] >= 12.0, f"{r['name']} discovery share {r['share']:.1f}% < 12%"
    assert s["min_inter"] > 0.05, f"robot-robot gap {s['min_inter']:.3f} m"
    assert s["precision"] >= 0.85, f"map precision {s['precision']:.2f} < 0.85"
    print(f"scenario {sc} ({name}): PASS  cov={s['cov']:.1f}%  t={s['t']:.1f}s  "
          f"gap={s['min_inter']:.2f}m  prec={s['precision'] * 100:.0f}%  wall={wall:.1f}s")
    return s


def main():
    results = []
    for sc in (1, 2, 3):
        results.append(run_scenario(sc))
    print(f"\nall {len(results)} scenarios passed")
    from drce.metrics import print_report
    for s in results:
        print_report(s)


if __name__ == "__main__":
    main()
