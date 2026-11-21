import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import fidelities, provisioning, single_probability
from paper_dashboard import contour_data, distance_data


class PaperFigureTests(unittest.TestCase):
    def test_contours_match_path_evaluation(self):
        for topology in ('chain', 'star'):
            for policy in (0, 1):
                params = dict(n=5, topology=topology, T=200., alpha=.21, d1=.95, d2=.95,
                              rho=.9, mem=.95, fiber=1.5, policy=policy, samples=10, seed=7)
                ds, dt = (20., 100.), (0., 50.)
                grids, exact = contour_data(params, (1000,), ds, dt)
                self.assertTrue(exact)
                for model in (1, 2):
                    hi, lo = grids[model, 1000]
                    for i, d in enumerate(ds):
                        r = fidelities(model=model, m=1000, distance=d, deltas=dt, **params)
                        for j in range(len(dt)):
                            self.assertAlmostEqual(hi[j, i], r['best'][j])
                            self.assertAlmostEqual(lo[j, i], r['worst'][j])

    def test_provisioning_distance_boundary(self):
        for topology in ('chain', 'star'):
            table = distance_data(5, topology, (1, 4000), .21, .95, .9)
            for row in table.itertuples():
                if row.m == 1:
                    self.assertTrue(math.isnan(row.max_edge_km))
                else:
                    p = single_probability(row.max_edge_km)
                    q = provisioning(5, row.model, row.m, p, topology)[0]
                    self.assertAlmostEqual(q, .99, places=8)
                    self.assertEqual(row.max_total_fiber_km, 4 * row.max_edge_km)
        table = distance_data(5, 'star', (4000,), 0., .95, .9)
        self.assertTrue(all(math.isinf(x) for x in table.max_edge_km))
