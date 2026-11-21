import math
from collections import Counter
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import edges, exposure, binomial_tail, demands, paths, provisioning, single_probability, fidelities, schedules


class ModelTests(unittest.TestCase):
    def test_binomial_against_enumeration(self):
        for m in [1,5,20,100]:
            for p in [0.,.001,.2,.7,1.]:
                for k in range(m+2):
                    expected=sum(math.comb(m,x)*p**x*(1-p)**(m-x) for x in range(k,m+1))
                    self.assertAlmostEqual(binomial_tail(m,p,k),expected,places=12)

    def test_demand_counts_from_paths(self):
        for n in range(2,13):
            ps=paths(n)
            counts=tuple(sum(path[0]<=i<path[-1] for path in ps) for i in range(1,n))
            self.assertEqual(counts,demands(n,2))
            self.assertEqual(demands(n,1),(len(ps),)*(n-1))
        self.assertEqual(demands(5,2),(4,6,6,4))

    def test_star_routes_demands_and_provisioning(self):
        for n in [2, 3, 5, 8]:
            routes = paths(n, 'star')
            self.assertEqual(len(routes), math.comb(n, 2))
            traversals = Counter(tuple(sorted(e)) for p in routes for e in zip(p, p[1:]))
            self.assertEqual(tuple(traversals[e] for e in edges(n, 'star')), (n-1,)*(n-1))
            self.assertEqual(demands(n, 2, 'star'), (n-1,)*(n-1))
            self.assertEqual(demands(n, 1, 'star'), (math.comb(n, 2),)*(n-1))
        p = single_probability(20)
        q = provisioning(5, 2, 10, p, 'star')[0]
        self.assertAlmostEqual(q, binomial_tail(10, p, 4)**4)
        self.assertAlmostEqual(provisioning(5, 1, 10, p, 'star')[0], p**40)

    def test_star_average_does_not_depend_on_order(self):
        order = paths(5, 'star')
        a = exposure(order, 5, 'star')
        b = exposure(order[::-1], 5, 'star')
        indirect = [len(p) == 3 for p in order]
        self.assertFalse(np.array_equal(a[indirect], b[indirect]))
        np.testing.assert_array_equal(np.sort(a[indirect], axis=0), np.sort(b[indirect], axis=0))
        r = fidelities(5, 2, 10, 20, [0, 10], topology='star')
        np.testing.assert_array_equal(r['best'], r['worst'])

    def test_numeric_examples(self):
        p=single_probability(100)
        self.assertAlmostEqual(p,.005806737977893178,places=15)
        self.assertAlmostEqual(binomial_tail(1000,p,10),.07068412793869405,places=10)
        self.assertAlmostEqual(provisioning(5,1,2000,p)[0],.27315542327399617,places=10)
        self.assertAlmostEqual(provisioning(5,2,2000,p)[0],.9438058644308,places=10)

    def test_impossible_and_deterministic(self):
        self.assertEqual(provisioning(5,1,9,.5)[0],0.)
        for model in [1,2]:
            self.assertEqual(provisioning(5,model,10,1.)[0],1.)
        np.testing.assert_allclose(fidelities(5,1,9,100,[0,10])['best'],.5)
        res=fidelities(2,2,1000,0,[0],d1=1,rho=1,mem=1,d2=1)
        np.testing.assert_allclose(res['best'],1.)

    def test_ordering_and_monotonicity(self):
        for n in [2,3,4,5,6,8]:
            a=fidelities(n,1,2000,100,[0,1,20],samples=10)
            b=fidelities(n,2,2000,100,[0,1,20],samples=10)
            self.assertTrue(np.all(a['best']<=b['best']+1e-14))
            self.assertTrue(np.all(a['best']>=a['worst']-1e-14))
            self.assertTrue(np.all(np.diff(a['best'])<=1e-14))
            self.assertTrue(np.all(a['worst']>=.5))
        self.assertEqual(len(schedules(5)[0]),720)
        self.assertTrue(schedules(5)[2])
        self.assertFalse(schedules(6,samples=10)[2])


if __name__=='__main__':
    unittest.main()
