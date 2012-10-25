#!/usr/bin/env python

import newton
import unittest
import numpy as N
import functions as F

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)
        
    def testSingleStep(self):
        # f(x) = a* x^2 + b * x + c
        a, b, c = 2, 3, 5
        def f(x):
            return a * x**2 + b * x + c
        x0 = 2.0
        #dx = 0.0001
        for dx in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            solver = newton.Newton(f, maxiter=2, dx=dx)
            x = solver.step(x0)
            self.assertAlmostEqual(x, x0-f(x0)/((f(x0+dx)-f(x0))/dx))
    

if __name__ == "__main__":
    unittest.main()
