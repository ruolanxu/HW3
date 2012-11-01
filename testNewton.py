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
        self.assertAlmostEqual(x, -2.0)
        
    def testSingleStep(self):
        # f(x) = a* x^2 + b * x + c
        a, b, c = 2, 3, -5
        def f(x):
            return a * x**2 + b * x + c
        x0 = 2.0
        for dx in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            solver = newton.Newton(f, maxiter=1, dx=dx)
            x = solver.step(x0)
            self.assertAlmostEqual(x, x0-f(x0)/((f(x0+dx)-f(x0))/dx))
            
    def testQuadratic(self):
        # f(x) = a* x^2 + b * x + c
        a, b, c = 2.0, 3.0, -5.0
        self.assertGreater(b**2-4*a*c, 0)
        f = F.Polynomial([a, b, c])
        solver = newton.Newton(f, maxiter=10000, dx=1e-3)
        x0 = [2.0, -4.5]
        x_sol = [((-1) * b + N.sqrt(b**2 - 4 * a * c)) / (2.0 * a), 
                 ((-1) * b - N.sqrt(b**2 - 4 * a * c)) / (2.0 * a)]
        for i in range(2):
            x = solver.solve(x0[i])
            self.assertAlmostEqual(x, x_sol[i])
   
    def testCos(self):
        # f(x) = cos(x) - x^3, x0 = 0.5, test if it converges
        def f(x):
            return N.cos(x) - N.power(x, 3.0)
        solver = newton.Newton(f, maxiter=1000000, dx=1e-3)
        x0 = 0.5
        x = solver.solve(x0)
        solver._maxiter = solver._maxiter - 1
        x1 = solver.solve(x0)
        self.assertAlmostEqual(x, x1)
 
    def testAnalyticalJacobianUsed(self):
        # Test if analitical Jacobian is used in Newton
        f = lambda x : x**2
        df = lambda x: 2*x
        x0 = 10.0
        solver = newton.Newton(f, maxiter=1, Df=df)
        x = solver.solve(x0)
        x_analytical = x0 - f(x0) / df(x0)
        self.assertAlmostEqual(x, x_analytical)
        
    
    @unittest.expectedFailure
    def testInfiniteCycle(self):
        # f(x) = x^3 - 2*x + 2, infinite cycle if x0 = 0
        def f(x):
            return x**3 - 2.0 * x + 2.0
        solver = newton.Newton(f, maxiter=1000, dx=1e-3)
        x0 = 0.0
        x = solver.solve(x0)
        solver._maxiter = solver._maxiter - 1
        x1 = solver.solve(x0)
        self.assertAlmostEqual(x, x1)

if __name__ == "__main__":
    unittest.main()
