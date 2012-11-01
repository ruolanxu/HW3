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
    
    def testLinear2D(self):
        # 2D linear equations
        A = N.matrix("1.0, 2.0; 3.0, 4.0")
        B = N.matrix("5.0; 7.0")
        x0 = N.matrix("0.0; 0.0")
        self.assertEqual(A.shape[1], x0.shape[0])
        self.assertEqual(B.shape, x0.shape)
        f = F.Linear(A, B)
        df = F.Linear_Df(A, B)     
        solver = newton.Newton(f, maxiter=1000, Df=df)
        x = solver.solve(x0)
        x_sol = N.matrix("3.0; -4.0")
        N.testing.assert_array_almost_equal(x, x_sol)
 
    def testAnalyticalJacobianUsed(self):
        # Test if analitical Jacobian is used in Newton
        f = lambda x : x**2
        df = lambda x: 2*x
        x0 = 10.0
        solver = newton.Newton(f, maxiter=1, Df=df)
        x = solver.solve(x0)
        x_analytical = x0 - f(x0) / df(x0)
        self.assertAlmostEqual(x, x_analytical)
        
    def testQuadraticWithAnalyticalJacobian(self):
        # f(x) = a* x^2 + b * x + c
        a, b, c = 2.0, 3.0, -5.0
        self.assertGreater(b**2-4*a*c, 0)
        f = F.Polynomial([a, b, c])
        df = F.Polynomial_Df([a, b, c])
        solver = newton.Newton(f, maxiter=10000, dx=1e-3, Df=df)
        x0 = [2.0, -4.5]
        x_sol = [((-1) * b + N.sqrt(b**2 - 4 * a * c)) / (2.0 * a), 
                 ((-1) * b - N.sqrt(b**2 - 4 * a * c)) / (2.0 * a)]
        for i in range(2):
            x = solver.solve(x0[i])
            self.assertAlmostEqual(x, x_sol[i])
            
    def testRadiusException(self):
        # Test if radius is excepted, an exception will be raised and passed
        f = lambda x : 3.0 * x + 6.0
        df = lambda x : 3.0
        r = 1
        solver = newton.Newton(f, tol=1.e-15, maxiter=1, Df=df, r=r)
        x = solver.solve(2.0)
        self.assertRaises("RadiusExceeded")        
    
    def testRadiusCubeRoot(self):
        # Test the cube root function for exceeding radius
        f = lambda x : N.power(x, 1.0/3.0)
        df = lambda x : 1.0/3.0 * N.power(x, -2.0/3.0)
        r = 10
        solver = newton.Newton(f, tol=1.e-15, maxiter=1000, Df=df, r=r)
        x = solver.solve(1.0)
        self.assertRaises("RadiusExceeded")     
        
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
