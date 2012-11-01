#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)
            
    def testPolynomial_Df(self):
        # p(x) = x^2 + 2x + 3, test if Polynomial_Df is correct
        p_Df = F.Polynomial_Df([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p_Df(x), 2*x + 2)
    
    def testLinear(self):
        # test if Linear function works
        A = N.matrix("1.0, 2.0; 3.0, 4.0")
        B = N.matrix("5.0; 6.0")
        x = N.matrix("2.0; 3.0")
        C = [13.0, 24.0]
        f = F.Linear(A, B)
        fx = f(x)
        for i in range(0, len(C)):
            self.assertAlmostEqual(fx[i], C[i])

    def testLinear_Df(self):
        # test if Linear_Df function works
        A = N.matrix("1.0, 2.0; 3.0, 4.0")
        B = N.matrix("5.0; 6.0")
        x = N.matrix("2.0; 3.0")
        Df = F.Linear_Df(A, B)
        Df_x = Df(x)
        self.assertEqual(Df_x.shape, A.shape)
        N.testing.assert_array_almost_equal(Df_x, A)
        
    def testPolynomial_Df_accurate(self):
        # Test if the analytical Jacobian for polynomial equation is accurate
        # by comparing with approximate Jacobian
        x0 = 2.0
        p_Df = F.Polynomial_Df([1, 2, 3])
        Df_analytical = p_Df(x0)
        f = F.Polynomial([1, 2, 3])
        Df_approximate = F.ApproximateJacobian(f, x0, dx=1.e-5)
        self.assertAlmostEqual(Df_analytical, Df_approximate, places=4)

                    
if __name__ == '__main__':
    unittest.main()



