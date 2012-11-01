# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import functions as F
# User-defined exception
class My_Exception(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    pass
    
class Newton(object):
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, Df=None, r=None):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian
        Df:      analytical Jacobian
        r:       tolerable radius ||x - x0||"""
        self._f = f
        self._tol = tol
        self._maxiter = maxiter
        self._dx = dx
        self._Df = Df
        self._r = r

    def solve(self, x0):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
        for i in xrange(self._maxiter):
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            x = self.step(x, fx)
            # Raise exception when radius is exceeded
            try:
                if self._r is not None and N.linalg.norm(x - x0) > self._r:
                    flag_exception = "RadiusExceeded"
                    raise My_Exception(flag_exception)
            except My_Exception as e:
                print 'My exception occurred: ' + e.value
        # If it doesn't converge after reaching maxiter
        try:
            fx = self._f(x)
            if N.linalg.norm(fx) >= self._tol:
                flag_exception = "NotConvergingAtMaxiter"
                raise My_Exception(flag_exception)
        except My_Exception as e:
            print 'My exception occurred: ' + e.value
        return x

    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        # If analytical Jacobian is given
        if self._Df is None:
            Df_x = F.ApproximateJacobian(self._f, x, self._dx)
        else:
            Df_x = self._Df(x)
        h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx))
        # wrong: return x + h
        return x - h

