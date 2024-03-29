import numpy as N

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        # original: Df_x[:,i] = (f(x + v) - fx)
        Df_x[:,i] = (f(x + v) - fx) / dx
    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans

    def __call__(self, x):
        return self.f(x)

class Polynomial_Df(object):
    """Callable analytical Jacobian of polynomial object
    For example: for p(x) = ax^2 + bx + c
    Df = 2ax + b"""
    def __init__(self, coeffs):
        self._coeffs = coeffs
        
    def __repr__(self):
        return "Polynomial_Df(%s)" % (", ".join([str(x) for x in self._coeffs]))
        
    def Df(self,x):
        a = self._coeffs[0]
        b = self._coeffs[1]
        ans = 2*a*x + b
        return ans
    
    def __call__(self, x):
        return self.Df(x)

class Linear(object):
    """Callable multi-variable linear system
    A * x + B = 0"""
    def __init__(self, A, B):
        self._A = A
        self._B = B
        
    def __repr__(self):
        return "Linear(A, B)"
    
    def f(self,x):
        return self._A * x + self._B
        
    def __call__(self, x):
        return self.f(x)
        
class Linear_Df(object):
    """Callable analytical Jacobian of multi-variable linear system
    For A * x + B = 0, the Jacobian is A"""
    def __init__(self, A, B):
        self._A = A
        self._B = B
    def __repr__(self):
        return "Linear_Df(A, B)"
    
    def Df(self,x):
        return self._A
        
    def __call__(self, x):
        return self.Df(x)
        