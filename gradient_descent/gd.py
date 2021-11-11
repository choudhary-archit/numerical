"""
This is a Python implementation of the famous gradient-descent method. Consider
a differentiable function F: R^n -> R, and suppose we are given a point a in
R^n. Gradient descent is a method to find a local minimum of f, and it works
as follows. Let DF: R^n -> R^n be the gradient of F; namely, the i-th component
of DF is the partial derivative of F with respect to the i-th coordinate. The
idea is that to get to the minimum we can keep moving in the direction which is
exactly opposite to the gradient at our point. So we can let x_0 = a, and
define x_n = x_{n-1} - c_{n-1} * DF(x_{n-1}) for n >= 1, where the c_n are
suitably chosen step-sizes. We need to choose these step-sizes carefully: if
all c_n are too small, the convergence will be very slow, but if some c_n is
too big then we can inadvertently leave the "pit" we are trying to find the
bottom of. In this implementation we use the Barzilai-Borwein method to find
the step-sizes c_n because of its simplicity. It was proposed by Jonathan
Barzilai and Jonathan Borwein (why do both of them have the same first name?)
in a 1986 paper titled "Two-Point Step Size Gradient Methods", published in the
IMA Journal of Numerical Analysis. Basically, the idea is to set

c_n = ((x_n - x_{n-1}) . (DF(x_n) - DF(x_{n-1})) / |DF(x_n) - DF(x_{n-1})|^2;

here we are taking a dot product in the numerator of two real n-dimensional
vectors, and a Euclidean L^2 norm in the denominator. But with this c_n is only
defined for n >= 1, so we have to pick c_0 arbitrarily. I just set c_0 = 0.01
in this implementation.

The gradient function DF could be represented as a N x 1 NumPy array here, with
each element of the array being a lambda function which takes a n x 1 NumPy
array as input where n is the dimension (representing a point in R^n) and gives
a float as output. But then we wouldn't be able to write things like `DF(x)`,
as it isn't correct syntax, expecting to find the gradient F: R^n -> R^n at
some point in R^n. So I wrote a simple class (`Gradient`) that allows us to do
just this. The primary `gd()` function takes as input the gradient DF, the
initial point `a`, and the number `N` of times that we wish to iterate. The
optional `sign`, if set to `1` instead of its default `-1`, gives us "gradient-
ascent". This is because we then move *in* the direction of the gradient rather
than its opposite.

By the way, it might help to read this code bottom-to-top instead of the usual
top-to-bottom.
"""


import numpy as np


"""
The following class allows us to evaluate gradient functions at points.
Initializing a `Gradient` object requires a NumPy array, of course. Note that
in Python, `func(x1, x2, ...)` is short for `func.__call__(x1, x2, ...)`, and
`.__call__()` is a built-in method for every object. See

https://developpaper.com/introduction-to-python-class-built-in-methods/

for a description of most of the important built-in methods.
"""

class Gradient:
    def __init__(self, grad_arr):
        self.grad_arr = grad_arr
        self.dim = grad_arr.shape[0]

    def __call__(self, x):
        y = np.zeros(x.shape)
        for index, _ in np.ndenumerate(self.grad_arr):
            y[index] = self.grad_arr[index](x)

        return y


"""
The `step_size()` function below implements the Barzilai-Borwein method.
It takes as input the gradient `DF` (which is a `Gradient` object) and a NumPy
array `X` of all the points that have been computed. Of course, this entire
array is not needed; only the last two elements. Note also that if `X` contains
only one element then we need to manually set the step-size. I choose the value
`0.01` for this, but you can use any sufficiently small number. For reference,
the formula was

c_n = ((x_n - x_{n-1}) . (DF(x_n) - DF(x_{n-1})) / |DF(x_n) - DF(x_{n-1})|^2.

"""

def step_size(DF, X):
    if X.shape[0] == 1:
        return 0.01

    dx = X[-1] - X[-2]
    dDF = DF(X[-1]) - DF(X[-2])

    return np.dot(dx, dDF) / np.linalg.norm(dDF)**2


"""
The primary function, which actually implements gradient-descent. Takes the
gradient `DF` as input (a `Gradient` object), the initial point `a` (n x 1
NumPy array), and the number of steps `N` that we wish to compute up to. The
optional `sign` parameter, if set to `1`, will find us instead a local maximum
instead of a local minimum. The NumPy array `X` contains the computed points,
and each new point is appended to it as a new row by `numpy.vstack()`.
"""

def gd(DF, a, N, sign=-1):
    n = DF.dim
    X = np.zeros((1, n))
    X[0] = a

    for j in range(1, N):
        X = np.vstack([X, X[-1] + sign * step_size(DF, X) * DF(X[-1])])

    return X


"""
Below is a very simple example of how to use the `gd()` function. Note that
we only need the gradient function in the appropriate format to carry this out.
We don't need to define the original function F. But if you do not have the
gradient explicitly, and would rather compute the gradient from the original
function, take a look at SymPy and in particular its lambdify method.
"""

def main():
    # F = lambda x: (x[0] - 2)**4 + 3
    DF = Gradient(np.array([lambda x: 4 * (x[0] - 2)**3]))
    a = np.array([3])

    print(gd(DF, a, 20))

if __name__ == "__main__":
    main()
