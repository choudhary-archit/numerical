"""
This is an implementation of the Explicit Runge-Kutta method, which is a
method that can be used to solve ODEs of the form y' = f(t, y). A fairly
detailed explanation can be found on Wikipedia, but let me explain it here
for completeness' sake.

Suppose we are given a function f: R^2 -> R and real numbers t_0, y_0, and
we wish to find a differentiable function y: R -> R such that y(t_0) = y_0
and y'(t) = f(t, y(t)) for all reals t. It may be impossible to solve for y
exactly, for instance when f(u, v) = u + v for all reals u, v, in which case
we must revert to numerical approximations.

ERK is as follows. Pick a positive integer s, pick an s x s lower triangular
matrix A with zeros on the diagonal, and pick two 1 x s matrices (vectors)
B and C such that c_0 = 0 (assuming they are labelled in the obvious manner).
Also pick a number h, which will function as our time-step. Suppose we wish
to compute y(t) at N equally spaced values of t, namely t_0 + n*h for
n = 0 to N-1. We know already that y(t_0) = y_0, so suppose we have computed
y(t_0), ..., y(t_n), and we now wish to compute y(t_{n+1}).

We compute, in the order described below,

k_0 = f(t_n, y(t_n)),
k_1 = f(t_n + hc_1, y(t_n) + h(a_{21}k_1)),
k_2 = f(t_n + hc_2, y(t_n) + h(a_{31}k_1 + a_{32}k_2)),
...
k_{s-1} = f(t_n + hc_{s-1}, y(t_n) + h(a_{s-1,1}k_1 +...+ a_{s-1,s-2}k_{s-2})),

and now let

y(t_{n+1}) = y(t_n) + h(b_0k_0 + b_1k_1 + ... b_{s-1}k_{s-1}).

Of course, one can't just pick A, B, C out of a hat. There are probably
criterions for which A, B, C give the best approximations, but I'm guessing
you aren't a mathematician; if you are then you'll know where to look. Also,
the higher the value of s (assuming A, B, C are chosen accordingly well), the
better the approximation. The most used value is s = 4, with A, B, C being

A = [[0  ,   0, 0, 0],
     [1/2,   0, 0, 0],
     [0  , 1/2, 0, 0],
     [0  ,   0, 1, 0]],

B = [1/6, 1/3, 1/3, 1/6],

C = [0, 1/2, 1/2, 1].

With these values of s, A, B, C, the method is called RK4.

"""


import numpy as np
import math


"""
The `ekr()` function takes as input the order `s`, the arrays `A`, `B`, `C`,
the function `f`, the initial values `t0`, `y0`, the time-step `h`, and the
number of steps `steps`. It returns a N x 2 array `ty`, which is of the form

[[t_0, y_0], [t_1, y_1], ..., [t_n, y_n], ..., [t_{steps - 1}, y_{steps - 1}]].

Let me go through how it works. We first initialize the required `ty` array as
a zero-filled NumPy array (note that you cannot write `np.array((steps, 2))` to
initialize this array, you have to write `np.zeros((steps, 2))` or
`np.ones((steps, 2))` instead), and then put the given initial values t0, y0
into this array. Then begins the main for-loop which executes ERK at each
`step`. In the for-loop, we set the values t = t_n and y = y_n which assume
to have already been computed (indeed, the start of the loop corresponds to
n = 0 and we already know t_0 and y_0). We then initialize a 1 x s array called
`K`, which will store the values k_0, k_1, ..., k_{s-1} at that particular
`step`. Of course, we already know that k_0 = f(t_n, y_n), so we set
`K[0] = f(t, y)`. We then iteratively define k_i for 1 <= i <= s-1, using the
equations mentioned at the beginning. For reference, the ith equation was

k_i = f(t_n + hc_i, y_n + h(a_{i,0}k_0 + a_{i,1}k_1 + ... + a_{i,i-1}k_{i-1})).

We can't write ... in Python, so we use the builtin `sum()` function. Finally,
once we have `K`, we exit out of this inner for-loop and compute `y_new` ---
recall the equation

y(t_{n+1}) = y(t_n) + h(b_0k_0 + b_1k_1 + ... + b_{s-1}k_{s-1})

from above --- and `t_new` and append these newly computed values to our `ty`
array. Once this outer for-loop finishes, we have computed the desired values
and happily return them in the form of a NumPy array.
"""

def ekr(s, A, B, C, f, t0, y0, h, steps):
    ty = np.zeros((steps, 2))
    ty[0, 0], ty[0, 1] = t0, y0

    for step in range(steps - 1):
        t = t0 + (h * step)
        y = ty[step, 1]

        K = np.zeros((s))
        K[0] = f(t, y)

        for i in range(1, s):
            t_inp = t + h * C[i]
            y_inp = y + h * sum([A[i,j] * K[j] for j in range(i-1)])

            K[i] = f(t_inp, y_inp)

        t_new = t + h
        y_new = y + h * sum([B[j] * K[j] for j in range(s)])

        ty[step + 1, 0] = t_new
        ty[step + 1, 1] = y_new

    return ty


"""
The `main()` function below is really just an example scenario using the values
for RK4, with the function f(u, v) = u + v^2. If you want to play around with
this, you can import this file into another Python file (make sure they are in
the same directory) and use the `erk()` function as you please. For example,
you could use numpy.random to generate random matrices for A, B, C, and compare
the results. You could possibly take `f` as user input, essentially turning this
into a calculator; see

https://stackoverflow.com/questions/43695459/getting-mathematical-function-as-users-input.

You could perhaps even use Matplotlib or something of the sort to graph the
resulting points (everyone likes pretty graphs). The possibilities are endless.
"""

def main():
    s = 4
    A = np.array([[0,0,0,0], [1/2,0,0,0], [0,1/2,0,0], [0,0,1,0]])
    B = np.array([1/6,1/3,1/3,1/6])
    C = np.array([0,1/2,1/2,1])

    f = lambda u, v: u + v
    t_init = 0
    t_final = 1
    h = 0.005
    steps = math.floor((t_final - t_init) / h)
    y0 = 1

    results = ekr(s, A, B, C, f, t_init, y0, h, steps)
    for n in range(len(results)):
        print(f"{n:02}: \t t = {results[n, 0]:.2f} \t y = {results[n, 1]:.2f}")

if __name__ == "__main__":
    main()
