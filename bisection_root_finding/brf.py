"""
This is a Python implementation of a binary-search-like method that can be
used to approximate solutions to equations of the form f(x) = 0. More
precisely, consider a continuous function f: [a, b] -> R which is nonzero at
the endpoints a and b, and moreover is such that f(a) and f(b) have different
signs. It is easy to prove the existence of some number x in (a, b) for which
we have f(x) = 0.

We can approximate the value of x as follows. Let us assume that f(a) > 0 and
f(b) < 0. We are searching for a value a <= x <= b such that f(x) = 0. Let
c = (a+b)/2. If f(c) >= 0, we need only need look in the interval [a, c], and
if f(c) < 0 we need only search in [c, b]. We can keep taking midpoints and
getting closer and closer to the actual value. This is the method that is
implemented below.

It is possible to code this both recursively and iteratively. Using iteration
will be better on the memory, because you don't have to store the previous
results in memory and can just throw them away each at each iteration. You
won't be able to recurse a thousand times; Python will freak out. But you can
iterate as much as you need to.

To avoid convoluting the code with edge cases, I have assumed below that f(a)
is positive while f(b) is negative in both of the functions `brf_recursive()`
and `brf_iterative()`.
"""


def brf_recursive(f, a, b, steps):
    if steps == 0:
        return a if abs(f(a)) <= abs(f(b)) else b

    c = (a + b) / 2

    if f(c) >= 0:
        return brf_recursive(f, c, b, steps - 1)
    else:
        return brf_recursive(f, a, c, steps - 1)


def brf_iterative(f, a, b, steps):
    left, right = a, b

    for _ in range(steps):
        mid = (left + right)/2

        if f(mid) >= 0:
            left = mid
        else:
            right = mid

    return left if abs(f(left)) <= abs(f(right)) else right


"""
Below is an example use of this method, with the function

f(x) = 1.57 - x^2(1-x).

According to WolframAlpha, there is a root at x = -0.907282. Our method gives
the approximation -0.9072821288535939, so yay.
"""

def main():
    f = lambda x: 1.57 - x**2 * (1 - x)
    a = 0
    b = -2

    print(brf_recursive(f, a, b, 100))
    print(brf_iterative(f, a, b, 100))

if __name__ == "__main__":
    main()
