
def find_x0(A, b, g):
    """ Find initial central point that fullfills Ax=b and g(x) < 0 """
    while g(x) >= 0:
        x = solve(A, b)
    return x

def barrier_method(f: Callable, A, b, g: Callable):
    """ Barrier method or Path-Following method """
    n = size(A, 1) # A is (mxn)
    x0 = find_x0(A, b, g)
    t = 1 # Choose t1 > 0
    for k in range(n):
        # Compute central point x(tk), use Newton's method starting at preivous central point
        xk = argmin(t, f, g, xk)
        if l/t < epsilon:
            return xk
        t = alpha * t   # Increase parameter t
        