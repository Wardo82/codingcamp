# Subgradient descent

Subgradient methods are the simplest among the most popular methods for dual optimization. 
They are useful for unconstrained optima of nondifferentiable functions that still fullfill the convex condition:

For the domain of f

    f(x) + g' (y-x) <= f(y)

One can still find a non-unique subgradient g.

## Subdifferential

The sub-differential is the set of all these subgradients g of f. Let this set be Sd

    Sd = {g | g is a subgradient of f}

The set Sd is a convex set, which means that it is closed under convex combination of the ubgradients of f.

    g1, g2 € Sd, lambda € [0, 1]
    lambda*g1 + (1-lambda)*g2 € Sd

### First order optimality condition

From this fact one can derive the first necessary condition for x* to be a global minimum.

    The 0 vector must be a member of the set of subdifferentials.

## The subgradient method

The algoritm becomes

    prev_x = initial_guess()
    x_min = prev_x

    for k in range(1:n):
        g = - grad(f(prev_x))                   # Choose negative gradient as search direction
        lambda = argmin(f(x_prev) + lambda*g)   # Determine optimal step size
        next_x = prev_x + lambda * g            # Perform update
        x_min = argmin(f(next_k), f(x_min))     # Store minimum

Lambda can be choosen according to Armijo's rule. This would, under certain continuity assumptions on f(x), guarantee convergence of the gradient of f to 0.

## Sources:

[First Order Methods (subgradient, projected gradient)](https://www.youtube.com/watch?v=x4IyNENx8L4)
