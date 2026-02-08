# Steepest Descent (also known as Gradient descent) algorithm

A fundamental algorithm useful for unconstrained optima of a differentiable cost function of several variables.

The princpile idea is to move along the direction for which the directional derivative is minimal.

The 'Gradient descent' naming comes from the fact that if a multivariate function is differentiable in the neighborhood of some point x_hat, then that function decreases fastest if one goes from x_hat into the direction of the negative gradient.

## Background

Let 

- f(.): function to minimize (no constraints)
- x:    current value (known)
- x*:   minimum (unknown)

The primary condition is that the function f(x) must be continuous, differentiable, and convex. The function’s gradient also needs to satisfy the requirements for Lipschitz continuity. This means that there should be a value k, such that the derivative f'(x) in its absolute value should always be lower than k.

If we meet these conditions, we can then compute the gradient \nabla f(\theta) of the function. Then, according to its sign, we update the parameters in the direction that decreases the cost function.

Assume that x* = x + delta(x)

We reformulate the problem using the second order Taylor series approximation:

    f(x*) = f(x+delta(x)) = f(x) + grad(f(x))'delta(x) + delta(x)' hessian(f(x)) delta(x)

and we look for the minimum delta(x) so that f(x+delta(x)) is minimum

## Minimization problem

So by the first derivative we find delta(x) that minimizes f(x+delta(x)).

    grad(f(x+delta(x))) = 0 # Gradient w.r.t to delta(x)
    grad(f(x) + grad(f(x))'delta(x) + delta(x)' hessian(f(x)) delta(x)) = 0
    grad(f(x)) + hessian(f(x)) delta(x) = 0
    delta(x) = - grad(f(x)) / hessian(f(x))

The update rule during iteration k becomes:

    next_x = prev_x - grad(f(prev_x)) / hessian(f(prev_x))

## The gradient descent algorithm

The algoritm becomes

    prev_x = initial_guess()
    for k in range(1:n):
        # If direction size is less than a predefined termination scalar, return current point.
        if hessian(f(prev_x)) < epsilon:
            return prev_x
        
        g = - grad(f(prev_x))                   # Choose negative gradient as search direction
        lambda = argmin(f(x_prev) + lambda*g)   # Determine optimal step size
        next_x = prev_x + lambda * g            # Perform update

Lambda can be choosen according to Armijo's rule. This would, under certain continuity assumptions on f(x), guarantee convergence of the gradient of f to 0.
