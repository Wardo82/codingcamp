# Solution methods for dual problems

This module deals with optimization problems where dealing with its dual results more beneficial. 

In particular, the dual problem may have smaller dimension and/or simpler constraints. It may also decompose into several subproblems that are easier to solve.

## Submodules

It is worth noting that in many cases obtaining the dual function in closed form is often not possible. More so, the dual function might not be differentiable for many type of problems.

In the module, approaches for solving nondifferentiable dual problems are considered.

- Subgradient methods
- Cutting plane method