import numpy as np
import numpy.polynomial as poly

from utils import *

# Algebra de polinomios:
f = poly.Polynomial([3, 2, 5, -4])  # f(x)
print(f"f(x) = {f}")
g = poly.Polynomial([0, 7, -1])        # g(x)
print(f"g(x) = {g}")

print(f"Suma: {f + g}")
print(f"Resta: {f - g}")
print(f"Multiplication: {f * g}")
print(f"Division: {f//g}")
print(f"Resto: {f % g}")
print("Divmod:")
quo, rem = divmod(f, g)
print(f"    Cociente: {quo} - Resto: {rem}")

# Minima informacion para definir un polinomio
f = poly.Polynomial([2, 3, -5, 1])

x = np.array([-1, 0, 1, 2])
plot_polynom(f, x)
A = build_vandermonde(x)
b = f(x)
print("Queremos resolver el sistema de equaciones Ac = b")
print("A:")
print(A)
print(f"b: {b}")
coeffs = np.linalg.solve(A, b)
print(f"Coeficientes: c = A^-1 b = {coeffs}")


x = np.array([1, 2, 3, 4])
plot_polynom(f, x)
A = build_vandermonde(x)
b = f(x)
print("A:")
print(A)
print(f"b: {b}")
coeffs = np.linalg.solve(A, b)
print(f"Coeficientes: c = A^-1 b = {coeffs}")

