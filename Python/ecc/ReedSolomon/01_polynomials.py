import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial as poly

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
def plot_polynom(f, points, begin: int = -2, end: int = 5):
    # Dominio continuo para el plot
    x = np.linspace(begin, end, 400)
    y = f(x)

    # Puntos específicos a marcar
    y_pts = f(points)

    # Plot
    plt.figure()
    plt.plot(x, y, label="f(x)")
    plt.scatter(points, y_pts, zorder=3, label="Puntos evaluados")
    plt.axhline(0)
    plt.axvline(0)
    plt.grid(True)
    plt.legend()

    plt.show()


# Conseguir el polinomio a partir de los puntos
def build_vandermonde(f, x):
    """
    f: numpy.polynomial.Polynomial
    x: iterable of evaluation points

    returns:
        A: Vandermonde matrix
        b: f evaluated at x
    """
    x = np.asarray(x, dtype=float)
    degree = f.degree()

    A = np.zeros((len(x), degree + 1), dtype=float)

    for i, xi in enumerate(x):
        for j in range(degree + 1):
            A[i, j] = xi ** j

    b = f(x)
    return A, b

f = poly.Polynomial([2, 3, -5, 1])

x = np.array([-1, 0, 1, 2])
plot_polynom(f, x)
A, b = build_vandermonde(f, x)
print("Queremos resolver el sistema de equaciones Ac = b")
print("A:")
print(A)
print(f"b: {b}")
coeffs = np.linalg.solve(A, b)
print(f"Coeficientes: c = A^-1 b = {coeffs}")


x = np.array([1, 2, 3, 4])
plot_polynom(f, x)
A, b = build_vandermonde(f, x)
print("A:")
print(A)
print(f"b: {b}")
coeffs = np.linalg.solve(A, b)
print(f"Coeficientes: c = A^-1 b = {coeffs}")

