import matplotlib.pyplot as plt
import numpy as np

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
def build_vandermonde(x):
    """
    x: iterable of evaluation points

    returns:
        A: Vandermonde matrix
    """
    x = np.asarray(x, dtype=float)

    A = np.zeros((len(x), len(x)), dtype=float)

    for i, xi in enumerate(x):
        for j in range(len(x)):
            A[i, j] = xi ** j
    return A
