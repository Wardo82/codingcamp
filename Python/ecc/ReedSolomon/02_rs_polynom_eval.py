"""
Del Articulo citado:

Protocol Specification

The encoding and decoding parties must come up with some fixed parameters of the coding protocol that are not message dependent:

    - Agree on an alphabet.
    - Agree on the length k of the message word.
    - Agree on the length n of the code word.
    The longer the difference in length between code word and message word, the more redundancy, and the more corrupted symbols can be error corrected.
    - Agree on how some polynomial p(x) should be constructed out of the symbols of the message word.
    In the original paper the message word symbols are used as coefficients of this polynomial, but that's not always the best choice.  See later sections…
    - Agree on values of x for which to evaluate the polynomial p(x)
    - Agree that the code word is formed by evaluating p(x) at the numbers of x that were agreed on in the previous step.

"""
import numpy as np
import numpy.polynomial as poly
import random

from utils import build_vandermonde

# The alphabet are integers.
# The size k of the message word m is 4.
k = 4
# We want to 2 redundant symbols, so the code word has length n of 6.
n = 6
# The polynomial p(x) is always evaluated for the following values of x: (−1,0,1,2,3,4). Note that there are as many values xi as there are symbols in a code word.
points = [-1, 0, 1, 2, 3, 4]

def encode(message, points):
    # Crea un polynomio donde los simbolos del mensaje son los coeficientes
    p = poly.Polynomial(message)

    # Evalua el polynomio en los puntos establecidos en el protocolo
    codeword = p(points)

    return codeword

message = [2, 3, -5, 1]
codeword = encode(message, points)
print(f"Mensaje: {message} -> Codigo: {codeword}")

def decode(codeword, points, k):
    n = len(codeword)
    # Empareja el mensaje codificado con los puntos preestablecidos en el protocol
    pairs = list(zip(points, codeword))
    # Toma k de n puntos para obtener los coefficientes del polynomio de grado k-1
    indexes = random.sample(range(0, n), k)
    x = [pairs[i][0] for i in indexes]
    b = [pairs[i][1] for i in indexes]

    # Igual que en 01_polynomials.py, resuelve Ac=b para obtener c, los coeficientes de p(x) que se uso para el mensaje.
    A = build_vandermonde(x)
    message = np.linalg.solve(A, b)
    return message

message = decode(codeword, points, k)
print(f"Mensaje recibido: {message}")

# Corrigiendo errores:
# Para corregir hasta s errores, necesitamos al menos 2s simbolos de redundancia (extra).

# TODO: A Simple Error Correcting Reed Solomon Decoder