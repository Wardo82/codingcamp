#import numpy as np
import copy
import math
from typing import List

def dot_product(v1: List, v2: List):
    """
    Computes the dot product between two vectors. They must be of the same length.
    """
    n = len(v1)
    prod = []
    for i in range(n):
        prod.append(v1[i]*v2[i])

    return sum(prod)

def length(vector):
    """ Computes the euclidian norm of vector """
    return math.sqrt()

class Matrix:

    def __init__(self, num_rows = 0, num_cols = 0):
        self.values = []
        self.num_rows = num_rows
        self.num_cols = num_cols
        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                row.append(0.0)
            self.values.append(row)

    def is_square(self) -> bool:
        return (self.num_rows == self.num_cols)

    def set_diagonal(self, value=1):
        assert self.is_square(), "set_diagonal only available to square matrix"
        for i in range(self.num_cols):
            self.values[i][i] = value

    def transpose(self):
        res = Matrix(self.num_rows, self.num_cols)
        for i in range(self.num_cols):
            new_col = self.values[i]    # Former rows are new columns
            for j in range(self.num_rows):
                res.values[j][i] = new_col[j]
        return res

    def det(self):
        """ Computes the determinant of the matrix if it is square.

        TODO: This is my naive approach. Try the laplace expansion or guassian elminiation.
        """
        assert self.num_rows == self.num_cols, "Determinant only available to square matrix"

        # Compute positive parts
        positive_part = 0
        for i in range(self.num_rows):
            i = (i%self.num_rows)
            for j in range(self.num_cols):
                j = (j%self.num_cols)

        # Compute negative parts
        negative_part = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):

        return positive_part - negative_part

    def __matmul__(self, M):
        """ Implements Matrix multiplication operator @ """
        res = Matrix(self.num_rows, M.num_cols)
        for i in range(self.num_rows):
            for j in range(M.num_cols):
                left_vector = self.values[i]
                right_vector = [row[j] for row in M.values]
                res.values[i][j] = dot_product(left_vector, right_vector)
        return res

    def __mul__(self, M):
        """ Implement elementwise multiplication operator * of two matrices """
        res = Matrix(self.num_rows, self.num_cols)
        if isinstance(M, Matrix):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    res.values[i][j] = self.values[i][j] * M.values[i][j]
        elif isinstance(M, int):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    res.values[i][j] = self.values[i][j] * M
        return res

    def __add__(self, M):
        """ Implements elementwise addition operator + of two matrices """
        res = Matrix(self.num_rows, self.num_cols)
        if isinstance(M, Matrix):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    res.values[i][j] = self.values[i][j] + M.values[i][j]
        elif isinstance(M, int):
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    res.values[i][j] = self.values[i][j] + M
        return res

    def __eq__(self, M) -> bool:
        equal = True
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                equal &= (self.values[i][j] == M.values[i][j])
                if not equal:
                    return equal
        return equal

    def __ne__(self, M) -> bool:
        return not (self == M)

    def __str__(self) -> str:
        str_rep = ""
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                str_rep = f"{str_rep}{self.values[i][j]}\t"
            str_rep = f"{str_rep}\n"
        return str_rep

def from_2d_array(two_d_array: List[List]):
    """
    Creates a Matrix from a 2D array.
    The input must be a list of m lists of n elements.
    m is the number of rows and n the number of columns.
    """
    assert type(two_d_array) == list and type(two_d_array[0]) == list
    num_rows = len(two_d_array)
    num_cols = max([len(col) for col in two_d_array])
    print(f"Input matrix shape is: ({num_rows}, {num_cols})")
    A = Matrix(num_rows, num_cols)
    A.values = copy.deepcopy(two_d_array)
    return A

def eye(n: int) -> Matrix:
    """ Returns an identity matrix of size n x n """
    res = Matrix(n, n)
    res.set_diagonal()
    return res

if __name__ == '__main__':

    Zero = Matrix(4, 4);    # Zero matrix by default
    One  = Zero + 1; print(One)
    A = from_2d_array(two_d_array=[[2, -1, -2], [-4, 6, 3], [-4, -2, 8]])
    B = from_2d_array(two_d_array=[[2, 1, -2], [4, 6, -3], [4, 2, 8]])

    print(A)
    print((A@A))

    I = eye(3)
    test_equality = (A == A) & (A.transpose().transpose() == A) & (B != A)
    print(f"Test equality operator: {test_equality}")
    test_matmul = (A@I == A)
    print(f"Test matmul operator: {test_matmul}")
    test_identity = (I@I == I) & ((I+I) == (I*2))
    print(f"Test identity operator: {test_identity}")

    print(dot_product(
        [1, 2, 3],
        [4, 5, 6]
        ))
