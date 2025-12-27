from field import Field

class Matrix:
    def __init__(self, values, field: Field):
        self.values = values
        self.field: Field = field
        self.num_rows = len(values)
        self.num_cols = len(values[0])

    def __matmul__(self, other):
        F = self.field
        result = [[F.zero() for _ in range(other.num_cols)] for _ in range(self.num_rows)]
        for i in range(self.num_rows):
            for j in range(other.num_cols):
                for k in range(self.num_cols):
                    result[i][j] = F.add(
                        result[i][j],
                        F.mul(self.values[i][k], other.values[k][j])
                    )
        return Matrix(result, F)

    def __str__(self) -> str:
        str_rep = ""
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                str_rep = f"{str_rep}{self.values[i][j]}\t"
            str_rep = f"{str_rep}\n"
        return str_rep

if __name__ == '__main__':
    from field import GFp, GFpm, GF2m
    Matrix(field=GFp)
    Matrix(field=GFpm)
    Matrix(field=GF2m)

    F = GF2m(8, 0x11d)
    A = Matrix([[1,2],[3,4]], F)
    B = Matrix([[5,6],[7,8]], F)
    C = A @ B

#TODOs: Next step
# Implement:
# - row-reduction over `Field`
# - nullspace computation

# verify dual RS codes
# compute syndromes
# implement decoding algorithms
