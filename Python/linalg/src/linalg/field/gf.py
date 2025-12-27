from field import Field

class GFp(Field):
    """ Implementation of Galois Field GFp """

    def __init__(self, p):
        self.p = p

    def zero(self): return 0
    def one(self): return 1

    def add(self, a, b): return (a + b) % self.p
    def sub(self, a, b): return (a - b) % self.p
    def mul(self, a, b): return (a * b) % self.p

    def inv(self, a):
        return pow(a, self.p - 2, self.p)

    def pow(self, a, n):
        return pow(a, n, self.p)

if __name__ == "__main__":
    p: int = 7
    F: Field = GFp(p)

    # -----------------------
    # Basic identities
    # -----------------------
    for a in range(p):
        assert F.add(a, F.zero()) == a
        assert F.mul(a, F.one()) == a

    # -----------------------
    # Additive inverse
    # -----------------------
    for a in range(p):
        assert F.add(a, F.sub(0, a)) == 0

    # -----------------------
    # Multiplicative inverse
    # -----------------------
    for a in range(1, p):
        inv = F.inv(a)
        assert F.mul(a, inv) == 1

    # -----------------------
    # Commutativity
    # -----------------------
    for a in range(p):
        for b in range(p):
            assert F.add(a, b) == F.add(b, a)
            assert F.mul(a, b) == F.mul(b, a)

    # -----------------------
    # Associativity
    # -----------------------
    for a in range(p):
        for b in range(p):
            for c in range(p):
                assert F.add(F.add(a, b), c) == F.add(a, F.add(b, c))
                assert F.mul(F.mul(a, b), c) == F.mul(a, F.mul(b, c))

    # -----------------------
    # Distributivity
    # -----------------------
    for a in range(p):
        for b in range(p):
            for c in range(p):
                left = F.mul(a, F.add(b, c))
                right = F.add(F.mul(a, b), F.mul(a, c))
                assert left == right

    # -----------------------
    # Power laws
    # -----------------------
    for a in range(1, p):
        assert F.pow(a, 0) == 1
        assert F.pow(a, 1) == a
        assert F.pow(a, p - 1) == 1  # Fermat

    # -----------------------
    # Inverse via power
    # -----------------------
    for a in range(1, p):
        assert F.inv(a) == F.pow(a, p - 2)

    print("All GF(p) tests passed.")
