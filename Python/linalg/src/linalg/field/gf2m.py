from field import Field

class GF2m(Field):
    def __init__(self, m, mod_poly):
        self.m = m
        self.mod = mod_poly  # integer bitmask

    def zero(self):
        return 0

    def one(self):
        return 1

    def add(self, a, b):
        return a ^ b

    def sub(self, a, b):
        return a ^ b

    def mul(self, a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & (1 << self.m):
                a ^= self.mod
        return result

    def inv(self, a):
        return self.pow(a, (1 << self.m) - 2)

    def pow(self, a, n):
        result = 1
        base = a
        while n:
            if n & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            n >>= 1
        return result
