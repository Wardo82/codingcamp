from field import Field

class GFpm(Field):
    def __init__(self, p, mod_poly):
        self.p = p
        self.mod = mod_poly
        self.m = len(mod_poly) - 1

    def zero(self):
        return (0,) * self.m

    def one(self):
        return (1,) + (0,) * (self.m - 1)

    def add(self, a, b):
        return tuple((a[i] + b[i]) % self.p for i in range(self.m))

    def sub(self, a, b):
        return tuple((a[i] - b[i]) % self.p for i in range(self.m))

    def mul(self, a, b):
        # polynomial multiplication
        tmp = [0] * (2 * self.m - 1)
        for i in range(self.m):
            for j in range(self.m):
                tmp[i + j] = (tmp[i + j] + a[i] * b[j]) % self.p

        # reduction modulo mod_poly
        for i in range(len(tmp) - 1, self.m - 1, -1):
            if tmp[i] != 0:
                for j in range(self.m):
                    tmp[i - self.m + j] = (
                        tmp[i - self.m + j] - tmp[i] * self.mod[j]
                    ) % self.p

        return tuple(tmp[:self.m])

    def inv(self, a):
        # exponentiation: a^(p^m - 2)
        return self.pow(a, self.p**self.m - 2)

    def pow(self, a, n):
        result = self.one()
        base = a
        while n:
            if n & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            n >>= 1
        return result
