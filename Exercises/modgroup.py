from math import gcd


class ModGroup:
    def __init__(self, x: int, p: int):
        if p <= 0:
            raise ValueError("p must ne positive")
        self.p = p
        self.x = x % p

    def _check_same_p(self, other):
        if not isinstance(other, ModGroup):
            return NotImplemented
        if self.p != other.p:
            raise ValueError(f"Different moduli: {self.p} vs {other.p}")

    def __repr__(self):
        return f"ModGroup({self.x}, {self.p})"

    # (a) multiplication
    def __mul__(self, other):
        self._check_same_p(other)
        return ModGroup(self.x * other.x, self.p)

    # (b) equality = congruence
    def __eq__(self, other):
        if not isinstance(other, ModGroup):
            return False
        if self.p != other.p:
            return False
        return (self.x % self.p) == (other.x % other.p)

    # (c) add/sub
    def __add__(self, other):
        self._check_same_p(other)
        return ModGroup(self.x + other.x, self.p)

    def __sub__(self, other):
        self._check_same_p(other)
        return ModGroup(self.x - other.x, self.p)

    # inverse + division
    def inv(self):
        if gcd(self.x, self.p) != 1:
            raise ValueError(
                f"No multiplicative inverse for {self.x} mod {self.p}")
        return ModGroup(pow(self.x, -1, self.p), self.p)

    def __truediv__(self, other):
        self._check_same_p(other)
        return self * other.inv()
