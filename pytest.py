from math import isclose

class Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        try:
            return isclose(other, self.expected, rel_tol=self.rel, abs_tol=self.abs)
        except TypeError:
            return False


def approx(expected, rel=1e-6, abs=1e-12):
    return Approx(expected, rel=rel, abs=abs)
