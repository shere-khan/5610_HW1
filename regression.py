import math


class Vector:
    def __init__(self, vec, cls):
        self.vec = vec
        self.cls = cls
        self.dist = None


def create_data(m, n):
    M = [Vector([5.5, 0.5, 4.5], 2),
         Vector([7.4, 1.1, 3.6], 0),
         Vector([5.9, 0.2, 3.4], 2),
         Vector([9.9, 0.1, 0.8], 0),
         Vector([6.9, -0.1, 0.6], 2),
         Vector([6.8, -0.3, 5.1], 2),
         Vector([4.1, 0.3, 5.1], 2),
         Vector([1.3, -0.2, 1.8], 1),
         Vector([4.5, 0.4, 2.0], 0),
         Vector([0.5, 0.0, 2.3], 1),
         Vector([5.9, -0.1, 4.4], 0),
         Vector([9.3, -0.2, 3.2], 0),
         Vector([1.0, 0.1, 2.8], 1),
         Vector([0.4, 0.1, 4.3], 1),
         Vector([2.7, -0.5, 4.2], 1)]

    return M


def dist(x, y):
    xy = math.sqrt(square_sum(x.vec, y.vec))
    return xy


def square_sum(x, y):
    xy = zip(x, y)
    xy = list(map(lambda r: math.pow(r[0] - r[1], 2), xy))
    tot = sum(xy)

    return tot


if __name__ == '__main__':
    M = create_data(3, 4)

    d_a = Vector([4.1, -0.1, 2.2], None)
    d_b = Vector([6.1, 0.4, 1.3], None)

    dists_a = list()
    dists_b = list()

    for m in M:
        m.dist = dist(d_a, m)
        dists_a.append(m)

    dists_a.sort(key=lambda x: x.dist)
    # dists_b.sort()

    ks_a = dists_a[:4]
    # ks_b = dists_b[:4]

    maj = [0, 0, 0]
    for i in ks_a:
        if i.cls == 2:
            maj[0] += 1
        elif i.cls == 1:
            maj[1] += 1
        elif i.cls == 0:
            maj[2] += 1

    classifier = max(maj)
