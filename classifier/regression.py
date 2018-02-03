import math


class VecData:
    def __init__(self, vec, label):
        self.vec = vec
        self.label = label
        self.dist = None

    def __repr__(self):
        return 'vector: {0}  dist: {1}  label: {2}'.format(self.vec, self.dist, self.label)


def create_data(m, n):
    M = [VecData(vec=[5.5, 0.5, 4.5], label=2),
         VecData([7.4, 1.1, 3.6], 0),
         VecData([5.9, 0.2, 3.4], 2),
         VecData([9.9, 0.1, 0.8], 0),
         VecData([6.9, -0.1, 0.6], 2),
         VecData([6.8, -0.3, 5.1], 2),
         VecData([4.1, 0.3, 5.1], 1),
         VecData([1.3, -0.2, 1.8], 1),
         VecData([4.5, 0.4, 2.0], 0),
         VecData([0.5, 0.0, 2.3], 1),
         VecData([5.9, -0.1, 4.4], 0),
         VecData([9.3, -0.2, 3.2], 0),
         VecData([1.0, 0.1, 2.8], 1),
         VecData([0.4, 0.1, 4.3], 1),
         VecData([2.7, -0.5, 4.2], 1)]

    return M


def dist(x, y):
    xy = math.sqrt(square_sum(x.vec, y.vec))
    return xy


def square_sum(x, y):
    xy = zip(x, y)
    xy = list(map(lambda r: math.pow(r[0] - r[1], 2), xy))
    tot = sum(xy)

    return tot


def calc_maj(a):
    maj = [0, 0, 0]
    for i in ks_a:
        if i.label == 2:
            maj[0] += 1
        elif i.label == 1:
            maj[1] += 1
        elif i.label == 0:
            maj[2] += 1

    return max(maj)


def read_file(input):
    f = open(input)
    for line in f:
        vec = line.split()


# if __name__ == '__main__':
#     M = create_data(3, 4)
#
#     d_a = VecData([4.1, -0.1, 2.2], None)
#     d_b = VecData([6.1, 0.4, 1.3], None)
#     D = [d_a, d_b]
#
#     for d in D:
#         dists = list()
#         for m in M:
#             m.dist = dist(d, m)
#             dists.append(m)
#
#         dists.sort(key=lambda x: x.dist)
#         for di in dists:
#             print(di)
#         print()
#         ks_a = dists[:3]
#         class_a = calc_maj(ks_a)
#         print('test label {0}'.format(class_a))
#         print()
