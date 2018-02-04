import math, os, copy


class VecData:
    def __init__(self, vec, label):
        self.vec = vec
        self.label = label
        self.dist = None

    def __repr__(self):
        return 'vector: {0}  dist: {1}  label: {2}'.format(self.vec, self.dist, self.label)


class KNN:
    def __init__(self):
        self.cm = list()
        for i in range(9):
            self.cm.append([])

    @staticmethod
    def read_data(f):
        a = list()
        label_set = set()
        for line in f:
            vec = line.split()
            vec = list(map(lambda x: int(x), vec))
            label = int(vec.pop(-1))
            label_set.add(label)
            a.append(VecData(vec, label))

        return a

    @staticmethod
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

    @staticmethod
    def dist(x, y):
        xy = math.sqrt(KNN.square_sum(x.vec, y.vec))
        return xy

    @staticmethod
    def square_sum(x, y):
        xy = zip(x, y)
        xy = list(map(lambda r: math.pow(r[0] - r[1], 2), xy))
        tot = sum(xy)

        return tot

    @staticmethod
    def calc_maj(ks_a):
        maj = [0, 0, 0]
        for i in ks_a:
            if i.label == 2:
                maj[0] += 1
            elif i.label == 1:
                maj[1] += 1
            elif i.label == 0:
                maj[2] += 1

        return max(maj)

    @staticmethod
    def read_file(input):
        f = open(input)
        for line in f:
            vec = line.split()

    def train(self, data_t, data_v_copy, data_v, k):
        for d_v in zip(data_v_copy, data_v):
            dists = list()
            for d_t in data_t:
                # Calculate the distance to each data point in the
                # training set and add that represented value to a list
                d_t.dist = KNN.dist(d_v[0], d_t)
                dists.append(d_t)

            # Get the predicted value from the list computed above
            if k == 1:
                class_a = max(dists, key=lambda x: x.label).label
            else:
                dists.sort(key=lambda x: x.dist)
                class_a = KNN.calc_maj(dists[:k])

            # Compare the predicted value (class_a) to the actual
            # value(d_v[1]) from the original validation data
            self.compare(class_a, d_v[1])

    def compare(self, pred, actual):
        if pred == actual:
            self.cm[pred][actual] += 1
        else:
            self.cm[pred][pred] += 1

    def precision_macro(self):
        pass

    def recall_macro(self):
        pass

    def recall_micro(self):
        # Compute TPs and FNs across all labels
        tps = None
        fns = None
        for i in range(len(self.cm)):
            row = self.cm[i]
            for j in range(len(row)):
                if i == j:
                    tps += self.cm[i][j]
                else:
                    fns += self.cm[i][j]

        return tps / (tps + fns)

    def f_measure_macro(self):
        pass


def problem2():
    M = KNN.create_data(3, 4)

    d_a = VecData([4.1, -0.1, 2.2], None)
    d_b = VecData([6.1, 0.4, 1.3], None)
    D = [d_a, d_b]

    for d in D:
        dists = list()
        for m in M:
            m.dist = KNN.dist(d, m)
            dists.append(m)

        dists.sort(key=lambda x: x.dist)
        # for di in dists:
        #     print(di)
        print()
        ks_a = dists[:3]
        class_a = KNN.calc_maj(ks_a)
        print('test label {0}'.format(class_a))
        print()


if __name__ == '__main__':
    # problem2()
    l = ['validate.txt', 'train.txt']
    data = list()
    for fn in l:
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(fn))) + '/' + fn
        f = open(__location__)

        data.append(KNN.read_data(f))

    data_v = data[0]
    data_t = data[1]

    data_v_copy = copy.deepcopy(data_v)

    # Train: calculate distances, assign labels,
    # and compare result with original
    knn = KNN()
    knn.train(data_t, data_v_copy, data_v, 1)

    # print('incorrect: {0}'.format(num_off))
