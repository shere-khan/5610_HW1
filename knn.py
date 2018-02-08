import math, os, copy, numpy, operator


class VecData:
    def __init__(self, vec, label):
        self.vec = vec
        self.label = label
        self.dist = None

    def __repr__(self):
        return 'dist: {1}  label: {2}'.format(self.vec, self.dist, self.label)


class KNN:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0

        # cm: the confusion matrix, a global field
        self.cm = list()
        for i in range(9):
            self.cm.append([])
            for j in range(9):
                self.cm[i].append(0)

    @staticmethod
    def read_data(f):
        a = list()
        for line in f:
            vec = line.split()
            vec = list(map(lambda x: int(x), vec))
            label = int(vec.pop(-1))
            a.append(VecData(numpy.asarray(vec), label))

        return a

    @staticmethod
    def create_data(m, n):
        M = [VecData([5.5, 0.5, 4.5], 2),
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
        return numpy.linalg.norm(x.vec - y.vec)

    @staticmethod
    def square_sum(x, y):
        xy = zip(x, y)
        xy = list(map(lambda r: math.pow(r[0] - r[1], 2), xy))
        tot = sum(xy)

        return tot

    def get_distances(self, data_t, data_v):
        dists_s = list()
        for d_v in data_v:
            dists = list()
            for d_t in data_t:
                d_t.dist = KNN.dist(d_v, d_t)
                dists.append(d_t)
            dists_s.append(dists)

        return dists_s

    def train(self, dists_s, data_v, k, out):
        for k in range(1, k):
            print('k: ' + str(k))
            for dists_val in zip(dists_s, data_v):
                class_a = KNN.prediction(dists_val[0], k)
                self.compare(class_a, dists_val[1].label)

            acc = self.accuracy()
            out.write('acc: ' + str(acc) + '\n')
            out.flush()
            print('acc: ' + str(acc))

            self.incorrect = 0
            self.correct = 0

    @staticmethod
    def prediction(l, k):
        l.sort(key=lambda x: x.dist)
        class_a = KNN.calc_maj(l[:k])

        return class_a

    @staticmethod
    def calc_maj(ks_a):
        maj = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
               7: 0, 8: 0, 9: 0}
        for i in ks_a:
            maj[i.label] += 1

        return numpy.argmax(list(maj.values()))

    def compare(self, pred, actual):
        # Updates the confusion matrix after comparing values
        if pred == actual:
            self.correct += 1
            self.cm[pred - 1][actual - 1] += 1
        else:
            self.incorrect += 1
            self.cm[pred - 1][pred - 1] += 1

    def precision_macro(self):
        pass

    def recall_macro(self):
        tp_a = None
        fn_a = None
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
        return 0

    def accuracy(self):
        return self.correct / (self.incorrect + self.correct)


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
        print()
        ks_a = dists[:3]
        class_a = KNN.calc_maj(ks_a)
        print('test label {0}'.format(class_a))
        print()


def get_file_location(fn):
    return os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(fn))) + '/' + fn


if __name__ == '__main__':
    # problem2()
    l = ['validate.txt', 'train.txt']
    # l = ['validate_cp.txt', 'train_cp.txt']
    data = list()
    for fn in l:
        f = open(get_file_location(fn))
        data.append(KNN.read_data(f))
        f.close()

    data_v = data[0]
    data_t = data[1]

    # Successively train against the training data and increase values of k
    scores = list()

    # train
    knn = KNN()
    dists_s = knn.get_distances(data_t, data_v)
    print('distances collected')
    out = open('out2.txt', 'w')

    for dists1 in dists_s:
        for dists2 in dists_s:
            for (d1, d2) in zip(dists1, dists2):
                if d1.dist != d2.dist:
                    print('success')

    # knn.train(dists_s, data_v, 20, out)

    out.close()
