import os
from classifier.regression import VecData


def read_data(f):
    a = list()
    for line in f:
        vec = line.split()
        label = int(vec.pop(-1))
        a.append(VecData(vec, label))

    return a


if __name__ == '__main__':
    l = ['validate.txt', 'train.txt']
    for fn in l:
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(fn))) + '/' + fn
        f = open(__location__)

        data = read_data(f)

        print(len(data))

    data_v = l[0]
    data_t = l[1]

