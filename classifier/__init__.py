import os

if __name__ == '__main__':
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname('validate.txt')))
    f = open(__location__ + '/validate.txt')

    i = 0
    for line in f:
        i += 1

    print(i)
    print('Hello world')
