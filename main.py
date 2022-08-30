import numpy as np
import matplotlib.pyplot as plt
import math
from gauss_newton import gauss_newton
from preprocessing import get_data


# P - the vector of parameters to optimize over
def f(P, x):
    a, b, c = P
    return a * math.exp(b * (x + a)) + c


# e = y - f(P, x)
def de_da(P, x):
    a, b, c = P
    return - math.exp(b * (x + a)) * (a * b + 1)


def de_db(P, x):
    a, b, c = P
    return - a * a * math.exp(b * (x + a))


def de_dc(P, x):
    return -1.


def main():
    X, Y = get_data('data_gn_challenge.csv', 8, 12)

    P_init = np.array([[-0.1], [-20.], [1.]])
    de_dp = [de_da, de_db, de_dc]
    P_acc = 1e-14
    P_ans = gauss_newton(X, Y, f, de_dp, P_init, P_acc)
    print('parameters: ', P_ans)

    Y_ans = [f(P_ans, x) for x in X]

    y_mean = np.mean(Y)
    SS_res = 0
    SS_tot = 0
    for i in range(len(Y)):
        SS_res += (Y[i] - Y_ans[i]) ** 2
        SS_tot += (Y[i] - y_mean) ** 2
    R2 = 1 - SS_res / SS_tot

    print('R2: ', R2)

    plt.scatter(X, Y)
    plt.scatter(X, Y_ans)
    plt.show()


if __name__ == '__main__':
    main()