import numpy as np


def gauss_newton(X, Y, f, de_dp, P_init, P_acc):
    assert len(X) == len(Y), "The length of X and Y should match"
    assert len(de_dp) == len(P_init), "The number of parameters and partial derivatives should match"

    J = np.zeros(shape=(len(X), len(de_dp)))
    E = np.zeros(shape=(len(X), 1))

    P_cur = P_init

    while True:
        for i in range(len(X)):
            for j in range(len(de_dp)):
                J[i][j] = de_dp[j](P_cur, X[i])
            E[i] = Y[i] - f(P_cur, X[i])

        pinvJ = np.linalg.pinv(J)
        step = np.matmul(pinvJ, E)
        P_next = np.subtract(P_cur, step)
        P_abs_diff = np.linalg.norm(np.subtract(P_cur, P_next))
        if P_abs_diff < P_acc:
            return P_next
        P_cur = P_next