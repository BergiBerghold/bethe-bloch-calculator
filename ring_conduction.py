import numpy as np
from numpy import pi, log
import matplotlib.pyplot as plt
import scipy.integrate as integrate


r_2 = 0.014
lam = 167
t = 25e-6

A = 8.2e6 * 0.01432
sigma = 2.75e-3


def gaussian(x):
    return A * np.exp( -x**2 / (2 * sigma**2) )


def P(r_1, r_2):
    res, abs_err = integrate.quad(lambda x: x * gaussian(x), r_1, r_2)

    if abs_err != 0:
        if abs(res / abs_err) < 1e6:
            print(res, abs_err)
            print('Large Integration Error')

    return 2 * np.pi * res


def delta_T(r, step):
    return P(r-step, r) * log(r_2/r) / (2*pi*lam*t) / step


def total_T(r):
    res, abs_err = integrate.quad(lambda x: delta_T(x, 1e-7), r, r_2)

    if abs_err != 0:
        if abs(res / abs_err) < 1e6:
            print(res, abs_err)
            print('Large Integration Error')

    return res


X = np.linspace(1e-6, 0.014, 1000)
Y = np.array([total_T(x) for x in X])

plt.xlabel("Radius [mm]")
plt.ylabel('Temperature difference ΔT [K]')

plt.plot(X * 1000, Y)
plt.show()


# X = np.linspace(1e-6, 0.014, 1000)
# Y = np.array([delta_T(x, 1e-7) * (X[1]-X[0]) for x in X])
#
# # plt.plot(X, Y)
# # plt.show()
#
# Y2 = []
# for x in X[::-1]:
#     value = np.sum(Y[X > x])
#     Y2.append(value)
#
# Y2 = Y2[::-1]
#
# plt.plot(X, Y2)
# plt.show()