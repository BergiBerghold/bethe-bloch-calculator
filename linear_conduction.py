import numpy as np
from numpy import pi, log
import matplotlib.pyplot as plt
import scipy.integrate as integrate


X_tot = 116e-6
lam = 18.6

P = 8.2e6 * 0.068 / X_tot


def delta_T(x):
    return P * (X_tot-x) / (lam)


def total_T(r):
    res, abs_err = integrate.quad(lambda x: delta_T(x), r, X_tot)

    if abs_err != 0:
        if abs(res / abs_err) < 1e6:
            print(res, abs_err)
            print('Large Integration Error')

    return res


X = np.linspace(0, X_tot, 1000)
Y = np.array([total_T(x) for x in X])

plt.xlabel("x [mm]")
plt.ylabel('Temperature [K]')

plt.plot(X * 1000, Y)
# plt.show()

print(max(Y), max(Y) * 2)
print(8.2e6 * 0.068 * X_tot / lam)

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