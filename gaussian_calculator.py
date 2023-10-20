import numpy as np
from numpy import pi, log
import matplotlib.pyplot as plt
import scipy.integrate as integrate

sigma = 2.75e-3
A = 5.6 / 2 * pi * sigma**2

def gaussian(x):
    return A * np.exp( -x**2 / (2 * sigma**2) )


def I(r_1, r_2):
    res, abs_err = integrate.quad(lambda x: x * gaussian(x), r_1, r_2)

    if abs_err != 0:
        if abs(res / abs_err) < 1e6:
            print(res, abs_err)
            print('Large Integration Error')

    return 2 * np.pi * res

# print(I(0, 5e-3) / I_total)


X = np.linspace(-0.014, 0.014, 1000)
Y = gaussian(X) * 1e6

plt.plot(X, Y, color='orange')
plt.ylabel('Heat Flux [W/mm^2]')
plt.show()