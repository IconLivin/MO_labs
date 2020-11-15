import numpy
from sympy import *
from sympy.plotting import plot3d
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x1, x2 = symbols("x1 x2")
f = 20 * (cos(3 * x1) - x2) ** 2 + (x2 - 4 * x1) ** 2
g1 = x2 + x1 - 0.1
g2 = x2 - x1 - 0.25
g3 = -1 * (x1 - 1) ** 2 - 2 * (x2 - 1) ** 2 + 5

dx = (-1.5, 2.0)
dy = (-2.0, 2.0)

bound = 2.0
x1_, x2_ = np.meshgrid(np.linspace(-1.5, 2.0, 500), np.linspace(-2.0, 2.0, 500))
f_ = 20 * (np.cos(3 * x1_) - x2_) ** 2 + (x2_ - 4 * x1_) ** 2
g1_ = x2_ + x1_ - 0.1
g2_ = x2_ - x1_ - 0.25
g3_ = -1 * (x1_ - 1) ** 2 - 2 * (x2_ - 1) ** 2 + 5

plt.figure()
plt.pcolormesh(x1_, x2_, (g1_ <= 0) & (g2_ <= 0) & (g3_ <= 0), alpha=0.1)
contours = plt.contour(x1_, x2_, f_, 25)
plt.clabel(contours, inline=True, fontsize=8)
contours = plt.contour(x1_, x2_, g1_, (0,), colors="r")
contours = plt.contour(x1_, x2_, g2_, (0,), colors="y")
contours = plt.contour(x1_, x2_, g3_, (0,), colors="g")

initial_guess = numpy.array([1.1, -1.2])


def con_g1(x):
    return x[0] + x[1] - 0.1


def con_g2(x):
    return x[1] - x[0] - 0.25


def con_g3(x):
    return -1 * (x[0] - 1) ** 2 - 2 * (x[1] - 1) ** 2 + 5


def func(x):
    return 20 * (cos(3 * x[0]) - x[1]) ** 2 + (x[1] - 4 * x[0]) ** 2


cons = [{"type": "eq", "fun": con_g3}]


res = opt.minimize(func, initial_guess, constraints=cons)

print(res.x)

plt.scatter(res.x[0], res.x[1])

z = func(res.x)

print("Minimum: ", res.x[0], res.x[1], z)

df_dx = [diff(f, x1), diff(f, x2)]
print(df_dx)

dg2_dx = [diff(g2, x1), diff(g2, x2)]
print(dg2_dx)

dg3_dx = [diff(g3, x1), diff(g3, x2)]
print(dg3_dx)


def res1(x1, x2):
    return -1 * (32 * x1 - 8 * x2 - 120 * (-x2 + cos(3 * x1)) * sin(3 * x1)) / (2 - 2 * x1)


def res2(x1, x2):
    return -1 * (-8 * x1 + 42 * x2 - 40 * cos(3 * x1)) / (4 - 4 * x2)


print(res1(*res.x), res2(*res.x))

plot3d(f, (x1, dx[0], dx[1]), (x2, dy[0], dy[1]))

plt.show()