from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import sympy
import numpy
import pylab
import matplotlib.pyplot as plt
from scipy import optimize
from sympy.vector import CoordSys3D, gradient

sympy.init_printing()


def makeData():
    x, y = numpy.mgrid[-1.5:2:25j, -2:2:25j]

    z = func(x, y)
    return x, y, z


def makeLims():
    x, y = numpy.mgrid[-1.5:2:25j, -2:2:25j]

    z = [lim1(x, y), lim2(x, y), lim3(x, y)]
    return z


def func(x1, x2):
    return 20 * (numpy.cos(3 * x1) - x2) ** 2 + (x2 - 4 * x1) ** 2


def grad(x, y):
    return 32 * x - 8 * y - 120 * (-y + numpy.cos(3 * x)) * numpy.sin(3 * x), (
        -8 * x + 42 * y - 40 * numpy.cos(3 * x)
    )


def lim1(x, y):
    return x + y - 0.1


def lim2(x, y):
    return -1 * (x - 1) ** 2 - 2 * (y - 1) ** 2 + 5


def lim3(x, y):
    return y - x - 0.25


def func1(x):
    return func(*x)


def cons_f(x):
    return [-1 * (x[0] - 1) ** 2 - 2 * (x[1] - 1) ** 2]


def cons_J(x):
    return [-2 * (x[0] - 1), -4 * (x[1] - 1)]


def make_surface():
    x = numpy.arange(-1.5, 2, 0.01)
    y = numpy.arange(-2, 2, 0.01)

    resx = []
    resy = []
    for i in x:
        for j in y:
            l1 = lim1(i, j)
            l2 = lim2(i, j)
            l3 = lim3(i, j)
            if (
                l1 <= 0
                and l2 <= 0
                and l3 <= 0
                and abs(max(l1, l2)) - abs(min(l1, l2)) <= numpy.finfo(float).eps
                and abs(max(l2, l3)) - abs(min(l2, l3)) <= numpy.finfo(float).eps
                and abs(max(l1, l3)) - abs(min(l1, l3)) <= numpy.finfo(float).eps
            ):
                if i in resx:
                    index = resx.index(i)
                    if resy[index] < j:
                        resy[index] = j
                else:
                    resx.append(i)
                    resy.append(j)
    return resx, resy


R = CoordSys3D("")

gradX, gradY = sympy.symbols(("x1 x2"))

s1 = 20 * (sympy.cos(3 * R.x) - R.y) ** 2 + (R.y - 4 * R.x) ** 2

bounds = optimize.Bounds([-1.5, -2], [2, 2])

linear_constraint = optimize.LinearConstraint(
    [[1, 1], [-1, 1]], [-numpy.inf, -numpy.inf], [0.1, 0.25]
)

nonlinear_constraint = optimize.NonlinearConstraint(
    cons_f, -numpy.inf, -5, jac=cons_J, hess="2-point"
)

_gradient = str(gradient(s1)).replace(".", "")

_gradient = _gradient.replace("**", "^")

_gradient = _gradient.replace("*", " * ")

_gradient = _gradient.replace("/", " / ")

print(_gradient)

x, y, z = makeData()

lims = makeLims()

begin = (0, 1.5)

res = optimize.minimize(
    func1,
    x0=begin,
    method="trust-constr",
    jac=optimize.rosen_der,
    hess=optimize.rosen_hess,
    constraints=[linear_constraint, nonlinear_constraint],
    options={"verbose": 1},
    bounds=bounds,
)

print("Minimum=[", res.x[0], res.x[1], func(res.x[0], res.x[1]), "]")
f, ax = plt.subplots()

gradx, grady = grad(*begin)

surfx, surfy = make_surface()

lx = numpy.arange(-1.5, 2, 0.1)

ly21 = 1 - numpy.sqrt((5 - (lx - 1) ** 2) / 2)
ly22 = 1 + numpy.sqrt((5 - (lx - 1) ** 2) / 2)
ly3 = lx + 0.25
ly1 = 0.1 - lx

ly4 = numpy.minimum(numpy.abs(ly1), numpy.abs(ly3))

cs = ax.contour(x, y, z, levels=10)
ax.clabel(cs)

ax.plot(lx, ly1, lx, ly3, linestyle="--")

ax.plot(lx, ly21, lx, ly22, linestyle="--", color="g")

ax.plot(surfx, surfy, color="black")

ax.scatter(res.x[0], res.x[1], func(*res.x))
# ax.arrow(begin[0], begin[1], -gradx, -grady, head_length = 10)

fig = pylab.figure()

axes = Axes3D(fig)

axes.scatter3D(begin[0], begin[1], func(begin[0], begin[1]), s=50, color="blue")
axes.scatter3D(res.x[0], res.x[1], func(*res.x), s=100, color="red")

axes.plot_surface(x, y, z, color="green", alpha=0.7)

"""
for i in range(3):
    axes.plot_surface(x, y, lims[i], color="yellow", alpha=0.3)
"""

pylab.show()
plt.show()
