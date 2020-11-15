from sympy import *
from sympy.plotting import plot3d
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x1, x2 = symbols('x1 x2')
f = 20*(cos(3*x1) - x2)**2 + (x2 - 4*x1)**2
g1 = x2 + x1 - 0.1
g2 = x2 - x1 - 0.25
g3 = -1*(x1 - 1)**2 - 2*(x2 - 1)**2 + 5

dx = (-1.5, 2.0)
dy = (-2.0, 2.0)
plot3d(f,(x1, dx[0], dx[1]), (x2, dy[0], dy[1]))

bound=2.0
x1_, x2_ = np.meshgrid(np.linspace(-1.5, 2.0, 500), np.linspace(-2.0, 2.0, 500))
f_  = 20*(np.cos(3*x1_) - x2_)**2 + (x2_ - 4*x1_)**2
g1_ = x2_ + x1_ - 0.1
g2_ = x2_ - x1_ - 0.25
g3_ = -1*(x1_ - 1)**2 - 2*(x2_ - 1)**2 + 5

plt.figure()
plt.pcolormesh(x1_, x2_, (g1_ <= 0) & (g2_ <= 0) & (g3_ <= 0), alpha=0.1)
contours = plt.contour(x1_, x2_, f_, 25)
plt.clabel(contours, inline=True, fontsize=8)
contours = plt.contour(x1_, x2_, g1_, (0,), colors='r')
contours = plt.contour(x1_, x2_, g2_, (0,), colors='y')
contours = plt.contour(x1_, x2_, g3_, (0,), colors='g')

plt.show()