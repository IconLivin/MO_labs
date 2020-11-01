from mpl_toolkits.mplot3d import Axes3D
import scipy as sc, sympy, numpy, pylab
from scipy import optimize
from sympy.vector import CoordSys3D, gradient

sympy.init_printing()

def makeData():
    x = numpy.arange(-10, 10, 0.1)
    y = numpy.arange(-10, 10, 0.1)
    xgrid, ygrid = numpy.meshgrid(x, y)

    zgrid = func(xgrid, ygrid)
    return xgrid,ygrid, zgrid

def func(x1, x2):
    return (numpy.sin(x1) * numpy.sin(x2)) / (x1 * x2)

def func1(x):
    return func(*x)

R = CoordSys3D('')

gradX, gradY = sympy.symbols(('x y'))

s1 = ((sympy.sin(R.x) * sympy.sin(R.y)) / (R.x * R.y))

print(gradient(s1))

x, y , z = makeData()

begin = (2,1)

res = optimize.minimize(func1, x0=begin)

fig = pylab.figure()
axes = Axes3D(fig)

axes.scatter3D(begin[0], begin[1], func(begin[0], begin[1]), s=100, color='blue')
axes.plot_surface(x, y, z, color='green', alpha=0.2)
axes.scatter3D(res.x[0],res.x[1], func(res.x[0],res.x[1]), s=100, color='red')
pylab.show()



