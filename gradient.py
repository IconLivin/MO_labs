import numpy
from sympy import cos, symbols, diff, sin
from sympy.plotting import plot3d
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt

"""Если ты читаешь это, то ты ВОР! Не стыдно? Зачем воруешь? Лаааадно. на первый раз прощаю"""


x1, x2 = symbols("x1 x2")

"""Уравнение с листочка"""
f = 20 * (cos(3 * x1) - x2) ** 2 + (x2 - 4 * x1) ** 2

"""Ограничения g1,g2,g3"""
g1 = x2 + x1 - 0.1
g2 = x2 - x1 - 0.25
g3 = -1 * (x1 - 1) ** 2 - 2 * (x2 - 1) ** 2 + 5


"""Ограничения по x,y """
dx = (-1.5, 2.0)
dy = (-2.0, 2.0)


"""Эта часть отделяет допустимую область (та что не в сетке)"""
x1_, x2_ = np.meshgrid(np.linspace(-1.5, 2.0, 500), np.linspace(-2.0, 2.0, 500))

"""Уравнение с листочка"""
f_ = 20 * (np.cos(3 * x1_) - x2_) ** 2 + (x2_ - 4 * x1_) ** 2

"""Ограничения g1,g2,g3"""
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


"""Первое приближение, можно понять по картинке куда сходятся линии"""
initial_guess = numpy.array([1.1, -1.2])

plt.scatter(initial_guess[0], initial_guess[1], color="r")

"""Уравнения огрниченний где параметр вектор"""


def con_g1(x):
    return x[0] + x[1] - 0.1


def con_g2(x):
    return x[1] - x[0] - 0.25


def con_g3(x):
    return -1 * (x[0] - 1) ** 2 - 2 * (x[1] - 1) ** 2 + 5


"""Функция с теми же критериями"""


def func(x):
    return 20 * (cos(3 * x[0]) - x[1]) ** 2 + (x[1] - 4 * x[0]) ** 2


"""Активные ограничения(зависит от начального приближения, можно выбрать любой, если поставить два, то он будет стремится найти решение на пересечении)"""
cons = [{"type": "eq", "fun": con_g3}]

"""Поиск минимума"""
res = opt.minimize(func, initial_guess, constraints=cons)

print(res.x)

"""Вывод минимума на график линии уровня"""
plt.scatter(res.x[0], res.x[1])

"""Получение решения по оси z"""
z = func(res.x)

print("Minimum: ", res.x[0], res.x[1], z)

"""Вычисления градиента функции по x,y"""
df_dx = [diff(f, x1), diff(f, x2)]
print(df_dx)

"""Вычисления градиента ограничений по x,y"""
dg2_dx = [diff(g2, x1), diff(g2, x2)]
print(dg2_dx)

dg3_dx = [diff(g3, x1), diff(g3, x2)]
print(dg3_dx)

"""Эта формула напрямую следует из уравнения KKT, -grad F(x) = lambda * grad g3(x); Отсюда выражаем лямбда"""


def res1(x1, x2):
    return (
        -1 * (32 * x1 - 8 * x2 - 120 * (-x2 + cos(3 * x1)) * sin(3 * x1)) / (2 - 2 * x1)
    )


"""Тоже что и выше только по y"""


def res2(x1, x2):
    return -1 * (-8 * x1 + 42 * x2 - 40 * cos(3 * x1)) / (4 - 4 * x2)


"""Вывод лямбд (не должны быть <0, если меньше - меняйте активные ограничения)"""
print(res1(*res.x), res2(*res.x))


"""Вывод графиков"""
plot3d(f, (x1, dx[0], dx[1]), (x2, dy[0], dy[1]))

plt.show()