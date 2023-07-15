from copy import deepcopy
from functools import partial
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

# delta_t = 0.01
# init_delta = 1.05
# target_t = 10
# delta_phi = 2 * np.pi / N

# This is what the result should look like:
# delta = [[all the points of the circle at t=0], ..., [all the points of the cicle at t=i*delta_t], ...]

# We already know the value of delta[0]:
# for j in range(N+1): delta[0][j] = init_delta

# And hence we can calculate d(delta)/dt at t=i*delta_t for each angle:
# for j in range(N+1):
#   d_delta_dt = (1/3) * (Re/Fr) * ((3 * delta[j]**2 * (delta[(j+1) % N] - delta[(j-1) % N])) / (4*np.pi/N) * np.cos(j*delta_t + t) - delta[j]**3 * np.sin(j*delta_t + t))

# Having d(delta)/dt for t=0 with the magic of Runge-Kutta we can get the values of delta for t=delta_t,
# And then repeating the previous step we'll get values for t=delta_t*2
# Rince and repeat until the target value of target_t

# phi = np.arange(0, 2 * np.pi, N + 1)  # Разбиение круга на N лучей
# delta = np.full(shape=N, fill_value=init_delta, dtype=np.int)  # Начальная высота поверхности
# d_delta_dt = (1/3) * (Re/Fr) * ((3 * delta[j]**2 * (delta[(j+1) % N] - delta[(j-1) % N])) / (2*delta_phi) * np.cos(j*delta_t + t) - delta[j]**3 * np.sin(j*delta_t + t))

class Runge:
    def __init__(self, step, target, init):
        self.step = step
        self.target = target
        self.init = init

    def increment(self, f, values, t):
        k0 = self.step * f(values, t)
        k1 = self.step * f(values + k0 / 2, t + self.step / 2)
        k2 = self.step * f(values + k1 / 2, t + self.step / 2)
        k3 = self.step * f(values + k2, t + self.step)
        return (k0 + 2*k1 + 2*k2 + k3) / 6

    def runge_method(self, f, init_values):
      values = init_values
      for t in np.arange(self.init, self.target, self.step):
          values = values + self.increment(f, values, t)
      return np.array(values)

def f(boundary_conditions, values, t): # values = (eps, Z, R, M)
    N, Re, Fr, delta_phi = itemgetter('N', 'Re', 'Fr', 'delta_phi')(boundary_conditions)
    # print(values)
    f = np.zeros([N]) # уравнения системы, где f[i] = d(delta)/dt при phi = 2*np.pi/N
    for j in range(N):
        f[j] = (1/3) * (Re/Fr) * ((3 * values[j]**2 * (values[(j+1) % N] - values[(j-1) % N])) / (2*delta_phi) * np.cos(j*delta_phi + t) - values[j]**3 * np.sin(j*delta_phi + t))

    return f


delta_t = 0.01  # Шаг по времени для метода Рунге-Кутты
init_delta = 0.2 # Начальное значение delta
target_t = 1  # Искомое t
Re = 12.38  # Число Рейнольдса
Fr = 1.23  # Число Фруда
N = 720  # Количество лучей
delta_phi = 2 * np.pi / N # Угол между n и n+1 лучами

runge = Runge(delta_t, target_t, 0)

boundary_conditions = {
    "N": N,
    "Re": Re,
    "Fr": Fr,
    "target_t": target_t,
    "init_delta": init_delta,
    "delta_phi": delta_phi
}

init_values = np.full(shape=N, fill_value=init_delta)
func = partial(f, boundary_conditions)

result = runge.runge_method(func, init_values)

r = [i + 1 for i in result]
phi = np.arange(0, 2 * np.pi, delta_phi)

print(r)

# r от φ в полярной системе координат
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(phi, r)
ax.plot(phi, np.ones_like(phi), color="black")
ax.grid(True)

# r от t в прямоугольной системе координат
# plt.figure()
# plt.plot(t,r)
# plt.xlabel('Время, t')
# plt.ylabel('Радиус, r')
# plt.legend('π', loc='best')
# plt.grid(True)

plt.show()



