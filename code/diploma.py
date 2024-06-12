from copy import deepcopy
from functools import partial
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import interpolate

np.set_printoptions(precision=3, suppress=True)

def highlight_print(*args):
    print("\033[91m", *args, "\033[0m")

polar_to_cartesian = lambda r, phi: (r * np.cos(phi), r * np.sin(phi))
cartesian_to_polar = lambda x, y: (np.sqrt(x**2 + y**2), np.arctan2(y, x))

def smoothen(r, phi):
  x, y = polar_to_cartesian(r, phi)
  tck, _ = interpolate.splprep([x, y], s=1, per=True)
  xi, yi = interpolate.splev([i / 2 / np.pi for i in phi], tck)
  polar = cartesian_to_polar(xi, yi)
  return polar

def smoothen_all(values, delta_phi):
  phi = np.arange(0, 2 * np.pi, delta_phi)
  return [smoothen(values[i], phi)[0] for i in range(len(values))]

class Runge:
    def __init__(self, step, target, init):
        self.step = step
        self.target = target
        self.init = init

    def increment(self, f, values, t):
        values1 = f(values, t)
        k0 = self.step * values1
        k1 = self.step * f(values + k0 / 2, t + self.step / 2)
        k2 = self.step * f(values + k1 / 2, t + self.step / 2)
        k3 = self.step * f(values + k2, t + self.step)
        return (k0 + 2*k1 + 2*k2 + k3) / 6

    def runge_method(self, f, init_values, callback = lambda *x: x[0]):
      values = init_values
      for t in np.arange(self.init, self.target, self.step):
          values = callback(values + self.increment(f, values, t), t)
      return np.array(values)


def shooting_method(order, ray_index, values, ray_quantity, delta):
    def getValue(increment = 0):
        return values[(ray_index + increment) % ray_quantity]
    
    if (order == 1):
        return (getValue(1) - getValue(-1)) / (2*delta)
    if (order == 2):
        return (getValue(1) - 2*getValue() + getValue(-1)) / (delta**2)
    if (order == 3):
        # return (getValue(1) - 3*getValue(0) + 3*getValue(-1) - getValue(-2)) / (delta**3)
        return (getValue(2) - 2*getValue(1) + 2*getValue(-1) - getValue(-2)) / (2*delta**3)
    if (order == 4):
        return (getValue(2) - 4*getValue(1) + 6*getValue() - 4*getValue(-1) + getValue(-2)) / (delta**4)
    raise Exception("Not implemented!")

def f(boundary_conditions, values, t): # values = (eps, Z, R, M)
    N, Re, Fr, We, delta_phi = itemgetter('N', 'Re', 'Fr', 'We', 'delta_phi')(boundary_conditions)
    # print(t)
    if (next(filter(lambda x: abs(x) > 3, values[0]), None) != None):
        highlight_print("Current time: ", t)
        raise Exception("Delta is too large")

    f = np.zeros(shape=(2, N))
    for j in range(N):
        # dn_x = d^n(x)/dφ^n
        [phi, delta, B,
         d1_B,
         d1_delta,
        #  d2_delta,
         d3_delta,
        #  d4_delta
         ] = [
            j * delta_phi, values[0][j], values[1][j],
            shooting_method(1, j, values[1], N, delta_phi),
            shooting_method(1, j, values[0], N, delta_phi),
            # shooting_method(2, j, values[0], N, delta_phi),
            shooting_method(3, j, values[0], N, delta_phi),
            # shooting_method(4, j, values[0], N, delta_phi),
        ]

        if (d3_delta > 1):
            print(t)

        # f[0][j] = (1/3) * (Re/Fr) * (3 * delta**2 * d1_delta * np.cos(phi + t) - delta**3 * np.sin(phi + t))

        # f[0][j] = (1/3) * (Re/Fr) * (3 * delta**2 * d1_delta * np.cos(phi + t) - delta**3 * np.sin(phi + t)) - 1/3*Re/We*(3 * d1_delta * delta**2 * (d1_delta + d3_delta) + delta**3 * (d2_delta + d4_delta))

        # f[0][i] = d(delta)/dt при phi = j*2*np.pi/N
        f[0][j] = 1/3 * (delta * d1_B + B * d1_delta)

        # f[0][i] = d(B)/dt при phi = j*2*np.pi/N
        # f[1][j] = (B**2 / 15*delta + 2*B - 3) * d1_delta + (7*B/15 + 2*delta) * d1_B + 3/Fr * np.cos(phi + t) - 3/Re * (B / delta**2)
        f[1][j] = (B**2 / 15*delta + 2*B - 3) * d1_delta + (7*B/15 + 2*delta) * d1_B - 3/We * (d1_delta + d3_delta) + 3/Fr * np.cos(phi + t) - 3/Re * (B / delta**2)

    return  f

delta_t = 0.01  # Шаг по времени для метода Рунге-Кутты
init_delta = 0.1 # 0.05 # Начальное значение delta
target_t = 0.69  # Искомое t

N = 600 # Количество лучей
delta_phi = 2 * np.pi / N # Угол между n и n+1 лучами
phi = np.linspace(0, 2 * np.pi, N)

# nu = 1.006e-6 # Коэффициент вязкости воды
nu = 1.11e-3 # Коэффициент вязкости глицерина

R0 = 0.035
# R0 = 0.025 # Начальный радиус цилиндра

rotation_speed = 5 * np.pi # Скорость вращения

# ro = 1000 # Плотность воды
ro = 1260 # Плотность глицерина

# sigma = 73e-3 # Поверхностное натяжение воды
sigma = 59.4e-3 # Поверхностное натяжение глицерина

g = 9.81 # Ускорение свободного падения

Re = R0**2 * rotation_speed / nu # Число Рейнольдса
Fr = R0 * rotation_speed**2 / g # Число Фруда
We = ro * R0**3 * rotation_speed**2 / sigma # Число Вебера

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(phi, np.ones_like(phi), color="black", label="Цилиндр")

for init_delta in [0.1]:
# for target_t in [2]:
    runge = Runge(delta_t, target_t, 0)

    boundary_conditions = {
        "N": N,
        "Re": Re,
        "Fr": Fr,
        "We": We,
        "target_t": target_t,
        "init_delta": init_delta,
        "delta_phi": delta_phi
    }

    init_values = [np.full(shape=N, fill_value=init_delta), np.full(shape=N, fill_value=0)]
    func = partial(f, boundary_conditions)

    smoothener = lambda values, t: smoothen_all(values, delta_phi) if t > 0.65 and t != 0 else values
    delta = runge.runge_method(func, init_values)[0]

    r = [i + 1 for i in delta]
    tetta = [i + target_t for i in phi]

    # r от φ в полярной системе координат
    ax.plot(tetta, r, label="τ=" + f'{target_t}' + 'c  Fr=' + f'{Fr:.3f}' + '  Re=' + f'{Re:.3f}' + '  δ⁰(φ)=const=' + f'{init_delta}')

# r от t в прямоугольной системе координат
# t = np.arange(0, target_t, delta_t)
# plt.figure()
# plt.plot(t, r)
# plt.xlabel('Время, t')
# plt.ylabel('Радиус, r')
# plt.legend('π', loc='best')
# plt.grid(True)

ax.grid(True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    ncols=2, mode="expand", borderaxespad=0.)
plt.show()





