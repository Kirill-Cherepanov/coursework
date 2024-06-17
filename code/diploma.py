from copy import deepcopy
from functools import partial
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

def highlight_print(*args):
    print("\033[91m", *args, "\033[0m")

# polar_to_cartesian = lambda r, phi: (r * np.cos(phi), r * np.sin(phi))
# cartesian_to_polar = lambda x, y: (np.sqrt(x**2 + y**2), np.arctan2(y, x))

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
        try:
            values = callback(values + self.increment(f, values, t), t)
        except Exception as e:
            break
      return (np.array(values), t)


def shooting_method(order, ray_index, values, ray_quantity, delta):
    def getValue(increment = 0):
        return values[(ray_index + increment) % ray_quantity]
    
    if (order == 1):
        return (getValue(1) - getValue(-1)) / (2*delta)
    if (order == 2):
        return (getValue(1) - 2*getValue() + getValue(-1)) / (delta**2)
    if (order == 3):
        return (getValue(2) - 2*getValue(1) + 2*getValue(-1) - getValue(-2)) / (2*delta**3)
    if (order == 4):
        return (getValue(2) - 4*getValue(1) + 6*getValue() - 4*getValue(-1) + getValue(-2)) / (delta**4)
    raise Exception("Not implemented!")

def f(boundary_conditions, values, t): # values = (eps, Z, R, M)
    N, Re, Fr, We, delta_phi, init_delta = itemgetter('N', 'Re', 'Fr', 'We', 'delta_phi', 'init_delta')(boundary_conditions)

    if (next(filter(lambda x: abs(x) > init_delta * 5, values[0]), None) != None):
        highlight_print("Overflow at time: ", t)
        raise Exception("Delta overflow")

    dx_dt = np.zeros(shape=(2, N))
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

        # dx_dt[0][i] = d(delta)/dt при phi = j*2*np.pi/N
        # dx_dt[1][i] = d(B)/dt при phi = j*2*np.pi/N

        # 3.2
        # dx_dt[0][j] = (1/3) * (Re/Fr) * (3 * delta**2 * d1_delta * np.cos(phi + t) - delta**3 * np.sin(phi + t))

        # 3.3
        dx_dt[0][j] = 1/3 * (delta * d1_B + B * d1_delta)
        dx_dt[1][j] = (B**2 / 15*delta + 2*B - 3) * d1_delta + (7*B/15 + 2*delta) * d1_B - 3/We * (d1_delta + d3_delta) + 3/Fr * np.cos(phi + t) - 3/Re * (B / delta**2)

    # print(3/Fr * np.cos(1.5 * np.pi))
    return dx_dt

# Регулировка погрешности
delta_t = 0.005  # Шаг по времени для метода Рунге-Кутты
N = 1080 # Количество лучей

# Начальные условия
init_delta = 0.08 # Начальное значение delta
target_t = 3  # Искомое t

rotation_speed = 10 * 2 * np.pi # Скорость вращения

R0 = 0.035
# R0 = 0.025 # Начальный радиус цилиндра

nu = 1.11e-3 # Коэффициент вязкости глицерина
rho = 1260 # Плотность глицерина
sigma = 59.4e-3 # Поверхностное натяжение глицерина
# nu = 1.006e-6 # Коэффициент вязкости воды
# rho = 1000 # Плотность воды
# sigma = 73e-3 # Поверхностное натяжение воды

g = 9.81 # Ускорение свободного падения

phi = np.linspace(0, 2 * np.pi, N)
delta_phi = 2 * np.pi / N # Угол между n и n+1 лучами

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(phi, np.ones_like(phi), color="black", label="Цилиндр")

disturbances = True

# for init_delta in [0.5, 0.1, 0.2]:
for target_t in [2.1]:
# for disturbances in [True]:
    Re = R0**2 * rotation_speed / nu # Число Рейнольдса
    Fr = R0 * rotation_speed**2 / g # Число Фруда
    We = rho * R0**3 * rotation_speed**2 / sigma # Число Вебера
    
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

    init_r = np.array([init_delta + np.sin(i * 8) * init_delta / 20 for i in phi]) if disturbances else np.full(shape=N, fill_value=init_delta)
    init_B = np.full(shape=N, fill_value=0)
    init_values = [init_r, init_B]
    func = partial(f, boundary_conditions)

    values, result_t = runge.runge_method(func, init_values)
    delta = values[0]

    # r = [i + 1 for i in delta]
    r = [i  + 1 for i in delta]
    tetta = [i + result_t for i in phi]

    # r от φ в полярной системе координат
    ax.plot(tetta, r, label="τ=" + f'{result_t:.2f}' + ' Fr=' + f'{Fr:.3f}' + ' We=' + f'{We:.3f}' + '  Re=' + f'{Re:.3f}' + '  δ⁰=' + f'{init_delta:.2f}' + '  ω=' + f'{rotation_speed:.2f} рад/с')

ax.grid(True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                    ncols=1, mode="expand", borderaxespad=0.)
plt.show()





