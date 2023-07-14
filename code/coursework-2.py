from copy import deepcopy
import math
from numpy import array, zeros, arange
import matplotlib.pyplot as plt
from operator import itemgetter
from functools import partial

def radToDeg(rad):
    return (rad % (2 * math.pi)) / math.pi * 180

class Runge:
    def __init__(self, step):
        self.step = step

    def increment(f, values, step):
        k0 = step * f(values)
        k1 = step * f(values + k0 / 2)
        k2 = step * f(values + k1 / 2)
        k3 = step * f(values + k2)
        return (k0 + 2*k1 + 2*k2 + k3) / 6

    def runge_method(self, f, init_values):
      curr_t = 0
      curr_value = init_values
      t = [curr_t] # подготовка списка t
      values = [curr_value] # подготовка списка values
      while curr_value[0] <= 1: # внесение результатов расчёта в массивы t, values
          # расчёт в точке t0 значений initValues
          curr_value += Runge.increment(f, curr_value, self.step)
          curr_t += self.step # приращение времени

          t.append(curr_t) # заполнение массива t
          values.append(deepcopy(curr_value)) # заполнение массива values
      return array(t), array(values)

def f(boundary_conditions, values): # (h, h', M)
    We, Eu = itemgetter('We', 'Eu')(boundary_conditions)

    f = zeros([3])
    f[0] = values[1] # h'
    f[1] = (2 * values[1]**2 + 2 - We * (2 * Eu + values[0]**2 - 1) * values[0] * (values[1]**2 + 1)**(3/2)) / (2*values[0]) # h''
    f[2] = math.fabs(math.pi * (1 - values[0]**2)) # dM/dx
    return f

# Неизменяемые значения
sigma = 0.07
density = 1260
cylinder_r = 0.025

delta_eu = 0.0005
step = 0.002

runge = Runge(step)

Z0 = 0; M0 = 0; R0 = 1; h0 = 1

# Изменяемые значения
rotation_speed = 2 * math.pi # 6.875
eps0 = - math.pi / 6
target_mass = 0.3

# for eps0 in [ -math.pi / 9,-math.pi / 12, -math.pi / 6, -math.pi / 3 ]: # 1
# for rotation_speed in [math.pi, 2 * math.pi, 3 * math.pi, 4 * math.pi]: # 2
# for rotation_speed in [4.2 * math.pi, 4.4 * math.pi, 4.7 * math.pi, 5 * math.pi]: # 3
# for rotation_speed in [2 * math.pi, 6 * math.pi, 8 * math.pi]: # 4
for target_mass in [0.25, 0.3, 0.35]: # 5,6
    print('mass: ', target_mass)

    L = []
    eps = arange(-math.pi / 3, -math.pi / 12, math.pi / 50)
    for eps0 in eps:
        print('eps: ', eps0)
    
        boundary_conditions = {
            "We": density * cylinder_r**3 * rotation_speed**2 / sigma,
            "Eu": 0, # Initial Eu value
            "eps0": eps0,
            "Z0": Z0,
            "M0": M0,
            "R0": R0,
            "h0": h0
        }
        
        init_values = [boundary_conditions["h0"], math.tan(eps0), boundary_conditions['M0']]
        
        func = partial(f, boundary_conditions)

        curr_mass = 1000
        while (math.fabs(target_mass - curr_mass) > 0.005):
            t, values = runge.runge_method(func, init_values)
            curr_mass = values[len(values) - 1][2]
            boundary_conditions["Eu"] += delta_eu
            # print(curr_mass)

        L.append(step * len(values))
        # z = [i for i in arange(0, len(values) * step, step)]
        # r = [value[0] for value in values]


    plt.plot(eps, L, label='M: ' + str(target_mass) + 
             ' | Eu: ' + str(round(boundary_conditions["Eu"], 2)) + 
             ' | We: ' + str(round(boundary_conditions["We"], 2)))
    
    # plt.plot(eps, L, label=
    #         # 'ε0: ' + str(int(radToDeg((eps0)))) + '°' + 
    #         # 'ω: ' + str(round(rotation_speed, 2)) +
    #         # 'M: ' + str(target_mass) + 
    #         ' | Eu: ' + str(round(boundary_conditions["Eu"], 2)) + 
    #         ' | We: ' + str(round(boundary_conditions["We"], 2)))
        
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0.)
# plt.xlabel("X")
# plt.ylabel("Y")
plt.xlabel("ε")
plt.ylabel("L(ε)")
# plt.axhline(y=1, color='black', linestyle='-')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.show()