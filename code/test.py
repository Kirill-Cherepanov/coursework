from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame({'measure':[10, -5, 15,20,20, 20,15,5,10], 'angle':[0,45,90,135,180, 225, 270, 315,360]})

x=df.measure[:-1] * np.cos(df.angle[:-1]/180*np.pi)
y=df.measure[:-1] * np.sin(df.angle[:-1]/180*np.pi)
x = np.r_[x, x[0]]
y = np.r_[y, y[0]]

# x = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0])
# y = np.array([1, 2, 1, 0, 1, 2, 1, 0, 1])

polar_to_cartesian = lambda r, phi: (r * np.cos(phi), r * np.sin(phi))
cartesian_to_polar = lambda x, y: (np.sqrt(x**2 + y**2), np.arctan2(y, x))
def smoothen(r, phi):
  x, y = polar_to_cartesian(r, phi)
  tck, _ = interpolate.splprep([x, y], s=100, per=True)
  xi, yi = interpolate.splev(np.linspace(0, 1, len(phi)), tck)
  return cartesian_to_polar(xi, yi)

# Initialise the spider plot
plt.figure(figsize=(12,8))
ax = plt.subplot(polar=True)

r, phi = smoothen(df.measure, df.angle/180*np.pi)

print(len(phi))

# Plot data
ax.plot(df.angle/180*np.pi, df['measure'], linewidth=1, linestyle='solid', label="Spider chart")
# ax.plot(angles, values, linewidth=1, linestyle='solid', label='Interval linearisation')
ax.plot(phi, r, linewidth=1, linestyle='solid', label='Smooth interpolation')
# ax.plot(phi, r, linewidth=1, linestyle='solid', label='Smooth interpolation')

ax.legend()

# Fill area
# ax.fill(angles, values, 'b', alpha=0.1)

plt.show()