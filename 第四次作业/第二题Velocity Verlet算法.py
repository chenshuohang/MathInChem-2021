import numpy as np
import matplotlib.pyplot as plt
M = 400
t = np.linspace(0,10*np.pi,M)
dt = t[1] - t[0]
x = np.zeros(M)
x[0] = 5
p = np.zeros(M)
p[0] = 6
for i in range(1,M,1):
    tmp = p[i-1] - x[i-1] * dt /2
    x[i] = x[i-1] + tmp * dt
    p[i] = tmp - x[i] * dt /2
Energy = p**2 / 2 + 0.5 * x ** 2
plt.figure(num = 4)
plt.plot(t,Energy)
plt.ylim((0,40))
plt.savefig("E-t图1.pdf", format = 'pdf')