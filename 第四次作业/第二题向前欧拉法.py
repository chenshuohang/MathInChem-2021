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
    x[i] = x[i-1] + p[i-1] * dt
    p[i] = p[i-1] + dt * (-x[i])
plt.figure(num = 1)
plt.plot(t,x)
plt.savefig("x-t图.pdf", format = 'pdf')
plt.figure(num = 2)
plt.plot(t,p)
plt.savefig("p-t图.pdf", format = 'pdf')
plt.figure(num = 3)
plt.plot(p,x)
plt.savefig("x-p图.pdf", format = 'pdf')
Energy = p**2 / 2 + 0.5 * x ** 2
plt.figure(num = 4)
plt.plot(t,Energy)
plt.ylim((0,40))
plt.savefig("E-t图.pdf", format = 'pdf')