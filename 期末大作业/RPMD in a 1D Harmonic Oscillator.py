# RPMD in a 1D Harmonic Oscillator
import numpy as np, matplotlib.pyplot as plt, scipy.linalg as la
kB = 3.166811564e-6   # constant
# defining potential and its derivative
pes_funcs = {
    'Harmonic Oscillator': lambda x: 0.5 * x ** 2 ,
    'Double Well': lambda x : 0.5 * (x**2 - 1) **2
}
def V(x,n,temp,m):
    v = pes_funcs['Harmonic Oscillator'](x)
    vtot = 0
    deltax = np.zeros((n))
    for i in range(0,n):
        deltax[i] = x[i-1] - x[i]
    vspring = np.dot(deltax,np.dot(np.diag(m/2 * (temp*kB)**2 * n), deltax.T))
    vtot += np.sum(v)
    vtot += vspring
    return vtot
def dV(x,n,temp,m):
    dx = 1e-8
    der_potential = np.zeros(n)
    for i in range(0,n):
        v0 = V(x,n,temp,m)
        x[i] += dx
        v = V(x,n,temp,m)
        der_potential[i] = (v-v0)/dx
        x[i] -= dx
    return der_potential
# initial settings
Num = 10 ; Temperature = 300
mass = np.ones((Num),dtype = np.float64)*1836.152672
Nt = 10000
t = np.linspace(0,500,Nt); dt = t[1] - t[0]
Nrand = 10
res = np.zeros((Nrand,Nt))
fig = plt.figure()
ax = plt.gca()
for l in range(0,1):
    for nrand in range(0,Nrand):
        x0 = np.random.normal(loc = 1, scale = (kB*Temperature)**0.5, size=Num)
        p0 = -dV(x0,Num,Temperature,mass)
        # calculation of the motion, using Velocity Verlet Algorithm.
        xtn = np.zeros((Nt, Num))
        ptn = np.zeros((Nt, Num))
        Htn = np.zeros((Nt))
        xtn[0] = x0 ; ptn[0] = p0
        Htn[0] = np.sum(ptn[0]**2/(2*mass)) + V(xtn[0],Num,Temperature,mass)
        for i in range(1,Nt,1):
            tmp = ptn[i-1] - dt/2 * (dV(xtn[i-1],Num,Temperature,mass))
            xtn[i] = xtn[i-1] + dt * (tmp/mass)
            ptn[i] = tmp - dt/2 * (dV(xtn[i],Num,Temperature,mass))
            Htn[i] = np.sum(ptn[i]**2/(2*mass)) + V(xtn[i],Num,Temperature,mass)
        ytn = xtn.T
        res[nrand] = np.average(xtn,axis =1)
        averagex = np.average(res, axis=0)
    # plotting the motion of the particles
    plt.plot(t, averagex)
plt.show()