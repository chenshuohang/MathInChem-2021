import numpy as np
import sympy as sp
x, k , c= sp.symbols('x, k, c')
phi = []
phi.append(c * sp.exp(-k * x**2))
for i in range(0,20,1):
    tmp = sp.diff(phi[i],x)
    phi.append(tmp)
    print(sp.latex(phi[i]))
    print('\n')