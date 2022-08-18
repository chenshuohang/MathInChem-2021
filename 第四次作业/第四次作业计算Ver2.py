import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import diag
#此处假定Planck常数 h = 1，M = 180，N = 1000
#定义探究的势箱L长度
L = 5
#M为取的基组个数
M = 180
#定义n作为量子数
n = np.linspace(1,M,M)
#x的离散化
N = 1001 
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
#定义波函数
phinx = np.sqrt(2/L) * np.sin(np.pi/L * np.outer(n,x+L/2) )

def f(x):
    return 2 * x**2
#动能项和势能项
T = np.diag( n**2/(8*L**2) )
V = np.dot( phinx, np.dot(np.diag(f(x)), phinx.T) ) * dx
H = T + V

eig = np.linalg.eigh(H)
np.set_printoptions(formatter={'float': '{: .6f}'.format})

eigen = eig[0]
vectors = eig[1]
print(eigen[0:20])

#连续输出前20个本征波函数
for quantum in range(0,20,1):
    co = vectors[:,quantum]
    y = np.dot(co,phinx)
    plt.figure(num = quantum+1)
    plt.plot(x,y)
    name = 'Figure0' + str(quantum + 1) +'.pdf'
    plt.savefig(name, format='pdf')