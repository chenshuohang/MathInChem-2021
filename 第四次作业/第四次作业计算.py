import numpy as np
import matplotlib.pyplot as plt
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

#定义波函数
def phi(n0,x):
    result = np.sqrt(2/L) * np.sin( n0 * np.pi * (x + L/2 ) / L )
    return result

#这个函数仅便于势能计算，被积函数
def adduct(m0,n0,x):
    result = phi(n0,x) * phi(m0,x) * 2 * x * x 
    return result

#定义动能n2h2/8mL2.
def Ek(n0):
    result = n0**2 / (8 * L * L)
    return result
#定义势能
def V0(n0, m0):
    result = L/N * sum(adduct(m0,n0,x)) - L/(2*N) * (adduct(m0,n0,-L/2)+adduct(m0,n0,L/2))
    return result
#T表示动能矩阵，是一个对角阵
T = np.diag(Ek(n))

#计算势能矩阵
V = np.zeros((M,M))
for i in range(0,M,1):
    for j in range(0,M,1):
        V[i][j] = V0(i+1, j+1)
#设置输出至六位小数点
np.set_printoptions(formatter={'float': '{: .6f}'.format})
#算出哈密顿矩阵并将其对角化
H = T + V
eig = np.linalg.eigh(H)

#找出能量本征态并排序，找出前20个以后输出本征能量
eigen = eig[0]
vectors = eig[1]
print(eigen)

#连续输出前20个本征波函数
for quantum in range(0,20,1):
    co = vectors[:,quantum]
    y = np.zeros((N))
    for i in range(0,M,1):
        y += co[i]*phi(i+1,x)
    plt.figure(num = quantum+1)
    plt.plot(x,y)
    name = 'Figure ' + str(quantum + 1) +'.pdf'
    plt.savefig(name, format='pdf')
