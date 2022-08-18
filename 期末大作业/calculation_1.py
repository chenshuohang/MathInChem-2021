from re import M, T
import numpy as np, scipy.linalg as la
import water
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import acos, exp, pi, sinh
flag = 2 # calculate all of v, dv and ddv
kB = 3.166811564e-6
# initial cartesian coordinates as [Hx Hy Hz Ox Oy Oz Hx Hy Hz]
cart = np.array([ 1.8112496379040899,      -0.0009820612379843,       0.0000000000000000,
				0.0012030107158682,       0.0015494457436835,       0.0000000000000000,
	 			-0.4497832071840029,      1.7545145430537885,       0.0000000000000000])
v, dv, ddv = water.h2opot(cart, flag)
m=np.array([1, 1, 1, 16.01102735, 16.01102735, 16.01102735, 1, 1, 1])*1836.152672
hessian = np.dot(np.diag(1/(m**0.5)), np.dot(ddv, np.diag(1/(m**0.5))))
eigen, normal = la.eig(hessian)
temp = 300

# select vibrational energy
veig = eigen[0:3]
vnormal = normal[:,0:3].T
omega = veig **0.5
freq = omega/(2 * np.pi)
# Unit convertion
print('frequency in Hz\n', freq/2.418843265857e-17)
print('frequency in cm-1\n', freq/(29979245800*2.418843265857e-17))
# partition function
def Z_CM(T,w):
	return kB*T/w
def Z_QM(T,w):
	return(1/(2*sinh(w/(2*kB*T))))
print('partition function at 0K:')
for i in range(0,3):
	print('Vibrational type:',i+1,', Z_CM:',0,', Z_QM:',0)
print('partition function at 300K')
for i in range(0,3):
	print('Vibrational type:',i+1,', Z_CM:',Z_CM(300,omega[i].real),', Z_QM:',Z_QM(300,omega[i].real))

# calculation of bond length and bond angle
rOH1 = np.sqrt((cart[3]-cart[0])**2+(cart[4]-cart[1])**2+(cart[5]-cart[2])**2)
rOH2 = np.sqrt((cart[3]-cart[6])**2+(cart[4]-cart[7])**2+(cart[5]-cart[8])**2)
rHH = np.sqrt((cart[6]-cart[0])**2+(cart[7]-cart[1])**2+(cart[8]-cart[2])**2)
angHOH = acos((rOH1**2+rOH2**2-rHH**2)/(2*rOH1*rOH2)) 
print('rOH1 in meter\n',rOH1*5.2917721090380e-11,'\nrOH2 in meter\n',rOH2*5.2917721090380e-11)
print('angle-HOH in degree\n', angHOH/np.pi * 180)

# calculate the situation at 300K
Nrand = 10000
deltaQ = np.zeros((9,Nrand))
for i in range(0,3):
	deltaQ[i] = np.random.normal(loc = 0, scale = (kB * temp)**0.5/(omega[i]), size=Nrand)
deltax = np.dot(np.diag(1/(m**0.5)), np.dot(normal, deltaQ)).T
x = deltax + cart
R_OH1 = np.zeros((Nrand))
R_OH2 = np.zeros((Nrand))
ANG_HOH = np.zeros((Nrand))
R_OH1 = np.sqrt((x[:,3]-x[:,0])**2+(x[:,4]-x[:,1])**2+(x[:,5]-x[:,2])**2)
R_OH2 = np.sqrt((x[:,3]-x[:,6])**2+(x[:,4]-x[:,7])**2+(x[:,5]-x[:,8])**2)
R_H = np.sqrt((x[:,6]-x[:,0])**2+(x[:,7]-x[:,1])**2+(x[:,8]-x[:,2])**2)
for i in range(0, Nrand):
	ANG_HOH[i] = acos((R_OH1[i]**2+R_OH2[i]**2-R_H[i]**2)/(2*R_OH1[i]*R_OH2[i])) - angHOH
print('Fluctation of rOH1 in meter at 300K\n',np.std(R_OH1)*5.2917721090380e-11)
print('Fluctation of rOH2 in meter at 300K\n',np.std(R_OH2)*5.2917721090380e-11)
print('Fluctation of AngHOH in degree at 300K\n',np.std(ANG_HOH)/np.pi * 180)
v0, dv0, ddv0 = water.h2opot_arr(x.T,flag)
P = -dv0
Ek = np.dot( np.diag(1/(m**0.5)), np.average(P**2, axis = 1))
print('Average kinetic energy of hydrogen atom1 at 300K in a.u.\n', Ek[0]+Ek[1]+Ek[2])
print('Average kinetic energy of oxygen atom at 300K in a.u.\n', Ek[3]+Ek[4]+Ek[5])
print('Average kinetic energy of hydrogen atom2 at 300K in a.u.\n', Ek[6]+Ek[7]+Ek[8])

# the situation at 0K
deltaQ_0K = np.zeros((9,Nrand))
for i in range(0,3):
	deltaQ_0K[i] = np.random.normal(loc = 0, scale = (1/2*omega[i])**0.5, size=Nrand)
deltax_0K = np.dot(np.diag(1/(m**0.5)), np.dot(normal, deltaQ_0K)).T
x_0K = deltax_0K + cart
print(np.average(x_0K, axis = 0))
R_OH1_0K = np.zeros((Nrand))
R_OH2_0K = np.zeros((Nrand))
ANG_HOH_0K = np.zeros((Nrand))
R_OH1_0K = np.sqrt((x_0K[:,3]-x_0K[:,0])**2+(x_0K[:,4]-x_0K[:,1])**2+(x_0K[:,5]-x_0K[:,2])**2)
R_OH2_0K = np.sqrt((x_0K[:,3]-x_0K[:,6])**2+(x_0K[:,4]-x_0K[:,7])**2+(x_0K[:,5]-x_0K[:,8])**2)
R_H_0K = np.sqrt((x_0K[:,6]-x_0K[:,0])**2+(x_0K[:,7]-x_0K[:,1])**2+(x_0K[:,8]-x_0K[:,2])**2)
for i in range(0, Nrand):
	ANG_HOH_0K[i] = acos((R_OH1_0K[i]**2+R_OH2_0K[i]**2-R_H_0K[i]**2)/(2*R_OH1_0K[i]*R_OH2_0K[i]))
print('Fluctation of rOH1 in meter at 0K\n',np.std(R_OH1_0K)*5.2917721090380e-11)
print('Fluctation of rOH2 in meter at 0K\n',np.std(R_OH2_0K)*5.2917721090380e-11)
print('Fluctation of AngHOH in degree at 0K\n',np.std(ANG_HOH_0K)/np.pi * 180)
v00, dv00, ddv00 = water.h2opot_arr(x_0K.T,flag)
P_0K = -dv00
Ek0 = np.dot( np.diag(1/(m**0.5)), np.average(P_0K**2, axis = 1))
print('Average kinetic energy of hydrogen atom1 at 0K in a.u.\n', Ek0[0]+Ek0[1]+Ek0[2])
print('Average kinetic energy of oxygen atom at 0K in a.u.\n', Ek0[3]+Ek0[4]+Ek0[5])
print('Average kinetic energy of hydrogen atom2 at 0K in a.u.\n', Ek0[6]+Ek0[7]+Ek0[8])

# ploting animation diagram, 3D plot is too hard(for me,yeah.)
Nt = 100 ; t_final = 5
t = np.linspace(0,t_final,Nt)
xkt = np.zeros((Nt,9))
for i in range(0,3):
	fig, ax = plt.subplots()
	dot, = ax.plot([],[],'-o',lw=2)
	xH1 , yH1, xO, yO, xH2, yH2 = [cart[0]],[cart[1]],[cart[3]],[cart[4]],[cart[6]],[cart[7]]
	for k in range(0,Nt):
		update = cart + np.sin(10*t[k]) * np.dot(np.diag(1/(m**0.5)),vnormal[i]*50)
		xH1.append(update[0])
		yH1.append(update[1])
		xO.append(update[3])
		yO.append(update[4])
		xH2.append(update[6])
		yH2.append(update[7])
	tit = 'vibrational mode whose freuency is {Fr:.2f} cm-1'
	def init():
		ax.set_xlim(-3,3)
		ax.set_ylim(-3,3)
		ax.set_title(tit.format(Fr = float(freq[i]/(29979245800*2.418843265857e-17))))
		return dot,
	def update_dot(p):
		newx = [xH1[p],xO[p],xH2[p]]
		newy = [yH1[p],yO[p],yH2[p]]
		dot.set_data(newx,newy)
		return dot,
	ani = animation.FuncAnimation(fig, update_dot, range(0,Nt), interval = 1,init_func=init)
	ani.save('Vibration mode'+str(i+1)+'.gif',fps=1000)
	plt.show()