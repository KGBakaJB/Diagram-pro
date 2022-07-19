# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:45:56 2021

@author: Кирилл
"""

import matplotlib.pyplot as plt
import math 
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick

po = 1000
g = 9.81
b = 1.5
h = 0.025
k = po*g
E = 6*10**9

I = (b*h**3)/12
a = pow(k/(4*E*I),0.25)
D = E*h**3/(12*(1-0.35**2))
a1 = pow(k/(4*D),0.25)
x = np.arange(0,9.1,0.05)
def w_arr(x,P):
    w_left = []

    for i in range(len(x)):
        w = (P*a*math.exp(-a*x[i])*(math.cos(a*x[i])+ math.sin(a*x[i])))/(2*k)
        
        #print("x = ", x[i],'\n',"w=",w)
        w_left.append(w)

    return np.array(w_left)


def w_arr2(x,P):
    w_left = []

    for i in range(len(x)):
        w = ((P/b)*(a1/(2*k))*math.exp(-a1*x[i])*(math.cos(a1*x[i])+ math.sin(a1*x[i])))
        #print("x = ", x[i],'\n',"w=",w)
        w_left.append(w)

    return np.array(w_left)


color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.4)
color3 = plt.cm.viridis(0.7)
color4 = plt.cm.viridis(.9)

ENG = True
fig = plt.figure()       
ax1 = fig.add_subplot()
plt.rcParams.update({'font.size': 16})
ax1.tick_params(labelsize = 20)
if ENG == False:
    ax1.set_xlabel('Длина ледяной пластины $\mathit{l}$, м',size = 20)
    ax1.set_ylabel('Прогиб $\mathit{w}$, мм',size = 20)
    kg = "кг"
else:
    ax1.set_xlabel('Distanse on ice field $\mathit{l}$, m', size = 20)
    ax1.set_ylabel('Deflection $\mathit{w}$, mm',size = 20)
    kg = "kg"


fig.subplots_adjust(left=0.09,right=0.95, top= 0.97, bottom=0.2)


# ax1.plot(x,w_arr(x,-11.32)*1000, c=color1, label ="11.32"+kg)
# ax1.plot(-1*x,w_arr(x,-11.32)*1000, c=color1)

ax1.plot(x,w_arr2(x,-11.32)*1000, c='dodgerblue', label ="11.32"+kg)
ax1.plot(-1*x,w_arr2(x,-11.32)*1000, c='dodgerblue')




# ax1.plot(x,w_arr(x,-16.32)*1000, c=color2, label ="16.32"+kg)
# ax1.plot(-1*x,w_arr(x,-16.32)*1000, c=color2)

ax1.plot(x,w_arr2(x,-16.32)*1000, c='orange', label ="16.32"+kg)
ax1.plot(-1*x,w_arr2(x,-16.32)*1000, c='orange')




# ax1.plot(x,w_arr(x,-21.32)*1000, c=color3, label ="21.32"+kg)
# ax1.plot(-1*x,w_arr(x,-21.32)*1000, c=color3)

ax1.plot(x,w_arr2(x,-21.32)*1000, c='g', label ="21.32"+kg)
ax1.plot(-1*x,w_arr2(x,-21.32)*1000, c='g')




# ax1.plot(x,w_arr(x,-27.32)*1000, c=color4, label ="27.32"+kg)
# ax1.plot(-1*x,w_arr(x,-27.32)*1000, c=color4)

ax1.plot(x,w_arr2(x,-27.32)*1000, c='r', label ="27.32"+kg)
ax1.plot(-1*x,w_arr2(x,-27.32)*1000, c='r')


ax1.legend(loc = 'lower center',
            mode = 'expand',
            borderaxespad = 20,
            prop={'size':16},
            bbox_to_anchor=(0.5,-0.82),
            ncol = 10
            )
plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(1))

plt.grid()
plt.show()

