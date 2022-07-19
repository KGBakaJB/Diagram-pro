# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:20:32 2020

@author: Кирилл
"""

import numpy as np
from stl import mesh
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
#from mpl_toolkits.mplot3d.axes3d import proj3d
filename = askopenfilename()
my_mesh = mesh.Mesh.from_file(filename)
data = np.array(my_mesh)
# for i in range(len(data[0, :])):
#     print(i, '=')
#     print(data[:, i])

P1 = np.column_stack((data[:, 0],
                data[:, 1],
                data[:, 2]))
P2 = np.column_stack((data[:, 3],
                data[:, 4],
                data[:, 5]))
P3 = np.column_stack((data[:, 6],
                data[:, 7],
                data[:, 8]))
fig = plt.figure()

ax = fig.gca(projection ='3d')
'''
0 - длина
1 - ширина
2 - высота
'''
# P1[:,2] = np.abs(P1[:,2])
# P2[:,2] = np.abs(P2[:,2])
# P3[:,2] = np.abs(P3[:,2])


def find_waterline(Z, k, dk = 0.02):

    '''

    Параметры
    ----------
    Z : TYPE
        ARRAY.
    k : TYPE, float or int. Высота ватерлинии от ОП 
    или координата шпангоута/батокса
    
    dk : TYPE, float or int
        DESCRIPTION. The default is 0.02. Поправка на определение координаты

    Возвращает
    -------
    Y : TYPE 
        ARRAY
        DESCRIPTION. Массив точек лежащих в области k+-dk

    '''
    Z = np.abs(np.copy(Z))
    Z[((k - dk) < Z) & (Z <= (k + dk))] *= -1
    Y = np.where(Z < 0)
    Y = Y[0]
    return Y

#Попытка сделать разный масштаб у осей графика
x_scale = 2
y_scale = 0.5
z_scale = 2
scale = np.diag([x_scale, y_scale, z_scale, 1.0])
scale = scale * (1.0/scale.max())
scale[3,3] = 1.0


def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)


ax.get_proj = short_proj
n = 2

X = find_waterline(P1[:, 2], n)
#print(P1[X, 1])
#print('1000 = ',P1[((np.abs(P1[:, 1] - 1000)).argmin()), 1])
#print(P1[:,1][np.argpartition(P1[:,1], n)[:n]])

ax.scatter(P1[X, 0],#Координата длины
            P1[X, 1],#Координата ширины
            P1[X, 2], color = "green")#Координата высоты
X = find_waterline(P2[:, 2], n)
ax.scatter(P2[X, 0],
            P2[X, 1],
            P2[X, 2])
X = find_waterline(P3[:, 2], n)
ax.scatter(P3[X, 0],
            P3[X, 1],
            P3[X, 2])

# ax.plot_surface(P1[:, 0],#Координата длины
#             P1[:, 1],#Координата ширины
#             P1[:, 2])#Координата высоты

# ax.scatter(P2[:, 0],
#             P2[:, 1],
#             P2[:, 2])

# ax.scatter(P3[:, 0],
#             P3[:, 1],
#             P3[:, 2])



print('0 = ',P2[0, 0],P2[-1, 0])
print('1 = ',P2[0, 1],P2[-1, 1])
print('2 = ',P2[0, 2],P2[-1, 2])

plt.show()