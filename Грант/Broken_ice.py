# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:46:19 2022

@author: Кирилл
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import Ice_methods as ice


                    # tgФ1  tgФ2   jлтф   jсф
Geom_16 = np.array([0.530, 0.494, 3.328, 1.197,# 0 1 2 3
                    #Флт   Фс     Фп     Фпт
                    5.323, 1.677, 0.085, 0.726,# 4 5 6 7
                    #Фи    Фит    Фик     Фитк
                    0.079, 0.255, 0.247, 0.317,# 8 9 10 11
                    #Фг    Фгт    Ф'пн   Ф'пнт
                    0.008, 0.028, 0.430, 1.734,# 12 13 14 15
                    #Ф'пк  Ф'птк   Ф'гн   Ф'гтн
                    0.157, 0.780, 0.062, 0.199,# 16 17 18 19
                    #Ф'гк   Фгтк
                    0.302, 0.290])             # 20 21

                    # tgФ1  tgФ2   jлтф   jсф
Geom_tanker = np.array([0.706, 4.704, 2.138, 0.722,# 0 1 2 3
                    #Флт   Фс     Фп     Фпт
                    5.336, 3.781, 0.004, 0.027,# 4 5 6 7
                    #Фи    Фит    Фик     Фитк
                    0.232, 0.405, 0.067, 0.234,# 8 9 10 11
                    #Фг    Фгт    Ф'пн   Ф'пнт
                    0.001, 0.002, 0.644, 1.080,# 12 13 14 15
                    #Ф'пк  Ф'птк   Ф'гн   Ф'гтн
                    0.358, 1.504, 0.273, 0.452,# 16 17 18 19
                    #Ф'гк   Фгтк
                    0.035, 0.128])             # 20 21

Geom = Geom_tanker
P_w = 140
v0 = 6.11   
v = np.arange(0, 7, 1)
B = 11.2
h = np.arange(0.1, 0.8, 0.1)
fig = plt.figure()
ax = fig.add_subplot()
col = np.array([0.1, 1, 1])

for i in h:
       col[0] += 0.1
       col[1] -= 0.1
       col[2] -= 0.1

       ax.plot(v, ice.R_Kalinina(i, B, v, Geom) , label = "Толщина льда " + str(round(i,2)) + " м", color = tuple(col))
       ax.plot(v, ice.R_Zuev(i, B, v), linestyle = 'dashed', color = tuple(col), linewidth = 2)
       
                            
ax.plot(v, ice.Thrust_line(P_w, v, v0), color = 'black')      
ax.set_xlabel('Скорость, м/c')
ax.set_ylabel('Сопротвление\nТяга, кН')
ax.set_xlim(0, 6)
ax.xaxis.set_major_locator(mtick.MultipleLocator(0.5))

plt.grid()
ax.legend()
plt.show()