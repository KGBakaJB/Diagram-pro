# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:30:37 2022

@author: Кирилл
"""

import pandas as pd
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
from numba import jit




def open_and_parse(path):
    file= open(path)
    data=pd.read_csv(file, sep='\s+' ,decimal=".", names =['X', 'Y', 'Z']) # читаю файл оффсетов freeship
    data = data.dropna()#убираю значения NaN 
    file.close()
    data = np.array(data.values, dtype=float)
    data = np.around(data, decimals = 4) # округление до 4 знака после запятой
    return data


def Parse_ling(path):

    @jit(parallel=True, fastmath=True)
    def enumeration(A):
        data=[]
        i=0
        for row in A:
            if len(row)==3:
                data+=[row]
                i+=1
        return(data)
    
    
    file= open(path)
    values = pd.read_csv(file, header=None) #Читаем txt-файл
    value = [*values[0].str.findall(r"[-+]?\d*\.\d+|\d+")]   #Ищем числа в полученных данных
    return np.array(enumeration(value), dtype=np.float32)   #Ищем среди полученных чисел координаты точек


def sterms(data):
    A = np.unique(data[:,0], return_index = True) #получить индексы уникальных элементов массива X
    return np.split(data, indices_or_sections = A[1])# получаем разбитый массив шпангоутов

    
def cut_waterline(sterm, waterline):
    sterm = np.sort(sterm,axis = 0)
    A = np.where(sterm[:,2] < waterline)[0]# массив координат где Z меньше чем заданая ватерлиния
    inter_Y=np.interp(waterline, sterm[A[-2]:A[-1]+2, 2], sterm[A[-2]:A[-1]+2, 1])#интерполирование
    return np.vstack([sterm[A] ,np.array([sterm[0,0], inter_Y, waterline])]) # соединение массивов
    

def ice_shirt_sterm(sterm, B=2):
    sterm = np.sort(sterm,axis = 0)
    l = np.sqrt(np.square(sterm[1:,0]-sterm[:-1,0])+np.square(sterm[1:,1]-sterm[:-1,1])+np.square(sterm[1:,2]-sterm[:-1,2]))
    print(l)
    #print(sterm)
    print(np.sum(l))
    input()

    
filename=askopenfilename()
data = open_and_parse(filename)
ice_shirt_sterm(sterms(data)[20], 4)


#строим корпус:
# fig = plt.figure()
# ax = fig.gca(projection ='3d')
# ax.scatter(data[5000:10000, 0],#Координата длины
#             data[5000:10000, 1],#Координата ширины
#             data[5000:10000, 2], color = "green")#Координата высоты


# ax.set_xlim(None, 10)
# ax.set_ylim(None, 40)
# ax.set_zlim(None, 40)
plt.show()