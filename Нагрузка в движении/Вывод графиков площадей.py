# -*- coding: utf-8 -*-
"""
Created on Sun May  2 20:04:09 2021

@author: Кирилл
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter.filedialog import askopenfilename
import tkinter as tk 
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit

def open_datafile(path):
    """
    Открыть файл с экспериментальными данными
    Запрашивает: путь к файлу(строковый тип), левая граница промежутка, правая граница промежутка 
    Возвращает массив из 3-х столбцов: время, сила, прогиб
    """
    
    try:
        file= open(path, encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data=pd.read_csv(file,sep='\s+' ,decimal="." ,header = None)
        print(data)
        data=np.array(data.values)    #перевод значений в массив Numpy
        float(data[-5,0])
        float(data[-10,0])
        
    except:
        file= open(path,encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data=pd.read_csv(file,sep='\s+' ,decimal=",", header = None ) #Читаю из тестового документа в качестве сепаратора: все пробелы
        data=np.array(data.values) 
        #data = np.flip(data, axis = 1)#перевод значений в массив Numpy
        
    #у переменной data сейчас тип dataframe
    file.close()
    return data
data = open_datafile(askopenfilename())
print(data)
N = len(data[:,0])     # число экспериментов
print(N)
sigma = 3   # стандартное отклонение наблюдаемых значений
k = 0.5     # теоретическое значение параметра k
b = 2       # теоретическое значение параметра b

x = data[:,0]

y = data[:,1]

y2 = data[:,2]

y3 = data[:,3]

# вычисляем коэффициенты
def coef (x,y):
    global N
    global kk
    global bb
    mx = x.sum()/N
    my = y.sum()/N
    a2 = np.dot(x.T, x)/N
    a11 = np.dot(x.T, y)/N
    
    kk = (a11 - mx*my)/(a2 - mx**2)
    bb = my - kk*mx
    return(np.array([kk*z+bb for z in range(N)]))

#def func (k,b):
    

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.9)
color3 = plt.cm.viridis(0.5)


root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
fig = plt.figure(figsize=(width/100., height/100.))       
plt.rcParams.update({'font.size': 16})
ax1 = fig.add_subplot(2,2,1)

Xmaj = 0.2
ENG = 1
dep = 0.8
hump = 0.064
zero = 0


plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(mtick.MultipleLocator(Xmaj/2))
plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(Xmaj))


if ENG == False:
    ax1.set_xlabel('Скорость, м\с')
    ax1.set_ylabel('Площадь, $м^2⋅10^{-3}$')
else:
    ax1.set_xlabel('Velocity, m\s')
    ax1.set_ylabel('Area, $m^2⋅10^{-3}$')


ax1.set_xlim(0, data[np.argmax(data[:,0]),0]*1.1)
ax1.set_ylim(0, 12)

if ENG == True:
    ax1.scatter(x, y, s=55, c=color1, label = "Area of the depression, $A_{dep}$")
    ax1.scatter(0.3,5, s = 85, c = color1, marker = "X")
    ax1.scatter(x, y2, s = 55, c=color2, label = "Area of the hump, $A_{hump}$")
    ax1.scatter(0.3,3, s = 85, c = color2, marker = "X")
else:
    ax1.scatter(x, y, s=55, c=color1, label = "Площадь впадины, $A_{вп}$ ")
    ax1.scatter(zero,dep, s=85, c = color1, marker = "X")
    ax1.scatter(x, y2, s = 55, c=color2, label = "Площадь выгиба, $A_{гор}$")
    ax1.scatter(zero,hump, s=85, c = color2, marker = "X")
plt.grid()


ax2 = fig.add_subplot(2,2,2)
plt.rcParams.update({'font.size': 16})


plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(Xmaj))
plt.gca().xaxis.set_minor_locator(mtick.MultipleLocator(Xmaj/2))
plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(5))


ax2.set_xlim(0, data[np.argmax(data[:,0]),0]*1.1)
ax2.set_ylim(0, 35)
if ENG == False:
    ax2.set_ylabel('Отношение площадей, $A_{вп}$/$A_{гор}$')
    ax2.set_xlabel('Скорость, м\с')
    ax2.scatter(x, y3, s = 55, c=color3, label = "Отношение площадей, $A_{вп}$/$A_{гор}$")
    #ax2.scatter(zero,dep/hump,s = 85, c=color3, marker = "X")
else:
    ax2.set_ylabel('Area ratio, $A_{hump}$/$A_{dep}$')
    ax2.set_xlabel('Velocity, m\s')
    ax2.scatter(x, y3, s = 55, c=color3, label = "Area ratio, $A_{hump}$/$A_{dep}$")
    ax2.scatter(0.3,17,s = 85, c=color3, marker = "X")
plt.grid()


#ax1.plot(coef(x,y), label = str("%.3f" %float(kk)) + "x+" + str("%.3f" %float(bb)) )
#ax1.plot(coef(x,y2), label = str("%.3f" %float(kk)) + "x+" + str("%.3f" %float(bb)) )

ax1.legend(loc = 'lower center',
           mode = 'expand',
           borderaxespad = 0,
           bbox_to_anchor=(0.5,-0.4),
           ncol = 9
           )

plt.show()