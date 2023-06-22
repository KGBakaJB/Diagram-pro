# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:18:29 2021

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

def Xeis(P, v, H = 0.5, h = 0.0094): # 0.4 0.00956
    g = 9.81
    X = v/(g*H)**0.5
    
    po = 1
    po_l =0.92
    puas = 0.35 # коэффициент Пуассона
    E = 5.8 * 10**6 #модуль упругости пресного льда
    D = E * h**3 / (12 * (1 - puas**2))
    alf = (po * g / D)**0.25
    alf = (po_l*h*g*H/(po*g*D)**0.5)**0.5
    w0= P/(2 * (2**0.5) * (D*po*g)**3)
    print(w0*(2**0.5/((1-X**2)**0.5 * (2*((1-X**2)**0.5)-alf**2*X**2))))
    return w0*(2**0.5/((1-X**2)**0.5 * (2*((1-X**2)**0.5)-alf**2*X**2)))
    
RIGHT = 1.7
KG5 = open_datafile(askopenfilename())
KG10 = open_datafile(askopenfilename())
KG15 = open_datafile(askopenfilename())
KG21 = open_datafile(askopenfilename())

def func(x,k,b):
    return k*2.718**(x*b)


root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
fig = plt.figure(figsize=(width/100., height/100.))
fig.subplots_adjust(left=0.09,right=0.95, top= 0.97, bottom=0.1)
plt.rcParams.update({'font.size': 18})
plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))


plt.xlim(0, KG21[np.argmax(KG21[:,0]),0]*1.1)
plt.ylim(0, KG21[np.argmax(KG21[:,0]),1]*1.1)

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.4)
color3 = plt.cm.viridis(0.7)
color4 = plt.cm.viridis(.9)


popt,pcov = curve_fit(func,KG5[:,0],KG5[:,1], maxfev=10**6)
k,b = popt
kg = "кг"
Stat = "Cтатический прогиб"
zero = 0.001
KG5 = np.sort(KG5, axis = 0)
KG10 = np.sort(KG10, axis = 0)
KG15 = np.sort(KG15, axis = 0)
KG21 = np.sort(KG21, axis = 0)
widh = 3
print(k,'/n',b)
plt.scatter(KG5[:,0],KG5[:,1], label = "11,32 "+kg)
plt.plot(KG5[:,0],func(KG5[:,0],*popt), label = '$w_{max}$=%.3f$e^{%.3f v}$'%(k,b))
# plt.plot(np.arange(KG5[0,0], RIGHT, 0.01), Xeis(111,np.arange(KG5[0,0], RIGHT, 0.01) ), c = color1, linestyle = "dashed", linewidth = widh)


popt,pcov = curve_fit(func,KG10[:,0],KG10[:,1], maxfev=10**6)
k,b = popt
print(k,'/n',b)
plt.scatter(KG10[:,0],KG10[:,1], label = "16,32 "+kg)
plt.plot(KG10[:,0],func(KG10[:,0],*popt), label = '$w_{max}$=%.3f$e^{%.3f v}$'%(k,b))
# plt.plot(np.arange(KG10[0,0], RIGHT, 0.01), Xeis(160,np.arange(KG10[0,0], RIGHT, 0.01) ), c = color2, linestyle = "dashed", linewidth = widh)


popt,pcov = curve_fit(func,KG15[:,0],KG15[:,1], maxfev=10**6)
k,b = popt
print(k,'/n',b)
plt.scatter(KG15[:,0],KG15[:,1], label = "21,32 "+kg)
plt.plot(KG15[:,0],func(KG15[:,0],*popt), label = '$w_{max}$=%.3f$e^{%.3f v}$'%(k,b))
# plt.plot(np.arange(KG15[0,0], RIGHT, 0.01), Xeis(209,np.arange(KG15[0,0], RIGHT, 0.01) ), c = color3, linestyle = "dashed", linewidth = widh)

popt,pcov = curve_fit(func,KG21[:,0],KG21[:,1], maxfev=10**6)
k,b = popt
print(k,'/n',b)
plt.scatter(KG21[:,0],KG21[:,1], label = "27,32 "+kg)
plt.plot(KG21[:,0],func(KG21[:,0],*popt), label = '$w_{max}$=%.3f$e^{%.3f v}$'%(k,b))
# plt.plot(np.arange(KG21[0,0], RIGHT, 0.01), Xeis(268,np.arange(KG21[0,0], RIGHT, 0.01) ), c = color4, linestyle = "dashed", linewidth = widh)

plt.scatter(zero,0.278, c = 'dodgerblue', s = 95, marker = 'X')
plt.scatter(zero,0.401, c = 'orange', s = 95, marker = 'X')
plt.scatter(zero,0.524, c = 'g', s = 95, marker = 'X')
plt.scatter(zero,0.672, c = 'r', s = 95, marker = 'X')

plt.xlabel("Load travel velocity $\it{v}$, m/s")
plt.ylabel("Maximum deflection of the ice cover $w_{max}$, mm")
"""
plt.xlabel("Скорость движения нагрузки $\it{v}$, м/c")
plt.ylabel("Максимальный прогиб ледяного покрова $w_{max}$, мм")
"""
plt.grid()
# plt.legend(loc = 'lower center',
#             mode = 'expand',
#             borderaxespad = 20,
#             prop={'size':16},
#             bbox_to_anchor=(0.5,-1),
#             ncol = 4
#             )
plt.show()

