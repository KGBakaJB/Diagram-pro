# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:41:14 2022

@author: Кирилл
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit


def open_datafile(path,a=1,b=20000000):
    """
    Открыть файл с экспериментальными данными
    Запрашивает: путь к файлу(строковый тип), левая граница промежутка, правая граница промежутка 
    Возвращает массив из 3-х столбцов: время, сила, прогиб
    """
    
    try:
        file= open(path, encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data=pd.read_csv(file,sep='\s+' ,decimal="." )
        data=np.array(data.values)    #перевод значений в массив Numpy
        float(data[-5,0])
        float(data[-10,0])
        
    except ValueError:
        file= open(path, encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data = pd.read_csv(file,sep='\s+' ,decimal="," ) #Читаю из тестового документа в качестве сепаратора: все пробелы
        data = np.array(data.values) 
        

        
    #у переменной data сейчас тип dataframe
    file.close()
    return data
def plot_aprx(Koef,data):
    global ax
    global F_DV

    def apr(h, k):
        return k * h ** 2
    if Koef[-1] == 0:
        pass
    else:
        
        h = np.arange(0.001,0.25, 0.001)
        new_data = np.vstack([data,data2])
        new_data = np.sort(new_data, axis = 0)
        popt, pcov = curve_fit(apr, new_data[:,0], new_data[:,1], maxfev=10**6)
        
        k = popt
        b = 2
        
        # ax2.plot(h, apr(h, *popt), linewidth = 2)
        ax2.grid()
        
        
        ax.plot(h, apr(h, *popt), color = 'black', label = Koef[0] + '=$10^{6}$×%.3f$h^{%.1f}$'%(k,b), linewidth = 2)
        # ax.plot(h, apr(h, *popt), color = 'black', label = Koef[0] + '=%.3f$h^{%.1f}$'%(k,b), linewidth = 2) 
        ax.legend(loc=2, prop={'size': 58})
label_dic = {'1': 'Максимальная сила, Н', '2':'Прогиб при максимальной силе, м' , '3':'Максимальный прогиб, м',
             '4':'Работа критической части диаграммы, Дж', '5': 'Работа закритической части диаграммы, Дж',
             '6': 'Общая работа разрушения, Дж', '7': ' Коэффициент формы критической части диаграммы',
             '8': 'Коэффициент формы закритической части диаграммы', '9': 'Коэффициент формы диаграммы разрушения',
             '10': 'Отношение критической работы разрушения к общей работе', '11': 'Прогиб льда в закритической части,м' }

label_dic = {'1': '$F_{max}$, Н', '2':'$W_{cr}$, м' , '3':'$W_{max}$, м',
             '4':'$A_{cr}$, Дж', '5': '$A_{subcr}$, Дж',
             '6': '$A_{tot}$, Дж', '7': '$K_{cr}$',
             '8': '$K_{subcr}$', '9': '$K_{Atot}$',
             '10': '$A_{cr}$/$A_{tot}$', '11': '$W_{subcr}$,м' }

dic_DV = {"1":['$F_{max}$' , 0.2786, 2],
           '2':['$W_{cr}$' ,0.4614 , 0.5],
           '3':['$W_{max}$' ,0.6198  , 0.5],
           "4":['$A_{cr}$' , 7.417, 2.5],# на 10^5
           '5':['$A_{subcr}$',1.752 , 2.5 ],#на 10^5
           '6':['$A_{tot}$',9.169 , 2.5],#на 10^5
           '7':['$K_{cr}$',0 , 0],#0.577
           '8':['$K_{subcr}$',0 ,0 ],#0.427
           '9':['$K_{tot}$',0 , 0],#0.531
           '10':['$A_{cr}$/$A_{tot}$',0 , 0],
           '11':['$W_{subcr}$' ,0 ,0 ]
           }

dic_KN = {"1":['$F_{max}$' , 2.73, 2],
           '2':['$W_{max}$' ,9.247  , 0.2],
           '3':['$W_{end}$' ,0  , 0],
           "4":['$A_{1}$' , 0.0042, 2.5],
           '5':['$A_{2}$',0.0168 , 2.5 ],#
           '6':['$A_{sum}$',0.0384 , 2.35],#
           '7':['$K_{A1}$',0 , 0],
           '8':['$K_{A2}$',0 ,0 ],
           '9':['$K_{Asum}$',0 , 0],
           '10':['$A_{1}$/$A_{sum}$',0 , 0],
           '11':['$W_{A2}$' ,0 ,0 ]
           
           }
SIZE= 58
X = 0.05
DIC = dic_DV
F_DV = 1/1000
F_KN = 1/1000



filename=askopenfilename()# Вызов окна открытия файла
data=open_datafile(filename) #Открыть файл
data[:,0] *= 0.01
data[:,1] *= F_DV
filename=askopenfilename()# Вызов окна открытия файла
data2=open_datafile(filename) #Открыть файл
data2[:,0] *= 0.001
data2[:,1] *= F_KN


fig = plt.figure()
ax = fig.add_axes([0.16, 0.18, 0.78, 0.76]) # создаем объект figure и axes [left, bottom, width, height] 
# ax2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
ax2 = fig.add_axes([0.001, 0.001, 0.001, 0.001])
ax.tick_params(labelsize=SIZE - 2)
ax2.tick_params(labelsize=SIZE - 2)

ax.set_xlabel('h, м', size = SIZE)
ax.set_ylabel(label_dic[filename.split('/')[-1].split('.')[0]] + "×$10^{3}$", size = SIZE)#+ "×$10^{6}$"
ax.set_xlim(0, 0.25)

ax.grid(linewidth=1)
ax.xaxis.set_major_locator(mtick.MultipleLocator(X))# цена деления
ax.yaxis.set_major_locator(mtick.MultipleLocator(5))# цена деления


# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.d×$10^{3}$"))




ax.scatter(data[:,0], data[:,1], s = 190)
ax.scatter(data2[:,0], data2[:,1], s = 190)
h = np.arange(0.001,0.25, 0.001)
lis = DIC[filename.split('/')[-1].split('.')[0]]

ax.plot(h, 100*lis[1]*h**lis[-1], linewidth = 2,label = lis[0] + '=$10^{5}$×%.3f$h^{%.1f}$'%(lis[1], lis[-1]) )
# ax.plot(h,lis[1]*h**lis[-1], linewidth = 2,label = lis[0] + '=%.3f$h^{%.1f}$'%(lis[1], lis[-1]) )
# ax.plot(h, 0.577*np.ones(len(h)), linewidth = 2,label = lis[0] + '=%.3f'%(0.577) )
ax.legend(loc=2, prop={'size': 52})


ax2.xaxis.set_major_locator(mtick.MultipleLocator(0.01))
# ax2.scatter(data[:,0], data[:,1], s = 70, c = 'm')
# ax2.scatter(data2[:,0], data2[:,1], s = 70, c = 'c')
ax2.set_xlim(0, 0.05)
ax2.set_ylim(0, 120)
# plot_aprx(DIC[filename.split('/')[-1].split('.')[0]], data)
ax.set_ylim(0, None)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

#fig.savefig(filename[0:-4]+'.png')
plt.show()