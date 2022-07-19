# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:05:12 2021
@author: Кирилл
"""

from matplotlib import pyplot as plt
import numpy as np 
import matplotlib.ticker as mtick
import pandas as pd
import tkinter as tk # use tkinter for python 3
import os
import math 
from tkinter.filedialog import askopenfilename
root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
folder =  askopenfilename() #папка с файлами
folder = os.path.dirname(folder) #Переход к директории выше чем выбранный файл
direct = os.path.dirname(folder)
files = os.listdir(folder) #формируем список путей к файлам
direct_files = os.listdir(direct) 
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
        
    except:
        file= open(path,encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data=pd.read_csv(file,sep='\s+' ,decimal="," ) #Читаю из тестового документа в качестве сепаратора: все пробелы
        data=np.array(data.values) 
        #data = np.flip(data, axis = 1)#перевод значений в массив Numpy
        
    #у переменной data сейчас тип dataframe
    file.close()
    return data[a:(b+1)]

fig = plt.figure(figsize=(width/100., height/100.))       
ax4 = fig.add_subplot()

plt.rcParams.update({'font.size': 12})
vel = []
w = []
w1 = []
name = []
area1 = []
area2 = []
lines = ['dashed', 'dashdot','solid','dotted']
i = 0
col = 0 
fig.subplots_adjust(left=0.09,right=0.95, top= 0.97, bottom=0.2)
for f in files:
    
    color1 = plt.cm.viridis(col)
  
    data_S = open_datafile(folder + '/' + f) 
    print("Читаю ", f)
    data_S = data_S[::500]

   


    print(len(data_S))
    data_save = np.copy(data_S)
    sensor_len = 6 
    index = (np.abs(data_S[:,1]-np.argmin(data_S[:,1]))).argmax()
    ax4.set_xlabel('Distanse on ice field, m',size = 20)
    ax4.set_ylabel('Deflection $\mathit{w}$, mm',size = 20)
    # ax4.set_xlabel('Длина ледяной пластины $\mathit{l}$, м',size = 20)
    # ax4.set_ylabel('Прогиб $\mathit{w}$, мм',size = 20)
    
    ax4.tick_params(labelsize = 20)
    
    w1.append(np.abs(data_S[np.argmin(data_S[:,1]),1]))
    data_S[:,0]= data_S[:,0]-data_S[index,0]
    data_S[:,1] = data_S[:,1] - data_S[int(np.argwhere(data_S[:, 0] == data_S[0, 0])[-1]),1]
  #  print("%.4f" %float(data_S[int(np.argmin(data_S[:,1])), 1]))
    data_S = data_S[int(np.argwhere(data_S[:, 0] == data_S[0, 0])[-1]):
                    int(np.argwhere(data_S[:, 0] == data_S[-1, 0])[0])]
  #  print('min=',np.abs(np.abs(data_S[0:np.argmin(data_S[:,1]),1])-0.004) )
    
    try:
        left_indx = np.argmin(np.abs(np.abs(data_S[0:np.argmax(data_S[0:np.argmin(data_S[:, 1]), 1]), 1])))
    except:
        left_indx = 0
        #left_array = np.abs(data_S[np.argmin(data_S[0:np.argmax(data_S[0:np.argmin(data_S[:,1]), 1]), 1])
#                                   :np.argmax(data_S[0:np.argmin(data_S[:,1]), 1])])
        #left_indx = left_array[np.argmin(left_array[:, 1]), 0]
       # left_indx = np.abs(data_S[:,0] - left_indx*(-1)).argmin()
        
    mid_array = np.abs(data_S[np.argmax(data_S[0:np.argmin(data_S[:, 1]), 1]):
                              np.argmin(data_S[:, 1])])
    mid_indx = mid_array[np.argmin(mid_array[:, 1]), 0]
    mid_indx = np.abs(data_S[:,0] - mid_indx*(-1)).argmin()
    
    """
    if data_S[-1, 1] < 0:
        data_S = np.vstack([data_S,[(((0 - data_S[-1, 1]) / (data_S[-5000, 1] - data_S[-1, 1])) * (data_S[-5000, 0] - data_S[-1, 0]) + data_S[-1, 0]),0]])
        """
    if data_S[-1,1] > 0:
        right_array = np.abs(data_S[np.argmin(data_S[:, 1]):-1])
        right_indx = right_array[np.argmin(right_array[:, 1]), 0]
        right_indx = np.abs(data_S[:,0] - right_indx).argmin()
       # right_indx = np.abs(data_S[-1:np.argmin(data_S[:,1]):-1,1]).argmin()
    else:
        right_indx = -1
    if data_S[-1, 0] > 12:
        data_S[-1,0] = data_S[-2,0]
        
    #Area = np.trapz(y = data_S[mid_indx:right_indx, 1], x = data_S[mid_indx:right_indx, 0])/np.trapz(y = data_S[left_indx:mid_indx, 1], x = data_S[left_indx:mid_indx, 0])
    

    ax4.plot(data_S[:, 0],
              data_S[:, 1], 
              label = ("Velocity: " + f.split('_')[-2]+" m/s \n" + "Deflection: "+ str("%.3f" %float(f.split(" ")[0])) + " mm" ))
    
    # ax4.plot(data_S[:, 0],
    #          data_S[:, 1], 
    #          #label = ("Скорость: " + f.split('_')[-2]+" м/с \n" + "Прогиб: "+ str("%.3f" %float(f.split(" ")[0])) + " мм" ),linewidth = 1.8, c = 'black', linestyle = lines[i])
    #          label = ("Скорость: " + f.split('_')[-2]+" м/с \n" + "Прогиб: "+ str("%.3f" %float(f.split(" ")[0])) + " мм" ),linewidth = 1.8)
    i += 1
             
             
             
    # ax4.scatter(data_S[left_indx,0],data_S[left_indx,1],color='orange', s=50, marker='o')
    # ax4.scatter(data_S[mid_indx,0],data_S[mid_indx,1],color='red', s=50, marker='o')
    # ax4.scatter(data_S[right_indx,0],data_S[right_indx,1],color='blue', s=50, marker='o')
    
    
    vel.append(f.split('_')[-2])
    w.append(np.abs(data_S[np.argmin(data_S[:,1]),1]))
    name.append(f.split("_")[0].split(" ")[-2] +':'+ f.split("_")[0].split(" ")[-1])
    data_S[:,1] = data_S[:,1]/1000
    area1.append(np.trapz(y = data_S[mid_indx:right_indx, 1], x = data_S[mid_indx:right_indx, 0]))
    area2.append(np.trapz(y = data_S[left_indx:mid_indx, 1], x = data_S[left_indx:mid_indx, 0]))
    #ax4.scatter(data_S[np.argmax(data_S[0:np.argmin(data_S[:, 1]), 1]),0],data_S[np.argmax(data_S[0:np.argmin(data_S[:, 1]), 1]),1],color='blue', s=50, marker='o')
    np.savetxt( str("%.4f" %float(np.abs(data_S[int(np.argmin(data_S[:,1])), 1]))) + f[0:-4] + '.txt', data_save)

po = 1000
g = 9.81
b = 1.5
h = 0.025
k = po*g
E = 6*10**9

I = (b*h**3)/12
D = E*h**3/(12*(1-0.35**2))
a1 = pow(k/(4*D),0.25)
x = np.arange(0,9.1,0.05)
def w_arr(x,P):
    w_left = []
    
    for i in range(len(x)):
        w = ((P/b)*(a1/(2*k))*math.exp(-a1*x[i])*(math.cos(a1*x[i])+ math.sin(a1*x[i])))
        #print("x = ", x[i],'\n',"w=",w)
        w_left.append(w)
    return np.array(w_left)


# ax4.plot(x,w_arr(x,-11.32)*1000,linewidth = 3, c = plt.cm.viridis(0), label ="Static deflection")
M = 26.32
ax4.plot(x,w_arr(x,-M)*1000,linewidth = 3, c = 'black', label ="Статический прогиб")
ax4.plot(-1*x,w_arr(x,-M)*1000,linewidth = 3, c = 'black')

Df = pd.DataFrame(data = {"Скорость":vel, "Площадь впадины": area1, "Площадь горба": area2})
# Df.to_excel('excel.xlsx', float_format="%.9f")
ax4.xaxis.set_major_locator(mtick.MultipleLocator(1))
ax4.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
plt.xlim(-7,5)
plt.ylim(-1.8,1.6)

print(i)
plt.grid()
#fig.subplots_adjust(left=0.08,right=0.95, top= 0.97, bottom=0.2)

ax4.legend(loc = 'lower center',
            mode = 'expand',
            borderaxespad = 20,
            prop={'size':16},
            bbox_to_anchor=(0.5,-0.82),
            ncol = 10
            )
plt.show()