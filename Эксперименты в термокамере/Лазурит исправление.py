# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:50:35 2020

@author: 5104
"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import pandas as pd  
import numpy as np
import scipy as sp
import scipy.signal
from tkinter.filedialog import askopenfilename
from matplotlib.widgets import Button, Slider, RadioButtons
#from tkinter.filedialog import asksaveasfilename

def open_datafile(path,a=1,b=20000000):
    """
    Открыть файл с экспериментальными данными
    Запрашивает: путь к файлу(строковый тип), левая граница промежутка, правая граница промежутка 
    Возвращает массив из 3-х столбцов: время, сила, прогиб
    """
    file= open(path,encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
    data=pd.read_csv(file,sep='\s+' )    #Читаю из тестового документа в качестве сепаратора: все пробелы
    #у переменной data сейчас тип dataframe
    file.close()
    data=np.array(data.values)    #перевод значений в массив Numpy
    return data[a:(b+1)]



def calculate_tar (l, r):
    l=int(l)
    r=int(r)

    global data

    data=data1[l:r]
      
        
def plot_F(graph_axes,graph_axes2):
    graph_axes.clear()
    graph_axes.grid()
    graph_axes2.clear()
    graph_axes2.grid()
    graph_axes.plot(data[:,0],data[:,1])
    graph_axes2.plot(data[:,0],data[:,2])
    graph_axes.set_xlabel('Время, секунды')
    graph_axes.set_title('Показания датчика силы до обработки')
    graph_axes.set_ylabel('Показания, Вольты')
    graph_axes.legend(['Нефильтрованные показания','Медианный фильтр','Медианный+Баттерворта фильтры'])
    plt.draw()
    

def onButtonClicked_save(event):
        np.savetxt(data_file[0:-4]+'.txt',data)    
        #filename=asksaveasfilename(defaultextension=".txt",filetypes=(("Текстовый файл",".txt"),("All Files","*.*")))
        #np.savetxt(fname=filename,X=data_mod) #сохранение обработанных данных диаграммы разрушения     


def onButtonClicked_сalc(event):
    global l_s, r_s, graph_axes,graph_axes2

    calculate_tar(l_s.val,r_s.val) 
    plot_F(graph_axes, graph_axes2)   


def interact_point (graph_axes,graph_axes2,l,r):
    l=int(l)
    r=int(r)
    graph_axes.clear()
    graph_axes2.clear()
    graph_axes.grid()
    graph_axes2.grid()
    graph_axes.plot(data1[:,0], data1[:,1])
    graph_axes2.plot(data1[:,0], data1[:,2])
    graph_axes.set_xlabel('Время, секунды')
    graph_axes.set_title('Показания датчика силы')
    graph_axes.set_ylabel('Показания, Вольты')
    graph_axes.scatter(data1[l,0],data1[l,1],color='red', s=50, marker='o') #Точка по которой строили прямую и считали D и E
    graph_axes.scatter(data1[r,0],data1[r,1],color='red', s=50, marker='o') #Вторая точка прямой упругой зоны
    graph_axes2.scatter(data1[l,0],data1[l,2],color='red', s=50, marker='o') #Точка по которой строили прямую и считали D и E
    graph_axes2.scatter(data1[r,0],data1[r,2],color='red', s=50, marker='o') #Вторая точка прямой упругой зоны
    plt.draw()


def Change_slider(value):
    interact_point(graph_axes,graph_axes2, l_s.val,r_s.val)


def add_figets():
    global fig, graph_axes,graph_axes2, data, i

    i=0
    data=open_datafile(data_file)
    fig,(graph_axes,graph_axes2) =plt.subplots(2,1)
    graph_axes.grid()
    graph_axes2.grid()
    # оставляем снизу графика место под виджеты
    fig.subplots_adjust(left=0.07,right=0.95, top= 0.97, bottom=0.29)

        # Создание кнопки "Пересчет"
    axes_button_add_1=plt.axes([0.1,0.02,0.1,0.045])# координаты
    global button_add_1
    button_add_1=Button(axes_button_add_1,'Пересчёт')


        # Создание кнопки "Сохранить"
    axes_button_save=plt.axes([0.202,0.02,0.1,0.045])# координаты
    global button_save
    button_save=Button(axes_button_save,'Cохранить')



   
    #Создание слайдеров
     # координаты слайдеров
    ax_L=plt.axes([0.07,0.08,0.85,0.01]) 
    ax_R=plt.axes([0.07,0.11,0.85,0.01])
   
    # Вызов слайдеров 
    global l_s, r_s

    l_s=Slider(ax_L,'Левая точка',1,int(len(data[:,0]-100)),valinit=1,valfmt='%1.0f',)
    r_s=Slider(ax_R,'Правая точка',1,int(len(data[:,0]-100)),valinit=int(len(data[:,0]-10000)),valfmt='%1.0f')
   

def start():
    global i
    i=0
    calculate_tar(l_s.val,r_s.val) 
    plot_F(graph_axes,graph_axes2)  
data_file=askopenfilename()  
data1=open_datafile(data_file)  
add_figets()
start()

button_add_1.on_clicked(onButtonClicked_сalc)# вызов функции события при нажатии на кнопку
button_save.on_clicked(onButtonClicked_save)# вызов функции события при нажатии на кнопку 

l_s.on_changed(Change_slider)
r_s.on_changed(Change_slider)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
