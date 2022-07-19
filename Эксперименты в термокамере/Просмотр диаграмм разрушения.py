# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:53:33 2020

@author: 5104
"""


import matplotlib.pyplot as plt 
import numpy as np  

from matplotlib.widgets import Button, Slider

from tkinter.filedialog import askopenfilename


def open_datafile(path):
    """
    Открыть файл с данными диаграммы разрушения
    Запрашивает: путь к файлу(строковый тип)
    Возвращает массив из 2-х столбцов: сила, прогиб (диаграмма без "хвостов")
    """
    import pandas as pd
    file= open(path,encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
    data=pd.read_csv(file,sep='\s+' )    #Читаю из тестового документа в качестве сепаратора: все пробелы
    #у переменной data сейчас тип dataframe
    file.close()
    data=np.array(data.values)    #перевод значений в массив Numpy
    supercritical=np.argwhere(data[:,1]>data[np.argmax(data[:,0]),1])[:,0] #массив, содержащий индексы элементов закритической части диаграммы
    if np.sum(data[supercritical,0]<0)>0: #Проверяем пересекает ли 0 диаграмма в закритической части
        qes=np.argwhere(data[supercritical,0]<0)[0]+np.argmax(data[:,0]) #Если да, то учитываем только значения до 0
    else: #Если нет, то учитываем только какое-то количество элементов
        qes=np.array(len(data[:,0]))-1 #второе слагаемое управляет количеством учитываемых элементов закритической части
        
    return data[np.logical_and(data[:,1]>=0, data[:,1]<=data[qes,1])] #возвращаем значения диаграммы разрушения без "хвостов"

def k_line(x2_arg):
    """
    Функция коэффициентов уравнения прямой по двум точкам, проходящей через начало координат
    Запрашивает: x2
    Возвращает: k
    """
    x2=data[x2_arg,1]
    y2=data[x2_arg,0]
    return y2/x2

def line (y,k,b):
    """
    Функция уравнения прямой вида y=kx+b
    Запрашивает y, k, b
    Возвращает x
    """
    return (y-b)/k
def two_point_line (arg_1, arg_2):
    """
    Функция уравнения прямой проходящей через 2 точки 
    Запращивает аргументы двух точек 
    Возвращает значение х на этой прямой когда у равно нулю
    """
    x1=data[arg_1,1]
    y1=data[arg_1,0]
    x2=data[arg_2,1]
    y2=data[arg_2,0]
    return (((0-y1)*(x2-x1))/(y2-y1))+x1

def corect_w(data,w):
    """
    Корректировка нуля перемещения
    Запрашивает: массив данных, корректировочное значение
    Возвращает откорректированный массив
    """
    return data[:,1]-w
def onelayer_h (h,d):
    """
    Вычисляет приведённую толщину 1-слойного композитного ледяного покрова без переморозки
    Запрашивает толщину проморозки h, диаметр шариков d
    Возвращает приведённую толщину льда
    """
    R=d/2
    pi=3.1416
    return h+0.3333*pi*R-0.1442*pi*h-((0.1626*pi*pow(h,2))/R)+((0.0833*pi*pow(h,3))/pow(R,2))

 #Диаметр гранул, мм

 #Значение (приведённой) толщины льда, мм

"""
kernel=5000 #Индекс точки в массиве данных, по которым будет строиться прямая упругой зоны
kernel_2=1000#Индекс второй точки в массиве данных, по которым будет строиться прямая упругой зоны
kernel_A=20000 #Индекс точки в массиве данных, до которой будет вычисляться работа разрушения
"""

filename=askopenfilename()
h_ice=2
if h_ice<10:
    gran_d=3 
else:
    gran_d=20
data=open_datafile(filename) #Открыть файл с данными диаграммы разрушения
data[0,0]=0
data[np.size(data[:,0])-1,0]=0
F_max=np.argmax(data[:,0]) #индекс значения максимальной силы в массиве
"""
a=data[:,1]
a0=0.75#точка для определения Е
kernel_2=list(a).index(float(a.flat[np.abs(a-a0).argmin()]))
"""
def addPlot (graph_axes,kernel,kernel_1):
    graph_axes.clear()
    graph_axes.grid()
    kernel=int(kernel)
    kernel_1=int(kernel_1)
   
    data[:,1]=corect_w(data,two_point_line(kernel, kernel_1))  
    graph_axes.set_xlim([0,int(data[-1,1])+2])
    graph_axes.set_ylim([0,int(data[F_max,0])+2])
    graph_axes.plot(data[:,1],data[:,0])
     #Строим диаграмму разрушения с отметками
    
    graph_axes.set_xlabel('Прогиб w, миллиметры')
    graph_axes.set_ylabel('Сила F, Ньютоны')
    graph_axes.set_title('Диаграмма разрушения ($\o_{гранул}$ = %r мм, $h_{пром}$ = %r мм)'%(gran_d,h_ice))
    graph_axes.annotate('max F=%.4g Н, w=%.4g мм' %(data[F_max,0],data[F_max,1]), xy=(data[F_max,1],data[F_max,0]),xytext=(data[F_max,1]+1,data[F_max,0]-.3), size=10)
    graph_axes.scatter(data[F_max,1],data[F_max,0],color='orange', s=30, marker='o')
    k_D=k_line(kernel) #Получаем коэффициент прямой упругой зоны
    graph_axes.plot((line(-0.6,k_D,0),line(data[F_max,0]-(data[F_max,0]/1.5),k_D,0)),(-0.6,data[F_max,0]-(data[F_max,0]/1.5)), linestyle = '--', linewidth=1, color = 'darkmagenta') #Строим прямую упругой части графика
  


    
    plt.draw()
def interact_point(graph_axes,kernel,kernel_1):
    graph_axes.clear()
    graph_axes.grid()
    kernel=int(kernel)
    kernel_1=int(kernel_1)
    graph_axes.plot(data[:,1],data[:,0])
    graph_axes.scatter(data[kernel,1],data[kernel,0],color='orange', s=30, marker='o') #Точка по которой строили прямую и считали D и E
    graph_axes.scatter(data[kernel_1,1],data[kernel_1,0],color='orange', s=30, marker='o') #Вторая точка прямой упругой зоны
    
    plt.draw()
if __name__=='__main__':
    def onButtonClicked(event):
        #Обработчик событий нажатия на кнопку
        global kernel_S
        global kernel_1_S 
        global kernel_A_S
        global graph_axes
        #атрибуд val необходим для получения значения с ползунков
        addPlot(graph_axes,kernel_S.val,kernel_1_S.val)
        np.savetxt(filename[0:-4]+'_new.txt',data)
    def Change_slider(value):

        interact_point(graph_axes,kernel_S.val,kernel_1_S.val)
    
    # создаем окно с графиком
    
     
    fig,graph_axes=plt.subplots()
    graph_axes.grid()
    # оставляем снизу графика место под виджеты
    fig.subplots_adjust(left=0.07,right=0.95, top= 0.975, bottom=0.15)
   

    # Создание кнопки "Пересчет"
    axes_button_add=plt.axes([0.6,0.02,0.1,0.1])# координаты
    button_add=Button(axes_button_add,'SAVE')
    button_add.on_clicked(onButtonClicked)# вызов функции события при нажатии на кнопку
    #Создание слайдеров
     # координаты слайдеров
    ax_kernel=plt.axes([0.05,0.05,0.4,0.01]) 
    ax_kernel_1=plt.axes([0.05,0.02,0.4,0.01])

    # Вызов слайдеров 
    kernel_S=Slider(ax_kernel,'1) E и D',1,int(len(data[:,0]-100)/3),valinit=501,valfmt='%10.0f')
    kernel_1_S=(Slider(ax_kernel_1,'2)  E и D',1,int(len(data[:,0]-100)/3),valinit=502,valfmt='%10.0f'))

    kernel_S.on_changed(Change_slider)
    kernel_1_S.on_changed(Change_slider)


    plt.show()
    """
def update (val):
    plt.set_clim(kernel.val,kernel_1.val,kernel_A.val)
    fig.canvas.draw_idle()
    


kernel.on_changed(update)
kernel_1.on_changed(update)
kernel_A.on_changed(update)
plt.show()
"""