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

def onRadioButtonsClicked(label):
    """
    Обработчик события при клике по типу датчика
    """
    global F_k
    dic_F_k = {'1 кг' : -41.274,'3 кг': -107.3, '5 кг': 198.05, '10 кг':  -173.61 , '2 кг': -95.791}
    return dic_F_k[label]

def tar_F(V, null_point=0, k =1 ):
    """
    Преобразование показаний датчика силы (Вольты в Ньютоны)
    Тарировочное уравнение вида F(V)=kV+b
    Запрашивает: показания датчика силы (Вольты), k, номер начальной точки (стандартное значение 0)
    b вычисляется таким образом, чтобы начальная точка переместилась в ноль
    Возвращает F(массив)
    2 кг = -95.824
    3кг = -107,3
    5кг маленький=-195.57
    тарировка 01.06.21:
    5 кг = -259,47
    10кг=-172,41
    10 кг 24.06.2021: -378.42  ###-173.61 20.06/2023
    улица -571.2
    
    к опытам по конусам тарировка от 06.2022:
        1 кг: -41.274
        3 кг: -107.3
        5 кг(большой): 87.85
        
    """
    k = onRadioButtonsClicked(radiobuttons.value_selected)
    b=-k*V[null_point]
    
    
    return np.array(k*V+b)#тарировочное уравнение
           

def tar_w(V, k=-7.0225, null_point=0):
    """
    Преобразование показаний датчика перемещений (Вольты в миллиметры)
    Тарировочное уравнение вида w(V)=kV+b
    Запрашивает: показания датчика перемещений (Вольты), k, номер начальной точки (стандартное значение 0)
    Возвращает w(массив)
    старый датчик -2.646 
    новый зелёный датчик -6,8581
    До конусов:-7.0406 (тоже зеленый)
    конусы: 3.3455
    улица -15.4992
    
    
    """
    b=-k*V[null_point]  
    return np.array(k*V+b) #Тарировочное уравнение


def dig_noise(data,kernel_size=51):
    """
    Устранение "цифрового шума" с помощью медианного фильтра
    Запрашивает: 1-мерный массив, количество точек
    Возвращает сглаженный 1-мерный массив
    """
    return sp.signal.medfilt(data, kernel_size)


def butter_lowpass(normal_cutoff, order=5):
    """
    Настройка фильтра Баттерворта
    """
    return sp.signal.butter(order, normal_cutoff, btype='low', analog=False)


def butter_lowpass_filtfilt(data, normal_cutoff, order=5):
    """
    Низкочастотный фильтр Баттерворта. Вызывает функцию настройки фильтра Баттерворта.
    Запрашивает: 1-мерный массив данных, нормальную отсечку, порядок
    Возвращает: 1-мерный отфильтрованный массив
    """
    b, a = butter_lowpass(normal_cutoff, order=order)
    return sp.signal.filtfilt(b, a, data)


def calculate_tar (l, r, bat, med, f_p, bat_w):
    l=int(l)
    r=int(r)
    bat=float(bat)
    bat_w=float(bat_w)
    med=round(int(med),0)
    f_p=round(int(f_p),0)
    global F_median, data_filt, data, data_mod, i, data_start, med_old, bat_old, l_old, r_old, filt_old, bat_old_w

    if med % 2==0:
        med=med+1

    if i==0:
        med_old = med
        bat_old = bat 
        bat_old_w = bat_w
        i=1

    data=data1[l:r]
    if i==1 or med_old!=med or bat!=bat_old or bat_w != bat_old_w:
        #Прогоняем показания датчика силы через медианный фильтр
        F_median=dig_noise(data[:,1],med) 
        #Прогоняем показания датчиков силы и перемещения через низкочастотный фильтр Баттерворта
        data_filt=np.column_stack((data[:,0],butter_lowpass_filtfilt(F_median, bat),butter_lowpass_filtfilt(data[:,2], bat_w)))
        data_mod=np.column_stack((tar_F(data_filt[:,1],null_point=f_p),tar_w(data_filt[:,2],null_point=0))) #Переводим Вольты в Ньютоны и миллиметры
        if i==1:
            filt_old=np.copy(data_filt)
    else:
        data_mod=np.column_stack((tar_F(data_filt[:,1],null_point=f_p),tar_w(data_filt[:,2],null_point=0))) #Переводим Вольты в Ньютоны и миллиметры
      
        
def plot_F(graph_axes):
    graph_axes.clear()
    graph_axes.grid()
    graph_axes.plot(data[:,0],data[:,1])
    graph_axes.plot(data_filt[:,0],F_median)
    graph_axes.plot(data_filt[:,0],data_filt[:,1])
    graph_axes.set_xlabel('Время, секунды')
    graph_axes.set_title('Показания датчика силы до обработки')
    graph_axes.set_ylabel('Показания, Вольты')
    graph_axes.legend(['Нефильтрованные показания','Медианный фильтр','Медианный+Баттерворта фильтры'])
    plt.draw()
    
    
def plot_w(graph_axes):
    graph_axes.clear()
    graph_axes.plot(data[:,0],data[:,2])
    graph_axes.plot(data_filt[:,0],data_filt[:,2])
    graph_axes.grid()
    graph_axes.set_xlabel('Время, секунды')
    graph_axes.set_title('Показания датчика перемещения до обработки')
    graph_axes.set_ylabel('Показания, Вольты')
    graph_axes.legend(['Нефильтрованные показания','Фильтр Баттерворта'])
    plt.draw()


def plot_F_tar(graph_axes):
    graph_axes.clear()
    graph_axes.plot(data[:,0],data_mod[:,0])
    graph_axes.grid()
    graph_axes.set_xlabel('Время, секунды')
    graph_axes.set_title('Показания датчика силы после обработки')
    graph_axes.set_ylabel('Сила, Ньютоны')
    plt.draw()


def plot_w_tar(graph_axes) :  
    graph_axes.clear()
    graph_axes.plot(data[:,0],data_mod[:,1])
    graph_axes.grid()
    graph_axes.set_xlabel('Время, секунды')
    graph_axes.set_title('Показания датчика перемещения после обработки')
    graph_axes.set_ylabel('Прогиб, миллиметры')
    plt.draw()


def plot_destract(graph_axes):    
    graph_axes.clear()
    graph_axes.plot(data_mod[:,1],data_mod[:,0])
    
    graph_axes.text(data_mod[np.argmax(data_mod[:,0]), 1], data_mod[np.argmax(data_mod[:,0]), 0], "$F_{max}=$" + str(round(data_mod[np.argmax(data_mod[:,0]), 0], 3)))
    graph_axes.grid()
    graph_axes.set_xlabel('Прогиб, миллиметры')
    graph_axes.set_ylabel('Сила, Ньютоны')
    graph_axes.set_title('Диаграмма разрушения')
    plt.draw()


def onButtonClicked_save(event):
        np.savetxt(data_file[0:-4]+'_Результаты.txt',data_mod)    
        #filename=asksaveasfilename(defaultextension=".txt",filetypes=(("Текстовый файл",".txt"),("All Files","*.*")))
        #np.savetxt(fname=filename,X=data_mod) #сохранение обработанных данных диаграммы разрушения     


def onButtonClicked_сalc(event):
    global l_s, r_s, bat_s, med_s, f_s, graph_axes, bat_s_w

    calculate_tar(l_s.val,r_s.val,bat_s.val,med_s.val,f_s.val, bat_s_w.val) 
    plot_F(graph_axes)   


def onButtonClicked_per(event):   
    plot_w(graph_axes)


def onButtonClicked_F(event):   
    plot_F(graph_axes)


def onButtonClicked_Diag(event):
    plot_destract(graph_axes)


def onButtonClicked_Ftar(event):
    plot_F_tar(graph_axes)


def onButtonClicked_wtar(event):
    plot_w_tar(graph_axes)
    

def interact_point (graph_axes,f_p,l,r):
    f_p=int(f_p)
    l=int(l)
    r=int(r)
    graph_axes.clear()
    graph_axes.grid()
    graph_axes.plot(filt_old[:,0],filt_old[:,1],color='green')
    graph_axes.set_xlabel('Время, секунды')
    graph_axes.set_title('Показания датчика силы')
    graph_axes.set_ylabel('Показания, Вольты')
    graph_axes.scatter(data_filt[f_p,0],data_filt[f_p,1],color='red', s=30, marker='o') #Точка по которой считали работу разрушения
    graph_axes.scatter(data1[l,0],data1[l,1],color='orange', s=30, marker='o') #Точка по которой строили прямую и считали D и E
    graph_axes.scatter(data1[r,0],data1[r,1],color='orange', s=30, marker='o') #Вторая точка прямой упругой зоны
    plt.draw()


def Change_slider(value):
    interact_point(graph_axes,f_s.val,l_s.val,r_s.val)


def add_figets():
    global fig, graph_axes, data, i

    i=0
    data=open_datafile(data_file)
    fig,graph_axes=plt.subplots(num = "Tar Interact")
    graph_axes.grid()
    # оставляем снизу графика место под виджеты
    fig.subplots_adjust(left=0.07,right=0.95, top= 0.97, bottom=0.29)

        # Создание кнопки "Пересчет"
    axes_button_add_1=plt.axes([0.1,0.02,0.1,0.045])# координаты
    global button_add_1
    button_add_1=Button(axes_button_add_1,'Пересчёт')

            # Создание кнопки "Перемещение"
    axes_button_add_2=plt.axes([0.508,0.02,0.1,0.045])# координаты
    global button_add_2
    button_add_2=Button(axes_button_add_2,'Перемещение')

            # Создание кнопки "Cила"
    axes_button_add_3=plt.axes([0.406,0.02,0.1,0.045])# координаты
    global button_add_3
    button_add_3=Button(axes_button_add_3,'Сила')
            # Создание кнопки "Диаграмма"
    axes_button_add_4=plt.axes([0.304,0.02,0.1,0.045])# координаты
    global button_add_4
    button_add_4=Button(axes_button_add_4,'Диаграмма')
        # Создание кнопки "Сохранить"
    axes_button_save=plt.axes([0.202,0.02,0.1,0.045])# координаты
    global button_save
    button_save=Button(axes_button_save,'Cохранить')

    axes_button_Ftar=plt.axes([0.61,0.02,0.1,0.045])# координаты
    global button_Ftar
    button_Ftar=Button(axes_button_Ftar,'Истиная сила')

    axes_button_wtar=plt.axes([0.712,0.02,0.15,0.045])# координаты
    global button_wtar
    button_wtar=Button(axes_button_wtar,'Истиное перемещение')
    
    # Создание переключателя для типа датчика
    global radiobuttons
    
    for i in data_file.split('/')[-2].split(' ')[-1]:
        if i == '1':
            act_sensor = 0
            break
        elif i == '3':
            act_sensor = 1
            break
        elif i == '5':
            act_sensor = 2
            break
        else:
            act_sensor = 4

    axes_radiobuttons = plt.axes([-0.02, 0.4, 0.11, 0.11], frameon=False, aspect='equal')# координаты left bottom width height
    radiobuttons= RadioButtons(axes_radiobuttons,['1 кг', '3 кг', '5 кг', '10 кг','2 кг'], activecolor='black', active = act_sensor)
    onRadioButtonsClicked(radiobuttons.value_selected)
    radiobuttons.on_clicked(onRadioButtonsClicked)
    #Создание слайдеров
     # координаты слайдеров
    ax_L=plt.axes([0.07,0.08,0.85,0.01]) 
    ax_R=plt.axes([0.07,0.11,0.85,0.01])
    ax_bat=plt.axes([0.07,0.17,0.85,0.01])
    ax_med=plt.axes([0.07,0.2,0.85,0.01])
    ax_f_p=plt.axes([0.07,0.23,0.85,0.01])
    ax_bat_w = plt.axes([0.07,0.14,0.85,0.01])# координаты left bottom width height
    # Вызов слайдеров 
    global l_s, r_s, bat_s, med_s, f_s, bat_s_w

    l_s=Slider(ax_L,'Левая точка',1,int(len(data[:,0]-100)),valinit=10501,valfmt='%1.0f',)
    r_s=Slider(ax_R,'Правая точка',1,int(len(data[:,0]-100)),valinit=int(len(data[:,0]-10000)),valfmt='%1.0f')
    bat_s=(Slider(ax_bat,' Баттерворт F',0.001,0.02,valinit=0.0055,valfmt='%1.4f'))
    med_s=(Slider(ax_med,' Медианный',101,2001,valinit=501,valfmt='%1.0f'))
    f_s=(Slider(ax_f_p,'Ноль Силы',1,int(len(data[:,0]-100)),valinit=1000,valfmt='%1.0f'))
    bat_s_w=(Slider(ax_bat_w,' Баттерворт W',0.001,0.02,valinit=0.0052,valfmt='%1.4f'))

def start():
    global i
    i=0
    calculate_tar(l_s.val,r_s.val,bat_s.val,med_s.val,f_s.val, bat_s_w.val) 
    plot_F(graph_axes)  
data_file=askopenfilename()  
data1=open_datafile(data_file)  
add_figets()
start()
button_add_1.on_clicked(onButtonClicked_сalc)# вызов функции события при нажатии на кнопку
button_add_2.on_clicked(onButtonClicked_per)# вызов функции события при нажатии на 
button_add_3.on_clicked(onButtonClicked_F)
button_add_4.on_clicked(onButtonClicked_Diag)
button_save.on_clicked(onButtonClicked_save)# вызов функции события при нажатии на кнопку 
button_Ftar.on_clicked(onButtonClicked_Ftar)# вызов функции события при нажатии на кнопку 
button_wtar.on_clicked(onButtonClicked_wtar)# вызов функции события при нажатии на кнопку 

f_s.on_changed(Change_slider)
l_s.on_changed(Change_slider)
r_s.on_changed(Change_slider)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
