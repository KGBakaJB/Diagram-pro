# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, Slider, RadioButtons
from tkinter.filedialog import askopenfilename
from GP_ice import onelayer_h, cone_h, multylayer
F_1 = 1
W_1 = 1

def open_datafile(path,a=1,b=20000000):
    """
    Открыть файл с экспериментальными данными
    Запрашивает: путь к файлу(строковый тип), левая граница промежутка, правая граница промежутка 
    Возвращает массив из 3-х столбцов: время, сила, прогиб
    """
    
    try:
        file= open(path,encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data=pd.read_csv(file,sep='\s+' ,decimal="." )
        data=np.array(data.values)    #перевод значений в массив Numpy
        float(data[-5,0])
        float(data[-10,0])
        
    except ValueError:
        file= open(path,encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data=pd.read_csv(file,sep='\s+' ,decimal="," ) #Читаю из тестового документа в качестве сепаратора: все пробелы
        data=np.array(data.values) 
        data = np.flip(data, axis = 1)#перевод значений в массив Numpy
        
    #у переменной data сейчас тип dataframe
    file.close()
    return data[a:(b+1)]



def k_line(x2_arg):
    """
    Функция коэффициентов уравнения прямой по двум точкам, проходящей через начало координат
    Запрашивает: x2
    Возвращает: k
    """
    x2=data[x2_arg, 1]
    y2=data[x2_arg, 0]
    return y2/x2


def line (y, k, b):
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
    Возвращает значение X на этой прямой когда Y равно нулю
    """
    
    x1=data[arg_1, 1]
    y1=data[arg_1, 0]
    x2=data[arg_2, 1]
    y2=data[arg_2, 0]
    
    return (((0-y1)*(x2-x1))/(y2-y1))+x1


def corect_w(data, w):
    """
    Корректировка нуля перемещения
    Запрашивает: массив данных, корректировочное значение
    Возвращает откорректированный массив
    """
    
    return data[:, 1]-w

def onRadioButtonsClicked(label):
    """
    Обработчик события при клике по radiobuttons типу гранул
    """
    global gran_d
    value_dic = {'20 мм' : 20,'10 мм' : 10, '3 мм' : 3,'Конусы': 2,'Натурный лёд': 0 }#словарь со значениями диаметров
    gran_d=value_dic[label] # вызов значения из словаря label это то что приходит в функцию при клике на кнопку
    
    
def onRadioButtonsClickedLayer(label):
    global layer
    layer_dic = {"1 слой": 1, "2 слоя": 2, "3 слоя": 3}#словарь со значениями количества слоев гранул
    layer = layer_dic[label]# вывод в переменную layer значения количества слоев
  

def onRadioButtonsClicked_k(label):
    """
    Обработчик события при клике по типу нагружения
    """
    global Nag
    dic_nag = {'Канал' : True,'Пролом': False}
    Nag=dic_nag[label]


def two_point_line_z(arg_1, arg_2):
    """
    Функция построения прямой для продолжения закритической части
    диаграммы разрушения. Работает через уравнение прямой через две точки
    запрашивает координаты точек. Cравнивает их значения из массива данных с
    исскуственно созданными массивом для работы с закритической частью диаграммы.
    Выдает массив значения координат Y

    """
    global x_val
    global X
    x1=data[arg_1, 1]
    # сравнение аргументов с массивом и нахождение индекса точки в исскуственном массиве
    index = (np.abs(x_val-x1)).argmin()
    index2= (np.abs(x_val-arg_2)).argmin()
    y1=data[arg_1, 0]
    if index<index2:
        X=x_val[index:index2]
    else:
        X=x_val[index2:index]
    x2=x_val[index2]

    return (((X-x1)*(0-y1))/(x2-x1))+y1# Y выраженый из уравнения прямой через 2 точки

def F_arh(data):
    from math import pi
    
    height_of_stamp = 0.015 # высота штампа
    diametr = 0.05 #метры
    
    S = (pi*diametr**2) / 4
    if Nag == False:
        data = data[F_max+1, 0] - (ro * S * height_of_stamp )#V*ro

    else:
        index = (np.abs(data[:, 1] - h_ice)).argmin() # ищем индекс точки касания штампа воды
        #в массиве данных перемещения
        while data[index, 1] < h_ice:
            index += 1
        index2 = (np.abs(data[:, 1] - (h_ice + height_of_stamp))).argmin()# ищем индекс точки полного 
        #погружения штампа в воду в массиве данных перемещения
        
        if data[index2, 1] < (h_ice + height_of_stamp):
            index2 += 1
        data[index:index2, 0]  -= ro * S * (data[index:index2, 1] - h_ice)
        data[index2:, 0] -= height_of_stamp * ro * S
 

# добавить ,kernel_A если возвращать работу разрушения на 4 позицию здесь
def addPlot (graph_axes,kernel,kernel_1,kernel_A_end,kernel_Fmax,kernel_L):
    """
    Функция строющая окончательный граффик, принимает значения со слайдеров
    """
    graph_axes.clear()
    graph_axes.grid()
    plt.draw()   # очистка поля графика
    fig.subplots_adjust(left=0.08,right=0.95, top= 0.95, bottom=0.1)
    #h_ice=round(h_ice_S.val, 2) #Толщина промороженного слоя округление до 1 знака после запятой
    if gran_d!=0: 
        
        if layer == 1:
            h=round(onelayer_h(h_ice,gran_d),2) # нахождение приведенной толщины льда и её округление
            if gran_d == 2:
                h = round(cone_h(h_ice),2)# нахождение приведенной толщины композитного льда c конусами и её округление
                
        elif layer == 2:
            h = round(multylayer(h_ice, gran_d, layer),2)# нахождение приведенной толщины двухслойного композитного льда и её округление
           
        elif layer == 3:
            h = round(multylayer(h_ice, gran_d, layer),2)# нахождение приведенной толщины трехслойного композитного льда и её округление
                        
    else:
        h=h_ice
    kernel=int(kernel)
    kernel_1=int(kernel_1)
   # kernel_A=int(kernel_A)
    kernel_A_end=float(kernel_A_end)-two_point_line(kernel, kernel_1)# Можно использовать float т.к. далее это число используется только в исскуственным массивом
    Fmax=int(kernel_Fmax)
    kernel_L=int(kernel_L) #int необходим там где идет работа с конкретной величиной индекса в массиве данных
    
    data[:,1]=corect_w(data,two_point_line(kernel, kernel_1))  # коректировка перемещения и как следствие массива данных
    
    if gran_d != 0:
        F_arh(data)

    # лимиты отображения области
    graph_axes.set_xlim([0,float(kernel_A_end+1)])  
    if data[F_max, 0] < 8:
        graph_axes.set_ylim([0,float(data[F_max,0])+1])
        if data[F_max, 0] < 4.5:
            graph_axes.set_ylim([0,float(data[F_max,0])+0.5])
            if  data[F_max, 0] < 1:
                graph_axes.set_ylim([0,float(data[F_max,0])+0.1])
        
    else:
        graph_axes.set_ylim([0,float(data[F_max,0])+4])
        if data[F_max, 0] > 100:
            graph_axes.set_ylim([0,float(data[F_max,0])+10])
            if data[F_max, 0] > 800:
                graph_axes.set_ylim([0,float(data[F_max,0])+100])
        
    if Nag==False: # Работает только когда нагружение по схеме центральный пролом
        D=(pow(data[kernel,0]/(8*data[kernel,1]*0.001),2))/(ro*g) #Вычисление цилиндрической жёсткости ледяной пластины, Н/м
        E=(D*12*(1-pow(puas,2)))/pow(h/1000,3) #Вычисление модуля упругости, Па
        r1=1/((ro*g/D)**0.25)# вычисление линейного размера пластины, м
        if gran_d == 0:
            D=(pow((1000 * data[kernel,0]) / (8 * data[kernel,1] * 0.01), 2))/(ro * g)
            E=(D*12*(1-pow(puas,2)))/pow(h/100,3)
            r1=1/((ro*g/D)**0.25)
    if gran_d != 0:

        A1=np.trapz(y=data[:Fmax,0],x=(data[:Fmax,1]/1000))   # Определение работы до Fmax
        # Работа поссле Fmax складывается из работы от точки Fmax до точки L cо слайдера и работы под прямой посчитаной от точки L до A_end.
        #Массив Х есть срез искуственного массива которые считается в функции two_point_line_z
        A2=np.trapz(y=data[Fmax:kernel_L,0],x=(data[Fmax:kernel_L,1]/1000))+np.trapz(y=two_point_line_z(kernel_L, kernel_A_end),x=X/1000)
        A_sum=A1+A2
        kp1=A1/(data[F_max,0]*data[Fmax,1]/1000)
        kp2=A2/(data[F_max,0]*(kernel_A_end-data[Fmax,1])/1000)
        kp_sum=A_sum/(data[F_max,0]*kernel_A_end/1000)
        k_a=A2/A_sum
        Wmax = A_sum / (data[F_max, 0] * kp_sum / 1000)
        WA2 =  A2/(data[F_max, 0] *kp2/1000)
    else:
        data1 = np.copy(data)
        data1[:,0] *= 1000

        A1=np.trapz(y=data1[:Fmax,0],x=(data1[:Fmax,1] / 100))   # Определение работы до Fmax
        # Работа поссле Fmax складывается из работы от точки Fmax до точки L cо слайдера и работы под прямой посчитаной от точки L до A_end.
        #Массив Х есть срез искуственного массива которые считается в функции two_point_line_z
        A2=np.trapz(y=data1[Fmax:kernel_L,0],x=(data1[Fmax:kernel_L,1]) / 100)+np.trapz(y=two_point_line_z(kernel_L, kernel_A_end),x=X/100)
        A_sum=A1+A2
        kp1=A1/(data1[F_max,0]*data1[Fmax,1] / 100)
        kp2=A2/(data1[F_max,0]*(kernel_A_end-data1[Fmax,1]) / 100)
        kp_sum=A_sum/(data1[F_max,0]*kernel_A_end / 100)
        k_a = A2/A_sum
        Wmax = A_sum / (data[F_max, 0] * kp_sum/10)
        WA2 =  A2/(data[F_max, 0] *kp2/10)
    #здесь , A_p
    if gran_d == 0:
        ser=[filename[-6:-4],gran_d, h, data[F_max, 0], data[Fmax, 1], A1, A2, A_sum, kp1, kp2, kp_sum, k_a, Wmax, WA2]
    else:
        ser=[gran_d,h_ice, h, data[F_max, 0], data[Fmax, 1], A1, A2, A_sum, kp1, kp2, kp_sum, k_a, Wmax, WA2]
    if Nag==False:
        ser+= [D, E, r1]
    Res_pd=pd.Series(ser)
    Res_pd.to_excel(filename[0:-4]+'.xlsx', float_format="%.6f", index=False, header=False)

    graph_axes.plot(data[:,1],data[:,0])
     #Строим диаграмму разрушения с отметками
    if gran_d == 0:
        graph_axes.set_xlabel('Прогиб w, сантиметры')
        graph_axes.set_ylabel('Сила F, КилоНьютоны') 
    else:
        graph_axes.set_xlabel('Прогиб w, миллиметры')
        graph_axes.set_ylabel('Сила F, Ньютоны')
        
        
    if gran_d != 0:
        if h_ice <= 3.6 and gran_d == 3:
            graph_axes.annotate('$F_{max}$=%.4g Н, w=%.4g мм' %(data[F_max,0],data[Fmax,1]), xy=(data[F_max,1],data[F_max,0]),xytext=(data[F_max,1]+.1,data[F_max,0]+.03))
        else:
            graph_axes.annotate('$F_{max}$=%.4g Н, w=%.4g мм' %(data[F_max,0],data[Fmax,1]), xy=(data[F_max,1],data[F_max,0]),xytext=(data[F_max,1]+0.5,data[F_max,0]+.1))
    else:
        if  data[F_max, 0] < 1:
            graph_axes.annotate('$F_{max}$=%.4g кН, w=%.4g см' %(data[F_max,0],data[Fmax,1]), xy=(data[F_max,1],data[F_max,0]),xytext=(data[F_max,1]+0.05,data[F_max,0]+.05))

        else:
            graph_axes.annotate('$F_{max}$=%.4g кН, w=%.4g см' %(data[F_max,0],data[Fmax,1]), xy=(data[F_max,1],data[F_max,0]),xytext=(data[F_max,1]+0.5,data[F_max,0]+.1))
    
    graph_axes.scatter(data[Fmax,1],data[Fmax,0],color='orange', s=30, marker='o')
    k_D=k_line(kernel) #Получаем коэффициент прямой упругой зоны
    
    
    graph_axes.scatter(data[kernel,1],data[kernel,0],color='orange', s=30, marker='o') #Точка по которой строили прямую и считали D и E
    graph_axes.scatter(data[kernel_1,1],data[kernel_1,0],color='orange', s=30, marker='o') #Вторая точка прямой упругой зоны
    graph_axes.scatter(data[Fmax,1],data[Fmax,0],color='red', s=30, marker='o') #Точка по которой считали работу по максимальной силе
    graph_axes.scatter(data[kernel_L,1],data[kernel_L,0],color='orange', s=30, marker='o')
    
    graph_axes.plot((line(-0.6,k_D,0),line(data[Fmax,0]-(data[Fmax,0]/3),k_D,0)),(-0.6,data[Fmax,0]-(data[Fmax,0]/3)), linestyle = '--', linewidth=1, color = 'darkmagenta') #Строим прямую упругой части графика
    graph_axes.plot((data[Fmax,1],data[Fmax,1]),(data[Fmax,0],0), linestyle = '--', linewidth=1, color = 'darkmagenta')
    graph_axes.plot([kernel_A_end,data[kernel_L,1]],[0,data[kernel_L,0]], linestyle = '--', linewidth=1, color = 'darkmagenta') #Строим прямую закритической части диаграммы
    
    fig2 = plt.figure()
    if Nag==False:
        fig2.text(0.4,0.8,'$r_{0}$ = %.3g м\nD = %.5g Н/м\nE = 'r'$%.4g\times10^3$ МПа' %(r1,D,E/pow(10,9)),size=14) #Выводим значения D и E
        print('$r_{0}$ = %.3g м\nD = %.5g Н/м\nE = 'r'$%.4g\times10^3$ МПа' %(r1,D,E/pow(10,9))) #Выводим значения D и E в консоль
        
    #здесь \n$A_р$ = %.5g Дж во 2 позицию и ,A_p
    fig2.text(0.4, 0.22, '$A_{1} = %.5g$ Дж\n$A_{2} = %.5g$ Дж\n$A_{Σ} =%.5g$ Дж\n$h_л =%.4g$ cм $k_{A1} = %.4g$\n$k_{A2} = %.4g$ \n$k_{AΣ} = %.4g$ \n$A_{2}/A_{Σ} = %.4g$ \n$W_{max} = %.4g$ мм \n$W_{A2} = %.4g$ мм' %(A1,A2,A_sum,h,kp1,kp2,kp_sum,k_a, Wmax, WA2),size=14)
    print('$A_{1} = %.5g$ Дж\n$A_{2} = %.5g$ Дж\n$A_{Σ} =%.5g$ Дж' %(A1,A2,A_sum))
    print('$k_{A1} = %.4g$\n$k_{A2} = %.4g$ \n$k_{AΣ} = %.4g$ \n$A_{2}/A_{Σ} = %.4g$ \n$W_{max} = %.4g$ мм \n$W_{A2} = %.4g$ мм ' %(kp1,kp2,kp_sum,k_a, Wmax, WA2))
    fig2.savefig(filename[0:-4]+ ' ' + str(h_ice) + ' мм' + ' данные' +'.png')
    fig2.clear()
    fig.savefig(filename[0:-4] + ' ' + str(h_ice) + ' мм' '.png')
    plt.draw()

# добавить ,kernel_A если возвращать работу разрушения на 4 позицию здесь
def interact_point(graph_axes,kernel,kernel_1,kernel_A_end,kernel_Fmax,kernel_L):
    """ Функция интерактивного взаимодействия с областью построения. Ничего не считает выводит только граффику"""
    graph_axes.clear()
    graph_axes.grid()

    kernel=int(kernel)
    kernel_1=int(kernel_1)
    #kernel_A=int(kernel_A)
    kernel_A_end=float(kernel_A_end)
    kernel_Fmax=int(kernel_Fmax)
    kernel_L=int(kernel_L)

    graph_axes.plot(data[:,1],data[:,0])
    graph_axes.scatter(data[kernel,1],data[kernel,0],color='orange', s=30, marker='o') #Точка по которой строили прямую и считали D и E
    graph_axes.scatter(data[kernel_1,1],data[kernel_1,0],color='orange', s=30, marker='o') #Вторая точка прямой упругой зоны

    graph_axes.scatter(data[kernel_Fmax,1],data[kernel_Fmax,0],color='red', s=30, marker='o') #Точка по которой считали работу по максимальной силе
    graph_axes.plot((data[kernel_Fmax,1],data[kernel_Fmax,1]),(data[kernel_Fmax,0],0), linestyle = '--', linewidth=1, color = 'darkmagenta')
    graph_axes.plot([kernel_A_end,data[kernel_L,1]],[0,data[kernel_L,0]], linestyle = '--', linewidth=1, color = 'darkmagenta') #Строим прямую упругой части графика
    graph_axes.scatter(data[kernel_L,1],data[kernel_L,0],color='orange', s=30, marker='o')
    plt.draw()
    
    
def onButtonClicked(event):
    #Обработчик событий нажатия на кнопку
    global kernel_S,kernel_1_S,kernel_A_S,kernel_A_end,kernel_Fmax,graph_axes,fig
    plt.clf()#очистка всего поля Figure
    graph_axes = fig.add_subplot(111)
    graph_axes.grid()
    #Если возвращать работу разрушения - добавить сюда 4 переменной ,kernel_A_S.val здесь
    addPlot(graph_axes,kernel_S.val,kernel_1_S.val,kernel_A_end.val,kernel_Fmax.val,kernel_L.val) 

    
def Change_slider(value):
     #Если возвращать работу разрушения - добавить сюда 4 переменной ,kernel_A_S.val здесь
    interact_point(graph_axes,kernel_S.val,kernel_1_S.val,kernel_A_end.val,kernel_Fmax.val,kernel_L.val)
   

def sliders():
    global kernel_S,kernel_1_S, kernel_A_S,kernel_A_end,h_ice_S, kernel_Fmax,kernel_L
    #Создание слайдеров
    # координаты слайдеров
    #ax_h=plt.axes([0.102,0.16,0.835,0.01])
    ax_kernel=plt.axes([0.102,0.14,0.375,0.01])
    ax_kernel_1=plt.axes([0.102,0.12,0.375,0.01])
    #ax_kernel_A=plt.axes([0.102,0.1,0.375,0.01])
    ax_kernel_end=plt.axes([0.562,0.1,0.375,0.01])
    ax_kernel_F=plt.axes([0.562,0.14,0.375,0.01])
    ax_kernel_L=plt.axes([0.562,0.12,0.375,0.01])
    kernel_S=Slider(ax_kernel,'Верхняя точка упр.зоны',1,int(len(data[:,0]-100)/3),valinit=501,valfmt='%10.0f')
    kernel_S.valtext.set_visible(False)
    kernel_1_S=(Slider(ax_kernel_1,'Нижняя точка упр.зоны',1,int(len(data[:,0]-100)/3),valinit=502,valfmt='%10.0f'))
    kernel_1_S.valtext.set_visible(False)
    kernel_A_end=(Slider(ax_kernel_end,'Полная работа',float(x_val[0]),float(x_val[-1]),valinit=float(data[-1,1]),valfmt='%0.01f'))
    kernel_A_end.valtext.set_visible(False)
    kernel_Fmax=(Slider(ax_kernel_F,'Максимальная сила',1,int(len(data[:,0]-100)),valinit=int(F_max),valfmt='%10.0f'))
    kernel_Fmax.valtext.set_visible(False)
    kernel_L=(Slider(ax_kernel_L,'Закрит. часть',int(F_max),int(len(data[:,0])),valinit=int(len(data[:,0])-1),valfmt='%10.0f'))
    kernel_L.valtext.set_visible(False)

def h_of_ice():
    global h_ice
    print("Введите толщину льда")
    h_ice=input()
    try:
        h_ice = float(h_ice)
    except:
        try:
            h_list=h_ice.split(',')
            h_ice=h_list[0] + '.' + h_list[1]
            h_ice=float(h_ice)
        except:
            h_list=h_ice.split('/')
            h_ice=h_list[0] + '.' + h_list[1]
            h_ice=float(h_ice) 
            
        
puas=0.35 #Значение коэффициента Пуассона льда
g=9.81 #Ускорение свободного падения
ro=1000 #Плотность воды, кг/куб.м
filename=askopenfilename()# Вызов окна открытия файла
data=open_datafile(filename) #Открыть файл с данными диаграммы разрушени
data = np.vstack((np.array([[0,0],[0,0]]),data))
data[:,0] *= F_1
data[:, 1] *=W_1
h_of_ice()
F_max=np.argmax(data[:, 0]) #индекс значения максимальной силы в массиве
x_val=np.arange(data[F_max,1],10*data[F_max,1]+0.01,0.01)# массив для работы с закритической частью диаграммы
# создаем окно с графиком
fig,graph_axes=plt.subplots()
graph_axes.grid()

# оставляем снизу графика место под виджеты
fig.subplots_adjust(left=0.08,right=0.95, top= 0.97, bottom=0.3)


# Создание переключателя для типа гранул
sliders()# Вызов слайдеров
axes_radiobuttons = plt.axes([-0.02, 0.6, 0.11, 0.11], frameon=False, aspect='equal' )# координаты left bottom width height
radiobuttons= RadioButtons(axes_radiobuttons,['20 мм', '10 мм', '3 мм', 'Конусы', 'Натурный лёд'], activecolor='black',active = 3)
radiobuttons.on_clicked(onRadioButtonsClicked)
onRadioButtonsClicked(radiobuttons.value_selected)# вызов функции события при нажатии на кнопку

# создание переключателя для количества слоев гранул
axes_radiobuttons_lay = plt.axes([-0.02, 0.5, 0.11, 0.11], frameon=False, aspect='equal' )# координаты left bottom width height
radiobuttons_lay= RadioButtons(axes_radiobuttons_lay,['1 слой', '2 слоя', '3 слоя'], activecolor='black',active = 0)
radiobuttons_lay.on_clicked(onRadioButtonsClickedLayer)
onRadioButtonsClickedLayer(radiobuttons_lay.value_selected)# вызов функции события при нажатии на кнопку



# Создание переключателя для типа нагружения

for i in filename.split('/')[-1]:
    if i == 'П' or i =='P' or  i =='Ц' or  i == "ЦП":
        act_nag = 1
        break
    else:
        act_nag = 0

axes_radiobuttons_k = plt.axes([-0.02, 0.4, 0.11, 0.11], frameon=False, aspect='equal' )# координаты left bottom width height
radiobuttons_k= RadioButtons(axes_radiobuttons_k,['Канал', 'Пролом'], activecolor='black', active = act_nag)
radiobuttons_k.on_clicked(onRadioButtonsClicked_k)
onRadioButtonsClicked_k(radiobuttons_k.value_selected)

plt.rcParams.update({'font.size': 19})
# Создание кнопки "Пересчет"
axes_button_add=plt.axes([0.1,0.02,0.1,0.04])# координаты left bottom width height
button_add=Button(axes_button_add,'Пересчёт')
button_add.on_clicked(onButtonClicked)

# При изменении значения ползунка вызывается функция с новым значением ползунка
kernel_S.on_changed(Change_slider)
kernel_1_S.on_changed(Change_slider)
#kernel_A_S.on_changed(Change_slider)
kernel_A_end.on_changed(Change_slider)
kernel_Fmax.on_changed(Change_slider)
kernel_L.on_changed(Change_slider)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
