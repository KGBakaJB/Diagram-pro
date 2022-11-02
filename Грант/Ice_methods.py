# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 11:56:32 2022

@author: Кирилл
"""
import math

def R_1 (h, B, v, Geom, g = 9.81, f = 0.15, po_l = 0.91, po = 1, Ci = None, k_id = 1.38):
    """
    Функция расчета импульсного сопротивления судна при движении в битых льдах.
    При вводе параметров в виде массивов данных, необходимо размеры входных массивов 
    привести к одному порядку.
    Parameters
    ----------
    h : TYPE - NP.Array, INT, FLOAT
        DESCRIPTION.Толщина ледяного покрова, м
                
    B : TYPE -  INT, FLOAT
        DESCRIPTION. Ширина судна, м
                
    v : TYPE - NP.Array, INT, FLOAT
        DESCRIPTION. Скорость движения судна, м\с
        
    Geom : TYPE - NP.Array
        DESCRIPTION.Одномерный массив геометрических характеристик корпуса судна
                
    g : TYPE, optional FLOAT
        DESCRIPTION. Ускорение свободного падения м\с2 Стандартное значение 9.81.
        
    f : TYPE, optional FLOAT
        DESCRIPTION. Коэффициент трения льда о корпус судна. Стандартное значение 0.24.
        
    po_l : TYPE, optional FLOAT
        DESCRIPTION. Плотность ледяного покрова т\м3. Стандартное значение 0.92.
        
    po : TYPE, optional FLOAT
        DESCRIPTION. Плотность воды т\м3. Стандартное значение 1.
        
    Ci : TYPE, optional FLOAT, INT
        DESCRIPTION.Коэффициент учитывающий присоединенные массы воды.
        Стандартное значение None, при не изменении стандартного значения 
        расчитывается по формуле внутри функции

    Returns
    -------
    TYPE FLOAT, NP.ARRAY
        DESCRIPTION. Функция возращает число равное импульсному сопротивлению судна кН.
        При вводе в функцию  одномерного массива с размером N возравщается массив размера N

    """
    if Ci == None:
        puas = 0.33 # коэффициент Пуассона
        E = 5 * 10**6 #модуль упругости пресного льда
        D = E * h**3 / (12 * (1 - puas**2))
        alf = (po * g / D)**0.25
        b = 0.312 / alf
        Ci = 1 + (0.068 * b / h) * (po / po_l)
        Ci = 1+(0.068* po)/(h * alf * po_l)
        
    return k_id * (Ci * po_l * h * (B/2) * (v**2) * (Geom[8] + f * Geom[9]))
    

def R_2 ( h, B, v, Geom, f = 0.15, po = 1, Cg = 2):
    """
    Функция расчитывающие сопротивление судна обусловленное рассеиванием энергии движущегося
    состающее из двух слагаемых: диссипативной составляющиией вследствии сопротивления воды
    раздвиганию льдин и диссипативной составляющией обусловленной трением льдин друг об друга.
    При вводе параметров в виде массивов данных, необходимо размеры входных массивов 
    привести к одному порядку.

    Parameters
   h : TYPE - NP.Array, INT, FLOAT
        DESCRIPTION.Толщина ледяного покрова, м
                
    B : TYPE -  INT, FLOAT
        DESCRIPTION. Ширина судна, м
                
    v : TYPE - NP.Array, INT, FLOAT
        DESCRIPTION. Скорость движения судна, м\с
     Geom : TYPE - NP.Array
        DESCRIPTION.Одномерный массив геометрических характеристик корпуса судна
    f : TYPE, optional FLOAT
        DESCRIPTION. Коэффициент трения льда о корпус судна. Стандартное значение 0.24.

    po : TYPE, optional FLOAT
        DESCRIPTION. Плотность воды т\м3. Стандартное значение 1.
        
    Cg : TYPE, optional FLOAT, INT
        DESCRIPTION.Коэффициент учитывающий гидродинамическое сопротивление 
        при раздвигании льдин.
        Стандартное значение 2
   Returns
    -------
    TYPE FLOAT, NP.ARRAY
        DESCRIPTION. Функция возращает число или массив числе равное сопротивлению
        воды раздвиганию льдин и трения льдин друг об друга , кН.
        При вводе в функцию  одномерного массива с размером N возравщается массив размера N


    """
    return Cg * po * (v**2) * h * (B/2) * (Geom[18] + f * Geom[19])


def R_3 (h, B, Geom, g = 9.81, f = 0.15, po_l = 0.91, po = 1, k_p = 0.7):
    """
    Функция расчета сопротивления судна обсуловленного притаплиыванием
    и поворачиванием льдин.    
    При вводе параметров в виде массивов данных, необходимо размеры входных массивов 
    привести к одному порядку.

    Parameters
    ----------
     h : TYPE - NP.Array, INT, FLOAT
        DESCRIPTION.Толщина ледяного покрова, м
                
    B : TYPE -  INT, FLOAT
        DESCRIPTION. Ширина судна, м
     Geom : TYPE - NP.Array
        DESCRIPTION.Одномерный массив геометрических характеристик корпуса судна
                
    g : TYPE, optional FLOAT
        DESCRIPTION. Ускорение свободного падения м\с2 Стандартное значение 9.81.
        
    f : TYPE, optional FLOAT
        DESCRIPTION. Коэффициент трения льда о корпус судна. Стандартное значение 0.15.
        
    po_l : TYPE, optional FLOAT
        DESCRIPTION. Плотность ледяного покрова т\м3. Стандартное значение 0.92.
        
    po : TYPE, optional FLOAT
        DESCRIPTION. Плотность воды т\м3. Стандартное значение 1.
        

    Returns
    -------
    TYPE FLOAT, NP.ARRAY
        DESCRIPTION. Возрвращает число или массив чисел от притапливания и 
        поворачивания льдин, кН
        При вводе в функцию  одномерного массива с размером N возравщается массив размера N
        

    """
    
    puas = 0.33 # коэффициент Пуассона
    E = 5 * 10**6 #Модуль упругости пресного льда
    D = E * h**3 / (12 * (1 - puas**2)) #Цилиндрическая жесткость
    alf = (po * g / D)**0.25 #Коэффициент упругого основания пластины
    b = 0.312 / alf # ширина льдины
    return k_p * (po-po_l) * g * h * b * B * (Geom[14] + f * Geom[15])


def R_Kalinina (h, B, v, Geom):
    return R_1(h, B, v, Geom) + R_2(h, B, v, Geom) + R_3(h, B, Geom)

    
def R_Zuev(h, B, v, po_l = 0.91, g = 9.82, S = 0.9):
    """
    

    Parameters
    ----------
    h : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    po_l : TYPE, optional
        DESCRIPTION. The default is 0.92.
    g : TYPE, optional
        DESCRIPTION. The default is 9.82.
    S : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    Fr_h = v/(g * h)**0.5

    return (po_l * g * B * h**2) * ((0.13 * B / h) + (1.3 * Fr_h) + (0.5*Fr_h**2)) * (2 - S) * S**2
def R_Sandakov (h, B ,v, L, an, a, Y2, k_raz, k, po_l = 0.91, po = 1.025, f = 0.15, g = 9.81):
    """
    Parameters
    ----------
     h : TYPE - NP.Array, INT, FLOAT
        DESCRIPTION.Толщина ледяного покрова, м
    B : TYPE -  INT, FLOAT
        DESCRIPTION. Ширина судна, м
    v : TYPE - NP.Array, INT, FLOAT
        DESCRIPTION. Скорость движения судна, м\с
    L : TYPE -  INT, FLOAT
        DESCRIPTION. Длина судна, м
    an : TYPE -  INT, FLOAT
        DESCRIPTION. Коэффициент полноты носовой ветви КВЛ
    a : TYPE -  INT, FLOAT
        DESCRIPTION. Коэффициент полноты  КВЛ
    Y2 : TYPE -  INT, FLOAT
        DESCRIPTION. Угол фхода носовой ветви КВЛ
    k_raz : TYPE -  INT, FLOAT
        DESCRIPTION. Коэффициент зависящий от разрушенности ледяного покрова
    k : TYPE - NP.Array
        DESCRIPTION. Массив коэффициентов расчтивающихся по методу наименьших квадратов
    po_l : TYPE, optional FLOAT
        DESCRIPTION. Плотность ледяного покрова т\м3. Стандартное значение 0.92.
    po : TYPE, optional FLOAT
        DESCRIPTION. Плотность воды т\м3. Стандартное значение 1.
    f : TYPE, optional FLOAT
        DESCRIPTION. Коэффициент трения льда о корпус судна. Стандартное значение 0.15.
        
    g : TYPE, optional FLOAT
        DESCRIPTION. Ускорение свободного падения м\с2 Стандартное значение 9.81.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    k1 = k[0]
    k2 = k[1]
    k3 = k[2]
    k4 = k[3]
    puas = 0.33 # коэффициент Пуассона
    E = 5 * 10**6 #Модуль упругости пресного льда
    D = E * h**3 / (12 * (1 - puas**2)) #Цилиндрическая жесткость
    alf = (po * g / D)**0.25 #Коэффициент упругого основания пластины
    b = 0.312 / alf # ширина льдины
    Fr = v / (g * L)*0.5 # число Фруда
    return po_l * ((b *h)**0.5) * ((B / 2)**2) * (k_raz * k1*(1 + f * an * L / B) + k4 * f * a * L / B) + k2 * po_l * b * h * B * (f + an * math.tan(Y2)) * Fr + k3 * po_l * b * h * L * Fr**2 * math.tan(Y2)**2


def R_Gramuzov(h, B, v, Geom, H, hsn, kov = 3.38, k_pst = 0.3*10**-6, ksf = 1.5* 10 **-3, kcb = 0.5*10**-3, k_ost = 4.7, ksn = 0.3, Cg = 2, po = 1.025, po_l = 0.91, f = 0.15, g = 9.81):
    puas = 0.33 # коэффициент Пуассона
    E = 5 * 10**6 #Модуль упругости пресного льда
    d = E * h**3 / (12 * (1 - puas**2)) #Цилиндрическая жесткость
    a = (po * g / d)**0.25 #Коэффициент упругого основания пластины
    Ci = 1+(0.068* po)/(h * a * po_l)
    K3=kov*po_l*h*B*[Ci*(Geom[8]+f*Geom[9])+(Cg*po*Geom[12]/po_l*B*H)*(Geom[12]+f*Geom[13])]
    K4=k_pst*(h**4/d*a)*((1+f*Geom[2]) + ksf*Geom[3]*(d*a**2/h)*(Geom[1]/1+Geom[1]**2)**0.5 + 0.66*(1+f*Geom[4])*B*a+((kcb*Geom[5]*d*a**3*B)/h)) + k_ost*(po-po_l)*g*h*Geom[12]*(Geom[6]+f*Geom[7])+ksn*g*hsn*Geom[12]*(Geom[6]+f*Geom[7])
    return K3 * v*2 + K4

def R_Kahtelian (h, B, v, sig_i = 0.5*10**3, u_0 = 1.0, hj = 1.0, po_l = 0.91, g = 9.81):
    return 0.004*B*sig_i*h*u_0 + 3.6*po_l*g*B*u_0*h**2 + 0.25*h*v/hj*B**1.65


# def R_Ionov():
#     a1 = 
#     a2 =
    
#     def R1(h, B, L , f = 0.15, sig_i = 0.5*10**3): #узнать про f
#         return  0.014 * sig_i * h * (a1 * B + 2 * f * a2 * L)
#     def R2():
        
#         return 
#     def R3():
        
#         return 
#     return R1 + R2 + R3
    
def Thrust_line (P_w, v, v0, a2 = 0.6):
    a1 = 1 - a2
    return (P_w * (1 - a1 * (v/v0) -  a2 * (v/v0)**2))
    