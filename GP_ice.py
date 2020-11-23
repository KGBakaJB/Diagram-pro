# -*- coding: utf-8 -*-
from math import pi
from numpy import array
"""
Вычисление характеристик композитного льда
"""
def onelayer_h (h, d, ro_b=0.919, ro_w=1):
    """
    Вычисляет приведённую толщину 1-слойного композитного ледяного покрова
    Запрашивает толщину проморозки h, диаметр шариков d, плотность шарика ro_b=0,919, плотность воды ro_w=1
    Возвращает приведённую толщину льда
    """
    R=d/2
    S=pow(3,0.5)*pow(R,2)   #Площадь элементарного треугольника разбивки поля льда
    V_b=(4/3)*pi*pow(R,3)    #Объём шарика
    V_upseg=V_b*(1-ro_b/ro_w)   #Объём шарового сегмента, торчащего над поверхностью воды
    #Расчёт высоты, на которую шарик торчит из воды:
    A=pow(((2*ro_b+2j*pow(ro_b*(ro_w-ro_b),0.5))/ro_w)-1,1/3)
    h_upseg=((2-1j*pow(3,0.5)*(A-pow(A,-1))-(A+pow(A,-1)))/2).real*R

    def compare(h):
        if (h+h_upseg)<d:#Если без переморозки
            h_dowseg=d-h_upseg-h  #Расчёт высоты, на которую шарик торчит вниз под нижней поверхностью льда
            V_dowseg=pi*pow(h_dowseg,2)*(R-h_dowseg/3)  #Объём сегмента шарика, торчащий под поверхностью льда
            h_cond1=h+(V_upseg+V_dowseg)/(2*S)   #Приведённая толщина
        else:   #Если с переморозкой
            h_cond1=h+V_upseg/(2*S)
        return h_cond1
    
    
    try:
        h_cond = compare(h)
    except:
        h_cond = []       
        for hi in  h:
            h_cond.append(compare(hi))
                
    return h_cond