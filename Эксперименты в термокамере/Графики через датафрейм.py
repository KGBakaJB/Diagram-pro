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
import os
label_dic = {'Fmax': 'Максимальная сила, Н', 'Wcr':'Прогиб при максимальной силе, м' , 'w_max':'Максимальный прогиб, м',
             'A1':'Работа критической части диаграммы, Дж', 'A2': 'Работа закритической части диаграммы, Дж',
             'Asum': 'Общая работа разрушения, Дж', 'ka1': ' Коэффициент формы критической части диаграммы',
             'ka2': 'Коэффициент формы закритической части диаграммы', 'ksum': 'Коэффициент формы диаграммы разрушения',
             'A1/A': 'Отношение критической работы разрушения к общей работе', 'wsup': 'Прогиб льда в закритической части, м',
             'E':'Модуль упругости первого рода, Па', 'D':'Цилиндрическая жесткость, Нм','r':'Характерный линейный размер, м' }


label_dic_0 = {'Fmax': '$F_{max}$, N', 'Wcr':'$W_{cr}$, m' , 'w_max':'$W_{max}$, m',
             'A1':'$A_{cr}$, J', 'A2': '$A_{supcr}$, J',
             'Asum': '$A_{tot}$, J', 'ka1': '$K_{cr}$',
             'ka2': '$K_{supcr}$', 'ksum': '$K_{Atot}$',
             'A1_A': '$A_{cr}$/$A_{tot}$', 'wsup': '$W_{subcr}$, m',
             'E': 'E, Pa', 'D': 'D, Nm', 'r': 'r, m'}

label_dic_eng = {'Fmax': 'F_max, N', 'Wcr':'W_cr, m' , 'w_max':'W_max, m',
             'A1':'A_cr, J', 'A2': 'A_supcr, J',
             'Asum': 'A_tot, J', 'ka1': 'K_cr',
             'ka2': 'K_supcr', 'ksum': 'K_Atot',
             'A1_A': 'A_cr/A_tot', 'wsup': 'W_subcr, m',
             'E': 'E, Pa', 'D': 'D, Nm', 'r': 'r, m'}

label_dic = {'Fmax': '$F_{max}$, Н', 'Wcr':'$W_{cr}$, мм' , 'w_max':'$W_{max}$, мм',
             'A1':'$A_{cr}$, Дж', 'A2': '$A_{supcr}$, Дж',
             'Asum': '$A_{tot}$, Дж', 'ka1': '$K_{cr}$',
             'ka2': '$K_{supcr}$', 'ksum': '$K_{Atot}$',
             'A1_A': '$A_{cr}$/$A_{tot}$', 'wsup': '$W_{supcr}$, мм',
             'E': 'E, Па', 'D': 'D, Нм', 'r': 'r, м'}
range_names ={'Fmax': 1, 'Wcr':2, 'w_max':3,
             'A1':4, 'A2': 5,'Asum': 6,
             'ka1': 7,'ka2': 9,'ksum': 8,
             'A1_A': 9, 'wsup': 10,
             'E': 11, 'D': 12, 'r': 13} 


dic_apr_10_pr = {"Fmax":['$F_{max}$' , 2.448, 1.593],
           'Wcr':['$W_{cr}$' ,7.834 , 0.344],
           'w_max':['$W_{max}$' ,10.43, 0.404],
           "A1":['$A_{cr}$' , 0.011, 1.993],
           'A2':['$A_{supcr}$',0.0013, 2.534 ],
           'Asum':['$A_{tot}$',0.012 , 2.137],
           'ka1':['$K_{cr}$',0.63],
           'ka2':['$K_{supcr}$',0],
           'ksum':['$K_{tot}$',0.6],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           'E':['E',2*10**8, 2.106],
           'D':['D',2.52, 2.793],
           'r':['r',0.132, 0.691]
           }
dic_apr_10_1kan = {"Fmax":['$F_{max}$' , 2.552, 1.261],
           "A1":['$A_{cr}$', 0.0097, 1.623],
           'A2':['$A_{supcr}$',0.0018, 2.15 ],
           'Asum':['$A_{tot}$',0.015 , 1.551],
           'Wcr':['$W_{cr}$' ,6.651, 0.271],
           'w_max':['$W_{max}$' ,8.992, 0.387],
           'ka1':['$K_{cr}$',0.65],
           'ksum':['$K_{tot}$',0.61],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }

dic_apr_10_2kan = {"Fmax":['$F_{max}$' , 1.756, 1.375],
           "A1":['$A_{cr}$', 0.0061, 1.671],
           'A2':['$A_{supcr}$',0.0019, 2.389 ],
           'Asum':['$A_{tot}$',0.0075 , 2.001],
           'Wcr':['$W_{cr}$' ,6.6571, 0.199],
           'w_max':['$W_{max}$' ,7.824, 0.592],
           'ka1':['$K_{cr}$',0.62],
           'ksum':['$K_{tot}$',0.59],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }

dic_apr_10_3kan = {"Fmax":['$F_{max}$' , 3.208, 0.957],
           "A1":['$A_{cr}$', 0.012, 1.308],
           'A2':['$A_{supcr}$',0.002, 2.196 ],
           'Asum':['$A_{tot}$',0.013, 1.613],
           'Wcr':['$W_{cr}$' ,6.485, 0.27],
           'w_max':['$W_{max}$' ,6.441, 0.669],
           'ka1':['$K_{cr}$',0.65],
           'ksum':['$K_{tot}$',0.62],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_2_10_pr = {"Fmax":['$F_{max}$' , 1.713, 1.305],
           "A1":['$A_{cr}$', 0.0056, 1.796],
           'A2':['$A_{supcr}$',0.001, 2.289 ],
           'Asum':['$A_{tot}$',0.0044, 2.123],
           'Wcr':['$W_{cr}$' ,9.657, 0.2],
           'w_max':['$W_{max}$' ,4.33, 0.729],
           'ka1':['$K_{cr}$',0.78],
           'ksum':['$K_{tot}$',0.73],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           'D':['D',1.811, 2.747],
           'E':['E',4*10**7, 1.935],
           'r':['r',0.14, 0.612]
           }
dic_apr_2_10_1kan = {"Fmax":['$F_{max}$' , 1.797, 1.178],
           "A1":['$A_{cr}$', 0.003, 1.814],
           'A2':['$A_{supcr}$',0.0011, 2.032],
           'Asum':['$A_{tot}$',0.0035, 1.976],
           'Wcr':['$W_{cr}$' ,1.827, 0.692],
           'w_max':['$W_{max}$' ,5.553, 0.472],
           'ka1':['$K_{cr}$',0.79],
           'ksum':['$K_{tot}$',0.77],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_2_10_2kan = {"Fmax":['$F_{max}$' , 4.627, 0.737],
           "A1":['$A_{cr}$', 0.0042, 1.689],
           'A2':['$A_{supcr}$',0.0003, 2.686],
           'Asum':['$A_{tot}$',0.0027, 2.149],
           'Wcr':['$W_{cr}$' ,1.611, 0.803],
           'w_max':['$W_{max}$' ,1.136, 1.237],
           'ka1':['$K_{cr}$',0.8],
           'ksum':['$K_{tot}$',0.76],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_2_10_3kan = {"Fmax":['$F_{max}$' , 6.128, 0.636],
           "A1":['$A_{cr}$', 0.0076, 1.497],
           'A2':['$A_{supcr}$',0.0031, 1.704],
           'Asum':['$A_{tot}$',0.0103, 1.601],
           'Wcr':['$W_{cr}$' ,2.4, 0.673],
           'w_max':['$W_{max}$' ,3.928, 0.712],
           'ka1':['$K_{cr}$',0.79],
           'ksum':['$K_{tot}$',0.75],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_20_pr = {"Fmax":['$F_{max}$' , 1.021, 1.201],
           "A1":['$A_{cr}$', 0.0028, 1.812],
           'A2':['$A_{supcr}$',0.007, 0.463 ],
           'Asum':['$A_{tot}$',0.0055, 1.57],
           'Wcr':['$W_{cr}$' ,3.841, 0.576],
           'w_max':['$W_{max}$' ,7.476, 0.374],
           'ka1':['$K_{cr}$',0.77],
           'ksum':['$K_{tot}$',0.71],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           'D':['D',0.038, 4.43],
           'E':['E',19.82*10**4, 4.333],
           'r':['r',0.055, 0.956]
           }
dic_apr_20_1kan = {"Fmax":['$F_{max}$' , 1.015, 0.993],
           "A1":['$A_{cr}$', 0.0032, 1.341],
           'A2':['$A_{supcr}$',0.0002, 2.01],
           'Asum':['$A_{tot}$',0.0031, 1.489],
           'Wcr':['$W_{cr}$' ,5.562, 0.232],
           'w_max':['$W_{max}$' ,5.804, 0.309],
           'ka1':['$K_{cr}$',0.72],
           'ksum':['$K_{tot}$',0.7],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_20_2kan = {"Fmax":['$F_{max}$' , 0.936, 0.99],
           "A1":['$A_{cr}$', 0.0029, 1.337],
           'A2':['$A_{supcr}$',0.0002, 2.01],
           'Asum':['$A_{tot}$',0.0039, 1.37],
           'Wcr':['$W_{cr}$' ,5.08, 0.256],
           'w_max':['$W_{max}$' ,7.627, 0.269],
           'ka1':['$K_{cr}$',0.75],
           'ksum':['$K_{tot}$',0.68],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_20_3kan = {"Fmax":['$F_{max}$' , 1.141, 0.87],
           "A1":['$A_{cr}$', 0.0026, 1.382],
           'A2':['$A_{supcr}$',0.0002, 2.01],
           'Asum':['$A_{tot}$',0.002, 1.72],
           'Wcr':['$W_{cr}$' ,5.838, 0.228],
           'w_max':['$W_{max}$' ,4.157, 0.618],
           'ka1':['$K_{cr}$',0.7],
           'ksum':['$K_{tot}$',0.66],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }

dic_apr_1_3_pr = {"Fmax":['$F_{max}$' , 3.08, 1.466],
           "A1":['$A_{cr}$', 0.0187, 1.67],
           'A2':['$A_{supcr}$',0.036, 1.104],
           'Asum':['$A_{tot}$',0.0533, 1.403],
           'Wcr':['$W_{cr}$' ,9.246, 0.277],
           'w_max':['$W_{max}$' ,27.817, -0.046],
           'ka1':['$K_{cr}$',0.6],
           'ksum':['$K_{tot}$',0.61],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           'D':['D',7.387, 1.972],
           'E':['E',4*10**10, -0.696],
           'r':['r',0.166, 0.493]
           }
dic_apr_1_3_1kan = {"Fmax":['$F_{max}$' , 1.657, 1.838],
           "A1":['$A_{cr}$', 0.0097, 1.945],
           'A2':['$A_{supcr}$',0.0099, 2.134],
           'Asum':['$A_{tot}$',0.02, 2.049],
           'Wcr':['$W_{cr}$' ,8.503, 0.164],
           'w_max':['$W_{max}$' ,20.56, 0.17],
           'ka1':['$K_{cr}$',0.63],
           'ksum':['$K_{tot}$',0.595],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_1_3_2kan = {"Fmax":['$F_{max}$' , 1.713, 1.687],
           "A1":['$A_{cr}$', 0.0072, 2.029],
           'A2':['$A_{supcr}$',0.014, 1.683],
           'Asum':['$A_{tot}$',0.022, 1.837],
           'Wcr':['$W_{cr}$' ,6.637, 0.311],
           'w_max':['$W_{max}$' ,22.663, 0.023],
           'ka1':['$K_{cr}$',0.66],
           'ksum':['$K_{tot}$',0.64],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_1_3_3kan = {"Fmax":['$F_{max}$' , 2.817, 1.431],
           "A1":['$A_{cr}$', 0.0135, 1.855],
           'A2':['$A_{supcr}$',0.0352, 0.902],
           'Asum':['$A_{tot}$',0.037, 1.537],
           'Wcr':['$W_{cr}$' ,14.11, -0.345],
           'w_max':['$W_{max}$' ,67.22, -0.974],
           'ka1':['$K_{cr}$',0.65],
           'ksum':['$K_{tot}$',0.59],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_2_3_pr = {"Fmax":['$F_{max}$' , 1.289, 2.219],
           "A1":['$A_{cr}$', 0.003, 2.969],
           'A2':['$A_{supcr}$',0.0033, 3.094],
           'Asum':['$A_{tot}$',0.02, 2.461],
           'Wcr':['$W_{cr}$' ,4.297, 0.646],
           'w_max':['$W_{max}$' ,36.01, 0.037],
           'ka1':['$K_{cr}$',0.67],
           'ksum':['$K_{tot}$',0.63],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           'D':['D',0.0002, 7.868],
           'E':['E',2*10**6, 4.985],
           'r':['r',0.012, 1.967]
           }
dic_apr_2_3_1kan = {"Fmax":['$F_{max}$' , 0.496, 2.247],
           "A1":['$A_{cr}$', 0.0006, 3.123],
           'A2':['$A_{supcr}$',0.0021, 2.847],
           'Asum':['$A_{tot}$',0.0021, 3.077],
           'Wcr':['$W_{cr}$' ,1.302, 0.967],
           'w_max':['$W_{max}$' ,4.888, 0.912],
           'ka1':['$K_{cr}$',0.705],
           'ksum':['$K_{tot}$',0.71],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_2_3_2kan = {"Fmax":['$F_{max}$' , 0.227, 2.537],
           "A1":['$A_{cr}$', 0.0021, 2.486],
           'A2':['$A_{supcr}$',0.0008, 3.229],
           'Asum':['$A_{tot}$',0.0023, 2.957],
           'Wcr':['$W_{cr}$' ,10.383, 0.054],
           'w_max':['$W_{max}$' ,22.54, 0.204],
           'ka1':['$K_{cr}$',0.7],
           'ksum':['$K_{tot}$',0.68],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_2_3_3kan = {"Fmax":['$F_{max}$' , 1.062, 1.749],
           "A1":['$A_{cr}$', 0.0004, 3.328],
           'A2':['$A_{supcr}$',0.029, 1.231],
           'Asum':['$A_{tot}$',0.0075, 2.273],
           'Wcr':['$W_{cr}$' ,1.085, 1.259],
           'w_max':['$W_{max}$' , 8.973, 0.564],
           'ka1':['$K_{cr}$',0.74],
           'ksum':['$K_{tot}$',0.72],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_3_3_pr = {"Fmax":['$F_{max}$' , 8.938, 1.086],
           "A1":['$A_{cr}$', 0.0098, 2.114],
           'A2':['$A_{supcr}$',0.0185, 1.875],
           'Asum':['$A_{tot}$',0.028, 1.979],
           'Wcr':['$W_{cr}$' ,5.663, 0.478],
           'w_max':['$W_{max}$' ,14.782, 0.373],
           'ka1':['$K_{cr}$',0.71],
           'ksum':['$K_{tot}$',0.72],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           'D':['D',0.156, 3.947],
           'E':['E',9*10**9, 0.052],
           'r':['r',0.109, 0.729]
           }
dic_apr_3_3_1kan = {"Fmax":['$F_{max}$' , 0.127, 2.675],
           "A1":['$A_{cr}$', 0.0003, 3.292],
           'A2':['$A_{supcr}$',0.003, 2.43],
           'Asum':['$A_{tot}$',0.0022, 2.789],
           'Wcr':['$W_{cr}$' ,5.771, 0.349],
           'w_max':['$W_{max}$' ,135.48, -0.622],
           'ka1':['$K_{cr}$',0.74],
           'ksum':['$K_{tot}$',0.75],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_3_3_2kan = {"Fmax":['$F_{max}$' , 0.0235, 3.347],
           "A1":['$A_{cr}$', 0.0003, 3.213],
           'A2':['$A_{supcr}$',0.0005, 3.156],
           'Asum':['$A_{tot}$',0.0007, 3.229],
           'Wcr':['$W_{cr}$' ,9.007, 0.156],
           'w_max':['$W_{max}$' ,66.61, -0.334],
           'ka1':['$K_{cr}$',0.7],
           'ksum':['$K_{tot}$',0.735],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_3_3_3kan = {"Fmax":['$F_{max}$' , 0.0069, 3.845],
           "A1":['$A_{cr}$', 4*10**-6, 5.002],
           'A2':['$A_{supcr}$',5*10**-5, 4.017],
           'Asum':['$A_{tot}$',4*10**-5, 4.426],
           'Wcr':['$W_{cr}$' ,1.645, 0.895],
           'w_max':['$W_{max}$' , 38.51, -0.125],
           'ka1':['$K_{cr}$',0.73],
           'ksum':['$K_{tot}$',0.775],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_4_3_pr = {"Fmax":['$F_{max}$' , 8.919, 0.953],
           "A1":['$A_{cr}$', 0.022, 1.63],
           'A2':['$A_{supcr}$',0.011, 1.942],
           'Asum':['$A_{tot}$',0.0267, 1.845],
           'Wcr':['$W_{cr}$' ,5.835, 0.444],
           'w_max':['$W_{max}$' ,3.275, 0.946],
           'ka1':['$K_{cr}$',0.78],
           'ksum':['$K_{tot}$',0.82],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           'D':['D',268.7, 0.679],
           'E':['E',2*10**12, -2.188],
           'r':['r',0.407, 0.17]
           }
dic_apr_4_3_1kan = {"Fmax":['$F_{max}$' , 3.897, 1.071],
           "A1":['$A_{cr}$', 0.0004, 2.87],
           'A2':['$A_{supcr}$',0.0007, 2.814],
           'Asum':['$A_{tot}$',0.0012, 2.827],
           'Wcr':['$W_{cr}$' ,0.872, 1.094],
           'w_max':['$W_{max}$' ,2.433, 1.016],
           'ka1':['$K_{cr}$',0.79],
           'ksum':['$K_{tot}$',0.84],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_4_3_2kan = {"Fmax":['$F_{max}$' , 2.484, 1.213],
           "A1":['$A_{cr}$', 0.107, 0.717],
           'A2':['$A_{supcr}$',0.0002, 3.248],
           'Asum':['$A_{tot}$',0.0054, 2.204],
           'Wcr':['$W_{cr}$' ,189.1, -0.983],
           'w_max':['$W_{max}$' ,5.87, 0.681],
           'ka1':['$K_{cr}$',0.815],
           'ksum':['$K_{tot}$',0.835],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
dic_apr_4_3_3kan = {"Fmax":['$F_{max}$' , 0.0017, 4.069],
           "A1":['$A_{cr}$', 4*10**-7, 5.502],
           'A2':['$A_{supcr}$',3*10**-5, 4.07],
           'Asum':['$A_{tot}$',2*10**-5, 4.494],
           'Wcr':['$W_{cr}$' ,1.015, 1.004],
           'w_max':['$W_{max}$' , 22.48, 0.194],
           'ka1':['$K_{cr}$',0.76],
           'ksum':['$K_{tot}$',0.78],
           'ka2':['$K_{supcr}$',0],
           'A1_A':['$A_{cr}$/$A_{tot}$',0],
           'wsup':['$W_{supcr}$' ,0],
           }
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
        float(data[-3,0])
        float(data[-10,0])
        
    except ValueError:
        file= open(path, encoding= 'ansi')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
        data = pd.read_csv(file,sep='\s+' ,decimal=",",names = ['h', 'h_pr', 'Fmax', 'Wcr', 'A1', 'A2', 'Asum','ka1','ka2', 'ksum', 'A1_A','w_max', 'wsup', 'D', 'E', 'r'] ) #Читаю из тестового документа в качестве сепаратора: все пробелы
        data = data.sort_values(by=['h'])
    #у переменной data сейчас тип dataframe
    file.close()
    return data


def aproximation(data, label):
    x = 1
    y = 1.25
    global dic   
    global apr
    global ax
    
    if len(dic[label]) == 2:
        ax.plot(data , np.ones(len(data))*dic[label][-1], c = 'r', label = str(dic[label][0]) + '=' + str(dic[label][-1]))
        legend = ax.legend(bbox_to_anchor=(x, y),frameon = 1,loc='upper right')
        frame = legend.get_frame()
        frame.set_color('white')
    else:
        apr = np.array(dic[label][-2] * data ** dic[label][-1])
        
        if label == 'E':
            ax.plot(data, dic[label][-2] * data ** dic[label][-1], c ='r', label = str(dic[label][0]) + '=' + '%s×10$^{%i}h_{fr}^{%s}$'%(str(dic[label][-2]/10**ten),ten,str(dic[label][-1])))
            legend = ax.legend(bbox_to_anchor=(x, y),frameon = 1,loc='upper right')
            frame = legend.get_frame()
            frame.set_color('white')
        else:
            ax.plot(data, dic[label][-2] * data ** dic[label][-1], c ='r', label = str(dic[label][0]) + '=' + '%s$h_{fr}^{%s}$'%(str(dic[label][-2]),str(dic[label][-1])))
            legend = ax.legend(bbox_to_anchor=(x, y),frameon = 1,loc='upper right')
            frame = legend.get_frame()
            frame.set_color('white')
    
    
dic = dic_apr_4_3_3kan
filename=askopenfilename() #Вызов окна открытия файла
data=open_datafile(filename) #Открыть файл
if 'канал' in filename:
    labels = ['Fmax', 'Wcr', 'A1', 'A2', 'Asum','ka1','ka2', 'ksum', 'A1_A','w_max', 'wsup']
else:    
    labels = ['Fmax', 'Wcr', 'A1', 'A2', 'Asum','ka1','ka2', 'ksum', 'A1_A','w_max', 'wsup', 'D', 'E', 'r']

for i in labels:
    if i == 'E':
        ten = np.log10(dic['E'][-2])//1
    fig, ax = plt.subplots(num = filename.split('/')[-1][:-4] + '_' + str(range_names[i]) + '_'  + i,figsize=(3.2, 2.4))
    fig.subplots_adjust(left = 0.2,right = 0.95, top = 0.865, bottom = 0.18)
    h_apr = np.arange(data['h'].min(), data['h'].max(), 0.1)
    # if dic[i][-1] != 0:
    #     aproximation(h_apr, i)
        # locs, labels = plt.yticks()
        # ax.clear()
    
    data.plot(ax = ax, x = 'h', y = i, kind = "scatter") 
    
    if dic[i][-1] != 0:
        aproximation(h_apr, i)
        
    
    ax.set_xlabel('$h_{fr}$, мм')
    ax.set_ylabel(label_dic[i])

    left, right = plt.xlim()
    

    if i in ['ka1','ka2', 'ksum', 'A1_A']:
        ax.set_xlim(0, int(right)+1)   
        ax.set_ylim(0, 1)
    else:
        ax.set_xlim(0, int(right)+1)
        ax.set_ylim(0, apr[np.argmax(apr)]*1.2)
        locs, labels = plt.yticks()
        ax.set_ylim(0, locs[-1])
       
        # index = (np.abs(locs-apr[-1])).argmin()
        
        # if locs[index] != locs[-1]:
        #     ax.set_ylim(0, locs[index+1])
            
        # else:
        #     ax.set_ylim(0, locs[-1])
        
    if i in ['D', 'E']:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        
    ax.xaxis.set_major_locator(mtick.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(1))
    plt.grid()
    folder = os.path.dirname(filename)
    plt.savefig(folder +'/' +filename[:-4].split(' ')[-2].split('/')[-1] + '_' +filename[:-4].split(' ')[-1]  + '_' + str(range_names[i]) + '_'  + i + '.png', dpi = 300)
# plt.show()  