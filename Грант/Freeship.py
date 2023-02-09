# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:30:37 2022

@author: Кирилл
"""

import pandas as pd
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import open3d as o3d
import pymeshfix
from pymeshfix._meshfix import PyTMesh
import pyvista as pv

def open_and_parse(path):
    file= open(path,encoding= 'ansi')
    data=pd.read_csv(file, sep='\s+' ,decimal=".", names =['X', 'Y', 'Z']) # читаю файл оффсетов freeship
    data = data.dropna()#убираю значения NaN 
    file.close()
    data = np.array(data.values, dtype=float)
    data = np.around(data, decimals = 4) # округление до 4 знака после запятой
    return data

def autocad_open(path):
    file= open(path,encoding= 'utf-8')#перекодировка из utf-8 в ansi из за странной ошибки в спайдере
    data=pd.read_csv(file, sep='\s+', names =[ 'Количество',	'Имя',	'X',	'Y',	'Z'])
    data = data.drop(columns =['Количество',	'Имя'])#Читаю из тестового документа в качестве сепаратора: все пробелы
    data = data.reindex(columns =['Z', 'X', 'Y'])
    data = np.array(data.values, dtype=float)
    data = np.around(data, decimals = 4)
    return data


def Parse_ling(path):

    @jit(parallel=True, fastmath=True)
    def enumeration(A):
        data=[]
        i=0
        for row in A:
            if len(row)==3:
                data+=[row]
                i+=1
        return(data)
    
    
    file= open(path)
    values = pd.read_csv(file, header=None) #Читаем txt-файл
    value = [*values[0].str.findall(r"[-+]?\d*\.\d+|\d+")]   #Ищем числа в полученных данных
    return np.array(enumeration(value), dtype=np.float32)   #Ищем среди полученных чисел координаты точек


def sterms(data):
    A = np.unique(data[:,0], return_index = True) #получить индексы уникальных элементов массива X
    return np.split(data, indices_or_sections = A[1])# получаем разбитый массив шпангоутов

    
def cut_waterline(sterm, waterline):
    sterm = np.sort(sterm,axis = 0)
    A = np.where(sterm[:,2] < waterline)[0]# массив координат где Z меньше чем заданая ватерлиния
    inter_Y=np.interp(waterline, sterm[A[-2]:A[-1]+2, 2], sterm[A[-2]:A[-1]+2, 1])#интерполирование
    return np.vstack([sterm[A] ,np.array([sterm[0,0], inter_Y, waterline])]) # соединение массивов
    

def ice_shirt_sterm(sterm, waterline, B):
    sterm = np.sort(sterm,axis = 0) # точки отсортированы от ДП к борту
    sterm = cut_waterline(sterm, waterline)[::-1] # массив точек шпангоута ниже заданной ватерлинии отсортированный от борта к ДП
    #расчет длины ветки шпангоута
    l = np.sqrt(np.square(sterm[1:,0]-sterm[:-1,0])+ np.square(sterm[1:,1]-sterm[:-1,1])+np.square(sterm[1:,2]-sterm[:-1,2]))
    if np.sum(l) < B/2: #если длина шпангоута меньше В/2
        pass # функция возвращает None
    else:
        w = 0
        for i in range(len(l)):
            w += l[i] 
            if w < B/2:
                continue
            else:
              w1 = w - l[i] # длина до ближайшей точки меньшей В/2
              sterm = sterm[:i] # массив длин меньше чем массив координат на единицу,
              #обрезаем по тот член на котором уже больше чем В/2
              break

    return np.vstack([sterm[:-1], sterm[-1] + ((sterm[-2] - sterm[-1])/(w-w1)) * (B/2 - w1)])

            
def o3d_ball_triangle(ST, ball):
    global pcd
    pcd = o3d.geometry.PointCloud() # создание облака точек
    pcd.points = o3d.utility.Vector3dVector(ST) # из Numpy в Open3D
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Зануление нормалей точек
    pcd.estimate_normals(fast_normal_computation = False) # Востановление нормалей
    pcd.orient_normals_consistent_tangent_plane(20) # Автоматическое ориентирование нормалей относительно последовательных касательных плоскостей
    nor = np.asarray(pcd.normals)*-1 # изменение ориентации нормалей
    pcd.normals = o3d.utility.Vector3dVector(nor) # добавление нормалей к облаку точек
    distances = pcd.compute_nearest_neighbor_distance() #вычисления расстояния от точки до ближайшего соседа в облаке точек
    avg_dist = np.mean(distances) # Среднее арифмитическое элементов массива
    radius = ball * avg_dist # Радиус шара
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, width=0, scale=1.2, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    
    # return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2])) # триангуляция методом катющегося шара
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, ball)


def clear_sterm_triangle(mesh):
    """
    Параметры
    ----------
    mesh : Тип: open3d.cpu.pybind.geometry.TriangleMesh Сетка с одной шпацией
        DESCRIPTION.

    Возвращает
    -------
    Тип: np.array
        Массив координат треугольников.

    """
    vert = np.asarray(mesh.vertices) # массив вершин
    trian = np.asarray(mesh.triangles) # массив треугольников
    #найдем уникальныe элементы в трехмерном массиве треугольников с вершинами
    uniq = np.unique(vert[trian][:,:,0][:,0], return_index = False) 
    non_zero = np.nonzero(vert[trian][:,:,0] - uniq[0])[0] # приведем к массиву их 0 и 1, найдем ненулевые элементы 
    args = np.where(np.logical_and(np.bincount(non_zero) > 0, np.bincount(non_zero) < 3)) 

    return trian[args]


def ice_shirt_surface(data, waterline, B):
    """
    Функция превращения координат теоретического чертежа в координаты 
    поверхности облегания льдом

    Parameters
    ----------
    data : TYPE np.ndarray
        массив с коррдинатами точек шпангоутов теоретического чертежа
    waterline :TYPE int, float 
        значение возвышения КВЛ от ОП в метрах
    B : TYPE int, float 
        Ширина судна, м

    Returns
    -------
    TYPE np.ndarray
        Возвращает массив точек поверхност облегания льдом

    """
    ST = np.array(sterms(data)) #разбиваем массив координат на массив шпангоутов
    cut_st = [] #создаем пустой список
    for st in ST:
        #обрезаем каждый шпангоут 
        try:
            cut_st.append(ice_shirt_sterm(st, waterline, B)) 
        except:
            continue
    
    return np.concatenate(cut_st)


def data_miror(data):
    data_r = np.column_stack([data[:,0], data[:,1]*-1, data[:,-1]])
    data_r = data_r[np.where(data_r[:,1] != 0)]
    data = np.vstack([data_r,data])
    return data[data[:,0].argsort()]
    
filename=askopenfilename()
data = open_and_parse(filename)

# data = ice_shirt_surface(data, 7, 15)
data = data_miror(data)
ST = sterms(data)

# i = 80
# data = np.vstack([ST[i], ST[i+1], ST[i+2], ST[i+3], ST[i+4], ST[i+5], ST[i+6], ST[i+7], ST[i+8]])
# # data = autocad_open(filename)

    
bpa_mesh = o3d_ball_triangle(data, 8)
bpa_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([pcd], point_show_normal = True)
bpa_mesh.paint_uniform_color([1., 0., 0.])
o3d.visualization.draw_geometries([bpa_mesh, pcd], mesh_show_wireframe = True, mesh_show_back_face = True, point_show_normal = False)
# o3d.visualization.draw_geometries([bpa_mesh,pcd], mesh_show_wireframe = True, mesh_show_back_face = True, point_show_normal = True)
vert = np.asarray(bpa_mesh.vertices) # массив вершин
faces = np.asarray(bpa_mesh.triangles) # массив треугольников

meshfix = pymeshfix.MeshFix(vert, faces)
meshfix.repair(verbose=True)
vert, faces = meshfix.v, meshfix.f
bpa_mesh.triangles = o3d.utility.Vector3iVector(faces)
bpa_mesh.vertices = o3d.utility.Vector3dVector(vert)

print(o3d.geometry.TriangleMesh.is_watertight(bpa_mesh))
bpa_mesh.paint_uniform_color([1., 0., 0.])
o3d.visualization.draw_geometries([bpa_mesh, pcd], mesh_show_wireframe = True, mesh_show_back_face = True, point_show_normal = False)
        

input()