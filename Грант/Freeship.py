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



def open_and_parse(path):
    file= open(path)
    data=pd.read_csv(file, sep='\s+' ,decimal=".", names =['X', 'Y', 'Z']) # читаю файл оффсетов freeship
    data = data.dropna()#убираю значения NaN 
    file.close()
    data = np.array(data.values, dtype=float)
    data = np.around(data, decimals = 4) # округление до 4 знака после запятой
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
    

def ice_shirt_sterm(sterm, B=2):
    sterm = np.sort(sterm,axis = 0)
    l = np.sqrt(np.square(sterm[1:,0]-sterm[:-1,0])+np.square(sterm[1:,1]-sterm[:-1,1])+np.square(sterm[1:,2]-sterm[:-1,2]))
    pass


    
filename=askopenfilename()
data = open_and_parse(filename)
ice_shirt_sterm(sterms(data)[20], 4)


#строим корпус:
fig = plt.figure()
ax = fig.gca(projection ='3d')
ax.scatter(data[:, 0],#Координата длины
            data[:, 1],#Координата ширины
            data[:, 2], color = "green")#Координата высоты



ST = np.array(sterms(data)[3:5])
print(ST)
ST = np.vstack([ST[0], ST[1]])
print('\n\n\n\n\n')
ST = data
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ST)
pcd.normals = o3d.utility.Vector3dVector(np.zeros(
    (1, 3)))  # invalidate existing normals

pcd.estimate_normals(fast_normal_computation = False)
pcd.orient_normals_consistent_tangent_plane(20)
nor = np.asarray(pcd.normals)*-1
pcd.normals = o3d.utility.Vector3dVector(nor)

o3d.visualization.draw_geometries([pcd], point_show_normal=True)


distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 5 * avg_dist
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))


o3d.visualization.draw_geometries([bpa_mesh])


# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# alpha = 0.03
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#         pcd, alpha, tetra_mesh, pt_map)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# 


# triangle = points.triangulate(inplace= False, progress_bar=True)
# print(triangle)
# triangle.plot(show_edges=True)
# ST = np.random.permutation(ST)

X = ST[:,0]
Y = ST[:,1]
Z = ST[:,2]
# Y = np.hstack([sterms(data)[13][:,1], sterms(data)[14][:,1], sterms(data)[15], sterms(data)[16][:,1]])

# X, Y = np.meshgrid(X, )


# grid = pv.StructuredGrid(X, Y, Z)
# grid.plot()
# ax.set_xlim(None, 10)
# ax.set_ylim(None, 40)
# ax.set_zlim(None, 40)
# plt.show()