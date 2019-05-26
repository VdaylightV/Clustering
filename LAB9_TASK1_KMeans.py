# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:10:42 2019

@author: Administrator
"""
#调库

import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

#导入数据并降维
filename = 'E:/大学课程/AI程序设计/实验部分/实验9 聚类-关联-异常/实验课聚类关联分析/house-votes-84.csv'
data_origin = pd.read_csv(filename, engine = 'python')

#导入数据

data_origin_matrix = data_origin.values
label_lst = []
for item in data_origin_matrix:
    if item[0]=='republican':
        label_lst.append(0)#共和党为0
    else:
        label_lst.append(1)#民主党为1
        
#把原始的标签提取出来
data_product=data_origin.drop('A',axis=1)
data_product_matrix = data_product.values
pri_X = data_product_matrix

lst_extern = []

for item in pri_X:
    lst_essen = []
    for member in item:
        if member == 'n':
            lst_essen.append(-1)
        elif member == 'y':
            lst_essen.append(1)
        else:
            lst_essen.append(0)
    lst_extern.append(lst_essen)

X = np.mat(lst_extern)

#把投票结果‘n’，‘y’，‘?’，分别用 −1 ， 1 ， 0 代替

pca = PCA(n_components = 2)
reduced_X = pca.fit_transform(X)

X_lst = []
Y_lst = []

for item in reduced_X:
    X_lst.append(item[0])
    Y_lst.append(item[1])
    
#数据降维，降成3维
    
clf = KMeans(n_clusters=2)
y_pred = clf.fit_predict(reduced_X)

#进行聚类

ax = plt.figure().add_subplot(111)
ax.scatter(X_lst,Y_lst, c=y_pred,marker='x')

#把聚类结果可视化

republic_x = []
republic_y = []

democrate_x = []
democrate_y = []

republic_x_pred = []
republic_y_pred = []

democrate_x_pred = []
democrate_y_pred = []

for i in range(435):
    if label_lst[i] == 0:
        republic_x.append(reduced_X[i][0])
        republic_y.append(reduced_X[i][1])
    else:
        democrate_x.append(reduced_X[i][0])
        democrate_y.append(reduced_X[i][1])

for i in range(435):
    if y_pred[i] == 0:
        republic_x_pred.append(reduced_X[i][0])
        republic_y_pred.append(reduced_X[i][1])
    else:
        democrate_x_pred.append(reduced_X[i][0])
        democrate_y_pred.append(reduced_X[i][1])
        
#绘制原来真实的类别图
        
ax = plt.figure().add_subplot(111)
ax.scatter(republic_x,republic_y,c='blue',marker='x')
ax.scatter(democrate_x,democrate_y,c='red',marker='x')

#下面尝试对该聚类结果进行评估

#首先找到两个聚类的中心
centroids=clf.cluster_centers_

#定义平面上两点求距离的函数
def dist(M,N):
    distance = math.sqrt((pow(M[0]-N[0],2)+pow(M[1]-N[1],2)))
    return distance

C_republic = len(republic_x_pred)#236
C_democrate = len(democrate_x_pred)#199

republic_coordinate_set = []
for i in range(C_republic):
    single_coordinate = []
    single_coordinate.append(republic_x_pred[i])
    single_coordinate.append(republic_y_pred[i])
    republic_coordinate_set.append(single_coordinate)
    
democrate_coordinate_set = []
for i in range(C_democrate):
    single_coordinate = []
    single_coordinate.append(democrate_x_pred[i])
    single_coordinate.append(democrate_y_pred[i])
    democrate_coordinate_set.append(single_coordinate)

sum_distance_republic = 0
for i in range(C_republic):
    for j in range(C_republic):
        if j>i:
            sum_distance_republic=sum_distance_republic+dist(republic_coordinate_set[i],republic_coordinate_set[j])
avg_republic_C = 2/((C_republic-1)*C_republic)*sum_distance_republic

sum_distance_democrate = 0
for i in range(C_democrate):
    for j in range(C_democrate):
        if j>i:
            sum_distance_democrate=sum_distance_democrate+dist(democrate_coordinate_set[i],democrate_coordinate_set[j])
avg_democrate_C = 2/((C_democrate-1)*C_democrate)*sum_distance_democrate

republic_distance_max = 0
for i in range(C_republic):
    for j in range(C_republic):
        if j>i:
            if dist(republic_coordinate_set[i],republic_coordinate_set[j]) > republic_distance_max:
                republic_distance_max = dist(republic_coordinate_set[i],republic_coordinate_set[j])

democrate_distance_max = 0
for i in range(C_democrate):
    for j in range(C_democrate):
        if j>i:
            if dist(democrate_coordinate_set[i],democrate_coordinate_set[j]) > democrate_distance_max:
                democrate_distance_max = dist(democrate_coordinate_set[i],democrate_coordinate_set[j])
                
distance_between_clusters = 200
for i in range(C_republic):
    for j in range(C_democrate):
        if dist(republic_coordinate_set[i],democrate_coordinate_set[j]) < distance_between_clusters:
            distance_between_clusters = dist(republic_coordinate_set[i],democrate_coordinate_set[j])

distance_between_two_centers = dist(centroids[0],centroids[1])

#下面检测真实的分类指标：
C_republic_real = len(republic_x)#168
C_democrate_real = len(democrate_x)#267

republic_coordinate_real_set = []
for i in range(C_republic_real):
    single_coordinate = []
    single_coordinate.append(republic_x[i])
    single_coordinate.append(republic_y[i])
    republic_coordinate_real_set.append(single_coordinate)
    
democrate_coordinate_real_set = []
for i in range(C_democrate_real):
    single_coordinate = []
    single_coordinate.append(democrate_x[i])
    single_coordinate.append(democrate_y[i])
    democrate_coordinate_real_set.append(single_coordinate)

sum_distance_republic_real = 0
for i in range(C_republic_real):
    for j in range(C_republic_real):
        if j>i:
            sum_distance_republic_real=sum_distance_republic_real+dist(republic_coordinate_real_set[i],republic_coordinate_real_set[j])
avg_republic_C_real = 2/((C_republic_real-1)*C_republic_real)*sum_distance_republic_real

sum_distance_democrate_real = 0
for i in range(C_democrate_real):
    for j in range(C_democrate_real):
        if j>i:
            sum_distance_democrate_real=sum_distance_democrate_real+dist(democrate_coordinate_real_set[i],democrate_coordinate_real_set[j])
avg_democrate_C_real = 2/((C_democrate_real-1)*C_democrate_real)*sum_distance_democrate_real

republic_distance_max_real = 0
for i in range(C_republic_real):
    for j in range(C_republic_real):
        if j>i:
            if dist(republic_coordinate_real_set[i],republic_coordinate_real_set[j]) > republic_distance_max_real:
                republic_distance_max_real = dist(republic_coordinate_real_set[i],republic_coordinate_real_set[j])
                
democrate_distance_max_real = 0
for i in range(C_democrate_real):
    for j in range(C_democrate_real):
        if j>i:
            if dist(democrate_coordinate_real_set[i],democrate_coordinate_real_set[j]) > democrate_distance_max_real:
                democrate_distance_max_real = dist(democrate_coordinate_real_set[i],democrate_coordinate_real_set[j])

distance_between_clusters_real = 200
for i in range(C_republic_real):
    for j in range(C_democrate_real):
        if dist(republic_coordinate_real_set[i],democrate_coordinate_real_set[j]) < distance_between_clusters_real:
            distance_between_clusters_real = dist(republic_coordinate_real_set[i],democrate_coordinate_real_set[j])    
print(distance_between_clusters_real)     