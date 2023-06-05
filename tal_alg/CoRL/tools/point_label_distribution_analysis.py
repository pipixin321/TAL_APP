import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data_path="/mnt/data1/zhx/dataset/THUMOS14/point_gaussian/point_labels.csv"
# data_path="/mnt/data1/zhx/dataset/THUMOS14/point_labels/point_labels_gaussian.csv"
data_path="/mnt/data1/zhx/dataset/THUMOS14/point_labels/point_labels_manual4.csv"
# data_path='/mnt/data1/zhx/dataset/ActivityNet13/point_labels/point_labels_gaussian.csv'
data=pd.read_csv(data_path)
# print(data)
alpha=(data['point']-data['start_frame'])/(data['stop_frame']-data['start_frame'])
print(alpha)
plt.figure()
plt.hist(alpha,bins=500,color='pink',edgecolor='b')
plt.xlim((-0.1,1.1))
plt.xticks(list(np.linspace(0,1,11)))
plt.savefig("/mnt/data1/zhx/Weakly_TAL/baseline/tools/figs/p_dist_manual4.png")