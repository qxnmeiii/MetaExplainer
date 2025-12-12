import nibabel as nib

from nilearn.image import resample_to_img
import os
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io
import matplotlib.cm as cm

def huatu(matrix,sstr):

    #matrix=np.load('D:/learning/23/23.7/FBNETGEN-main/matrix/UCLA/inter_learnable_matrix01-27-17-43-00.npy')
    df = pd.read_excel('D:/learning/neurosystem.xlsx')
    idx=df.iloc[:,0].values
    neuro=df.iloc[:,1].values

    new_conn_matrix = np.zeros((184, 184))
    for i, roi1 in enumerate(idx):
        for j, roi2 in enumerate(idx):
            new_conn_matrix[i, j] = matrix[roi1 - 1, roi2 - 1]
    #network=['a']
    #roi_classes = [1, 2, 3,4,5,6,7]
    sorted_indices = np.argsort(neuro)
    sorted_conn_matrix = new_conn_matrix[sorted_indices, :][:, sorted_indices]

    th = np.percentile(sorted_conn_matrix.reshape(-1), 80)
    sorted_conn_matrix[sorted_conn_matrix < th] = 0
    #m=neuro[sorted_indices]
    # scipy.io.savemat('matrix.mat', {'matrix': result})
    # scipy.io.savemat('matrix.mat', {'neuro': neuro})
    plt.imshow(sorted_conn_matrix, cmap="magma", interpolation='nearest')
    plt.plot([0, 183], [41, 41], color='black', linestyle='--', linewidth=1.5)
    plt.plot([41, 41], [0, 183], color='black', linestyle='--', linewidth=1.5)
    plt.plot([0, 183], [62, 62], color='black', linestyle='--', linewidth=1.5)
    plt.plot([62, 62], [0, 183], color='black', linestyle='--', linewidth=1.5)
    plt.plot([0, 183], [82, 82], color='black', linestyle='--', linewidth=1.5)
    plt.plot([82, 82], [0, 183], color='black', linestyle='--', linewidth=1.5)
    plt.plot([0, 183], [104, 104], color='black', linestyle='--', linewidth=1.5)
    plt.plot([104, 104], [0, 183], color='black', linestyle='--', linewidth=1.5)
    plt.plot([0, 183], [119, 119], color='black', linestyle='--', linewidth=1.5)
    plt.plot([119, 119], [0, 183], color='black', linestyle='--', linewidth=1.5)
    plt.plot([0, 183], [140, 140], color='black', linestyle='--', linewidth=1.5)
    plt.plot([140, 140], [0, 183], color='black', linestyle='--', linewidth=1.5)


    plt.xticks([41, 62, 82,104,119,140,183], ['VIS', 'SMN', 'DA','VA','Limbic','FPN','DMN'],fontsize=14)
    plt.yticks([41, 62, 82,104,119,140,183], ['VIS', 'SMN', 'DA','VA','Limbic','FPN','DMN'],fontsize=14)
    # 应用新的刻度和标签
    plt.plot([0, 183], [0, 183], color='blue', linestyle='--', linewidth=1)
    plt.colorbar()

    plt.savefig('hua/NYU_FB/02-22-11-25-05' +sstr +'.png')
    plt.clf()
    #plt.show()

