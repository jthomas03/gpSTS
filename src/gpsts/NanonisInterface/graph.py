# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

import time
from time import strftime
import numpy as np
import os
from Config import return_scandata


Vals = return_scandata()
imoff, impix, imsize = Vals.get_scan_conditions()
fil_path, imfile, channel, imdirection = Vals.get_file_info()
imdirectory = fil_path+'data'+'\\'+'impath'

def plot_2d_function(gp_optimizer_obj):
    plot_iput_dim = [1, int(impix[0][0])],[1, int(impix[0][0])]
    resolution = [100,100]
    xlabel = "x_label"
    ylabel = "y_label"
    zlabel = "z_label"
    gp_mean = True
    gp_var = False
    gp_mean_grad = False,
    objective_function = False
    costs = False
    entropy = False
    print("plotting dims:", plot_iput_dim)
    l = [len(plot_iput_dim[i]) for i in range(len(plot_iput_dim))]
    plot_indices = [i for i, x in enumerate(l) if x == 2]
    slice_indices = [i for i, x in enumerate(l) if x == 1]


    ##plot the current model
    import matplotlib.pyplot as plt
    x = np.linspace(plot_iput_dim[0][0],plot_iput_dim[0][1],resolution[0])
    y = np.linspace(plot_iput_dim[1][0],plot_iput_dim[1][1],resolution[0])
    x,y = np.meshgrid(x,y)
    model = np.zeros((x.shape))
    variance = np.zeros((x.shape))
    obj = np.zeros((x.shape))
    cost = np.zeros((x.shape))
    gp_grad = np.zeros((x.shape))
    entropy_array = np.zeros((x.shape))

    points = []
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.zeros((len(plot_iput_dim)))
            point[plot_indices[0]] = x[i,j]
            point[plot_indices[1]] = y[i,j]
            for k in range(len(slice_indices)):
                point[slice_indices[k]] = plot_iput_dim[slice_indices[k]][0]
            points.append(point)

    points = np.asarray(points)
    index = 0
    for i in range(len(x)):
        for j in range(len(y)):
            aa1 = gp_optimizer_obj.gp.posterior_mean(np.array([points[index]]))
            bb1 = gp_optimizer_obj.gp.posterior_covariance(np.array([points[index]]))
            aa = aa1["f(x)"][0]
            bb = bb1["v(x)"][0]
            model[i,j] = aa
            variance[i,j] = bb
            index += 1
    fig = plt.figure(1)
    xx = x[::1]
    yy = y[::-1]
    hb = plt.pcolor(yy, xx,model, cmap='inferno')
    print(model)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("gp mean model function")
    plt.savefig(imdirectory+'\\'+'gpmeanmodel'+'.png', bbox_inches='tight', dpi = 50)
    plt.close()

    fig = plt.figure(1)
    hb = plt.pcolor(yy, xx,variance, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points[:,plot_indices[0]], gp_optimizer_obj.points[:,plot_indices[1]])
    plt.title("gp variance function")
    plt.savefig(imdirectory+'\\'+'gpvariance'+'.png', bbox_inches='tight', dpi = 50)
    plt.close()
