# -*- coding: utf-8 -*-
#
# John C. Thomas 2022 gpSTS tutorial

import time
from time import strftime
import numpy as np
import os
from Config import return_vals
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import Config as conf


Vals = return_vals()
#read_file, write_file = Vals.get_read_write()
data_path, tar_path = Vals.get_path()
#dfile = Vals.get_testfile()
dpix, drange, spix = Vals.get_pix()
exp_name = Vals.get_run()
#imoff, impix, imsize = Vals.get_scan_conditions()
#fil_path, imfile, channel, imdirection = Vals.get_file_info()
imdirectory = data_path+'/impath/'+exp_name+'/'

def plot_2d_function(gp_optimizer_obj,ind):
    px = int(dpix[0])
    py = int(dpix[1])
    plot_iput_dim = [1, px],[1, py]
    resolution = [px,py]
    xlabel = "x_label"
    ylabel = "y_label"
    zlabel = "z_label"
    gp_mean = True
    gp_var = False
    gp_mean_grad = False
    objective_function = False
    costs = False
    entropy = False
    print("plotting dims:", plot_iput_dim)
    l = [len(plot_iput_dim[i]) for i in range(len(plot_iput_dim))]
    plot_indices = [i for i, x in enumerate(l) if x == 2]
    slice_indices = [i for i, x in enumerate(l) if x == 1]


    ##plot the current model
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
    plt.savefig(imdirectory+'gpmeanmodel_'+str(ind)+'.png', bbox_inches='tight', dpi = 50)
    np.save(imdirectory+'model_'+str(ind)+'.npy',model)
    plt.close()

    fig = plt.figure(1)
    hb = plt.pcolor(yy, xx,variance, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points[:,plot_indices[0]], gp_optimizer_obj.points[:,plot_indices[1]])
    plt.title("gp variance function")
    #plt.savefig(imdirectory+'gpvariance_'+str(ind)+'.png', bbox_inches='tight', dpi = 50)
    np.save(imdirectory+'variance_'+str(ind)+'.npy',variance)
    plt.close()
    
def plot_2drand_function(data,ind):
    pointsdg = []
    valuesdg = []
    for idx in range(0,len(data)):
        if data[idx][0]["measured"] == True:
            tmp = []
            out = data[idx][0]["position"]
            for para_name in conf.parameters:
                tmp.append(data[idx][0]["position"][para_name])
            pointsdg.append(tmp)
            valuesdg.append(data[idx][0]["measurement values"]["values"][0] )
    xpix, ypix = dpix[0], dpix[1]
    grid_xdg,grid_ydg=np.mgrid[0:xpix,0:ypix]
    filled_griddg=griddata(pointsdg, valuesdg, (grid_xdg, grid_ydg), method='linear')
   
    ### Filled Empties ###
    test = conf.dfil
    minval = 1e15
    for i in range(0,test.shape[0]):
        for j in range(0,test.shape[1]):
            if np.min(test[j][i]) < minval:
                minval = np.min(test[j][i])
    for i in range(0,xpix):
        for j in range(0,ypix):
            if np.isnan(filled_griddg[j][i]):
                filled_griddg[j][i] = minval
    
    fig = plt.figure(1)
    xg = np.linspace(1, test.shape[0], test.shape[0])
    yg = np.linspace(1, test.shape[1], test.shape[1])
    hb = plt.pcolor(xg,yg,filled_griddg, cmap='inferno')
    plt.title("random interpolated function")
    plt.savefig(imdirectory+'random_'+str(ind)+'.png', bbox_inches='tight', dpi = 50)
    np.save(imdirectory+'random_'+str(ind)+'.npy',filled_griddg)
    plt.close()
    