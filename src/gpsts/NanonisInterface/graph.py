# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

import time
from time import strftime
import numpy as np
#import Config
#global_config = Config
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
    #cb = fig.colorbar(hb)
    #cb.set_label(zlabel)
    plt.savefig(imdirectory+'\\'+'gpmeanmodel'+'.png', bbox_inches='tight', dpi = 50)
    plt.close()

    fig = plt.figure(1)
    hb = plt.pcolor(yy, xx,variance, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points[:,plot_indices[0]], gp_optimizer_obj.points[:,plot_indices[1]]) #0,1
    plt.title("gp variance function")
    #cb = fig.colorbar(hb)
    plt.savefig(imdirectory+'\\'+'gpvariance'+'.png', bbox_inches='tight', dpi = 50)
    plt.close()

def plot_2d_function_ind(gp_optimizer_obj, ind):
    plot_iput_dim = [1, 128],[1, 128]#[[1,21590],[1,21590]] ##to alter
    resolution = [128,128]
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
            #res = gp_optimizer_obj.gp.compute_posterior_fvGP_pdf(np.array([points[index]]), np.array([[0]]),
            #compute_posterior_covariances = True)
            #aa = res["posterior means"][0]
            #bb = res["posterior covariances"][0]
            aa1 = gp_optimizer_obj.gp.posterior_mean(np.array([points[index]]))
            bb1 = gp_optimizer_obj.gp.posterior_covariance(np.array([points[index]]))
            aa = aa1["f(x)"][0]
            bb = bb1["v(x)"][0]
            model[i,j] = aa
            variance[i,j] = bb
            index += 1
    fig = plt.figure(1, clear=True)
    xx = x[::1]
    yy = y[::-1]
    hb = plt.pcolor(yy, xx,model, cmap='inferno')
    print(model)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("gp mean model function")
    cb = fig.colorbar(hb)
    cb.set_label(zlabel)
    fig.savefig('movie/'+'gp_mean_model_fcn_'+str(ind)+'.png',dpi=fig.dpi)
    np.save('movie/'+'model_'+str(ind)+'.npy',model)
    plt.close()

    fig = plt.figure(1, clear=True)
    #xx = x[::-1]
    #yy = y[::-1]
    hb = plt.pcolor(yy, xx,variance, cmap='inferno')
    #print(variance)
    #print(variance.shape)
    #print(model.shape)
    #input()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points[:,plot_indices[0]], gp_optimizer_obj.points[:,plot_indices[1]]) #0,1
    plt.title("gp variance function")
    cb = fig.colorbar(hb)
    #plt.gca().invert_xaxis()
    fig.savefig('movie/'+'gp_variance_fcn_'+str(ind)+'.png',dpi=fig.dpi)
    np.save('movie/'+'variance_'+str(ind)+'.npy',variance)
    plt.close()
    #plt.show()

    fig = plt.figure(2, clear = True)
    z = []
    dmin = np.min(variance)
    dmax = np.max(variance)
    var = variance.flatten()
    for i in range(0,len(var)):
        z.append((var[i]-dmin)/(dmax-dmin))
    ax = plt.axes(projection='3d')
    ax.scatter3D(yy,xx,variance,c=z,cmap=plt.get_cmap('magma'))
    ax.set_axis_off()
    fig.savefig('movie/'+'gp_fig_fcn_'+str(ind)+'.png',dpi=fig.dpi)
    plt.close()
    #plt.show()


    #dataload = np.load('200226_154605_hyperspec_cl_se.npy', allow_pickle=True)
    #data_plot(coffee)
    #for i in coffee[:,0]:
    #    print(i)
    #dataload = np.load('../../../nanonis_proc/img/savedimg/goodtiptest.npy')
    def cofplot(data_array):
        fig = plt.figure(3)
        ax = Axes3D(fig)
        ax.view_init(85, 90)
        ax.set_title("test")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.scatter(data_array[:,0],data_array[:,1],data_array[:,2], marker='.', s=50)
        ax.scatter(gp_optimizer_obj.points[:,plot_indices[0]], gp_optimizer_obj.points[:,plot_indices[1]])
        plt.show()
    def image_plot(imagein):
        fig, ax = plt.subplots()
        z_min, z_max = imagein.min(), imagein.max()
        xx, yy = np.meshgrid(np.linspace(1, imagein.shape[0], imagein.shape[1]), np.linspace(1, imagein.shape[0], imagein.shape[1]))
        x = xx[::-1]
        y = yy[::]
        cout = ax.pcolormesh(x, y, imagein, cmap='plasma', vmin=z_min, vmax=z_max)
        plt.scatter(gp_optimizer_obj.points[:,plot_indices[0]], gp_optimizer_obj.points[:,plot_indices[1]], color='blue')
        ax.set_title('Nanonis Test Image')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(cout, ax=ax)
        plt.show()
    def image_plot2(imagein):
        fig, ax = plt.subplots()
        z_min, z_max = imagein.min(), imagein.max()
        xx, yy = np.meshgrid(np.linspace(1, imagein.shape[0], imagein.shape[1]), np.linspace(1, imagein.shape[0], imagein.shape[1]))
        x = xx[::-1]
        y = yy[::-1]
        cout = ax.pcolormesh(x, y, imagein, cmap='plasma', vmin=z_min, vmax=z_max)
        plt.scatter(gp_optimizer_obj.points[:,plot_indices[0]]/168.671875, gp_optimizer_obj.points[:,plot_indices[1]]/168.671875, color='blue')
        ax.set_title('CLSEM Test Image')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        fig.colorbar(cout, ax=ax)
        plt.show()
    def cmesh(data,l1,l2):
        dout = []
        idx = 0
        for i in range(0,l1):
            tmp = []
            for j in range(0,l2):
                tmp.append(data[idx,2])
                idx += 1
            dout.append(tmp)
        return np.array(dout)
    #datatest = cmesh(dataload,128,128)
    #image_plot2(datatest)
    #image_plot(dataload)