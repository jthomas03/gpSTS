# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

import sys
import Config
from gpsts.Classification.data_collect import InsertData, read_spec, closestval
import Config as conf
from Config import return_scandata
import numpy as np
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def main_post():
    fil_path = conf.nanonis_config['Nanonis_Settings']['FolderLocation']
    stsx = conf.nanonis_config['Nanonis_Settings']['STSbias']
    stsy = conf.nanonis_config['Nanonis_Settings']['STSsignal']
    srange = conf.nanonis_config['Nanonis_Settings']['SpectralRange']
    expname = conf.nanonis_config['Nanonis_Settings']['ExperimentName']
    savepath = conf.nanonis_config['Nanonis_Settings']['DataLocation']
    srangemin, srangemax = srange[0], srange[1]
    dpath = fil_path+'data'+'\\'
    parameters = return_scandata()
    imoff, impix, imsize = parameters.get_scan_conditions()
    didv_files = []
    for fname in os.listdir(dpath):
        if fname.endswith('.dat') & fname.startswith('dI_dV'):
            didv_files.append(fname)
    sorted(didv_files)
    flist = didv_files
    out1, x1, x2 = read_spec(dpath+flist[0])
    speclen = len(out1)
    ind, indx, indy = 0, 0, 0
    for i in out1[0]:
        if i == stsx:
            indx = ind
        elif i == stsy:
            indy = ind
        ind += 1
    try:
        assert indx != indy
    except Exception as e:
        print("data error")
        raise e
    xpix = int(impix[0][0])
    ypix = int(impix[0][1])
    delete = []
    out_a = np.zeros((xpix,ypix))
    test = []
    test_out = []
    test_outx = []
    test_outsum = []
    init1, x1, x2 = read_spec(dpath+flist[0])
    slen = 0
    for i in range(1,speclen):
        if (float(init1[i][0]) >= srangemin) and (float(init1[i][0]) <= srangemax):
            slen += 1
    out_3d = []
    for j in range(0,len(flist)-1):
        out1, x1, x2 = read_spec(dpath+flist[j])
        x1a = float(x1.split(' ')[1])
        x2a = float(x2.split(' ')[1])
        di = []
        dv = []
        for i in range(1,speclen):
            if (float(out1[i][0]) >= srangemin) and (float(out1[i][0]) <= srangemax): 
                dv.append(float(out1[i][indx]))
                di.append(float(out1[i][indy]))
        if x1a >= xpix or x2a >= ypix:
            delete.append(j)
        tmp = np.sum(di)
        test.append(tmp)
        test_out.append(di)
        test_outsum.append(tmp)
        xx = round(x1a) - 1
        yy = round(x2a) - 1
        out_a[yy][xx] = tmp
        tmp3d = []
        tmp3d.append(xx)
        tmp3d.append(yy)
        tmp3d.append(di)
        out_3d.append(tmp3d)
    ref = out_a.copy()
    points=np.where(out_a != 0)
    values=ref[points]
    grid_x,grid_y=np.mgrid[0:xpix,0:ypix]
    outmin = np.min(values)
    filled_grid2=griddata(points, values, (grid_x, grid_y), method='linear')
    tout = closestval(filled_grid2[0][0],test_outsum)
    itemindex = np.where(test_outsum == tout)

    out_a3 = np.zeros(((xpix),(ypix),(slen)))
    for i in range(0,filled_grid2.shape[0]):
        for j in range(0,filled_grid2.shape[1]):
            tout = closestval(filled_grid2[j][i],test_outsum)
            itemindex = np.where(test_outsum == tout)
            out_a3[j][i] = test_out[itemindex[0][0]]

    out_a5 = np.zeros(((xpix),(ypix)))
    for i in range(0,out_a3.shape[0]):
        for j in range(0,out_a3.shape[1]):
            tmp = np.sum(out_a3[j][i][:])#[:])
            out_a5[j][i] = tmp
    fig, ax = plt.subplots()
    ax.imshow(np.flipud(out_a5))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.show()
    ename = expname.replace(' ',"_")
    try:
        np.save(savepath+ename+'_3dsparse.npy',out_3d)
        np.save(savepath+ename+'_2dsparse.npy',out_a)
        np.save(savepath+ename+'_3dinter.npy',out_a3)
        np.save(savepath+ename+'_2dinter.npy',out_a5)
        print("Files saved in: "+str(savepath))
    except:
        print("Files failed to save: "+str(savepath))
