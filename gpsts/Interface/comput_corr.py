# -*- coding: utf-8 -*-
#
# John C. Thomas 2022 gpSTS tutorial

import numpy as np
import matplotlib.pyplot as plt

class Corr(object):
    def __init__(self, im1, im2):
        """Constructor"""
        assert im1.shape == im2.shape
        self.im1 = im1
        self.im2 = im2
        self.corrlen = ((self.im1.shape[0]*im1.shape[1])+1)
        self.corrs = []
        self.ind = []
    
    def get_corr(self,ind):
        def corr_vec(xx, yy, ll):
            return np.sum((xx-np.mean(xx, axis=0))*(yy-np.mean(yy, axis=0)), axis=0)/((ll-1)*np.std(xx, axis=0)*np.std(yy, axis=0))
        def specnorm(data):
            dmin = np.min(data)
            dmax = np.max(data)
            out = np.zeros(data.shape[0])
            for i in range(0,data.shape[0]):
                out[i] = (data[i] - dmin)/(dmax-dmin) 
            return out
        im1_corr = specnorm(self.im1.flatten())
        im2_corr = specnorm(self.im2.flatten())
        corrd = corr_vec(im1_corr,im2_corr,self.corrlen)
        self.corrs.append(corrd)
        self.ind.append(ind)

    def update_im1(self, im1):
        assert im1.shape == self.im2.shape
        self.im1 = im1
        
    def update_im2(self, im2):
        assert im2.shape == self.im1.shape
        self.im2 = im2

    def get_corr_data(self):
        return self.ind, self.corrs
        
    def plt_corr(self,path, title = ''):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)#, xticks=[], yticks=[])
        ax.set_xlabel('Nsteps', fontsize=16)
        ax.set_ylabel('Correlation', fontsize=16)
        ax.scatter(self.ind,self.corrs,c='c',linewidth=2) #$V_{S}$
        ax.plot(self.ind,self.corrs,c='r',linewidth=1, label = title)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16) # Size here overrides font_prop
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig(path+'.png', dpi=fig.dpi)
        plt.close()
    