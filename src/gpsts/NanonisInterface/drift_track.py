# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

import numpy as np

class DriftTrack(object):
    def __init__(self, im1, im2):
        """Constructor"""
        self.im1 = im1
        self.im2 = im2
        self.offx = 0
        self.offy = 0

    def corr_matrix(self, maxc1, maxr1, ws, ps):
        self.corr = []
        self.corrmax = 0
        self.ind = 0
        i1ext = np.pad(self.im1, [ws+ps,ws+ps], 'symmetric')
        i2ext = np.pad(self.im2, [ws+ps,ws+ps], 'symmetric')
        rows, cols = self.im1.shape[0], self.im1.shape[1]
        maxc1ext = np.array(maxc1)
        maxc1ext = maxc1ext + ws + ps
        maxr1ext = np.array(maxr1)
        maxr1ext = maxr1ext + ws + ps
        def corr_vec(xx, yy, ll):
            return np.sum((xx-np.mean(xx, axis=0))*(yy-np.mean(yy, axis=0)), axis=0)/((ll-1)*np.std(xx, axis=0)*np.std(yy, axis=0))
        cc = maxc1[0]
        rr = maxr1[0]
        t1 = [maxr1ext[0], maxc1ext[0]]
        tmp1 = i1ext[t1[1]-ps:t1[1]+ps,t1[0]-ps:t1[0]+ps]
        patch1 = tmp1.flatten()
        cnt = 0
        for col in range(-ws,ws):
            for row in range(-ws,ws):
                cnt += 1
                if ((rr + row < 1) or (rr+row>self.im1.shape[0])):
                    self.corr.append([col,row,cnt,-10])
                else:
                    if ((cc+col<1) or (cc+col>self.im1.shape[1])):
                        self.corr.append([col,row,cnt,-10])
                    else:
                        t2 = np.array((1,2))
                        t2[0], t2[1] = t1[0]+row, t1[1]+col
                        tmp2 = i2ext[t2[1]-ps:t2[1]+ps,t2[0]-ps:t2[0]+ps]
                        patch2 = tmp2.flatten()
                        corrd = corr_vec(patch1,patch2,(2*ps+1)**2)
                        self.corr.append([col,row,cnt,corrd])
        for i in range(0,len(self.corr)):
            if self.corr[i][3] > self.corrmax:
                self.corrmax = self.corr[i][3]
                self.ind = i
        self.offx = self.corr[self.ind][1]
        self.offy = self.corr[self.ind][0]
        
    def get_maxcorr(self):
        return self.offx, self.offy      
    
    def get_corrdata(self):
        return self.corr

    def update_im1(self, im1):
        self.im1 = im1

    def update_im2(self, im2):
        self.im2 = im2          

