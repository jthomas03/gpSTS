# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS
def read_spec(fil):
    data = []
    x1 = ''
    x2 = ''
    with open(fil,'r') as f:
        x1 = f.readline()
        x2 = f.readline()
        idx = 0
        for i in f:
            tmp = []
            if '[DATA]' in i:
                idx = 1
            elif idx ==1:
                idx = 2
            if idx == 2:
                out = i.split('\t')
                for j in out:
                    if '\n' in j:
                        sp = j.split('\n')
                        tmp.append(sp[0])
                    else:
                        tmp.append(j)
                data.append(tmp)
    return data, x1, x2

def closestval(k,data):
    return min(data,key=lambda i:abs(i-k))

class InsertData(object):
    def __init__(self):
        """Constructor"""
        self.didv_files = []
        self.specpath = specpath
    
    def add_path(self, specpath):
        self.specpath = specpath
        
    def update_files(self, specdatai, specdatav, xpix, ypix):
        tmp = []
        tmp.append(xpix)
        tmp.append(ypix)
        tmp.append(specdatai)
        tmp.append(specdatav)
        self.didv_files = specdata.append(tmp)
    
    def print_files(self):
        ind = 0
        for i in self.didv_files:
            ind += 1
        print(ind) 
