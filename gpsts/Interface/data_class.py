# -*- coding: utf-8 -*-
#
# John C. Thomas 2022 gpSTS_tutorial
import numpy as np
import uuid
import random
import time
import datetime


def takeclosest(num,collection):
    num = num
    out = min(collection,key=lambda x:abs(x-num))
    ind = 0
    for i in range(0,len(collection)):
        if collection[i] == out:
            ind = i
    return ind

class TestData(object):
    def __init__(self):
        """Constructor"""
        self.dpix = []
        self.fil_path = ''
        self.tpath = ''
        self.dfile = ''
        self.specrange = []
        self.specres = 0
        self.cfile = ''
        self.rfile = ''
        self.run = ''
    
    def get_path(self):
        return self.fil_path, self.tpath
    
    def get_run(self):
        return self.run
    
    def get_testfile(self):
        return self.dfile
    
    def get_pix(self):
        return self.dpix, self.specrange, self.specres
    
    def get_read_write(self):
        return self.cfile, self.rfile

    def update_file_info(self,fil_path, dfile, tpath, dpix, specrange, specres, cfile, rfile, run):
        self.fil_path = fil_path
        self.dfile = dfile
        self.tpath = tpath
        self.dpix = dpix
        self.specrange = specrange
        self.specres = specres
        self.cfile = cfile
        self.rfile = rfile
        self.run = run
        
        
class PointList(object):
    def __init__(self):
        """Constructor"""
        self.plist = []
    
    def add_pnt(self, point):
        self.plist.append(point)

    def get_list(self):
        return self.plist
    
class RandData(object):
    def __init__(self,conf):
        self.conf = conf
        self.data_set = []
        self.idx = 0
        self.visited = []
        
    def get_data(self):
        return self.data_set
    
    def read_data(self,new_data):
        if new_data[0]["measured"] == True:
            self.data_set.append(new_data)
    
    def update_file(self,location):
        np.save(location,self.data_set)
                    
    def write_command(self,location):
        com = []
        def get_command():
            def get_random(lower_limit,upper_limit):
                out = random.uniform(lower_limit, upper_limit)
                return out
            idx = 0
            com = []
            com.append({})
            com[idx]["position"] = {}
            for gp_idx in self.conf.random_process.keys():
                dim = \
                    self.conf.random_process[gp_idx]["dimensionality of return"]
                num = \
                    self.conf.random_process[gp_idx]["number of returns"]
                for para_name in self.conf.parameters:
                    lower_limit = self.conf.parameters[para_name]["element interval"][0]
                    upper_limit = self.conf.parameters[para_name]["element interval"][1]
                    com[idx]["position"][para_name] = get_random(lower_limit,upper_limit)
                com[idx]["cost"] = None
                com[idx]["measurement values"] = {}
                com[idx]["measurement values"]["values"] = np.zeros((num))
                com[idx]["measurement values"]["value positions"] = np.zeros((num, dim))
                com[idx]["time stamp"] = time.time()
                com[idx]["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
                com[idx]["measured"] = False
                com[idx]["id"] = str(uuid.uuid4())
            return com, idx
        write = False
        while write == False:
            if len(self.data_set) == 0:
                com, idx = get_command()
                self.visited.append(com[idx]["position"])
                np.save(location,com)
                write = True
            else:
                com, idx = get_command()
                if com[idx]["position"] not in self.visited:
                    self.visited.append(com[idx]["position"])
                    np.save(location,com)
                    write = True
                else:
                    write = False
        
    def get_dlen(self):
        return len(self.data_set)
        