# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

class ScanData(object):
    def __init__(self):
        """Constructor"""
        self.imoff = 0
        self.impix = 0
        self.imsize = 0
        self.fil_path = ''
        self.imfile = ''
        self.channel = ''
        self.imdirection = ''
        self.center_point = []
        self.search_window = []
        self.feature_window = []
        self.specrange = []
        
    def update_scan_conditions(self, imoff, impix, imsize):
        self.imoff = imoff
        self.impix = impix
        self.imsize = imsize
    
    def update_search_conditions(self, center_point, search_window, feature_window, specrange):
        self.center_point = center_point
        self.search_window = search_window
        self.feature_window = feature_window
        self.specrange = specrange

    def get_search_conditions(self):
        return self.center_point, self.search_window, self.feature_window, self.specrange
    
    def get_scan_conditions(self):
        return self.imoff, self.impix, self.imsize

    def get_file_info(self):
        return self.fil_path, self.imfile, self.channel, self.imdirection

    def update_file_info(self,fil_path, imfile, channel, imdirection):
        self.fil_path = fil_path
        self.imfile = imfile
        self.channel = channel
        self.imdirection = imdirection

class SpecCounter(object):
    def __init__(self):
        """Constructor"""
        self.counter = 0
        self.maxvalue = 0
        self.driftvar = 0
        
    def incr_counter(self):
        self.counter += 1
        
    def get_counter(self):
        return self.counter
    
    def reset_counter(self):
        self.counter = 0
    
    def update_maxcnt(self, maxvalue):
        self.maxvalue = maxvalue
    
    def update_driftvar(self, driftvar):
        self.driftvar = driftvar

    def get_driftvar(self):
        return self.driftvar

    def get_maxvalue(self):
        return self.maxvalue

class PointList(object):
    def __init__(self):
        """Constructor"""
        self.plist = []
    
    def add_pnt(self, point):
        self.plist.append(point)

    def get_list(self):
        return self.plist

class ImageInfo(object):
    def __init__(self, image):
        """Constructor"""
        self.image = image

    def update_img(self,new_image):
        self.image = new_image
    
    def get_image(self):
        return self.image

