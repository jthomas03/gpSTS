# -*- coding: utf-8 -*-
#
# John C. Thomas 2022 gpSTS - tutorial version

################################################################
######Run random STS collection Tutorial#######################
################################################################

import os
import numpy as np
import Config
import gpcam
gpcam.global_config = Config
import time
from time import strftime
import random
from scipy.interpolate import griddata
from gpcam.misc import delete_files
from gpsts.Interface.mainvis import gpvis
from gpsts.Interface.data_class import takeclosest, RandData
from Config import return_vals
import Config as conf

Vals = return_vals()
read_file, write_file = Vals.get_read_write()
data_path, tar_path = Vals.get_path()
dfile = Vals.get_testfile()
dpix, drange, spix = Vals.get_pix()
cfil, rfil = Vals.get_read_write()
drun = Vals.get_run()

write_file = data_path+'/new_experiment_command/'+cfil
read_file = data_path+'/new_experiment_result/'+rfil

vispath = data_path+'/current_data/'
dvispath = vispath+'Data_model_1.npy'

def random_main():
    """
    The main loop

    Parameters
    ----------
    path_new_experiment_command : Path
        Full path to where to look to read data

    path_new_experiment_result : Path
        Full path to where to write new commands
    """
    #########################################
    ######Prepare first set of Experiments###
    #########################################
    print("################################################")
    print("################################################")
    print("#################Version: 0.1###################")
    print("################################################")
    print("")
    start_time = time.time()
    start_date_time = strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    print("Date and time:       ", start_date_time)
    print("################################################")
    #########Initialize a set of random measurements####
    #########this will be the initial experiment data###
    print("################################################")
    print("Initializing data set...")
    print("################################################")
    delete_files()
    data = RandData(conf)
    print("################################################")
    print("Beginning random data collection...")
    print("################################################")
    
    data.write_command(write_file)
    ###############################################
    ###Begin Random collection loop################
    ###############################################
    test = 0
    while test < conf.likelihood_optimization_max_iter:
        print("Waiting for experiment device to read and subsequently delete last command.")
        time.sleep(2)
        if os.path.isfile(read_file):
            print("result received")
            a = np.load(read_file, encoding="ASCII", allow_pickle=True)
            data.read_data(a)
            data.update_file(dvispath)
            #for entry in a:
            #    if entry["measured"] == True: continue
            #    entry["measured"] = True
            #    xx = int(round(entry['position']['x1']))
            #    yy = int(round(entry['position']['x2']))
            #    entry["measurement values"]["values"] = np.array([sum(dtout[xx][yy][lpix:upix])]) 
            #    entry["measurement values"]["value positions"] = np.array([0])
            #    entry['measured'] = True
            os.remove(read_file)
            data.write_command(write_file)
            out = data.get_dlen()
            print(out)
            #np.save(write_file, a)
            #if os.path.isfile(vispath+'Data_model_1.npy'):
            #    mainvis(data_path = vispath+'Data_model_1.npy', ind=ind)
            #ind += 1
            #np.save(data_path+'/impath/'+drun+'/ind.npy',ind)
            print("command written")
            test += 1
    out = data.get_data()      
    print(out)