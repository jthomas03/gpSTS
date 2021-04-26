# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

#################################
##Bias Spectroscopy Collection###
#################################
import numpy as np
import Config
import os
import random
import time
from scipy.interpolate import griddata
import sys
import uuid
from Config import return_cnt, return_pntlist, return_image, return_scandata
from gpsts.NanonisInterface.nanonis_interface import Nanonis
from gpsts.NanonisInterface.drift_track import DriftTrack
from gpsts.NanonisInterface.mainvis import mainvis
import gpcam
gpcam.global_config = Config


def perform_experiment_overlap2(data):
    path_new_command = "data/new_experiment_command/"
    path_new_result = "data/new_experiment_result/"
    while os.path.isfile(path_new_command + "command.npy"):
        time.sleep(1)
        print("Waiting for experiment device to read and subsequently delete last command.")

    read_success = False
    write_success = False
    try:
        new_data = np.load(path_new_result + "result.npy", encoding="ASCII", allow_pickle=True)
        read_success = True
        for entry in data:
            if entry["measured"] == True: 
                continue
            else: 
                for val in new_data:
                    if entry['id'] == val['id']:
                        entry['measurement values']['values'] = val['measurement values']['values']
                        entry['measurement values']['value positions'] = val['measurement values']['value positions']
                        entry['measured'] = val['measured']
                    else:
                        None
        print("Successfully received updated data set from experiment device")
    except:
        print('data update failed')
        read_success = False
        
    while write_success == False:
        try:
            np.save(path_new_command + "command", data)
            write_success = True
            print("Successfully send data set to experiment device")
        except:
            time.sleep(1)
            print("Saving new experiment command file not successful, trying again...")
            write_success = False
    while read_success == False:
        try:
            new_data = np.load(
                path_new_result + "result.npy", encoding="ASCII", allow_pickle=True
            )
            read_success = True
            print("Successfully received data set from experiment device")
        except:
            print("New measurement values have not been written yet.")
            print("exception: ", sys.exc_info()[0])
            time.sleep(1)
            read_success = False

    return data


def perform_NanonisExp_BiasSpec(data):
    Vals = return_scandata()
    imoff, impix, imsize = Vals.get_scan_conditions()
    current_image = return_image()
    fil_path, imfile, channel, imdirection = Vals.get_file_info()
    imdirectory = fil_path+'data'+'\\'+'impath'
    specpath = fil_path+'data'+'\\'
    path_new_command = fil_path + 'data\\new_experiment_command\\'
    path_new_result = fil_path + 'data\\new_experiment_result\\'
    spec_counter = return_cnt()
    pix = float(impix[0][0])
    winsize = float(imsize[0][0])
    pixcnst = winsize/pix
    pntlist = return_pntlist()
    center_point, search_window, feature_window, spec_range = Vals.get_search_conditions()
    if (os.path.exists(path_new_command+"command.txt") == False) and (os.path.exists(path_new_result+"result.txt") == False):
        with open(path_new_command+'command.txt','w') as f:
            x1 = (pixcnst*0.5)+float(imoff[0][0]) 
            x2 = (pixcnst*0.5)+float(imoff[0][1]) 
            x1o, y1o = pix*0.5, pix*0.5
            pntlist.add_pnt([x1o,y1o])
            spec_counter.update_driftvar(1)
            f.write(str(x1)+'\t')
            f.write(str(x2)+'\t')
            f.write(str(float(spec_counter.get_driftvar()))+'\t')
            f.write(str(float(imoff[0][0]))+'\t')
            f.write(str(float(imoff[0][1]))+'\n')
            f.close()
            Nanonis.sxm_plot(current_image.get_image(),imdirectory,'current',pntlist.get_list())          
    while os.path.isfile(path_new_command + "command.txt"):
        time.sleep(1)
        print("Waiting for experiment device to read and subsequently delete last command.")
    while os.path.isfile(path_new_result + "result.txt") == False:
        time.sleep(1)
        print("awaiting results.")
    read_success = False
    write_success = False
    while read_success == False:
        time.sleep(5)
        new_data = []
        with open(path_new_result+'result.txt') as file:
            for i in file:
                tmp = i.split('\t')
                for j in range(0,len(tmp)-1):
                    new_data.append(float(tmp[j]))
                new_data.append(float(tmp[len(tmp)-1].split('\n')[0]))
        print(new_data)
        spec_counter.update_driftvar(int(new_data[2]))
        if spec_counter.get_driftvar() == 1:
            didv_files = []
            for fname in os.listdir(specpath):
                if fname.endswith('.dat') & fname.startswith('dI_dV'):
                    didv_files.append(fname)
            sorted(didv_files)
            imspec_filn = ''
            try:
                imspec_filn = didv_files[len(didv_files)-1]
            except:
                print('No spectroscopy data')
                assert os.path.isfile(specpath+imspec_filn)
            ###read in dI/dV###
            fil = specpath+imspec_filn
            out = Nanonis.retspec(fil)
            dv = []
            di = []
            for i in range(1,len(out)):
                if (float(out[i][0]) >= spec_range[0]) and (float(out[i][0]) <= spec_range[1]):
                    dv.append(float(out[i][0]))
                    di.append(float(out[i][1]))
            sumspec = sum(di)
            Nanonis.spectra_plot(dv,di,imdirectory,'lastspectra')
            for entry in data:
                if entry["measured"] == True: 
                    continue
                else: 
                    entry['measurement values']['values'] = np.array([sumspec])
                    entry['measurement values']['value positions'] = np.array([0])
                    entry['measured'] = True
            os.remove(path_new_result+'result.txt')        
            print("Successfully received updated data set from experiment device")
            spec_counter.incr_counter()
            xpix = (new_data[0]-float(imoff[0][0])+(winsize/2))/pixcnst
            ypix = (new_data[1]-float(imoff[0][1])+(winsize/2))/pixcnst
            print(spec_counter.get_counter())
            ###update *.dat file###
            if os.path.isfile(specpath + imspec_filn) == True:
                time.sleep(3)
                filn = open(fil,'r')
                lines = [line for line in filn]
                filn.close()
                filn = open(fil,'w')
                filn.write('xpix: '+str(xpix)+'\n')
                filn.write('ypix: '+str(ypix)+'\n')
                filn.write('xpos:'+str(new_data[0])+'\n')
                filn.write('ypos:'+str(new_data[1])+'\n')
                for line in lines:
                    filn.write(line)
                filn.close()
            read_success = True
        elif spec_counter.get_driftvar() == 2:
            path_image = fil_path+'data/'
            image_files = []
            for fname in os.listdir(path_image):
                if fname.endswith('.sxm'):
                    image_files.append(fname)
            sorted(image_files)
            imout = Nanonis.readimage(path_image+image_files[len(image_files)-1],channel,imdirection)
            corrmatrix = DriftTrack(current_image.get_image(),imout)
            corrmatrix.corr_matrix([center_point[0]],[center_point[1]],search_window,feature_window)
            offset = corrmatrix.get_corrdata()
            offx, offy = corrmatrix.get_maxcorr()
            os.remove(path_new_result+'result.txt')
            offxcorr, offycorr = offx*pixcnst, offy*pixcnst
            print(str(offxcorr)+', '+str(offycorr))
            newoffset = [[str(float(imoff[0][0])+offxcorr),str(float(imoff[0][1])+offycorr)]]
            print(newoffset)
            Vals.update_scan_conditions(newoffset, impix, imsize)
            imoff1, impix1, imsize1 = Vals.get_scan_conditions()
            with open(path_new_command+'command.txt','w') as f:
                x1o = data[len(data)-1]['position']['x1']
                x2o = data[len(data)-1]['position']['x2']  
                x1 = x1o*(pixcnst)+float(imoff1[0][0])-(winsize/2) 
                x2 = x2o*(pixcnst)+float(imoff1[0][1])-(winsize/2) 
                f.write(str(x1)+'\t')
                f.write(str(x2)+'\t')
                f.write(str(float(3))+'\t')
                f.write(str(float(imoff1[0][0]))+'\t')
                f.write(str(float(imoff1[0][1]))+'\n')
                f.close()
            while os.path.isfile(path_new_command + "command.txt"):
                time.sleep(1)
                print("Waiting for tool to update new location.")
            while os.path.isfile(path_new_result + "result.txt") == False:
                time.sleep(1)
                print("Waiting for tool to confirm updated location.")
            os.remove(path_new_result+'result.txt') 
            spec_counter.update_driftvar(1)
            read_success = True
        else:
            time.sleep(1)
            print('data update failed')
            read_success = False   
    while write_success == False:
        try:
            if spec_counter.get_counter() < spec_counter.get_maxvalue():
                with open(path_new_command+'command.txt','w') as f:
                    x1o = data[len(data)-1]['position']['x1']
                    x2o = data[len(data)-1]['position']['x2']  
                    x1 = x1o*(pixcnst)+float(imoff[0][0])-(winsize/2) 
                    x2 = x2o*(pixcnst)+float(imoff[0][1])-(winsize/2) 
                    pntlist.add_pnt([x1o,x2o])
                    print(x1)
                    print(x2)
                    f.write(str(x1)+'\t')
                    f.write(str(x2)+'\t')
                    f.write(str(float(1))+'\t')
                    f.write(str(float(imoff[0][0]))+'\t')
                    f.write(str(float(imoff[0][1]))+'\n')
                    f.close()
                print("Successfully sent data set to experiment device")
                Nanonis.sxm_plot(current_image.get_image(),imdirectory,'current',pntlist.get_list())
                print('no track')
                write_success = True
            elif spec_counter.get_counter() >= spec_counter.get_maxvalue():
                spec_counter.reset_counter()
                with open(path_new_command+'command.txt','w') as f:
                    x1o = data[len(data)-1]['position']['x1']
                    x2o = data[len(data)-1]['position']['x2']  
                    x1 = x1o*(pixcnst)+float(imoff[0][0])-(winsize/2) 
                    x2 = x2o*(pixcnst)+float(imoff[0][1])-(winsize/2) 
                    f.write(str(x1)+'\t')
                    f.write(str(x2)+'\t')
                    f.write(str(float(2))+'\t')
                    f.write(str(float(imoff[0][0]))+'\t')
                    f.write(str(float(imoff[0][1]))+'\n')
                    f.close()
                write_success = True
        except:
            time.sleep(1)
            print("Saving new experiment command file not successful, trying again...")
            write_success = False
    while os.path.isfile(path_new_result + "result.txt") == False:
        time.sleep(1)
        print("awaiting results.")
    try:
        with open(path_new_result+'result.txt') as file:
            bd = file.read()
        read_success = True
        print("Successfully received data set from experiment device.")
    except:
        print("New measurement values have not been written yet.")
        print("exception: ", sys.exc_info()[0])
        time.sleep(1)
        read_success = False
    """
    try:
        mainvis(data_path = '../data/current_data/Data_model_1.npy')
    except:
        print('model visualizations failed to update.')
    """
    return data
    

