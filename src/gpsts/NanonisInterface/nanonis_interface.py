# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json


class Nanonis(object):
    def __init__(self):
        """Constructor"""
    
    def readheader(path,imfile):
        """
        Reads SXM file header
        """
        assert os.path.exists(path+imfile)
        l = ''
        key = ''
        header = {}
        with open(path+imfile,'rb') as f:
            while l!=b':SCANIT_END:':
                l = f.readline().rstrip()
                if l[:1]==b':':
                    key = l.split(b':')[1].decode('ascii')
                    header[key] = []
                else:
                    if l: 
                        header[key].append(l.decode('ascii').split())
        imoff = header['SCAN_OFFSET']
        impix = header['SCAN_PIXELS']
        imsize = header['SCAN_RANGE']
        return imoff, impix, imsize
    
    def retspec(fil):
        dataf = []
        with open(fil,'r') as f:
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
                    dataf.append(tmp)
        return dataf

    def readimage(filein,dtype,direction):
        """
        Reads SXM file and returns array, corrects for scan direction
        """
        def parse_scan_header_table(table_list):
            table_processed = []
            for row in table_list:
                table_processed.append(row.strip('\t').split('\t'))
            keys = table_processed[0]
            values = table_processed[1:]
            zip_vals = zip(*values)
            return dict(zip(keys, zip_vals))
            
        def start_byte(fname):
            with open(fname, 'rb') as f:
                tag = 'SCANIT_END'
                byte_offset = -1
                for line in f:
                    entry = line.strip().decode()
                    if tag in entry:
                        byte_offset = f.tell()
                        break
                if byte_offset == -1:
                    print('SXM file read error')
            return byte_offset

        def read_raw_header(fname,fnamebyt):
            with open(fname, 'rb') as f:
                return f.read(fnamebyt).decode('utf-8', errors='replace')

        def parse_sxm_header(header_raw):
            header_entries = header_raw.split('\n')
            header_entries = header_entries[:-3]
            header_dict = dict()
            entries_to_be_split = ['scan_offset',
                                'scan_pixels',
                                'scan_range',
                                'scan_time']
            entries_to_be_floated = ['scan_offset',
                                    'scan_range',
                                    'scan_time',
                                    'bias',
                                    'acq_time']
            entries_to_be_inted = ['scan_pixels']
            entries_to_be_dict = [':DATA_INFO:']
            for i, entry in enumerate(header_entries):
                if entry in entries_to_be_dict:
                    count = 1
                    for j in range(i+1, len(header_entries)):
                        if header_entries[j].startswith(':'):
                            break
                        if header_entries[j][0] == '\t':
                            count += 1
                    header_dict[entry.strip(':').lower()] = parse_scan_header_table(header_entries[i+1:i+count])
                    continue
                if entry.startswith(':'):
                    header_dict[entry.strip(':').lower()] = header_entries[i+1].strip()
            for key in entries_to_be_split:
                header_dict[key] = header_dict[key].split()
            for key in entries_to_be_floated:
                if isinstance(header_dict[key], list):
                    header_dict[key] = np.asarray(header_dict[key], dtype=np.float)
                else:
                    if header_dict[key] != 'n/a': 
                        header_dict[key] = np.float(header_dict[key])
            for key in entries_to_be_inted:
                header_dict[key] = np.asarray(header_dict[key], dtype=np.int)
            return header_dict

        def load_data(fname,header,byte_offset,dataf,indir):
            channs = list(header['data_info']['Name'])
            nchanns = len(channs)
            nx, ny = header['scan_pixels'] 
            ndir = indir
            data_dict = dict()
            f = open(fname, 'rb')
            byte_offset += 4
            f.seek(byte_offset)
            scandata = np.fromfile(f, dtype='>f4')
            f.close()
            scandata_shaped = scandata.reshape(nchanns, ndir, ny, nx)
            for i, chann in enumerate(channs):
                chann_dict = dict(forward=scandata_shaped[i, 0, :, :],
                                    backward=scandata_shaped[i, 1, :, :])
                data_dict[chann] = chann_dict
            return data_dict
            
        byte = start_byte(filein)
        header = parse_sxm_header(read_raw_header(filein,byte))
        data = dict()
        if 'both' in header['data_info']['Direction']:
            data = load_data(filein,header,byte,'scan',2)
        else:
            data = load_data(filein,header,byte,'scan',1)
        channs = list(header['data_info']['Name'])
        if dtype not in channs:
            dtype = ''
        try:
            if header['scan_dir'] == 'up':
                return data[dtype][direction]
            elif header['scan_dir'] == 'down':
                return np.flipud(data[dtype][direction])
        except:
            print('SXM file read error')

    def sxm_plot(imdata, filloc, filname, collectedpnts):
        fig, ax = plt.subplots()
        z_min, z_max = imdata.min(), imdata.max()
        x, y = np.meshgrid(np.linspace(1, imdata.shape[1], imdata.shape[1]), np.linspace(1, imdata.shape[0], imdata.shape[0]))
        cout = ax.pcolormesh(x,y,imdata, cmap='gray', vmin=z_min, vmax=z_max,shading='auto')
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        plt.axis('scaled')
        if len(collectedpnts)>0:
            for i in range(0,len(collectedpnts)):
                if i == 0:
                    plt.plot(collectedpnts[i][0], collectedpnts[i][1], '.', color='blue')
                else: 
                    plt.plot(collectedpnts[i][0], collectedpnts[i][1], '.', color='red')
        fig.tight_layout()
        plt.savefig(filloc+'\\'+filname+'.png', bbox_inches='tight', dpi = 50)
        plt.close()

    def spectra_plot(xs1, ys1, filloc, filname):
        fig, ax = plt.subplots()
        ax.set_xlabel('Bias (V)')
        ax.plot(xs1,ys1,c='b',linewidth=2,label='Collected Spectra') 
        plt.savefig(filloc+'\\'+filname+'.png', bbox_inches='tight', dpi = 50)
        plt.close()

