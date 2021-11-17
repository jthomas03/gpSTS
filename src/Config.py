# -*- coding: utf-8 -*-
#
# John C. Thomas 2021 gpSTS

###########################################
###Configuration File######################
###for gpSTS steering of experiments######
###########################################
import os
import numpy as np
from gpsts.NanonisInterface.nanonis_interface import Nanonis
from gpsts.NanonisInterface.data_class import ScanData, SpecCounter, PointList, ImageInfo
from gpsts.NanonisInterface.kernel import kernel_l2
import json
###############################
###Initialize##################
###############################
nanonis_config = {
	"Nanonis_Settings": {
		"File": "gpSTSinit",
        "ExperimentName": "Test Out",
		"Version": "0.0.1",
		"ImageStart": "test_img001.sxm",
		"FolderLocation": "C:\\gpSTS\\src\\",
        "DataLocation": "C:\\gpSTS\\src\\data\\",
		"Channel": "Z",
		"ImDirection": "forward",
		"SpectralRange": [-1,1],
        "NumSpectralPoints": 1200,
		"Center_Point": [174,34],
		"Search_Window": 40,
		"Feature_Window": 20,
		"ScanCurrent": 30e-12,
		"SpecCurrent": 200e-12,
        "STSbias": "Bias calc (V)",
        "STSsignal": "Current (A)"
	},
    "Neural_Network": {
        "TrainingPath": "C:\\ML\\gpSTS_vSubmit\\gpSTS\\src\\train\\",
        "EpochNumber": 2,
        "ClassNumber": 4,
        "LearningRate": 0.001,
        "BatchSizeTrain": 5,
        "BatchSizeVal": 1,
        "BatchSizeTest": 1
    }
}
with open('data/'+str(nanonis_config['Nanonis_Settings']['File'])+'.json','w') as fil:
    json.dump(nanonis_config, fil, sort_keys = True, indent = 4, ensure_ascii = False)
Vals = ScanData()
Vals.update_file_info(nanonis_config['Nanonis_Settings']['FolderLocation'], 
    nanonis_config['Nanonis_Settings']['ImageStart'], nanonis_config['Nanonis_Settings']['Channel'], 
    nanonis_config['Nanonis_Settings']['ImDirection'])
Vals.update_search_conditions(nanonis_config['Nanonis_Settings']['Center_Point'],
    nanonis_config['Nanonis_Settings']['Search_Window'],nanonis_config['Nanonis_Settings']['Feature_Window'],
    nanonis_config['Nanonis_Settings']['SpectralRange'])
fil_path, imfile, channel, imdirection = Vals.get_file_info()
try:
    imoff, impix, imsize = Nanonis.readheader(fil_path+'data'+'\\',imfile)
except Exception as e:
    print('Error. Please save '+str(imfile)+' within '+str(fil_path)+'data\\')
    raise e
Vals.update_scan_conditions(imoff, impix, imsize)
imdirectory = fil_path+'data'+'\\'+'impath'
if not os.path.exists(imdirectory):
    os.makedirs(imdirectory)
datadirectory = fil_path+'data'
if not os.path.exists(datadirectory):
    os.makedirs(datadirectory)
def return_scandata(): 
    return Vals
spec_counter = SpecCounter()
spec_counter.update_maxcnt(10)
def return_cnt():
    return spec_counter
recorded_points = PointList()
def return_pntlist():
    return recorded_points
imout = Nanonis.readimage(fil_path+'data'+'\\'+imfile,channel,imdirection)
current_image = ImageInfo(imout)
def return_image():
    return current_image
Nanonis.sxm_plot(imout,imdirectory,'current',recorded_points.get_list())
center_point, search_window, feature_window, spec_range = Vals.get_search_conditions()
imx1, imx2 = int((center_point[0]-(feature_window/2))), int((center_point[0]+(feature_window/2)))
imy1, imy2 = int((center_point[1]-(feature_window/2))), int((center_point[1]+(feature_window/2)))
imtrack = imout[imx1:imx2,imy1:imy2]
Nanonis.sxm_plot(imtrack,imdirectory,'feature',recorded_points.get_list())
###############################
###General#####################
###############################
from controls import perform_NanonisExp_BiasSpec, perform_experiment_overlap2
from gpsts.NanonisInterface.graph import plot_2d_function

parameters = {
    "x1": {
        "element interval": [1,int(impix[0][0])], 
    },
    "x2": {
        "element interval": [1,int(impix[0][0])],
    },
}

######acquisition and cost functions#####################
def my_ac_func(x,obj):
  mean = obj.posterior_mean(x)["f(x)"]
  cov  = obj.posterior_covariance(x)["v(x)"]
  return mean + 3.0 * np.sqrt(cov)

gaussian_processes = {
    "model_1": {
        "kernel function": kernel_l2,
        "hyperparameters": [1.0,1.0,1.0],
        "hyperparameter bounds": [[1.0,100.0],[0.10,100.0],[0.10,100.0]],
        "input hyper parameters": [1.0,1.0,1.0],
        "output hyper parameters": [1.0],
        "input hyper parameter bounds": [[0.01,1000000.0],[0.01,10.0],[0.01,10.0]],
        "output hyper parameter bounds":[[0.9,1.1]],
        "number of returns": 1,
        "dimensionality of return": 1, 
        "variance optimization tolerance": 0.001,
        "adjust optimization threshold": [True,0.1],
        "steering mode": "covariance",
        "run function in every iteration": None,
        "data acquisition function": perform_NanonisExp_BiasSpec, 
        "acquisition function": my_ac_func,
        "objective function": None,
        "mean function": None,
        "cost function": None,
        "cost update function": None,
        "cost function parameters": {"offset": 10,"slope":2.0},
        "cost function optimization bounds": [[0.0,10.0],[0.0,10.0]],
        "cost optimization chance" : 0.1,
        "plot function": plot_2d_function,
        "acquisition function optimization tolerance": 0.001
    },
}
compute_device = "cpu"
sparse = False
compute_inverse = False
initial_likelihood_optimization_method = "global"
training_dask_client = False
prediction_dask_client = False
likelihood_optimization_tolerance = 1e-12
likelihood_optimization_max_iter = 200
automatic_signal_variance_range_determination = True
acquisition_function_optimization_method = "global"
chance_for_local_acquisition_function_optimization = 0.5
acquisition_function_optimization_population_size = 20
acquisition_function_optimization_max_iter = 20
global_likelihood_optimization_at = [200]
hgdl_likelihood_optimization_at = []
local_likelihood_optimization_at = []

breaking_error = 1e-18
########################################
###Variance Optimization################
########################################
objective_function_optimization_population_size = 20
likelihood_optimization_population_size = 20
number_of_suggested_measurements = 1
########################################
###Computation Parameters###############
########################################
global_kernel_optimization_frequency = 0.2  
local_kernel_optimization_frequency = 0.5  
gpu_acceleration = False  
rank_n_update = [False,0.2]  
gp_system_solver = "inv"  # "inv", "cg" or "minres"
switch_system_solver_to_after = [True, "cg", 5000]
###############################
###DATA ACQUISITION############
###############################
initial_data_set_size = 1
max_number_of_measurements = 10
#####################################################################
###############END###################################################
#####################################################################
