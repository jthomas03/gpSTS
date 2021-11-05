import numpy as np


def kernel_l2(x1,x2,hyper_parameters,obj):
    ################################################################
    ###standard anisotropic kernel in an input space with l2########
    ################################################################
    """
    x1: 2d numpy array of points
    x2: 2d numpy array of points
    obj: object containing kernel definition

    Return:
    -------
    Kernel Matrix
    """
    hps = hyper_parameters
    distance_matrix = np.zeros((len(x1),len(x2)))
    #you can always get some help:
    #help(obj.squared_exponential_kernel)
    #exit()
    for i in range(len(hps)-1):
        distance_matrix += abs(np.subtract.outer(x1[:,i],x2[:,i])/hps[1+i])**2
    distance_matrix = np.sqrt(distance_matrix)
    #return   hps[0] * obj.squared_exponential_kernel(distance_matrix,1)
    #return   hps[0] * obj.exponential_kernel(distance_matrix,1) + obj.periodic_kernel(abs(x1[:,-1]-x2[:,-1]), hps[-1])
    return   hps[0] *  obj.matern_kernel_diff1(distance_matrix,1)
    #return   hps[0] * obj.matern_kernel_diff2(distance_matrix,1)

