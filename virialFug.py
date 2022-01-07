import numpy as np
import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

def BMix(T, name):
    T = np.reshape(T, (T.shape[0],1))
    T0 = 100 #K
    # Hydrogen
    if name == 'Hydrogen':
        c = np.array([[32.6972, -223.354, 219.893, -114.099]])
        d = np.array([[-0.21, -1.50, -2.26, -3.21]])
    # Helium
    elif name == 'Helium':
        c = np.array([[55.57, -59.25, 13.32, -4.767]])
        d = np.array([[-0.347, -0.85, -1.45, -2.1]])
    # Carbon Monoxide
    elif name == 'CarbonMonoxide':
        c = np.array([[13.304, 113.21, -333.09, -261.86, -165.91]])
        d = np.array([[0.0, -0.5, -1.0, -3.0, -4.5]])
        # Old CO
        #c = np.array([[493.709, -579.466, -248.146, -271.885]])
        #d = np.array([[-0.45, -0.57, -2.0, -4.25]])
    elif name == 'Argon':
        c = np.array([[96.1591, -211.074, -96.4425, -12.6006]])
        d = np.array([[-0.31, -0.82, -2.24, -4.60]])
    else:
        print('Not valid name')
    return np.sum(c*(T/T0)**d, axis=1)*1E-6

def BWater(T):
    T = np.reshape(T, (T.shape[0],1))
    T0 = 100 #K
    c = np.array([0.34404, -0.75826, -24.219, -3978.2])
    d = np.array([-0.5, -0.8, -3.35, -8.3])
    return np.sum(c*(T/T0)**d, axis=1)*1E-3

def BSolute(T, name):
    # Get solute virial from refprop
    os.environ['RPPREFIX'] = r'/Users/jmy1/Documents/REFPROP/bin' #path to refprop
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    MOLAR_SI = RP.GETENUMdll(0,"MOLAR SI").iEnum
    nT = T.shape[0]
    B = np.zeros(nT)
    for i in range(nT):
        r = RP.REFPROPdll(name, 'TP', 'Bvir', MOLAR_SI, 0, 0, T[i], 0.1, [1.0])
        B[i] = r.Output[0]
    return B*1E-3

def virialFug(T,y,totdens,name):
    n = T.shape[0]
    # make sure sizes correct
    assert(y.shape == (n,2))
    assert(totdens.shape[0] == n)
    totdens = np.reshape(totdens, (n,1)) #make 2D if not already
    B = np.zeros((n, 2, 2))
    B[:,0,0] = BWater(T)
    B[:,1,1] = BSolute(T, name)
    B[:,0,1] = BMix(T, name)
    B[:,1,0] = B[:,0,1]
    Z = 1 + np.sum(y*np.sum(y[...,np.newaxis]*B, axis=1), axis=1)[...,np.newaxis]*totdens
    phiVir = np.exp(2*totdens*np.sum(y[...,np.newaxis]*B, axis=1) - np.log(Z))
    return phiVir

def Z(T,y,totdens,name):
    n = T.shape[0]
    # make sure sizes correct
    assert(y.shape == (n,2))
    assert(totdens.shape[0] == n)
    totdens = np.reshape(totdens, (n,1)) #make 2D if not already
    B = np.zeros((n, 2, 2))
    B[:,0,0] = BWater(T)
    B[:,1,1] = BSolute(T, name)
    B[:,0,1] = BMix(T, name)
    B[:,1,0] = B[:,0,1]
    Z = 1 + np.sum(y*np.sum(y[...,np.newaxis]*B, axis=1), axis=1)[...,np.newaxis]*totdens
    return np.squeeze(Z)
