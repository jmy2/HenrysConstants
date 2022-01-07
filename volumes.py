import numpy as np
import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

# Calculate volumes for pyonting corrections
def Vwater(T, P):
    nP = P.shape[0]
    T = np.atleast_1d(T)
    if T.shape[0] == 1:
        T = np.repeat(T,nP)
    else:
        assert(T.shape[0] == nP)
    os.environ['RPPREFIX'] = r'/Users/jmy1/Documents/REFPROP/bin' #path to refprop
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    MOLAR_SI = RP.GETENUMdll(0,"MOLAR SI").iEnum
    V = np.zeros(nP)
    for i in range(nP):
        r = RP.REFPROPdll('Water', 'TP', 'V', MOLAR_SI, 0, 0, T[i], P[i], [1.0])
        V[i] = r.Output[0]
    return V*1E-3 #m3/mol

def Vinf(T, P, name):
    Vref = 0.018068623941512676*1E-3 #m3/mol
    Vinf = 0
    if name == 'Hydrogen':
        Vinf = 26.1*1E-6 #m3/mol
    elif name == 'Helium':
        Vinf = 24.6*1E-6 #m3/mol
    elif name == 'CarbonMonoxide':
        Vinf = 36.7*1E-6 #m3/mol
    elif name == 'Argon':
        Vinf = 31.5*1E-6 #m3/mol
    # Calculate water densities
    Vcalc = Vwater(T,P)
    return Vinf*Vcalc/Vref #m3/mol

def Vinfsat(T, name): #partial molar volume in water at psat
    Vref = 0.018068623941512676*1E-3 #m3/mol
    Vinf = 0
    if name == 'Hydrogen':
        Vinf = 26.1*1E-6 #m3/mol
    elif name == 'Helium':
        Vinf = 24.6*1E-6 #m3/mol
    elif name == 'CarbonMonoxide':
        Vinf = 36.7*1E-6 #m3/mol
    elif name == 'Argon':
        Vinf = 31.5*1E-6 #m3/mol
    # Calculate water densities
    os.environ['RPPREFIX'] = r'/Users/jmy1/Documents/REFPROP/bin' #path to refprop
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    MOLAR_SI = RP.GETENUMdll(0,"MOLAR SI").iEnum
    nT = T.shape[0]
    Vcalc = np.zeros(nT)
    for i in range(nT):
        r = RP.REFPROPdll('Water', 'TQ', 'V', MOLAR_SI, 0, 0, T[i], 0.0, [1.0])
        Vcalc[i] = r.Output[0]*1E-3
    return Vinf*Vcalc/Vref #m3/mol
