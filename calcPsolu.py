import numpy as np
import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import virialFug as vf
import volumes as vl

# Function to calculate total pressure and density from solute partial pressure

# Inputs:
# T = temperature [=] K
# Psolu = solute partial pressure [=] MPa
# name = name of solute

# Outputs:
# P = total pressure [=] MPa
# totdens = total density [=] mol/m3

def calcPsolu(T, P, name):
    # Check inputs
    nT = T.shape[0]
    assert(P.shape[0] == nT)

    # Refprop setup
    os.environ['RPPREFIX'] = r'/Users/jmy1/Documents/REFPROP/bin' #path to refprop
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    MOLAR_SI = RP.GETENUMdll(0,"MOLAR SI").iEnum

    # Get pure water properties
    phi1L = np.zeros(nT)
    Psat = np.zeros(nT)
    wDens = np.zeros(nT)
    for i in range(nT):
        r1 = RP.REFPROPdll('Water', 'TQ', 'FC,P,D', MOLAR_SI, 0, 0, T[i], 0.0, [1.0])
        phi1L[i] = r1.Output[0]
        Psat[i] = r1.Output[1]
        r2 = RP.REFPROPdll('Water', 'TQ', 'FC,P,D', MOLAR_SI, 0, 0, T[i], 1.0, [1.0])
        wDens[i] = r2.Output[0]*1E3

    # Initial guesses for total density, partial pressure, y
    R = 8.3144626E-6 #m3MPa/Kmol
    Psolu = P - Psat #MPa
    totdens = wDens+Psolu/(R*T) #mol/m3
    y = np.stack([1-Psolu/P,Psolu/P], axis=1)

    # Iterate until Psolu and density converge
    Pconv = 0.0001 #MPa
    Dconv = 0.0001 #mol/m3
    dP = 1
    dDens = 1
    while (np.amax(dP) > Pconv or np.amax(dDens) > Dconv):
        # Get Z
        Z = vf.Z(T, y, totdens, name)
        # Calculate new rho
        newdens = P/(Z*R*T) #mol/m3
        dDens = np.abs(newdens-totdens)
        totdens = newdens
        # Get fugacity coefficients
        phiV = vf.virialFug(T, y, totdens, name)
        # Get Pyonting correction
        Nint = 5
        pts, wts = np.polynomial.legendre.leggauss(Nint)
        pyntW = np.zeros(nT)
        for i in range(nT):
            Pint = (P[i]-Psat[i])/2*pts + (P[i]+Psat[i])/2 
            # Get volumes
            V1L = vl.Vwater(T[i], Pint)
            # Calc integral
            dPdu = (P[i]-Psat[i])/2
            pyntW[i] = np.sum(V1L*dPdu*wts)/(R*T[i])

        # Calculate new P
        y1 = phi1L*Psat/(phiV[:,0]*P)*np.exp(pyntW)
        P2new = P*(1-y1)
        dP = np.abs(P2new-Psolu)
        Psolu = P2new
        y = np.stack([1-Psolu/P,Psolu/P], axis=1)

    # Return P and total density
    return Psolu, totdens

