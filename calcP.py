import numpy as np
import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import virialFug as vf
import volumes as vl
from scipy.optimize import fsolve

# Function to calculate total pressure and density from solute partial pressure

# Inputs:
# T = temperature [=] K
# Psolu = solute partial pressure [=] MPa
# name = name of solute

# Outputs:
# P = total pressure [=] MPa
# totdens = total density [=] mol/m3

def calcP(T, Psolu, name):
    # Check inputs
    nT = T.shape[0]
    assert(Psolu.shape[0] == nT)

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
        r1 = RP.REFPROPdll('Water', 'TQ', 'FC,P', MOLAR_SI, 0, 0, T[i], 0.0, [1.0])
        phi1L[i] = r1.Output[0]
        r2 = RP.REFPROPdll('Water', 'TQ', 'D,P', MOLAR_SI, 0, 0, T[i], 1.0, [1.0])
        wDens[i] = r2.Output[0]*1E3
        Psat[i] = r1.Output[1]

    # Initial guesses for total density, pressure, y
    R = 8.3144626E-6 #m3MPa/Kmol
    P = Psolu+Psat #MPa
    totdens = wDens+Psolu/(R*T) #mol/m3
    y = np.stack([1-Psolu/P,Psolu/P], axis=1)

    # Solve system of eqns at each T
    for i in range(nT):
        soln = fsolve(toSolveP, [totdens[i], P[i], y[i,0]], args=(T[i], Psolu[i], phi1L[i], Psat[i], name))
        totdens[i] = soln[0]
        P[i] = soln[1]
        y[i,:] = np.array([soln[2],1-soln[2]])

    # Return P and total density
    return P, totdens

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

    # Solve system of eqns at each T
    for i in range(nT):
        soln = fsolve(toSolvePsolu, [totdens[i], Psolu[i], y[i,0]], args=(T[i], P[i], phi1L[i], Psat[i], name))
        totdens[i] = soln[0]
        Psolu[i] = soln[1]
        y[i,:] = np.array([soln[2],1-soln[2]])

    # Return Psolu and total density
    return Psolu, totdens

def toSolveP(params, T, Psolu, phi1L, Psat, name): # params = dens, P, y1
    dens = np.array([params[0]])
    P = np.array([params[1]])
    y = np.array([params[2], 1-params[2]])[np.newaxis,...]
    T = np.array([T])
    return toSolve(dens, P, y, T, Psolu, phi1L, Psat, name)

def toSolvePsolu(params, T, P, phi1L, Psat, name): # params = dens, Psolu, y1
    dens = np.array([params[0]])
    Psolu = np.array([params[1]])
    y = np.array([params[2], 1-params[2]])[np.newaxis,...]
    T = np.array([T])
    return toSolve(dens, P, y, T, Psolu, phi1L, Psat, name)

def toSolve(dens, P, y, T, Psolu, phi1L, Psat, name):
    R = 8.3144626E-6 #m3MPa/Kmol
    # Each value of output should be zero
    output = np.zeros(3)
    Z = vf.Z(T, y, dens, name)
    output[0] = Z*dens*R*T/P - 1
    output[1] = P*(y[:,0]-1)+Psolu
    phiV = vf.virialFug(T, y, dens, name)
    # Get Pyonting correction
    Nint = 5
    pts, wts = np.polynomial.legendre.leggauss(Nint)
    Pint = (P-Psat)/2*pts + (P+Psat)/2 
    # Get volumes
    V1L = vl.Vwater(T, Pint)
    # Calc integral
    dPdu = (P-Psat)/2
    pyntW = np.sum(V1L*dPdu*wts)/(R*T)
    output[2] = phi1L*Psat/(phiV[:,0]*y[:,0])*np.exp(pyntW) - P
    return output
