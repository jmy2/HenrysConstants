import numpy as np
import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from virialFug import virialFug
import volumes as vl

# Function to calculate Henry's constant

# Inputs:
# T = temperature [=] K
# P = total pressure [=] MPa
# x2 = liquid mol fraction
# y = vapor mol fraction
# totdens = total molar density of vapor [=] mol/m3
# name = name of solute

# From virial EOS:
# phi1V = vapor fugacity coefficient of water
# phi2V = vapor fugacity coefficient of solute
# From REFPROP for water:
# phi1L = pure H2O liquid fugacity coefficient at saturation
# V1L = pure H2O liquid density (as function of P)
# From Plyasunov:
# V2Linf = partial molar volume at infinite dilution (as function of P)

# Output:
# Henrys constant [=] MPa
def calcH(T, P, x2, y, totdens, name):
    # Check that density less than 1/4 of critical density
    crit = CritDens(name)
    errors = totdens > 0.25*crit
    if errors.nonzero()[0].size > 0:
        #raise RuntimeError('Density at T = ' + str(T[errors.nonzero()]) + ' greater than 1/4 critical density')
        print('Warning: Density at T = ' + str(T[errors.nonzero()]) + ' equals ' + str(totdens[errors.nonzero()]) + ' and is greater than 1/4 critical density (' + str(crit) + ')')

    # Refprop setup
    os.environ['RPPREFIX'] = r'/Users/jmy1/Documents/REFPROP/bin' #path to refprop
    RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
    RP.SETPATHdll(os.environ['RPPREFIX'])
    MOLAR_SI = RP.GETENUMdll(0,"MOLAR SI").iEnum

    R = 8.3144626E-6 #m3MPa/molK
    nT = T.shape[0]
    # Get fugacity coefficients
    phiV = virialFug(T, y, totdens, name)
    Psat = np.zeros(nT)
    for i in range(nT):
        r1 = RP.REFPROPdll('Water', 'TQ', 'P', MOLAR_SI, 0, 0, T[i], 0.0, [1.0])
        Psat[i] = r1.Output[0]

    # Pyonting corrections
    # Integrate volumes from psat to p
    # Calculate p points from gaussian quadrature
    Nint = 5
    pts, wts = np.polynomial.legendre.leggauss(Nint)
    pyntInf = np.zeros(nT)
    for i in range(nT):
        Pint = (P[i]-Psat[i])/2*pts + (P[i]+Psat[i])/2 
        # Get volumes
        V2Linf = vl.Vinf(T[i], Pint, name)
        # Calc integral
        dPdu = (P[i]-Psat[i])/2
        pyntInf[i] = np.sum(V2Linf*dPdu*wts)/(R*T[i])

    # Activity coefficients
    gamma2 = 1

    # Calculate Henrys constants
    return phiV[:,1]*P*y[:,1]/(gamma2*x2)*np.exp(-pyntInf)

def CritDens(name):
    if name == 'Hydrogen':
        return 15.508*1E3 #mol/m3
    elif name == 'Helium':
        return 17.3837*1E3 #mol/m3
    elif name == 'CarbonMonoxide':
        return 10.85*1E3 #mol/m3
    else:
        raise RuntimeError('Not valid name')

