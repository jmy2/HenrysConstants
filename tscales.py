import numpy as np
from scipy.optimize import fsolve

def convert68to90(T):
    T = np.reshape(T, (T.shape[0],1))
    b = np.array([[-0.148759, -0.267408, 1.08076, 1.269056, -4.089591, -1.871251, 7.438081, -3.536296]])
    i = np.arange(1,9)[np.newaxis,...]
    return np.sum(b*((T-273.15)/630)**i, axis=1) + T[:,0]

def convert48to90(T):
    T = np.reshape(T, (T.shape[0]))
    T = T - 273.15 #Convert to C
    assert(np.amax(T) <= 630.74 and np.amin(T) >= 0)
    # First convert from 48 to 68, then convert to 90
    T68 = fsolve(solveFor68, T, args=(T))
    T68 = T68 + 273.15 #Convert to K
    return convert68to90(T68)

def solveFor68(t68, t48):
    phi = 0.045*(t68/100)*(t68/100-1)*(t68/419.58-1)*(t68/630.74-1)
    mu = 4.904e-7*t68*(t68-100)/(1-2.939e-4*t68) + phi
    return t68-mu - t48

