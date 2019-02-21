#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

delta = 0.1
t = 3
a = 1
X = 0.5

KPOINTS = 100
WPOINTS = 201
ws = np.linspace(-30, 30, WPOINTS)
ks_tmp = np.linspace(0, 2*np.pi/a, num=KPOINTS, endpoint=False)
ks = np.array(np.meshgrid(ks_tmp, ks_tmp)).reshape(2, -1).T

#eps_k = lambda a,t,kx,ky: -2*t*(np.cos(kx*a)+np.cos(ky*a))
#eps = np.mean([eps_k(a,t,kx,ky) for kx,ky in ks])
eps_k = lambda a,t,ks: -2*t*(np.cos(ks[:,0]*a)+np.cos(ks[:,1]*a))

eps = 10
mu = -eps/2
print('mu', mu)

siw = np.zeros(WPOINTS)
for i in range(10):
    print(i)
    siw_old = siw
    
    f = lambda w, s: np.mean(1/(w+mu-eps_k(a,t,ks)-s+1j*delta))
    giw = np.vectorize(f)(ws, siw)
    Giw = 1/(1/giw + siw)
    giw = (1-X)*Giw + X*(1/(1/Giw-eps))
    siw = 1/Giw - 1/giw
    
    Aiw = -1/np.pi * np.imag(giw)
    
    if (np.sum(np.abs(siw-siw_old)) < 1e-6):
        break

plt.plot(ws, Aiw)
plt.show()

