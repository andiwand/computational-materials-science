#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

delta = 0.1
t = 1
a = 1
X = 0.5

KPOINTS = 100
WPOINTS = 201

ws = np.linspace(-30, 30, num=WPOINTS)
ks_tmp = np.linspace(0, 2*np.pi/a, num=KPOINTS, endpoint=False)
ks = np.array(np.meshgrid(ks_tmp, ks_tmp)).reshape(2, -1).T
epss = [ 0, 1.5, 3, 5, 10, 10 ]
mus = [ 0, 0, 0, 0, 0, -5 ]

eps_k = lambda a,t,ks: -2*t*(np.cos(ks[:,0]*a)+np.cos(ks[:,1]*a))

for eps, mu in zip(epss, mus):
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

    plt.plot(ws, Aiw, label='mu %f eps %f' % (mu, eps) )

plt.legend()
plt.show()

