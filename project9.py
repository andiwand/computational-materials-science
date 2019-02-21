#!/usr/bin/env python3

import numpy as np
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
import pandas as pd
from scipy.misc import comb
from scipy import sparse

# from https://github.com/mwallerb/LAZY/blob/master/ED.ipynb
def get_annihilators(flavors, dtype=np.float64):
    dim_fock = 2**flavors
    fock_range = np.arange(dim_fock)

    c = [None] * flavors
    sign = np.ones(dim_fock, np.int8)
    bitmask = 1
    for i in range(flavors):
        # The operator c_i removes the state i, which is equivalent to
        # shifting the Fock state 2**i to the left or occupying the +2**i'th
        # side diagonal, but only if the state is indeed present (has_i).
        # Diagonal storage always starts at column 0 (the first rows in the
        # c operators being "above" the matrix)
        has_i = np.array(fock_range & bitmask, dtype=np.bool)
        c[i] = sparse.dia_matrix((has_i * sign, bitmask),
                                 shape=(dim_fock, dim_fock),
                                 dtype=dtype).todense()

        # The sign is (-1)**n, where n is the number of occupied states
        # smaller than (ordered before) i.  So, for the next step we flip
        # the sign of those elements where the current state is present.
        sign[has_i] *= -1
        bitmask <<= 1
    return c

def get_creators(annihilators):
    return [a.H for a in annihilators]

def get_ns(annihilators):
    it = zip(annihilators[::2], annihilators[1::2])
    return [au.H.dot(au) + ad.H.dot(ad) for au, ad in it]

def get_szs(annihilators):
    it = zip(annihilators[::2], annihilators[1::2])
    return [0.5 * (au.H.dot(au) - ad.H.dot(ad)) for au, ad in it]

def get_n(annihilators):
    return sum(get_ns(annihilators))

def get_sz(annihilators):
    return sum(get_szs(annihilators))

def get_s(annihilators):
    szs = get_szs(annihilators)
    it = zip(szs, szs[1:] + szs[:1])
    return 0.25 * sum(si.dot(sj) for si, sj in it)

def hamilton_ring(n, t, U, annihilators):
    c = annihilators
    h = c[0].dot(c[1]) + c[1].dot(c[0])
    sites = list(range(n))
    
    for i, j in zip(sites, sites[1:] + sites[:1]):
        for s in range(2):
            h += -t * (c[2*i+s].H.dot(c[2*j+s]) + c[2*j+s].H.dot(c[2*i+s]))
    
    for i in sites:
        h += U * (c[2*i+0].H.dot(c[2*i+1].H) * c[2*i+1].dot(c[2*i+0]))
    
    return h

def n_sector_mask(dim_fock, n):
    mask = np.empty(dim_fock, np.bool)
    for i in range(dim_fock):
        mask[i] = bin(i).count("1") == n
    return mask

def sz_sector_mask(dim_fock, sz):
    mask = np.empty(dim_fock, np.bool)
    for i in range(dim_fock):
        b = bin(i)[2:][::-1]
        sz_is = 0.5 * (b[::2].count("1") - b[1::2].count("1"))
        mask[i] = sz_is == sz
    return mask

def get_sector(operator, mask):
    return operator[mask,:][:,mask]

flavors = 8 # number of flavors
dim_fock = 2**flavors
fermions = 4 # number of fermions
dim = comb(flavors, fermions)

print('fock space dimension = %d' % dim_fock)
print('%d fermions sector dimension = %d' % (fermions, dim))

annihilators = get_annihilators(flavors)

#H = hamilton_ring(4, 1, 0, annihilators)
#print('H is hermitian = %r' % (H == HT).all())
#N = get_n(annihilators)
#Sz = get_sz(annihilators)
#print('H and N commutes = %r' % (H.dot(N) == N.dot(H)).all())
#print('H and Sz commutes = %r' % (H.dot(Sz) == Sz.dot(H)).all())

def part1(annihilators, fermions, t, Umin, Umax, Usteps):
    # spectrum for n fermions; U=Umin..Umax; t
    Us = np.linspace(Umin, Umax, num=Usteps)
    Es = []
    Ss = []
    mask = n_sector_mask(dim_fock, fermions)# & sz_sector_mask(dim_fock, 1)
    print('part1: sector dimension %d' % np.sum(mask))
    S = get_s(annihilators)
    S4 = get_sector(S, mask)
    for U in Us:
        print('part1: U=%f' % U)
        H = hamilton_ring(fermions, t, U, annihilators)
        H4 = get_sector(H, mask)
        w, v = np.linalg.eigh(H4)
        w[np.abs(w) < 1e-5] = 0
        v[np.abs(v) < 1e-5] = 0
        sorter = np.argsort(w)
        w = w[sorter]
        v = v[:,sorter]
        Es.append(w)
        s = multi_dot((v[:,0].T, S4, v[:,0]))[0,0]
        Ss.append(s)
    Es = np.array(Es)
    Ss = np.array(Ss)
    
    for i in range(min(100, Es.shape[1])):
        plt.plot(Us, Es[:,i], label='E%d' % i)
    plt.xlabel('U')
    plt.ylabel('Ei')
    plt.legend()
    plt.show()
    
    plt.plot(Us, Ss)
    plt.xlabel('U')
    plt.ylabel('S(gs)')
    plt.show()
#part1(annihilators, fermions, 1.0, 0.0, 4.0, 50)

def part2(annihilators, fermions, t, U, Tmin, Tmax, Tsteps):
    Ts = np.linspace(Tmin, Tmax, num=Tsteps)
    Ss = []
    mask = n_sector_mask(dim_fock, fermions)
    
    H = hamilton_ring(fermions, t, U, annihilators)
    H4 = get_sector(H, mask)
    w, v = np.linalg.eigh(H4)
    S = get_s(annihilators)
    S4 = get_sector(S, mask)
    
    for T in Ts:
        Z = np.sum(np.exp(-w/T))
        s = 1/Z * sum(multi_dot((v[:,i].T, S4, v[:,i]))[0,0] * np.exp(-w[i]/T) for i in range(len(w)))
        Ss.append(s)
    Ss = np.array(Ss)
    
    plt.plot(Ts, Ss)
    plt.ylabel('S')
    plt.xlabel('T')
    plt.show()
part2(annihilators, fermions, 1.0, 1.0, 0.01, 10.0, 500)

