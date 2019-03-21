#!/usr/bin/env python3

import numpy as np
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
from scipy.misc import comb
from scipy import sparse

def get_annihilators(flavors, format=None, dtype=np.int8):
    dim_fock = 2**flavors
    c = [None] * flavors
    for i in range(flavors):
        eye = sparse.identity(2**i, format=format, dtype=dtype)
        zero = sparse.csr_matrix(eye.shape, dtype=dtype)
        for j in range(i):
            c[j] = sparse.bmat([[c[j], zero], [zero, -c[j]]], format=format, dtype=dtype)
        c[i] = sparse.bmat([[zero, eye], [zero, zero]], format=format, dtype=dtype)
    return c

def get_creators(annihilators):
    return [a.H for a in annihilators]

def get_ns(annihilators):
    return [a.H.dot(a) for a in annihilators]

def get_nss(annihilators):
    ns = get_ns(annihilators)
    it = zip(ns[::2], ns[1::2])
    return [nu + nd for nu, nd in it]

def get_szs(annihilators):
    it = zip(annihilators[::2], annihilators[1::2])
    return [0.5 * (au.H.dot(au) - ad.H.dot(ad)) for au, ad in it]

def get_n(annihilators):
    return sum(get_ns(annihilators))

def get_sz(annihilators):
    return sum(get_szs(annihilators))

def get_s(annihilators):
    szs = get_szs(annihilators)
    return 0.25 * sum(si.dot(sj) for si, sj in zip(szs, szs[1:] + szs[:1]))

def hamilton_ring(t, U, annihilators):
    c = annihilators
    h = sparse.csr_matrix(c[0].shape, dtype=c[0].dtype)
    sites = list(range(len(annihilators) // 2))
    
    for i, j in zip(sites, sites[1:] + sites[:1]):
        for s in range(2):
            h += t * (c[2*i+s].H.dot(c[2*j+s]) + c[2*j+s].H.dot(c[2*i+s]))
    
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
    result = operator[mask,:][:,mask]
    if sparse.issparse(result):
        result = result.toarray()
    return result

def check_annihilators(annihilators):
    flavors = len(annihilators)
    dim_fock = 2**flavors

    g = ((i, j) for i in range(flavors) for j in range(i, flavors))
    for i, j in g:
        ci, cj = annihilators[i], annihilators[j]

        if ci.shape != (dim_fock, dim_fock):
            print('ERROR ci has invalid shape for i=%i' % i)
        if cj.shape != (dim_fock, dim_fock):
            print('ERROR cj has invalid shape for j=%i' % i)

        r = ci.dot(cj) + cj.dot(ci)
        if len(sparse.find(r)[2]) > 0:
            print('ERROR ci*cj+cj*ci != 0 for i=%i, j=%i' % (i, j))

        r = ci.dot(cj.H) + cj.H.dot(ci)
        if i != j and len(sparse.find(r)[2]) > 0:
            print('ERROR ci*cj.H+cj.H*ci != 0 for i=%i, j=%i' % (i, j))
        elif i == j and (r != np.eye(dim_fock)).any():
            print('ERROR ci*ci.H+ci.H*ci != 1 for i=%i' % i)

def check_H(annihilators):
    H = hamilton_ring(1, 1, annihilators)
    if not np.allclose(H.A, H.H.A):
        print('ERROR H is not hermitian')
    N = get_n(annihilators)
    if not np.allclose(H.dot(N).A, N.dot(H).A):
        print('ERROR H and N do not commute')
    Sz = get_sz(annihilators)
    if not np.allclose(H.dot(Sz).A, Sz.dot(H).A):
        print('ERROR H and Sz do not commute')

def part1(annihilators, mask, t, Us):
    S = get_s(annihilators)
    S = get_sector(S, mask)

    Es = []
    Sgs = []
    for U in Us:
        print('part1: U=%f' % U)
        H = hamilton_ring(t, U, annihilators)
        H = get_sector(H, mask)
        w, v = np.linalg.eigh(H)

        w[np.abs(w) < 1e-5] = 0
        v[np.abs(v) < 1e-5] = 0
        sorter = np.argsort(w)
        w = w[sorter]
        v = v[:,sorter]

        s = multi_dot((v[:,0].T, S, v[:,0]))

        Es.append(w)
        Sgs.append(s)
    Es = np.array(Es)
    Sgs = np.array(Sgs)
    
    for i in range(Es.shape[1]):
        plt.plot(Us, Es[:,i], label='E%d' % i)
    plt.xlabel('U')
    plt.ylabel('Ei')
    #plt.legend()
    plt.show()
    
    plt.plot(Us, Sgs)
    plt.xlabel('U')
    plt.ylabel('S(gs)')
    plt.show()

def part2(annihilators, mask, t, U, Ts):
    Ss = []
    
    H = hamilton_ring(t, U, annihilators)
    H = get_sector(H, mask)
    w, v = np.linalg.eigh(H)
    S = get_s(annihilators)
    S = get_sector(S, mask)

    sorter = np.argsort(w)
    w = w[sorter]
    v = v[:,sorter]

    for T in Ts:
        print('part2: T=%f' % T)
        Z = np.sum(np.exp(-w/T))
        s = 1/Z * sum(multi_dot((v[:,i].T, S, v[:,i])) * np.exp(-w[i]/T) for i in range(len(w)))
        Ss.append(s)
    Ss = np.array(Ss)

    plt.plot(Ts, Ss)
    plt.ylabel('S')
    plt.xlabel('T')
    plt.show()


fermions = 4 # number of fermions
sites = 4 # number of sites
flavors = 2*sites # number of flavors
dim_fock = 2**flavors
dim = comb(flavors, fermions)
states = np.arange(dim_fock)

t = 1.0
U = 1.0
Us = np.linspace(0.0, 4.0, num=100)
Ts = np.linspace(0.0, 1.0, num=1000)

print('fock space dimension = %d' % dim_fock)
print('%d fermions sector dimension = %d' % (fermions, dim))

annihilators = get_annihilators(flavors)

check_annihilators(annihilators)
check_H(annihilators)

mask = n_sector_mask(dim_fock, fermions)# & sz_sector_mask(dim_fock, 0)

print('sector dimension %d' % np.sum(mask))

part1(annihilators, mask, t, Us)
part2(annihilators, mask, t, U, Ts)

