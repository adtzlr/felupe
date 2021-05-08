# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:06:47 2021

@author: adutz
"""

import numpy as np

array_like = (tuple, list, np.array)

def cube(a=(0,0,0), b=(1,1,1), n=(2,2,2)):
    
    dim = 3
    
    # check if number "n" is scalar or no. of nodes per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)
        
    # generate points for each axis
    a = np.array(a)
    b = np.array(b)
    n = np.array(n)
    
    X = [np.linspace(A, B, N) for A, B, N in zip(a[::-1], b[::-1], n[::-1])]
    
    # make a grid
    grid = np.meshgrid(*X, indexing="ij")[::-1]
    
    # generate list of node coordinates
    nodes = np.vstack([ax.ravel() for ax in grid]).T
    
    # prepare element connectivity
    a = []
    for i in range(n[1]):
        a.append(np.repeat(np.arange(i*n[0], (i+1)*n[0]), 2)[1:-1].reshape(-1,2))
    
    b = []
    for j in range(n[1]-1):
        d = np.hstack((a[j], a[j+1][:,[1,0]]))
        b.append(np.hstack((d, d + n[0]*n[1])))
        
    c = [np.vstack(b) + k*n[0]*n[1] for k in range(n[2]-1)]

    # generate element connectivity
    connectivity = np.vstack(c)
    
    return nodes, connectivity

def rectangle(a=(0,0), b=(1,1), n=(2,2)):
    
    dim = 2
    
    # check if number "n" is scalar or no. of nodes per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)
        
    # generate points for each axis
    X = [np.linspace(A, B, N) for A, B, N in zip(a, b, n)]
    
    # make a grid
    grid = np.meshgrid(*X)
    
    # generate list of node coordinates
    nodes = np.vstack([ax.ravel() for ax in grid]).T
    
    # prepare element connectivity
    a = []
    for i in range(n[1]):
        a.append(np.repeat(np.arange(i*n[0], (i+1)*n[0]), 2)[1:-1].reshape(-1,2))
    
    b = []
    for j in range(n[1]-1):
        b.append(np.hstack((a[j], a[j+1][:,[1,0]])))
    
    # generate element connectivity
    connectivity = np.vstack(b)
    
    return nodes, connectivity

nodes, connectivity = rectangle(n=(3,2))
nodes, connectivity = cube(b=(1,1,1), n=(32,32,32))