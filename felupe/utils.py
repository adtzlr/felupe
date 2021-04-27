# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from .solve import partition, solve
from .helpers import identity

from .doftools import apply, partition as dofpartition

def dofresiduals(domain, r, rref, dof1=None):
    
    rx =    r.toarray().reshape(domain.nnodes, domain.ndim)
    ry = rref.toarray().reshape(domain.nnodes, domain.ndim)
    
    ry[ry == 0] = np.nan
    
    rxy = rx/ry
    
    rxy[domain.mesh.elements_per_node == 1] = \
        rx[domain.mesh.elements_per_node == 1]
        
    if dof1 is None:
        return rxy
    else:
        return rxy.ravel()[dof1]
    
def newtonrhapson(domain, u, u0ext, f_P, f_A, dof1, dof0, 
              maxiter=20, 
              tol={"u": 1e-6, 
                   "r": 1e-6,
                   "r_dof": np.inf,
                  }
              ):
    
    # deformation gradient at integration points
    F = identity(domain.grad(u)) + domain.grad(u)
    
    # PK1 stress and elasticity matrix
    P = f_P(F)
    A = f_A(F)
    
    # residuals and elasticity matrix components
    r_aie   = domain.integrate(P)
    K_aibke = domain.integrate(A)
    
    # assembly
    r    = domain.asm(    r_aie   )
    K    = domain.asm(    K_aibke )
    
    r = r.toarray()[:,0].reshape(domain.nnodes, domain.ndim)
    
    converged = False
    
    for iteration in range(maxiter):
        
        system = partition(u, r.ravel(), K, dof1, dof0)
        du = solve(*system, u0ext)
        
        if np.any(np.isnan(du)):
            break
        else:
            u += du
        
        # deformation gradient at integration points
        F = identity(domain.grad(u)) + domain.grad(u)
        
        # PK1 stress and elasticity matrix
        P = f_P(F)
        
        # residuals and stiffness matrix components
        r_aie   = domain.integrate(P)
        
        # assembly
        r = domain.asm(r_aie).toarray()[:,0].reshape(domain.nnodes, 
                                                     domain.ndim)
        
        norm_r = np.linalg.norm(r.ravel()[dof1])
        norm_u = np.linalg.norm(du)
        
        if tol["r_dof"] != np.inf:
            rref = domain.asm(abs(r_aie  ))
            norm_rr = np.linalg.norm(
                dofresiduals(domain, r, rref, dof1)
                )
        else:
            norm_rr = -1
            
        print(f"#{iteration+1:2d}: |r|={norm_r:1.3e} (|δu|={norm_u:1.3e})")

        if norm_u < tol["u"] and norm_r < tol["r"] and norm_rr < tol["r_dof"]:
            converged = True
            break
        else:
            # elasticity matrix
            A = f_A(F)
            
            # stiffness matrix components
            K_aibke = domain.integrate(A)
            
            # assembly
            K = domain.asm(K_aibke)

    return u, F, P, A, r, K, converged


def incsolve(u, domain, bounds, move, P, A, boundid=-1, filename="out",
             maxiter=8):
    
    res = []
    
    # dofs to dismiss and to keep
    dof0, dof1 = dofpartition(domain.dof, bounds)
    
    # solve newton iterations and save result
    for increment, move_t in enumerate(move):
        
        print(f"\nINCREMENT {increment+1:2d}   (move={move_t:1.3g})")
        # set new value on boundary
        bounds[boundid].value = move_t
        
        # obtain external displacements for prescribed dofs
        u0ext = apply(u, domain.dof, bounds, dof0)
        
        results = newtonrhapson(domain, u, u0ext, P, A, dof1, dof0,
                                maxiter=maxiter)
        
        result, converged = results[:-1], results[-1]
        u = result[0]
        
        if not converged:
            # reset counter for last converged increment and break
            increment = increment - 1
            break
        else:
            # save results and go to next increment
            res.append(results)
            domain.save(*result, filename= filename+".vtk")
            domain.save(*result, filename= filename+f"_{increment+1:d}.vtk")
            print("SAVED TO FILE")
        
    return res
