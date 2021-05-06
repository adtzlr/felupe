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
import felupe as fe

tol = 1e-6
move = np.linspace(0,1,3)#[1:]
a = -4
b = 5

mesh = fe.mesh.Cube(a=(-1,0,-1), b=(1,1,1), n=(16,16,16)) #Quadratic

z = mesh.nodes.copy()

#mask = np.logical_or(mesh.nodes[:,0] > 0.1, mesh.nodes[:,1] > 0.1)
#keep =  np.arange(mesh.nnodes)[mask]
#select = np.array([np.all(np.isin(conn, keep)) for conn in mesh.connectivity])
#mesh.connectivity = mesh.connectivity[select]

L = 115
B = 240
H = 26

L = 95
B = 220
H = 26

dL = 10
dB = 10

D = 100
dD = 10

# round
#r = np.sqrt(mesh.nodes[:,0]**2 + mesh.nodes[:,1]**2)
#mask = np.logical_or(abs(mesh.nodes[:,0]) > 0.5, abs(mesh.nodes[:,1]) > 0.5)
#r[~mask] = (r[~mask] - 2/3) / r[~mask].max() * r[~mask] + 2/3
#z[:,0] /= (0.05+r)
#z[:,1] /= (0.05+r)
#z[:,0] *= D/2# * (1+dD/D*mesh.nodes[:,2]**4)
#z[:,1] *= D/2# * (1+dD/D*mesh.nodes[:,2]**4)

# height
z[:,2] *= H/2

z[:,0] *= L/2*(1+2*dL/L*mesh.nodes[:,2]**4)# * (1+mesh.nodes[:,1]**3/4)
z[:,1] *= B/2*(1+2*dB/B*mesh.nodes[:,2]**4)# * (1-mesh.nodes[:,0]**3/4)


mesh.nodes = z
#mesh.connectivity = np.vstack((mesh.connectivity[:-1], 
#                               mesh.connectivity[-1:-1]))
mesh.update(mesh.connectivity)

region  = fe.Region(mesh, fe.element.Hex1(), fe.quadrature.Linear(dim=3)) 
region0 = fe.Region(mesh, fe.element.Hex0(), fe.quadrature.Linear(dim=3)) 

dV = region.dV

# u at nodes
u = fe.Field(region , 3)
p = fe.Field(region0, 1)
J = fe.Field(region0, 1, values=1)

# load constitutive material formulation
mat = fe.constitution.NeoHooke(mu=1.0, bulk=5000.0)

# boundaries
f0 = lambda x: np.isclose(x,-H/2)
f1 = lambda x: np.isclose(x, H/2)
bounds = fe.doftools.symmetry(u, (0,1,0))
bounds["bot" ] = fe.Boundary(u, skip=(0,0,0), fz=f0)
bounds["top" ] = fe.Boundary(u, skip=(0,0,1), fz=f1)
bounds["move"] = fe.Boundary(u, skip=(1,1,0), fz=f1)

bounds2 = fe.doftools.symmetry(u, (0,1,0))
bounds2["bot" ] = fe.Boundary(u, skip=(0,0,0), fz=f0)
bounds2["top" ] = fe.Boundary(u, skip=(1,0,1), fz=f1)
bounds2["fix" ] = fe.Boundary(u, skip=(1,1,0), fz=f1, value=a*move[-1])
bounds2["move"] = fe.Boundary(u, skip=(0,1,1), fz=f1, value=b*move)

results1 = fe.utils.incsolve((u,p,J), region, mat.f, mat.A, bounds, a*move,
                            tol=tol)

results2 = fe.utils.incsolve(results1[-1].fields, region, mat.f, mat.A, 
                             bounds2, 5*move,
                            tol=tol)

fe.utils.savehistory(region, [*results1, *results2])

force_z = np.array([res.r[bounds[ "move"].dof].sum() for res in results1])
force_x = np.array([res.r[bounds2["move"].dof].sum() for res in results2])

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

f_z = interp1d(
    a*move[:len(force_z)], 
    force_z, 
    kind="quadratic"
)
z = np.linspace(0,a*move[:len(force_z)][-1])
plt.plot(a*move[:len(force_z)], 2*force_z, 'o')
plt.plot(z, 2*f_z(z),'C0--')

f_x = interp1d(
    b*move[:len(force_x)], 
    force_x, 
    kind="quadratic"
)
x = np.linspace(0,b*move[:len(force_x)][-1])
plt.figure()
plt.plot(b*move[:len(force_x)], 2*force_x, 'o')
plt.plot(x, 2*f_x(x),'C0--')

print("c_Z0 = ", (np.diff(2*f_z(z))/np.diff(z))[0])
print("c_X0 = ", (np.diff(2*f_x(x))/np.diff(x))[0])

print("c_Z = ", (np.diff(2*f_z(z))/np.diff(z))[-1])
print("c_X = ", (np.diff(2*f_x(x))/np.diff(x))[-1])
print("V = ", region.volume().sum())