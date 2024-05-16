# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:05:23 2024

@author: z0039mte
"""

import felupe as fem

mesh = fem.Cube(n=3)
region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

# import tensortrax.math as tm

# @fem.total_lagrange
# def neo_hooke_total_lagrange(C, mu=1):
#     CG = tm.linalg.det(C)**(-1/3) * C
#     return mu * tm.special.dev(CG) @ tm.linalg.inv(C)

# umat = fem.Hyperelastic(neo_hooke_total_lagrange, mu=1)

# umat = fem.OgdenRoxburgh(material=fem.NeoHooke(mu=1), r=3, m=1, beta=0)
umat = fem.Hyperelastic(
    fem.morph,  # _representative_directions,
    p=[0.039, 0.371, 0.174, 2.41, 0.0094, 6.84, 5.65, 0.244],
    # p=[0.011, 0.408, 0.421, 6.85, 0.0056, 5.54, 5.84, 0.117],
    nstatevars=86,
)
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)


move = fem.math.linsteps([0, 1, 0, 1], num=5) * 0.7
step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)

job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
job.evaluate()
fig, ax = job.plot(
    xlabel="Displacement $u$ in mm $\longrightarrow$",
    ylabel="Normal Force $F$ in N $\longrightarrow$",
)

solid.plot("Principal Values of Cauchy Stress").show()
