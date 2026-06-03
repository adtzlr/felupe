# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from pyfluids import HumidAir, InputHumidAir
from scipy.constants import g


class FreeConvection:
    r"Free convection heat transfer formulation for flat plates."

    def __init__(self):
        self.T0 = 273.15
        self.P0 = 101325


    def _pyfluids_units():
        check = HumidAir().factory()
        if str(check.units_system) == 'SIWithCelsiusAndPercents':
            dt_ = 0  # use °C
            rh_ = 1  # use %
        else:
            dt_ = 273.15  # use K
            rh_ = 100  # use absolute value
        return dt_, rh_


    @np.vectorize
    def _rayleigh(ts_c, ti_c, length_, rh=10):
        """
        Calculate dimensionless Rayleigh number Ra.
        """
        dtk, rhf = self._pyfluids_units()
        tm_c = (ts_c + ti_c)/2

        # Humid air properties at p0 and Tm (indoors).
        air = HumidAir().with_state(
                  InputHumidAir.pressure(self.P0),
                  InputHumidAir.temperature(tm_c + dtk),
                  InputHumidAir.relative_humidity(rh/rhf),
              )
        rho = air.density
        cp = air.specific_heat
        uv = air.kinematic_viscosity  # m^2/s
        k = air.conductivity  # W/(m K)
        alpha = k/(rho*cp)  # m^2/s thermal diffusivity
        beta = 1/(tm_c + self.T0)

        # Eqn. 9.25, page 571.
        ra = g*beta*abs(ts_c - ti_c)*length_*length_*length_/alpha/uv

        return ra

    @np.vectorize
    def _nusselt_horizontal(ra, pr, hflux='z+'):
        """
        Calculate dimensionless Nusselt number Nu for horizontal plates for various
        cases of heat flux direction.
        """
        if hflux == 'z+':  # warm plate, top face or cold plate, bottom face
            if ra < 1E04:
                nu = 0.54*math.pow(1E04,0.25)
            elif (1E04 <= ra <= 1E07) and (pr >= 0.7):
                nu = 0.54*math.pow(ra,0.25)
            elif 1E07 < ra <= 1E11:
                nu = 0.15*math.pow(ra,0.33333)
            else:
                nu = 0.15*math.pow(1E11,0.33333)
        else:  # warm plate, bottom face or cold plate, top face
            if ra < 1E04:
                nu = 0.52*math.pow(1E04,0.2)
            elif 1E04 <= ra <= 1E09 and pr >= 0.7:
                nu = 0.52*math.pow(ra,0.2)
            else:
                nu = 0.52*math.pow(1E09,0.2)
        return nu


    @np.vectorize
    def hc_fun(ts, tamb):
        """
        Calculate convection coefficient for horizontal plate.
        """
        dtk, rhf = self._pyfluids_units()
        tm_c = (ts + tamb)/2
        rh = 50

        air = HumidAir().with_state(
                  InputHumidAir.pressure(self.P0),
                  InputHumidAir.temperature(tm_c + dtk),
                  InputHumidAir.relative_humidity(rh/rhf),
              )
    
        l = 1.25  # assume slab size 5 x 5 m^2 for h_c
        ra = rayleigh(ts, tamb, l)
        if ts > tamb:
            nu = nusselt_horizontal(ra, air.prandtl, hflux='z+')
        else:
            nu = nusselt_horizontal(ra, air.prandtl, hflux='z-')
    
        return(nu*air.conductivity/l)
