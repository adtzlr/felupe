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

import math

from pyfluids import HumidAir, InputHumidAir
from scipy.constants import g


class FreeConvection:
    r"""Free convection heat transfer formulation for flat plates.

    <<todo: add top/bottom of plate switch! >>

    Parameters
    ----------
    plate_width : float
        Horizontal plate width in (m).
    plate_length : float
        Horizontal plate length in (m).
    p_abs: float (optional, default 101325)
        Absolute (total) air pressure in (Pa).
    rh : float (optional, default 50 %)
        Relative humidity of air in (%).

    Notes
    -----
    This class represents ...

    References
    ----------
    ..  [1] F. P. Incropera, D. P. DeWitt, and et. al., Fundamentals of Heat
            and Mass Transfer, 6th Edition. John Wiley & Sons, 2007;
            ISBN 0-471-45728-0.

    See Also
    --------
    felupe.thermal.SolidBodyThermal : A thermal solid body for heat conduction.
    felupe.thermal.SolidBodySurfaceConvection : Detailed surface convection
      heat transfer.


    """

    def __init__(self, plate_width, plate_length, p_abs=101325, rh=50):
        self.T0 = 273.15  # 0 °C in Kelvin, for °C <=> K conversion
        self.plate_width = plate_width
        self.plate_length = plate_length
        self.rh = rh
        self.p_abs = p_abs

        # Characteristic length for horizontal plate.
        self.length = self.plate_width*self.plate_length/\
                           (2*(self.plate_width+self.plate_length))

    def _pyfluids_units(self):
        check = HumidAir().factory()
        if str(check.units_system) == 'SIWithCelsiusAndPercents':
            dt_ = 0  # use °C
            rh_ = 1  # use %
        else:
            dt_ = 273.15  # use K
            rh_ = 100  # use absolute value
        return dt_, rh_


    def _rayleigh(self, ts_c, ti_c, length_, rh=10):
        """
        Calculate dimensionless Rayleigh number Ra.
        """
        dtk, rhf = self._pyfluids_units()
        tm_c = (ts_c + ti_c)/2

        # Humid air properties at p0 and Tm (indoors).
        air = HumidAir().with_state(
                  InputHumidAir.pressure(self.p_abs),
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

    def _nusselt_horizontal(self, ra, pr, hflux='z+'):
        """
        Calculate dimensionless Nusselt number Nu for horizontal plates for various
        cases of heat flux direction.
        """
        if hflux == 'z+':  # warm plate, top face or cold plate, bottom face
            if ra < 1E04:
                nu = 0.54*math.pow(1E04, 0.25)
            elif (1E04 <= ra <= 1E07) and (pr >= 0.7):
                nu = 0.54*math.pow(ra, 0.25)
            elif 1E07 < ra <= 1E11:
                nu = 0.15*math.pow(ra, 0.33333)
            else:
                nu = 0.15*math.pow(1E11, 0.33333)
        else:  # warm plate, bottom face or cold plate, top face
            if ra < 1E04:
                nu = 0.52*math.pow(1E04, 0.2)
            elif 1E04 <= ra <= 1E09 and pr >= 0.7:
                nu = 0.52*math.pow(ra, 0.2)
            else:
                nu = 0.52*math.pow(1E09, 0.2)
        return nu


    def hc_fun(self, ts, tamb):
        """
        Calculate convection coefficient for horizontal plate.
        """
        dtk, rhf = self._pyfluids_units()
        tm_c = (ts + tamb)/2

        air = HumidAir().with_state(
                  InputHumidAir.pressure(self.p_abs),
                  InputHumidAir.temperature(tm_c + dtk),
                  InputHumidAir.relative_humidity(self.rh/rhf),
              )

        ra = self._rayleigh(ts_c=ts, ti_c=tamb, length_=self.length)
        if ts > tamb:
            nu = self._nusselt_horizontal(ra, air.prandtl, hflux='z+')
        else:
            nu = self._nusselt_horizontal(ra, air.prandtl, hflux='z-')

        return(nu*air.conductivity/self.length)
