# SPDX-License-Identifier: GPL-3.0-or-later

from ._neo_hooke_nearly_incompressible import NeoHooke
from ._volumetric_strain_energy import _volumetric_strain_energy


class Volumetric(NeoHooke):
    r"""Neo-Hookean material formulation with deactivated shear modulus, i.e. the
    volumetric part of the strain energy density function :math:`U(J)` only.

    Parameters
    ----------
    bulk : float or None, optional
        Bulk modulus of the volumetric part of the strain energy density function. It is
        required if ``volumetric`` is a kind and it must be None if ``volumetric`` is an
        instance, which holds its own parameters (default is None).
    parallel : bool, optional
        A flag to evaluate the gradient and the hessian in chunks of cells by a pool of
        threads (default is False).
    volumetric : str or VolumetricStrainEnergy, optional
        The volumetric part of the strain energy density function :math:`U(J)`, given by
        a kind with the bulk modulus, ``"volume-ratio"``, ``"stretch"`` or ``"log"``, or
        by an instance of :class:`~felupe.VolumetricStrainEnergy`, e.g. of a custom
        subclass. Default is ``"volume-ratio"``.

    Examples
    --------
    The volumetric part is combined with a material formulation for the distortional
    part of the strain energy density function.

    >>> import felupe as fem
    >>>
    >>> umat = fem.NeoHooke(mu=1.0) & fem.Volumetric(bulk=5000, volumetric="stretch")

    See Also
    --------
    felupe.NeoHooke : Nearly-incompressible isotropic hyperelastic Neo-Hookean material
        formulation.
    felupe.VolumetricStrainEnergy : The volumetric part of the strain energy density
        function as a function of the volume ratio.
    """

    def __init__(self, bulk=None, parallel=False, volumetric="volume-ratio"):
        if _volumetric_strain_energy(volumetric, bulk) is None:
            raise ValueError(
                f'volumetric="{volumetric}" requires the bulk modulus, e.g. bulk=5000.'
            )

        super().__init__(mu=None, bulk=bulk, parallel=parallel, volumetric=volumetric)
