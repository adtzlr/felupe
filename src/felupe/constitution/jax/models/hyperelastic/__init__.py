from ._blatz_ko import blatz_ko
from ._miehe_goektepe_lulei import miehe_goektepe_lulei
from ._mooney_rivlin import mooney_rivlin
from ._storakers import storakers
from ._third_order_deformation import third_order_deformation
from ._van_der_waals import van_der_waals
from ._yeoh import yeoh

__all__ = [
    "blatz_ko",
    "miehe_goektepe_lulei",
    "mooney_rivlin",
    "storakers",
    "third_order_deformation",
    "van_der_waals",
    "yeoh",
]

# default (stable) material parameters
blatz_ko.kwargs = dict(mu=1)
miehe_goektepe_lulei.kwargs = dict(mu=0, N=100, U=0, p=2, q=2)
mooney_rivlin.kwargs = dict(C10=0, C01=0)
storakers.kwargs = dict(mu=[0], alpha=[2], beta=[1])
third_order_deformation.kwargs = dict(C10=0, C01=0, C11=0, C20=0, C30=0)
van_der_waals.kwargs = dict(mu=0, beta=0, a=0, limit=100)
yeoh.kwargs = dict(C10=0, C20=0, C30=0)
