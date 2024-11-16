from ._miehe_goektepe_lulei import miehe_goektepe_lulei
from ._mooney_rivlin import mooney_rivlin
from ._storakers import storakers
from ._third_order_deformation import third_order_deformation
from ._yeoh import yeoh

__all__ = [
    "miehe_goektepe_lulei",
    "mooney_rivlin",
    "storakers",
    "third_order_deformation",
    "yeoh",
]

# default (stable) material parameters
miehe_goektepe_lulei.kwargs = dict(mu=0, N=100, U=0, p=2, q=2)
mooney_rivlin.kwargs = dict(C10=0, C01=0)
storakers.kwargs = dict(mu=[0], alpha=[2], beta=[1])
third_order_deformation.kwargs = dict(C10=0, C01=0, C11=0, C20=0, C30=0)
yeoh.kwargs = dict(C10=0, C20=0, C30=0)
