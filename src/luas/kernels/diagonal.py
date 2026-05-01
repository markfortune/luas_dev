import jax.numpy as jnp
import numpy as np
from jax import vmap
from typing import Callable
from luas.luas_types import JAXArray, Scalar, PyTree, is_scalar
import luas.kernels.covtype as covtype

__all__ = [
    "Independent",
    "Noise",
]
    
def KroneckerDelta(sigma: JAXArray | Scalar = None) -> JAXArray:
    if sigma is not None:
        if is_scalar(sigma):
            return covtype.ScaledIdentity(diag = sigma**2)
        else:
            return covtype.Diagonal(diag = sigma**2)
    else:
        return covtype.Identity()


def Noise(sigma: JAXArray | Scalar) -> JAXArray:
    r"""Matern 5/2 kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{5} \frac{|x - y|}{L} + \frac{5|x - y|^2}{3L^2}\Bigg) \exp\Bigg( -\sqrt{5}\frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    if is_scalar(sigma):
        return covtype.ScaledIdentity(wn_diag = sigma**2)
    else:
        return covtype.Diagonal(wn_diag = sigma**2)

 