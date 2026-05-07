import jax.numpy as jnp
from .luas_types import JAXArray, Scalar
from ...old_code.kernels import evaluate_kernel
from sklearn.gaussian_process.kernels import Matern
from luas.kernels import covtype

__all__ = [
    "Matern",
]

def Matern(scale: Scalar, nu: Scalar, axes: JAXArray | None = None):
    r"""Squared exponential kernel function, also known as the radial basis function,
    used with ``luas.kernels.evaluate_kernel`` to build a covariance matrix.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{|x - y|^2}{2L^2}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    return covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(matern_calc, x, y, scale, nu))


def matern_calc(x: JAXArray, y: JAXArray, scale: Scalar, nu: Scalar) -> JAXArray:
    kernel = Matern(length_scale = scale, nu = nu)
    
    if x.ndim == 1:
        x = x.reshape((x.size, 1))
    if y.ndim == 1:
        y = y.reshape((y.size, 1))
        
    return kernel.__call__(x, y)

