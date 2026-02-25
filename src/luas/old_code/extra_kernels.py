import jax.numpy as jnp
from jax import vmap
from typing import Callable
from .luas_types import JAXArray, Scalar
from .kernels import evaluate_kernel, distanceL1, distanceL2Sq
from sklearn.gaussian_process.kernels import Matern

__all__ = [
    "SHO_gran",
    "powered_exp",
    "matern",
]

def SHO_gran(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Exponential kernel function, also known as the Matern 1/2 kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(\frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    return evaluate_kernel(SHO_gran_calc, x, y, L)


def SHO_gran_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by ``luas.kernels.exp`` to evaluate the Exponential kernel function. 
    
    """

    delta_t = distanceL1(x, y, L).sum()
    return jnp.sqrt(2)*jnp.exp(-delta_t/jnp.sqrt(2))*jnp.cos(delta_t/jnp.sqrt(2) - jnp.pi/4.)


def powered_exp(x: JAXArray, y: JAXArray, L_k: Scalar, k: Scalar) -> JAXArray:
    r"""Powered exponential kernel function, a family of kernel functions which
    include the exponential and squared exponential kernels as special cases. 
    Equivalent to the exponential kernel for k = 1 and the squared exponential kernel
    for k = 2 (although the length scales will differ by sqrt(2) because 2 is not in
    the denominator inside the exponent in this function).
    Used with evaluate_kernel to build a covariance matrix.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{|x - y|^k}{L^k}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L_k (Scalar): Length scale to power k
        k (Scalar): Exponent which can take any positive real values.
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    
    return evaluate_kernel(powered_exp_calc, x, y, L_k, k)
    
    
def powered_exp_calc(x: JAXArray, y: JAXArray, L_k: Scalar, k: Scalar) -> JAXArray:
    """Function used by squared_exp to evaluate the squared exponential kernel function. 
    """

    tau_sq = jnp.sum(jnp.power(jnp.abs(x - y), k)/L_k)
    return jnp.exp(-tau_sq.sum())



def matern(x: JAXArray, y: JAXArray, L: Scalar, nu: Scalar) -> JAXArray:
    kernel = Matern(length_scale = L, nu = nu)
    
    if x.ndim == 1:
        x = x.reshape((x.size, 1))
    if y.ndim == 1:
        y = y.reshape((y.size, 1))
        
    return kernel.__call__(x, y)



def triangular(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Triangular kernel, used with evaluate_kernel
    to build covariance matrices which have bounded covariance.
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        JAXArray: Covariance between two input vectors
    """
    
    return evaluate_kernel(triangular_calc, x, y, L)


def triangular_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by cosine to evaluate the triangular kernel function. 
    """
        
    delta_t = distanceL1(x, y, L).sum()
    return jnp.heaviside(1 - delta_t, 0.)*(1 - delta_t)
