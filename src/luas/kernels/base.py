import jax.numpy as jnp
import numpy as np
import jax
from jax import vmap
from jax.scipy.special import gammaln
from typing import Callable
from luas.luas_types import JAXArray, Scalar, PyTree, is_scalar
import luas.kernels.covtype as covtype

__all__ = [
    "KroneckerDelta",
    "Noise",
    "evaluate_kernel",
    "distanceL1",
    "distanceL2Sq",
    "Exp",
    "SquaredExp",
    "Matern32",
    "Matern52",
    "RationalQuadratic",
    "ExpSineSquared",
    "Cosine",
    "PoweredExp",
]

from luas.kernels.diagonal import KroneckerDelta, Noise

def evaluate_kernel(kernel_fn: Callable, x: JAXArray, y: JAXArray, *args, axes = None) -> JAXArray:
    """Uses the ``jax.vmap`` function to efficiently build the covariance matrix from
    a given kernel function.
    
    Args:
        kernel_fn (Callable): The desired kernel function to use
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        l (Scalar): Length scale
    
    Returns:
        JAXArray: The constructed covariance matrix
        
    """
    if axes is not None:
        K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, *args), in_axes = -1)(y[axes, :]), in_axes = -1)(x[axes, :])
    else:
        K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, *args), in_axes = -1)(y), in_axes = -1)(x)
    return K
    

def distanceL1(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Evaluates the L1 norm of two input vectors divided by a length scale.
    
    .. math::

        L1(x, y) = \frac{|x - y|}{L}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: L1 norm between two input vectors
        
    """
    
    return jnp.sum(jnp.abs(x - y)/L)


def distanceL2Sq(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    r"""Evaluates the Squared L2 norm of two input vectors divided by the length scale ``L``.
    
    .. math::

        L2^2(x, y) = \frac{|x - y|^2}{L^2}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: L2 norm between two input vectors
        
    """
    
    return jnp.sum(jnp.square(x - y)/L**2)
    
    
def SquaredExp(scale: Scalar, sigma: Scalar = 1., axes: JAXArray | None = None) -> JAXArray:
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
    
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(squared_exp_calc, x, y, scale))
ExpSquared = SquaredExp
    
def squared_exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by ``luas.kernels.squared_exp`` to evaluate the squared exponential kernel function. 
    
    """

    tau_sq = distanceL2Sq(x, y, L)
    return jnp.exp(-0.5 * tau_sq.sum())


def Exp(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
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
    
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(exp_calc, x, y, scale))

def exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by ``luas.kernels.exp`` to evaluate the Exponential kernel function. 
    
    """

    delta_t = distanceL1(x, y, L).sum()
    return jnp.exp(-delta_t)

def Custom(kf: Callable, hp: PyTree, params = None, kf_args = (), kf_kwargs = {}) -> JAXArray:
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
        L (Scalar): Length scale
        k (Scalar): Exponent which can take any positive real values between [0, 2]
        
    Returns:
        Scalar: Covariance between two input vectors
    """
    custom_kf = lambda _, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs: kf(hp, x1, x2,
                                                                                            *kf_args, **kf_kwargs,
                                                                                            **kwargs)
    return covtype.General(custom_kf, params = params)


def Fixed(cov_mat: JAXArray, params = None, ignore_idx = False, sigma: Scalar = 1.) -> JAXArray:

    def kf(hp, x1, x2, full = True, row_idx = None, col_idx = None, **kwargs):

        if not full and not ignore_idx:
            return sigma**2 * cov_mat[jnp.ix_(row_idx, col_idx)]
        else:
            return sigma**2 * cov_mat
        
    return covtype.General(kf, params = params)


def PoweredExp(scale: Scalar, k: Scalar, sigma: Scalar = 1.) -> JAXArray:
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
        L (Scalar): Length scale
        k (Scalar): Exponent which can take any positive real values between [0, 2]
        
    Returns:
        Scalar: Covariance between two input vectors
    """

    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(powered_exp_calc, x, y, scale, k))
    
    
def powered_exp_calc(x: JAXArray, y: JAXArray, L: Scalar, k: Scalar) -> JAXArray:
    """Function used by powered_exp to evaluate the powered exponential kernel function. 
    """

    tau_sq = jnp.power(distanceL1(x, y, L), k)
    return jnp.exp(-tau_sq.sum())


def Linear(alpha: JAXArray, sigma: Scalar = 1.) -> JAXArray:
    r"""Matern 3/2 kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{3} \frac{|x - y|}{L}\Bigg) \exp\Bigg( -\sqrt{3} \frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    return covtype.Outer(alpha = sigma * alpha)


def Constant(const: JAXArray) -> JAXArray:
    r"""Matern 3/2 kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{3} \frac{|x - y|}{L}\Bigg) \exp\Bigg( -\sqrt{3} \frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    return covtype.Outer(alpha = jnp.sqrt(const))


def Matern32(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Matern 3/2 kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{3} \frac{|x - y|}{L}\Bigg) \exp\Bigg( -\sqrt{3} \frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """

    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(matern32_calc, x, y, scale))


def matern32_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern32 to evaluate the Matern 3/2 kernel function. 
    
    """

    delta_t = jnp.sqrt(3)*distanceL1(x, y, L).sum()
    return (1+delta_t)*jnp.exp(-delta_t)
    

def Matern52(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
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
    
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(matern52_calc, x, y, scale))

def matern52_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern52 to evaluate the Matern 5/2 kernel function. 
    
    """

    delta_t = jnp.sqrt(5)*distanceL1(x, y, L).sum()
    return (1+delta_t+jnp.square(delta_t)/3)*jnp.exp(-delta_t)
    

def MaternHalfInt(scale: Scalar, double_nu: int, sigma: Scalar = 1.) -> JAXArray:
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
    p = (double_nu - 1)//2
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(matern_half_int_calc, x, y, scale, p))


def matern_half_int_calc(x: JAXArray, y: JAXArray, scale: Scalar, p: int) -> JAXArray:
    """Function used by matern52 to evaluate the Matern 5/2 kernel function. 
    
    """

    f = jnp.sqrt(2*p + 1)
    delta_t = f*distanceL1(x, y, scale).sum()

    # const = jax.scipy.special.factorial(p)/jax.scipy.special.factorial(2*p)
    # poly_term = 0
    # for i in jnp.arange(p+1):
        # bin_coeff = jax.scipy.special.factorial(p + i)/(jax.scipy.special.factorial(i)*jax.scipy.special.factorial(p-i))
        # poly_term += bin_coeff * (2*delta_t)**(p-i)

    const = gammaln(p + 1) - gammaln(2*p + 1)

    ind = jnp.arange(p+1)
    poly_term = jnp.exp(gammaln(p + ind + 1) - gammaln(ind + 1) - gammaln(p - ind + 1))
    poly_term *= (2*delta_t)**(p-ind)

    return jnp.exp(const - delta_t) * poly_term.sum()

    
def RationalQuadratic(scale: Scalar, alpha: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Rational quadratic kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \frac{|x - y|^2}{2 \alpha L^2}\Bigg)^{-\alpha}
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        alpha (Scalar): Scale mixture parameter
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(rational_quadratic_calc, x, y, scale, alpha))


def rational_quadratic_calc(x: JAXArray, y: JAXArray, L: Scalar, alpha: Scalar) -> JAXArray:
    """Function used by rational_quadratic to evaluate the rational quadratic kernel function. 
    
    """

    tau_sq = distanceL2Sq(x, y, L).sum()
    return (1. + 0.5*tau_sq/alpha)**(-alpha)


def ExpSineSquared(gamma: Scalar, period: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Exponential sine squared kernel, used with evaluate_kernel
    to build covariance matrices which have periodic covariance.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{2 \sin^2(\pi(x - y)/P)}{L^2}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        P (Scalar): Period
        
    Returns:
        JAXArray: Covariance between two input vectors
        
    """
    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(exp_sine_squared_calc, x, y, gamma, period))


def exp_sine_squared_calc(x: JAXArray, y: JAXArray, gamma: Scalar, period: Scalar) -> JAXArray:
    """Function used by exp_sine_squared to evaluate the exponential sine squared kernel function.
    
    """

    sine_sq = jnp.square(jnp.sin(jnp.pi*(x - y)/period))
    return jnp.exp(-gamma * sine_sq).sum()
    

def Cosine(period: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Cosine kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices which have periodic covariance.
    
    .. math::

        k(x, y) = \cos\Bigg(\frac{2\pi|x - y|}{P}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        P (Scalar): Period
        
    Returns:
        JAXArray: Covariance between two input vectors
        
    """

    return covtype.General(lambda hp, x, y, **kwargs: sigma**2 * evaluate_kernel(cosine_calc, x, y, period))


def cosine_calc(x: JAXArray, y: JAXArray, period: Scalar) -> JAXArray:
    """Function used by cosine to evaluate the cosine kernel function. 
    
    """
        
    delta_t = distanceL1(x, y, period).sum()
    return jnp.cos(2*jnp.pi*delta_t)

