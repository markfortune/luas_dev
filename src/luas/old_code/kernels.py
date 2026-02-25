import jax.numpy as jnp
import numpy as np
from jax import vmap
from typing import Callable
from .luas_types import JAXArray, Scalar, PyTree, is_scalar
import luas.covtype

__all__ = [
    "evaluate_kernel",
    "distanceL1",
    "distanceL2Sq",
    "Custom",
    "Exp",
    "SquaredExp",
    "PoweredExp",
    "Matern32",
    "Matern52",
    "MaternNuHalf",
    "RationalQuadratic",
    "ExpSineSquared",
    "Cosine",
    "Constant",
    "Outer",
]

def evaluate_kernel(kernel_fn: Callable, x: JAXArray, y: JAXArray, *args) -> JAXArray:
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
    
    K = vmap(lambda x1: vmap(lambda y1: kernel_fn(x1, y1, *args))(y))(x)
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



def Custom(kf: Callable, hp: PyTree, params = None) -> JAXArray:
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

    return luas.covtype.General(lambda x1, x2, **kwargs: kf(hp, x1, x2, **kwargs), params = params)



def Exp(scale: Scalar) -> JAXArray:
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
    
    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(exp_calc, x, y, scale))

def exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by ``luas.kernels.exp`` to evaluate the Exponential kernel function. 
    
    """

    delta_t = distanceL1(x, y, L).sum()
    return jnp.exp(-delta_t)

    
def SquaredExp(scale: Scalar) -> JAXArray:
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
    
    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(squared_exp_calc, x, y, scale))
ExpSquared = SquaredExp
    
def squared_exp_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by ``luas.kernels.squared_exp`` to evaluate the squared exponential kernel function. 
    
    """

    tau_sq = distanceL2Sq(x, y, L)
    return jnp.exp(-0.5 * tau_sq.sum())


def PoweredExp(scale: Scalar, k: Scalar) -> JAXArray:
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

    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(powered_exp_calc, x, y, scale, k), params = jnp.array([scale, k]))
    
    
def powered_exp_calc(x: JAXArray, y: JAXArray, L: Scalar, k: Scalar) -> JAXArray:
    """Function used by squared_exp to evaluate the squared exponential kernel function. 
    """

    tau_sq = jnp.sum(jnp.power(jnp.abs(x - y), k)/jnp.power(L, k))
    return jnp.exp(-tau_sq.sum())



def Matern32(scale: Scalar) -> JAXArray:
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

    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(matern32_calc, x, y, scale))


def matern32_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern32 to evaluate the Matern 3/2 kernel function. 
    
    """

    delta_t = jnp.sqrt(3)*distanceL1(x, y, L).sum()
    return (1+delta_t)*jnp.exp(-delta_t)
    

def Matern52(scale: Scalar) -> JAXArray:
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
    
    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(matern52_calc, x, y, scale))

def matern52_calc(x: JAXArray, y: JAXArray, L: Scalar) -> JAXArray:
    """Function used by matern52 to evaluate the Matern 5/2 kernel function. 
    
    """

    delta_t = jnp.sqrt(5)*distanceL1(x, y, L).sum()
    return (1+delta_t+jnp.square(delta_t)/3)*jnp.exp(-delta_t)
    
def MaternNuHalf(scale: Scalar, nu) -> JAXArray:
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
    
    return evaluate_kernel(matern_p_calc, x, y, L, p)


def matern_p_calc(x: JAXArray, y: JAXArray, L: Scalar, p) -> JAXArray:
    """Function used by matern52 to evaluate the Matern 5/2 kernel function. 
    
    """

    f = jnp.sqrt(2*p + 1)
    delta_t = f*distanceL1(x, y, L).sum()

    const = jax.scipy.special.factorial(p)/jax.scipy.special.factorial(2*p)

    poly_term = 0
    for i in jnp.arange(p+1):
        bin_coeff = jax.scipy.special.factorial(p + i)/(jax.scipy.special.factorial(i)*jax.scipy.special.factorial(p-i))
        poly_term += bin_coeff * (2*delta_t)**(p-i)
    return const * poly_term * jnp.exp(-delta_t)

  
def RationalQuadratic(scale: Scalar, alpha: Scalar) -> JAXArray:
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
    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(rational_quadratic_calc, x, y, scale, alpha))


def rational_quadratic_calc(x: JAXArray, y: JAXArray, L: Scalar, alpha: Scalar) -> JAXArray:
    """Function used by rational_quadratic to evaluate the rational quadratic kernel function. 
    
    """

    tau_sq = distanceL2Sq(x, y, L).sum()
    return (1. + 0.5*tau_sq/alpha)**(-alpha)


def ExpSineSquared(scale: Scalar, P: Scalar) -> JAXArray:
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
    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(exp_sine_squared_calc, x, y, scale, P))


def exp_sine_squared_calc(x: JAXArray, y: JAXArray, L: Scalar, P: Scalar) -> JAXArray:
    """Function used by exp_sine_squared to evaluate the exponential sine squared kernel function.
    
    """

    tau_sq = (jnp.sum(jnp.square(jnp.sin(jnp.pi*(x - y)/P)/L))).sum()
    return jnp.exp(-2.0 * tau_sq)
    

def Cosine(P: Scalar) -> JAXArray:
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

    return luas.covtype.General(lambda hp, x, y, **kwargs: evaluate_kernel(cosine_calc, x, y, P))


def cosine_calc(x: JAXArray, y: JAXArray, P: Scalar) -> JAXArray:
    """Function used by cosine to evaluate the cosine kernel function. 
    
    """
        
    delta_t = distanceL1(x, y, P).sum()
    return jnp.cos(2*jnp.pi*delta_t)


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
    
    return luas.covtype.Outer(alpha = const)

  
    
def Outer(alpha: JAXArray) -> JAXArray:
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
    
    return luas.covtype.Outer(alpha = alpha)

