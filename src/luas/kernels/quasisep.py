import jax.numpy as jnp
import jax
from jax import vmap
import numpy as np
from typing import Callable
import tinygp
import tinygp.kernels.quasisep
import equinox as eqx

from luas.luas_types import JAXArray, Scalar, PyTree, is_scalar
import luas.kernels.covtype as covtype
import luas.kernels.tinygp_ext
from luas.kernels.tinygp_ext import HandleIdx, ScaledKernel

__all__ = [
    "Exp",
    "Matern32",
    "Matern52",
    "Matern",
    "Outer",
    "Constant",
    "SHO",
    "Cosine",
    "KroneckerDelta",
    "Noise",
]

from luas.kernels.diagonal import KroneckerDelta, Noise



def CustomTinygp(kf_tinygp: Callable, params = None) -> JAXArray:
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

    return covtype.GeneralQuasisepPlusNoise(HandleIdx(kf_tinygp), params = params)


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
    assert is_scalar(const)

    return covtype.Outer(alpha = jnp.sqrt(const))


def Linear(alpha: JAXArray, sigma: Scalar = 1., use_block: bool = True, const: None = None) -> JAXArray:
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
    assert alpha.ndim == 1

    if const is not None:
        tinygp_kf = luas.kernels.tinygp_ext.Linear(alpha = sigma * alpha, const = const)
        return covtype.GeneralQuasisepPlusNoise(tinygp_kf, use_block = use_block)
    else:
        return covtype.Outer(alpha)


def Banded(diag: JAXArray, off_diags: JAXArray, use_block: bool = True) -> covtype.CovType:
    
    symm_qsm = tinygp.noise.Banded(diag, off_diags).to_qsm()

    # Don't use HandleIdx here as need the location in matrix for non-stationary kernels
    return covtype.GeneralQuasisepPlusNoise(None, noise_model = symm_qsm, use_block = use_block)


def ConstantBlocks(endpoints: JAXArray, sigma: Scalar = 1., use_block: bool = True) -> JAXArray:
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

    tinygp_kf = luas.kernels.tinygp_ext.ConstantBlocks(endpoints = endpoints, sigma = sigma)
    return covtype.GeneralQuasisep(HandleIdx(tinygp_kf), use_block = use_block)


def Exp(scale: Scalar, sigma: Scalar = 1., fast_eigen = False) -> JAXArray:
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
    exp_kernel = tinygp.kernels.quasisep.Exp(scale = scale, sigma = sigma)
    return covtype.Exp(HandleIdx(exp_kernel), scale, sigma = sigma, fast_eigen = fast_eigen)


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
    
    return covtype.GeneralQuasisep(HandleIdx(tinygp.kernels.quasisep.Matern32(scale = scale, sigma = sigma)))
    

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
    
    return covtype.GeneralQuasisep(HandleIdx(tinygp.kernels.quasisep.Matern52(scale = scale, sigma = sigma)))


def MaternHalfInt(scale: Scalar, double_nu: int, sigma = 1.) -> JAXArray:
    r"""Matern half-integer kernel function
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{5} \frac{|x - y|}{L} + \frac{5|x - y|^2}{3L^2}\Bigg) \exp\Bigg( -\sqrt{5}\frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    tinygp_kf = luas.kernels.tinygp_ext.MaternHalfInt(scale = scale, double_nu = double_nu, sigma = sigma)
    return covtype.GeneralQuasisep(HandleIdx(tinygp_kf))


def SHO(omega: Scalar, quality: Scalar, sigma = 1.) -> JAXArray:
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
    
    return covtype.GeneralQuasisep(HandleIdx(tinygp.kernels.quasisep.SHO(omega = omega, quality = quality, sigma = sigma)))


def Cosine(scale: Scalar, sigma: Scalar = 1.) -> JAXArray:
    r"""Cosine kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices which have periodic covariance.
    
    .. math::

        k(x, y) = \cos\Bigg(\frac{2\pi|x - y|}{P}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        scale (Scalar): Period
        
    Returns:
        JAXArray: Covariance between two input vectors
        
    """

    return covtype.PeriodicQuasisep(HandleIdx(tinygp.kernels.quasisep.Cosine(scale = scale, sigma = sigma)))



def ExpSineSquaredApprox(
    period: JAXArray | float,
    gamma: JAXArray | float,
    sigma: Scalar = 1.,
    order: int = 5,
    use_block: bool = True,
):
    # Note defaults to an order of 5 which can be inaccurate
    kf_tinygp = luas.kernels.tinygp_ext.ExpSineSquaredApprox(period=period, gamma=gamma,
                                                             sigma=sigma, order=order,
                                                             use_block = use_block)
    return covtype.PeriodicQuasisep(HandleIdx(kf_tinygp))


def QuasiperiodicApprox(
    period: JAXArray | float,
    gamma: JAXArray | float,
    decay_kernel: covtype.CovType,
    sigma: Scalar = 1.,
    order: int = 5,
    use_block: bool = True,
):
    kf_tinygp = luas.kernels.tinygp_ext.QuasiperiodicApprox(period=period, gamma=gamma, decay_kernel = decay_kernel.tinygp_kf,
                                                            sigma=sigma, order=order)
    return covtype.GeneralQuasisep(HandleIdx(kf_tinygp), use_block = use_block)


def SquaredExpApprox(scale: Scalar, sigma: Scalar = 1., order: int = 6, use_block = True) -> JAXArray:
    """Taylor-spectrum squared-exponential approximation as a quasisep kernel.

    This returns a ``covtype.GeneralQuasisep`` wrapper so it plugs straight into
    existing ``luas`` covariance composition patterns.
    """
    kf_tinygp = luas.kernels.tinygp_ext.SquaredExpApprox(scale=scale, sigma=sigma, order=order,
                                                         use_block = use_block)
    return covtype.GeneralQuasisep(HandleIdx(kf_tinygp))

# def SquaredExpApprox2(scale: Scalar, sigma: Scalar = 1.0, order: int = 6, use_block = True) -> JAXArray:
#     """Taylor-spectrum squared-exponential approximation as a quasisep kernel.

#     This returns a ``covtype.GeneralQuasisep`` wrapper so it plugs straight into
#     existing ``luas`` covariance composition patterns.
#     """
#     kf_tinygp = luas.kernels.tinygp_ext.SquaredExpApprox2(scale=scale, sigma=sigma, order=order,
#                                                          use_block = use_block)
#     return covtype.GeneralQuasisep(HandleIdx(kf_tinygp))
