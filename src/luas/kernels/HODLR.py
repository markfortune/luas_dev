import numpy as np
from typing import Callable
from luas.luas_types import Scalar, PyTree, is_scalar, Kernel
from luas.kernels.covtype import CovType, Diagonal, Identity, ScaledIdentity
import george
import jax.numpy as jnp

__all__ = [
    "Exp",
    "Matern32",
    "Matern52",
    "Constant",
    "Cosine",
]


class HODLR(CovType):
    """HODLR covariance backed by ``george.HODLRSolver`` (non-JAX backend).

    This class is useful for large dense-like kernels where hierarchical low-rank
    structure is exploited by george. Because george is not JAX-native, this
    class is best used with callback-aware higher-level kernels when JIT tracing
    is involved.
    """

    def __init__(self, kf, diag = 1e-14, wn_diag = 0., tol = 1e-10, params = None):
        self.kf_hodlr = kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.tol = tol
        self.params = params

        self.gp_hodlr = george.GP(kf, white_noise = np.log(self.diag + self.wn_diag),
                                  solver=george.HODLRSolver, tol=self.tol)

    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        return self.gp_hodlr.get_matrix(x1, x2) + np.diag((self.diag + wn*self.wn_diag)*np.ones(x1.shape[-1]))
    
    def decompose(self, x, wn = True, **kwargs):
        
        self.gp_hodlr.compute(np.asarray(x))
        self.logdet = self.gp_hodlr.solver.log_determinant
        return self, {"logdetK": self.logdet}

    def matrix_sqrt(self, R, **kwargs):
         raise Exception("george backend doesn't implement matrix_inv_sqrt for HODLR solver")

    def matrix_sqrt(self, R, **kwargs):
         raise Exception("george backend doesn't implement matrix_sqrt for HODLR solver")

    def dot_solve(self, R):
        return self.gp_hodlr.solver.dot_solve(R)

    def logL(self, R, stored_values, **kwargs):

        return self.gp_hodlr.log_likelihood(R)

    def inverse(self, R, **kwargs):
        
        return self.gp_hodlr.apply_inverse(R)

    def scale(self, c):
        return HODLR(self.kf_hodlr * c, diag = self.diag * c, wn_diag = self.wn_diag * c, tol = self.tol)

    def __add__(self, K):

        if type(K) in [HODLR]:
            K_sum = HODLR(self.kf_hodlr + K.kf_hodlr, diag = self.diag + K.diag,
                          tol = self.tol, wn_diag = self.wn_diag + K.wn_diag)
        elif type(K) in [Diagonal, Identity, ScaledIdentity]:
            K_sum = HODLR(self.kf_hodlr, diag = self.diag + K.diag,
                          tol = self.tol, wn_diag = self.wn_diag + K.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply kernel with other kernels"
            )
        return self.scale(other)


def CustomGeorge(kf_george: Callable, params = None):
    r"""Powered exponential kernel function, a family of kernel functions which
    include the exponential and squared exponential kernels as special cases. 
    Equivalent to the exponential kernel for k = 1 and the squared exponential kernel
    for k = 2 (although the length scales will differ by sqrt(2) because 2 is not in
    the denominator inside the exponent in this function).
    Used with evaluate_kernel to build a covariance matrix.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{|x - y|^k}{L^k}\Bigg)
    
    Args:
        L (Scalar): Length scale
        k (Scalar): Exponent which can take any positive real values between [0, 2]
        
    Returns:
        Scalar: Covariance between two input vectors
    """

    return HODLR(kf_george, params = params)
    

def Exp(scale: Scalar):
    r"""Exponential kernel function, also known as the Matern 1/2 kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(\frac{|x - y|}{L}\Bigg)
    
    Args:
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """

    return HODLR(george.kernels.ExpKernel(metric = scale**2))


def ExpSquared(scale: Scalar):
    r"""Exponential kernel function, also known as the Matern 1/2 kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(\frac{|x - y|}{L}\Bigg)
    
    Args:
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """

    return HODLR(george.kernels.ExpSquaredKernel(metric = scale**2))
SquaredExp = ExpSquared


def Matern32(scale: Scalar):
    r"""Exponential kernel function, also known as the Matern 1/2 kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(\frac{|x - y|}{L}\Bigg)
    
    Args:
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """

    return HODLR(george.kernels.Matern32Kernel(metric = scale**2))

def Matern52(scale: Scalar):
    r"""Exponential kernel function, also known as the Matern 1/2 kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(\frac{|x - y|}{L}\Bigg)
    
    Args:
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """

    return HODLR(george.kernels.Matern52Kernel(metric = scale**2))


def RationalQuadratic(scale: Scalar, alpha: Scalar):
    r"""Exponential kernel function, also known as the Matern 1/2 kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(\frac{|x - y|}{L}\Bigg)
    
    Args:
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """

    return HODLR(george.kernels.RationalQuadraticKernel(metric = scale**2, log_alpha = np.log(alpha)))


def Constant(const):
    r"""Matern 3/2 kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{3} \frac{|x - y|}{L}\Bigg) \exp\Bigg( -\sqrt{3} \frac{|x - y|}{L}\Bigg)
    
    Args:
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    return HODLR(george.kernels.ConstantKernel(log_constant = np.log(const)))



def Cosine(P: Scalar):
    r"""Cosine kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices which have periodic covariance.
    
    .. math::

        k(x, y) = \cos\Bigg(\frac{2\pi|x - y|}{P}\Bigg)
    
    Args:
        P (Scalar): Period
        
        
    """

    return HODLR(george.kernels.CosineKernel(log_period = np.log(P)))
