import jax.numpy as jnp
import jax
from jax import custom_jvp, jit
import numpy as np
from typing import Callable, Tuple, Union, Any, Optional
from .luas_types import JAXArray, Scalar, PyTree
from functools import partial

__all__ = [
    "make_vec",
    "make_mat",
    "kron_prod",
]


def make_vec(R: JAXArray) -> JAXArray:
    r"""Function for converting a matrix of shape ``(N_l, N_t)`` into
    a vector of shape ``(N_l * N_t,)``.
    
    .. math::

        \mathbf{R}_{ij} = r_{i N_l + j}
    
    Args:
        R (JAXArray): Matrix of shape ``(N_l, N_t)``
        
    Returns:
        JAXArray: A vector of shape ``(N_l * N_t,)``
        
    """
    
    return R.ravel("C")


def make_mat(
    r: JAXArray,
    N_l: int,
    N_t: int
) -> JAXArray:
    r"""Function for converting a vector of shape ``(N_l * N_t,)``
    into an array of shape ``(N_l, N_t)``.
    
    .. math::

        r_{i N_l + j} = \mathbf{R}_{ij}
    
    Args:
        r (JAXArray): Vector of shape ``(N_l * N_t,)``
        N_l (int): Size of wavelength/vertical dimension
        N_t (int): Size of time/horizontal dimension
        
    Returns:
        JAXArray: An array of shape ``(N_l, N_t)``
        
    """
    return r.reshape((N_l, N_t))


def kron_prod(
    A: JAXArray,
    B: JAXArray,
    R: JAXArray
) -> JAXArray:
    r"""Computes the matrix vector product of the kronecker product of two matrices
    ``A`` and ``B`` times a vector ``r``, stored as an ``(N_l, N_t)`` array ``R``.
    
    .. math::

        [\mathbf{A} \otimes \mathbf{B}] \vec{r} = \mathbf{A} \mathbf{R} \mathbf{B}^T
    
    Args:
        A (JAXArray): Matrix on the left side of the kronecker product.
        B (JAXArray): Matrix on the right side of the kronecker product.
        R (JAXArray): Vector to right multiply, stored as an ``(N_l, N_t)`` array.
        
    Returns:
        JAXArray: The result of the multiplication as a JAXArray array of shape ``(N_l, N_t)``.
    """
    
    return A @ R @ B.T

@partial(jit, static_argnums=1)
def cyclic_transpose(R, d):
    ndim = R.ndim
    axes = tuple((i + d) % ndim for i in range(ndim))
    return jnp.transpose(R, axes)


def vmap_for_tensors(f):
    def wrapped(R, **kwargs):
        *leading, N_l, N_t = R.shape
        
        R_flat = R.reshape(-1, N_l, N_t)
        
        f_new = lambda R: f(R, **kwargs)
        R_prime_flat = jax.vmap(f_new)(R_flat)
        
        return R_prime_flat.reshape(*leading, *R_prime_flat.shape[1:])
    
    return wrapped

    
def tensor_mult(kron_mat_list, X1, X2, R, **kwargs):

    gp_dim = len(kron_mat_list)
    R_prime = cyclic_transpose(R, 2)
    
    for d in range(gp_dim):
        method = getattr(kron_mat_list[d], "matmul")

        matmul_fn = lambda R, **kwargs: method(X1[d], X2[d], R, **kwargs)

        if gp_dim > 2:
            matmul_fn = vmap_for_tensors(matmul_fn)
            
        R_prime = matmul_fn(R_prime, **kwargs)
        R_prime = cyclic_transpose(R_prime, 1)
        
    R_prime = cyclic_transpose(R_prime, -2)

    return R_prime


def tensor_arb_op(kron_mat_list, method_name, R, **kwargs):

    gp_dim = len(kron_mat_list)
    R_prime = cyclic_transpose(R, 2)
    
    for d in range(gp_dim):
        method = getattr(kron_mat_list[d], "matmul")
        
        R_prime = method(X1[d], X2[d], R_prime, **kwargs)
        R_prime = cyclic_transpose(R_prime, 1)
        
    R_prime = cyclic_transpose(R_prime, -2)

    return R_prime
    

def kron_prod_dim_d(
    A: JAXArray,
    R: JAXArray,
    d: int,
) -> JAXArray:
    r"""Computes the matrix vector product of the kronecker product of N matrices
    which are all Identity matrices except for ``A`` at dim ``d``, times a tensor ``R``.
    Performs these using the cyclic matrix transpose as described in Saatchi (2011).
    
    .. math::

        [\mathbf{I} \otimes \dots \mathbf{A} \dots \mathbf{I}] \vec{R} = \mathbf{A} ((\mathbf{R})^T) \dots )^T
    
    Args:
        A (JAXArray): Matrix on the left side of the kronecker product.
        R (JAXArray): Tensor to right multiply, stored as an ndarray.
        
    Returns:
        JAXArray: The result of the multiplication as a JAXArray array of shape ``(N_l, N_t)``.
    """

    R_trans = cyclic_transpose(R, d)
    A_R = A @ R_trans
    return cyclic_transpose(A_R, -d)
    
def calc_total_size(x):
    # If it's a single array, just return its size
    if isinstance(x, jnp.ndarray):
        return x.shape[-1]
        
    # Otherwise assume it's an iterable of arrays
    return jnp.prod(jnp.array([xi.shape[-1] for xi in x]))


def calc_data_shape(X):
    # If it's a single array, just return its size
    if isinstance(X, jnp.ndarray) or isinstance(X, np.ndarray):
        return (X.shape[-1],)
        
    # Otherwise assume it's an iterable of arrays
    return sum([(x_i.shape[-1],) for x_i in X], ())



def read_K_list_2D(K_list, X):

    # Initialise for loop reading K_list
    dense_kron = None
    alpha_list = [] # np.zeros((X[self.fast_dim].shape[-1], self.N_alpha))
    beta_list = []

    low_rank_kernels_dim0 = []
    low_rank_kernels_dim1 = []
    for K_i in K_list:
        K_0 = K_i[0]
        K_1 = K_i[1]

        # rank_0 = K_0.rank(X[0])
        # rank_1 = K_1.rank(X[1])

        if isinstance(K_0, Outer) and isinstance(K_1, CovType):
            K_0, _ = K_0.decompose(X[0])
            alpha_list.append(K_0.alpha)
            low_rank_kernels_dim1.append(K_1)

        elif isinstance(K_0, CovType) and isinstance(K_1, Outer):
            K_1, _ = K_1.decompose(X[1])
            beta_list.append(K_1.alpha)
            low_rank_kernels_dim0.append(K_0)

        elif isinstance(K_0, CovType) and isinstance(K_1, CovType) and dense_kron is None:
            dense_kron = [K_0, K_1]

        elif isinstance(dense_kron, tuple):
            raise Exception("Can only have one dense kronecker term")
        
        else:
            raise Exception("All kernel terms must be valid CovType objects")
        
    if alpha_list:
        alpha_mat = jnp.stack(alpha_list, axis = 1)
        alpha_terms = (alpha_mat, low_rank_kernels_dim1)
    else:
        alpha_terms = None

    if beta_list:
        beta_mat = jnp.stack(beta_list, axis = 1)
        beta_terms = (beta_mat, low_rank_kernels_dim0)
    else:
        beta_terms = None

    return dense_kron, alpha_terms, beta_terms

