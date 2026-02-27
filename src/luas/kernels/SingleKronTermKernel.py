import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, hessian, vmap, custom_jvp, jit
from jax.flatten_util import ravel_pytree
from copy import deepcopy
from tqdm import tqdm
from typing import Callable, Tuple, Union, Any, Optional
from functools import partial

from luas.kernels.covtype import CovType
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import kron_prod, calc_total_size, calc_data_shape, cyclic_transpose, tensor_mult, vmap_for_tensors
from luas.jax_convenience_fns import array_to_pytree_2D, get_corr_mat


__all__ = [
    "SingleKronTermKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class SingleKronTermKernel(CovType):
    r"""Kernel class which solves for the log likelihood for any covariance matrix which
    is the sum of two kronecker products of the covariance matrix in each of two dimensions
    i.e. the full covariance matrix K is given by:
    
    .. math::
        K = K_l \otimes K_t + S_l \otimes S_t
    
    although we can avoid calculating ``K`` for many calculations implemented here.
        
    The ``Kl`` and ``Sl`` functions should both return ``(N_l, N_l)`` matrices which will be the covariance
    matrices in the wavelength/vertical direction.
    
    The ``Kt`` and ``St`` functions should both return ``(N_t, N_t)`` matrices which will by the covariance
    matrices in the time/horizontal direction.
    
    .. code-block:: python

        >>> from luas import LuasKernel, kernels
        >>> def Kl_fn(hp, x_l1, x_l2, wn = True):
        >>> ... return hp["h"]**2*kernels.squared_exp(x_l1, x_l2, hp["l_l"])
        >>> def Kt_fn(hp, x_t1, x_t2, wn = True):
        >>> ... return kernels.squared_exp(x_t1, x_t2, hp["l_t"])
        >>> # ... And similarly for Sl_fn, St_fn
        >>> kernel = LuasKernel(Kl = Kl_fn, Kt = Kt_fn, Sl = Sl_fn, St = St_fn)
        ... )
    
    See https://luas.readthedocs.io/en/latest/tutorials.html for more detailed tutorials on how to use.
        
    Args:
        Kl (Callable): Function which returns the covariance matrix Kl, should be of the form
            ``Kl(hp, x_l1, x_l2, wn = True)``.
        Kt (Callable): Function which returns the covariance matrix Kt, should be of the form
            ``Kt(hp, x_t1, x_t2, wn = True)``.
        Sl (Callable): Function which returns the covariance matrix Sl, should be of the form
            ``Sl(hp, x_l1, x_l2, wn = True)``.
        St (Callable): Function which returns the covariance matrix St, should be of the form
            ``St(hp, x_t1, x_t2, wn = True)``.
        use_stored_values (bool, optional): Whether to perform checks if any of the component
            covariance matrices have changed and to make use of previously stored values for
            the decomposition of those matrices if they're the same. If ``False`` then will
            not perform these checks and will compute the eigendecomposition of all matrices
            for every calculation.
    
    """
    
    def __init__(
        self,
        *Sigma,
        use_stored_values: Optional[bool] = False,
        data_shape: Optional[Tuple] = None,
    ):

        self.Sigma = Sigma
        self.dim = len(Sigma)
        
        # Define for consistency with other kernel objects
        self.K_list = []

        self.data_shape = data_shape
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            self.decompose = self.decompose_use_stored_values
        else:
            self.decompose = self.decompose_no_stored_values

    
    def evaluate(self, *X, **kwargs):

        dim = len(X)
        Sigma = self.Sigma[0].evaluate(X[0], X[0], **kwargs)
        
        for d in range(1, dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X[d], X[d], **kwargs))
        
        return Sigma
        
    
    def decompose_no_stored_values(
        self,
        *X: JAXArray,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

        total_size = calc_total_size(X)
        gp_dim = len(X)
        
        stored_values["logdetK"] =  0.
        sigma_decomp = ()
        for d in range(gp_dim):
            Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d])

            if gp_dim > 2:
                Sigma_d_new.matrix_sqrt = vmap_for_tensors(Sigma_d_new.matrix_sqrt)
                Sigma_d_new.matrix_inv_sqrt = vmap_for_tensors(Sigma_d_new.matrix_inv_sqrt)

            sigma_decomp += (Sigma_d_new,)
            stored_values["logdetK"] += (total_size/X[d].shape[-1])*stored_values_d["logdetK"]

        self.Sigma = sigma_decomp
        return self, stored_values

    
    def matrix_sqrt(self, R, transpose = 0):

        R_prime = cyclic_transpose(R, 2)

        for d in range(self.dim):
            R_prime = self.Sigma[d].matrix_sqrt(R_prime, transpose = transpose)
            R_prime = cyclic_transpose(R_prime, 1)
            
        R_prime = cyclic_transpose(R_prime, -2)
        
        return R_prime

    
    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:

        R_prime = cyclic_transpose(R, 2)

        for d in range(self.dim):
            R_prime = self.Sigma[d].matrix_inv_sqrt(R_prime, transpose = transpose)
            R_prime = cyclic_transpose(R_prime, 1)
            
        R_prime = cyclic_transpose(R_prime, -2)
        
        return R_prime
        

    def inverse(self, R):

        R_prime = self.matrix_inv_sqrt(R, transpose=0)
        K_inv_R = self.matrix_inv_sqrt(R_prime, transpose=1)

        return K_inv_R


    def eigendecomp(
        self,
        *X,
        **kwargs,
    ):

        dim = len(X)

        all_lam = jnp.ones(1)
        all_lam_shape = ()
        eigen_decomp_mats = ()
        
        for d in jnp.arange(dim):
            lam_d, Q_d = self.Sigma[d].eigendecomp(R_prime)

            all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), lam_d)
            all_lam_shape = all_lam.shape
            eigen_decomp_mats += (Q_d,)

        return all_lam, eigen_decomp_mats


    def dot_solve(self, R):
        
        L_inv_R = self.matrix_inv_sqrt(R, transpose = 0)
        return jnp.square(L_inv_R).sum()


    def logL(self, R, stored_values, **kwargs):
        
        return - 0.5 * self.dot_solve(R) - 0.5 * stored_values["logdetK"] - 0.5 * R.size * jnp.log(2*jnp.pi)

    
    def matmul(self, X1, X2, R, **kwargs):
        
        R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)
        
        return R_prime

        
    # def __add__(self, other):

    #     if self.data_shape is None:
    #         data_shape = self.dim

    #     if type(K) == SingleKronTermKernel:
    #         kernel_choice = kernel_selector(self.Sigma, K)
    #     else:
    #         raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
    #     return K_sum
        