import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

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
    
    def __init__(
        self,
        *Sigma,
        use_stored_values: Optional[bool] = False,
        data_shape: Optional[Tuple] = None,
        inv_dims: bool = False,
    ):

        self.Sigma = Sigma
        self.dim = len(Sigma)
        self.inv_dims = inv_dims

        if inv_dims:
            self.dot_solve = self.dot_solve_w_inv_dims
        
        # Define for consistency with other kernel objects
        self.K_list = []

        self.data_shape = data_shape
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            self.decompose = self.decompose_use_stored_values
        else:
            self.decompose = self.decompose_no_stored_values
    
    def evaluate(self, X, **kwargs):

        dim = len(X)
        Sigma = self.Sigma[0].evaluate(X[0], X[0], **kwargs)
        
        for d in range(1, dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X[d], X[d], **kwargs))
        
        return Sigma
    
    def decompose_no_stored_values(
        self,
        X: JAXArray,
        stored_values: Optional[PyTree] = {},
        full = True,
        **kwargs,
    ) -> PyTree:

        total_size = calc_total_size(X)
        gp_dim = len(X)
        
        stored_values["logdet"] =  0.
        sigma_decomp = ()
        for d in range(gp_dim):
            Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d])

            if gp_dim > 2:
                Sigma_d_new.matrix_sqrt = vmap_for_tensors(Sigma_d_new.matrix_sqrt)
                Sigma_d_new.matrix_inv_sqrt = vmap_for_tensors(Sigma_d_new.matrix_inv_sqrt)

            sigma_decomp += (Sigma_d_new,)
            stored_values["logdet"] += (total_size/X[d].shape[-1])*stored_values_d["logdet"]

        self.Sigma = sigma_decomp
        self.logdet = stored_values["logdet"]

        return self, stored_values
    
    def dot_solve_w_inv_dims(self, R):
        R_prime = cyclic_transpose(R, 2)
        R_T = R_prime.copy()

        for d in range(self.dim):
            try:
                R_prime = self.Sigma[d].matrix_inv_sqrt(R_prime, transpose = 0)
                R_T = self.Sigma[d].matrix_inv_sqrt(R_T, transpose = 0)
            except:
                R_prime = self.Sigma[d].inverse(R_prime)
            
            R_prime = cyclic_transpose(R_prime, 1)
            R_T = cyclic_transpose(R_T, 1)

        return (R_T * R_prime).sum()
    
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

    def eigendecomp(
        self,
        X,
        **kwargs,
    ):

        dim = len(X)

        all_lam = jnp.ones(1)
        all_lam_shape = ()
        eigen_decomp_mats = ()
        
        for d in jnp.arange(dim):
            lam_d, Q_d = self.Sigma[d].eigendecomp(X)

            all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), lam_d)
            all_lam_shape = all_lam.shape
            eigen_decomp_mats += (Q_d,)

        return all_lam, eigen_decomp_mats
    
    def matmul(self, X1, X2, R, **kwargs):
        
        R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)
        
        return R_prime
