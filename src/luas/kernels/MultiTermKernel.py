import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional
import tinygp

from luas import WhiteNoiseKernel, SingleKronTermKernel
from luas.kernels.covtype import Outer, GeneralQuasisepPlusNoise, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal
from luas.kernels.householder import HouseholderProduct, orthonormal_nullspace_gen
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import tensor_mult
from luas.kernels.tinygp_ext import ScaledKernel
from luas.kernels.BlockKernel import Block2x2Kernel
from luas.kernels.MixingMatQuasisep import MixingMatQuasisep

__all__ = [
    "MultiTermKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class MultiTermKernel(CovType):
    def __init__(
        self,
        Sigma,
        *K_list,
        fast_dim = None,
        never_reduce_dim = False,
        use_stored_values: Optional[bool] = True,
    ):
        assert fast_dim is not None # Must specify the fast dimension
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.fast_dim = fast_dim
        self.never_reduce_dim = never_reduce_dim
        self.N_alpha = len(K_list)

        self.logL_hessianable = self.logL
        self.decompose = self.decomp_no_stored_values


    def _rotate_to_fast_dim_wrapper(self, fn, fast_dim):

        def wrapped_fn(R, **kwargs):
            if fast_dim == 0:
                R_prime = R.T
            else:
                R_prime = R.copy()

            R_prime = fn(R_prime, **kwargs)

            if self.fast_dim == 0:
                R_prime = R_prime.T
                
            return R_prime

        return wrapped_fn
    

    def decomp_no_stored_values(
        self,
        X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = {},
        full = True,
        idx = None,
    ) -> PyTree:

        # should to something about idx and full
        assert full == True
        
        non_cel_vec = X[1-self.fast_dim]
        cel_vec = X[self.fast_dim]

        # Generate transformations
        self.Sigma_transf, stored_values_Sigma_transf = self.Sigma[1-self.fast_dim].decompose(non_cel_vec, full = True)

        # Computes the log determinant of K
        stored_values["logdet"] = cel_vec.shape[-1]*stored_values_Sigma_transf["logdet"]
        
        stored_values["non_cel_rank"] = 0
        for K_i in self.K_list:
            stored_values["non_cel_rank"] += K_i[1-self.fast_dim].rank(non_cel_vec)
        
        stored_values["A"] = jnp.zeros((non_cel_vec.shape[-1], stored_values["non_cel_rank"]))
        stored_values["cel_kernel_order"] = []
        
        col_i = 0
        for (i, K_i) in enumerate(self.K_list):
            if isinstance(K_i[1-self.fast_dim], Outer):
                K_i_non_cel, _ = K_i[1-self.fast_dim].decompose(non_cel_vec)
                stored_values["A"] = stored_values["A"].at[:, col_i].set(K_i_non_cel.alpha)
                stored_values["cel_kernel_order"].append(K_i[self.fast_dim])
                col_i += 1
            else:
                lam, Q = K_i[1-self.fast_dim].eigendecomp(non_cel_vec)

                rank_i = K_i[1-self.fast_dim].rank(non_cel_vec)
                lam_nonzero = lam[-rank_i:]

                if isinstance(Q, HouseholderProduct):
                    alpha_eigen = Q.V[:, -rank_i:] @ jnp.diag(jnp.sqrt(lam_nonzero))
                else:
                    alpha_eigen = Q[:, -rank_i:] @ jnp.diag(jnp.sqrt(lam_nonzero))
                
                for j in range(rank_i):
                    stored_values["A"] = stored_values["A"].at[:, col_i].set(alpha_eigen[:, j])
                    
                    # Could accept a list of kernels in K_i[self.fast_dim]
                    stored_values["cel_kernel_order"].append(K_i[self.fast_dim])
                    col_i += 1
        
        # Transform vectors 
        stored_values["J"] = self.Sigma_transf.matrix_inv_sqrt(stored_values["A"], transpose = 0)

        # If the total rank is less than the length of that dimension, we can reduce the dimension by exploiting sparsity
        self.reduce_dim = stored_values["non_cel_rank"] < non_cel_vec.shape[-1] and not self.never_reduce_dim

        if self.reduce_dim:
            stored_values["J"], U, self.householder_transform = orthonormal_nullspace_gen(stored_values["J"])
        
        # Handle Sigma mat in the fast_dim, likely just diagonal
        if isinstance(self.Sigma[self.fast_dim], (GeneralQuasisep, GeneralQuasisepPlusNoise)):
            add_basis = jnp.eye(stored_values["non_cel_rank"])
            stored_values["J"] = jnp.concatenate([stored_values["J"], add_basis], axis = 1)

            for j in range(stored_values["non_cel_rank"]):
                stored_values["cel_kernel_order"].append(self.Sigma[self.fast_dim])

            assert self.Sigma[self.fast_dim].noise_model is None # Not implemented
        else:
            assert isinstance(self.Sigma[self.fast_dim], (Identity, ScaledIdentity, Diagonal))

        cel_diag = (self.Sigma[self.fast_dim].diag + self.Sigma[self.fast_dim].wn_diag)*jnp.ones(cel_vec.shape[-1])

        if self.reduce_dim:
            total_cel_diag = jnp.kron(cel_diag, jnp.ones(stored_values["non_cel_rank"]))
        else:
            total_cel_diag = jnp.kron(cel_diag, jnp.ones(non_cel_vec.shape[-1]))
        
        kf_quasi2D = MixingMatQuasisep(mixing_mat = stored_values["J"], kernel_list = stored_values["cel_kernel_order"],
                                        diag = total_cel_diag, fast_dim = self.fast_dim)

        if self.reduce_dim:
            if isinstance(self.Sigma[self.fast_dim], (Identity, ScaledIdentity, Diagonal)):
                null_space_rank = non_cel_vec.shape[-1] - stored_values["non_cel_rank"]
                if self.fast_dim == 0:
                    rest_cel_diag = jnp.outer(cel_diag, jnp.ones(null_space_rank))
                else:
                    rest_cel_diag = jnp.outer(jnp.ones(null_space_rank), cel_diag)
                    
                kf_D = WhiteNoiseKernel(diag = rest_cel_diag)
            else:
                if self.fast_dim == 0:
                    kf_D = SingleKronTermKernel(self.Sigma[self.fast_dim], Identity())
                else:
                    kf_D = SingleKronTermKernel(Identity(), self.Sigma[self.fast_dim])

            # print("kf_A", kf_quasi2D, "kf_D_CAB", kf_D, "dim_split", 1-self.fast_dim, "split_loc", stored_values["non_cel_rank"], "fast_dim",self.fast_dim)
            self.kf_tilde = Block2x2Kernel(kf_A = kf_quasi2D, kf_D_CAB = kf_D,
                                           dim_split = 1-self.fast_dim, D_full = True,
                                           split_loc = stored_values["non_cel_rank"])
        else:
            self.kf_tilde = kf_quasi2D
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X)
        stored_values["logdet"] += stored_values["kf_tilde_stored"]["logdet"]

        self.logdet = stored_values["logdet"]

        return self, stored_values
        
    
    def transform_fn(self, R, transpose = 0):
        
        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        if transpose:
            R_prime = self.Sigma_transf.matrix_sqrt(R_prime, transpose = 1)

            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 0)
        else:
            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 1)
            R_prime = self.Sigma_transf.matrix_sqrt(R_prime, transpose = 0)

        if self.fast_dim == 0:
            R_prime = R_prime.T
            
        return R_prime


    def inv_transform_fn(self, R, transpose = 0):
        
        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        if transpose:
            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 1)
            R_prime = self.Sigma_transf.matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_transf.matrix_inv_sqrt(R_prime, transpose = 0)
            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 0)

        if self.fast_dim == 0:
            R_prime = R_prime.T
            
        return R_prime
    

    def matrix_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:

        if transpose:
            R_prime = self.transform_fn(R, transpose = 1)
            R_prime = self.kf_tilde.matrix_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.kf_tilde.matrix_sqrt(R, transpose = 0)
            R_prime = self.transform_fn(R_prime, transpose = 0)
        
        return R_prime


    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:

        if transpose:
            R_prime = self.kf_tilde.matrix_inv_sqrt(R, transpose = 1)
            R_prime = self.inv_transform_fn(R_prime, transpose = 1)
        else:
            R_prime = self.inv_transform_fn(R, transpose = 0)
            R_prime = self.kf_tilde.matrix_inv_sqrt(R_prime, transpose = 0)
        
        return R_prime


    def matmul(self, X1, X2, R, **kwargs):

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

        for K in self.K_list:
            K_R += tensor_mult(K, X1, X2, R, **kwargs)
        
        return K_R



