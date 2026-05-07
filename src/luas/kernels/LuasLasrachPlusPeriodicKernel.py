import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import CovType, Outer, Identity, ScaledIdentity, Diagonal
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import tensor_mult
import luas.kernels.tinygp_ext
from luas.kernels.covtype import Exp, GeneralQuasisep, GeneralQuasisepPlusNoise
import tinygp

__all__ = [
    "LuasLasrachPlusPeriodicKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

class LuasLasrachPlusPeriodicKernel(CovType):
    def __init__(
        self,
        Sigma,
        K,
        periodic_term,
        period = None,
        fast_dim: int | None = None,
        use_stored_values: Optional[bool] = False,
    ):
        assert fast_dim is not None # Must specify the fast_dim
        self.Sigma = Sigma
        self.K = K
        self.periodic_term = periodic_term
        self.Outer_peri = periodic_term[1-fast_dim]
        self.peri_kernel = periodic_term[fast_dim]
        self.peri_vector = None
        self.fast_dim = fast_dim
        self.period = period
        self.K_list = (K,periodic_term)
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            raise Exception("Use of precalculated stored values not yet implemented")
        
        assert period is not None
    

    def decompose(
        self,
        X: Tuple,
        stored_values: Optional[PyTree] = None,
    ) -> PyTree:

        # total_size = calc_total_size(X)
        total_size = X[0].shape[-1] * X[1].shape[-1]
        self.dim = len(X)
        stored_values = {} if stored_values is None else stored_values

        self.N_slow = X[1-self.fast_dim].shape[-1]
        self.N_fast = X[self.fast_dim].shape[-1]

        self.Sigma_slow, stored_values_slow = self.Sigma[1-self.fast_dim].decompose(X[1-self.fast_dim])
        stored_values["logdet"] = (total_size/X[1-self.fast_dim].shape[-1])*stored_values_slow["logdet"]

        K_slow = self.Sigma_slow.inv_sqrt_transform(self.K[1-self.fast_dim], X[1-self.fast_dim])
        self.Outer_peri = self.Sigma_slow.inv_sqrt_transform(self.Outer_peri, X[self.fast_dim])

        self.lam_K_slow, self.Q_K_slow = K_slow.eigendecomp(X[1-self.fast_dim])
        alpha_new = self.Q_K_slow.T @ self.Outer_peri.alpha_init


        span = 1.000001 * jnp.ptp(X[self.fast_dim])
        n_periods = span//self.period
        span = (n_periods + 1) * self.period

        x_long = jnp.kron(jnp.ones(self.N_slow), X[self.fast_dim])
        x_long += jnp.kron(span*jnp.arange(self.N_slow), jnp.ones(self.N_fast))

        block_endpoints =  X[self.fast_dim][-1] + 1e-10 + span*jnp.arange(self.N_slow)
        block_kernel = luas.kernels.quasisep.ConstantBlocks(endpoints=block_endpoints)
        
        lambda_2D = jnp.kron(jnp.sqrt(self.lam_K_slow), jnp.ones(self.N_fast))
        kernel_2D = block_kernel * self.K[self.fast_dim] * luas.kernels.quasisep.Linear(lambda_2D)
        if isinstance(self.Sigma[self.fast_dim], (Exp, GeneralQuasisep, GeneralQuasisepPlusNoise)):
            kernel_2D += block_kernel * self.Sigma[self.fast_dim]

        band_full = jnp.kron(alpha_new, jnp.ones(self.N_fast))

        peri_term = luas.kernels.tinygp_ext.SpecialMultiband(
                    kernel=self.peri_kernel.tinygp_kf,
                    band_amplitudes=band_full, # not reversable?
        )
        
        if self.peri_vector is None:
            self.peri_vector = jnp.kron(jnp.ones(self.N_slow), X[self.fast_dim])

            qsm_matrix = peri_term.to_symm_qsm((self.peri_vector,  jnp.arange(total_size)))
            qsm_matrix += kernel_2D.tinygp_kf.to_symm_qsm((x_long, jnp.arange(total_size)))
        else:
            kernel_2D += peri_term
            qsm_matrix = kernel_2D.tinygp_kf.to_symm_qsm((x_long, jnp.arange(total_size)))

        print(qsm_matrix)
        sigma_diag = self.Sigma[self.fast_dim].diag + self.Sigma[self.fast_dim].wn_diag
        K_diag = self.K[self.fast_dim].diag + self.K[self.fast_dim].wn_diag
        qsm_matrix += tinygp.noise.Diagonal(jnp.kron(jnp.ones(self.N_slow), sigma_diag * jnp.ones(self.N_fast))).to_qsm()
        qsm_matrix += tinygp.noise.Diagonal(jnp.kron(self.lam_K_slow, K_diag * jnp.ones(self.N_fast))).to_qsm()

        self.K_tilde = qsm_matrix.to_dense()

        self.factor = qsm_matrix.cholesky()
        stored_values["logdet"] += 2*jnp.sum(jnp.log(self.factor.diag.d))

        self.logdet = stored_values["logdet"]

        return self, stored_values
    
    
    def kf_tilde_matrix_sqrt(self, R, transpose = 0):
        R_shape = R.shape
        r = R.ravel()

        if transpose:
            r_prime = self.factor.transpose() @ r
        else:
            r_prime = self.factor @ r

        return r_prime.reshape(R_shape)

    def kf_tilde_inv_sqrt(self, R, transpose=0):
        R_shape = R.shape
        r = R.ravel()
        if transpose:
            r_prime = self.factor.transpose().solve(r)
        else:
            r_prime = self.factor.solve(r)

        return r_prime.reshape(R_shape)


    def transform_fn(self, R, transpose = 0):
        
        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        if transpose:
            R_prime = self.Sigma_slow.matrix_sqrt(R_prime, transpose = 1)
            R_prime = self.Q_K_slow.T @ R_prime
        else:
            R_prime = self.Q_K_slow @ R_prime
            R_prime = self.Sigma_slow.matrix_sqrt(R_prime, transpose = 0)

        if self.fast_dim == 0:
            R_prime = R_prime.T
            
        return R_prime


    def inv_transform_fn(self, R, transpose = 0):
        
        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        if transpose:
            R_prime = self.Q_K_slow @ R_prime
            R_prime = self.Sigma_slow.matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_slow.matrix_inv_sqrt(R_prime, transpose = 0)
            R_prime = self.Q_K_slow.T @ R_prime

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
            R_prime = self.kf_tilde_matrix_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.kf_tilde_matrix_sqrt(R, transpose = 0)
            R_prime = self.transform_fn(R_prime, transpose = 0)
        
        return R_prime


    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:

        if transpose:
            R_prime = self.kf_tilde_inv_sqrt(R, transpose = 1)
            R_prime = self.inv_transform_fn(R_prime, transpose = 1)
        else:
            R_prime = self.inv_transform_fn(R, transpose = 0)
            R_prime = self.kf_tilde_inv_sqrt(R_prime, transpose = 0)
        
        return R_prime


    def matmul(self, X1, X2, R, **kwargs):

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)
        K_R += tensor_mult(self.K, X1, X2, R, **kwargs)
        K_R += tensor_mult(self.periodic_term, X1, X2, R, **kwargs)
        
        return K_R
