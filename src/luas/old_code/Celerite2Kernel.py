import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Callable, Tuple, Any
import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

from .luas_types import Kernel, PyTree, JAXArray, Scalar
from .kronecker_fns import make_vec, make_mat
from .jax_convenience_fns import array_to_pytree_2D

import tinygp

__all__ = ["Celerite2Kernel"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])

    
def make_t(x_t, N_l):
    t = jnp.kron(x_t, jnp.ones(N_l))
    label = (jnp.kron(jnp.ones(x_t.shape[-1]), jnp.arange(N_l))).astype(int)
    return (t, label)


class Celerite2Kernel(Kernel):
    """
    
    Args:
        K (Callable, optional): Function which returns the covariance matrix K.
        decomp_fn (Callable, optional): Function which given the computed matrix K
            computes the Cholesky decomposition and log determinant of K.
            Defaults to performing general Cholesky decomposition.
            
    """
    
    def __init__(
        self,
        Kl_fns: Optional[Callable] = None,
        Kt_fns: Optional[Callable] = None,
        sigma_fn: Optional[Callable] = None,
        decomp_fn = None,
    ):
        
        self.Kl_fns = Kl_fns
        self.Kt_fns = Kt_fns
        self.diag_fn = lambda hp, x_l, x_t: jnp.power(10, 2*hp["sigma"])
        
        
        if decomp_fn is None:
            self.Kl_decomp_fn = jnp.linalg.cholesky
            
        # alias to maintain consistency with LuasKernel which has a separate fn for calculating the hessian
        self.logL_hessianable = self.logL
        
        
    def logL(self, hp, x_l, x_t, R, stored_values):
        
        x_t = make_t(x_t, x_l.shape[-1])
        
        kernel = 0.
        for i in range(len(self.Kl_fns)):
            Kl = self.Kl_fns[i](hp, x_l, x_l)
            L = self.Kl_decomp_fn(Kl)

            Kt = self.Kt_fns[i](hp, x_t, x_t)
            kernel += sum(Multiband(kernel=Kt, amplitudes=row) for row in L)

        gp = tinygp.GaussianProcess(kernel, x_t, diag=self.diag_fn(hp, x_l, x_t))

        return gp.log_probability(R.ravel("F")), stored_values
    