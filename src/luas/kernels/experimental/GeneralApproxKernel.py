import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Callable, Tuple, Any, Union

import jax
from jax import grad, value_and_grad, hessian, vmap, lax
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import make_vec, make_mat, kron_prod
from luas.jax_convenience_fns import array_to_pytree_2D
from luas.kernels.GeneralKernel import GeneralKernel

__all__ = ["GeneralApproxKernel"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


def lanczos_scan(mvm, v0, num_steps):
    """
    Fixed-length Lanczos tridiagonalization.
    Returns alphas (k,), betas (k-1,)
    """
    v0 = v0 / jnp.linalg.norm(v0)

    def step(carry, _):
        v, v_prev, beta_prev = carry

        w = mvm(v)
        alpha = jnp.dot(v, w)
        w = w - alpha * v - beta_prev * v_prev
        beta = jnp.linalg.norm(w)

        # Safe normalization (avoids NaNs)
        beta_safe = jnp.where(beta > 0, beta, 1.0)
        v_next = w / beta_safe

        carry_next = (v_next, v, beta)
        return carry_next, (alpha, beta)

    init = (v0, jnp.zeros_like(v0), 0.0)
    (_, _, _), (alphas, betas) = lax.scan(
        step, init, xs=None, length=num_steps
    )

    # betas[0] corresponds to beta_1, but T needs only k-1
    return alphas, betas[:-1]


def log_quadrature(alphas, betas):
    k = alphas.shape[0]

    T = jnp.diag(alphas)
    T = T + jnp.diag(betas, 1) + jnp.diag(betas, -1)

    eigvals, eigvecs = jnp.linalg.eigh(T)
    eigvals = jnp.clip(eigvals, 1e-15)

    return jnp.sum((eigvecs[0] ** 2) * jnp.log(eigvals))


def slq_probe(key, mvm, dim, num_steps):
    z = jax.random.rademacher(key, (dim,), dtype=jnp.float64)
    norm_sq = jnp.dot(z, z)

    alphas, betas = lanczos_scan(mvm, z, num_steps)
    quad = log_quadrature(alphas, betas)

    return norm_sq * quad


class GeneralApproxKernel(GeneralKernel):
    
    def __init__(
        self,
        Sigma,
        *K_list,
        use_stored_values = False,
        tol=1e-10,
        num_probes=80,
        lanczos_steps=30,
    ):
        # Function used to build the covariance matrix K
        self.Sigma = Sigma
        self.K_list = K_list
        self.tol = tol
        self.num_probes = num_probes
        self.lanczos_steps = lanczos_steps
        # alias to maintain consistency with LuasKernel which has a separate fn for calculating the hessian
        self.logL = jax.jit(lambda *args: self.approx_logL(*args, tol=tol, num_probes=num_probes, lanczos_steps=lanczos_steps))
     
    def _logdet_calc(self, key):

        N_l = self.X[0].shape[-1]
        N_t = self.X[1].shape[-1]

        K_mult_fn = lambda r: self.matmul(self.X, self.X, r.reshape((N_l, N_t))).ravel()
        
        keys = jax.random.split(key, self.num_probes)
        probe_vals = jax.vmap(lambda k: slq_probe(k, K_mult_fn, N_l*N_t, self.lanczos_steps))(keys)
        new_key = keys[-1]
        logdet = jnp.mean(probe_vals)

        return logdet, new_key
    
    def decompose(
        self,
        X,
        stored_values: Optional[PyTree] = {},
        **kwargs,
    ) -> PyTree:
        
        if not stored_values:
            stored_values["key"] = jax.random.PRNGKey(42)
        
        self.X = X
        self.logdet, stored_values["key"] = self._logdet_calc(stored_values["key"])
        stored_values["logdet"] = self.logdet

        return self, stored_values
    
    
    def dot_solve(self, R):
        K_inv_R = self.inverse(R)
        return (R * K_inv_R).sum()
    

    def inverse(
        self,
        R: JAXArray,
        **kwargs,
    ) -> JAXArray:

        K_mult_fn = lambda R: self.matmul(self.X, self.X, R, **kwargs)
        K_inv_R = jax.scipy.sparse.linalg.cg(K_mult_fn, R, tol=self.tol)[0]
        
        return K_inv_R


    def matrix_inv_sqrt(self, R, transpose=0):
        raise Exception("Not implemented")

    def matrix_sqrt(self, R, transpose=0):
        raise Exception("Not implemented")
