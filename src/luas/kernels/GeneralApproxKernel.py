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

__all__ = ["GeneralKernel", "general_cholesky"]

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




class GeneralApproxKernel(Kernel):
    
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
        # alias to maintain consistency with LuasKernel which has a separate fn for calculating the hessian
        self.logL = jax.jit(lambda *args: self.approx_logL(*args, tol=tol, num_probes=num_probes, lanczos_steps=lanczos_steps))
        self.logL_hessianable = self.logL

    
    def kf(self, hp, x_l1, x_l2, x_t1, x_t2, **kwargs):

        K = jnp.kron(self.Sigma[0].evaluate(hp, x_l1, x_l2, **kwargs), self.Sigma[1].evaluate(hp, x_t1, x_t2, **kwargs))

        for i in range(len(self.K_list)):
            K += jnp.kron(self.K_list[i][0].evaluate(hp, x_l1, x_l2, **kwargs), self.K_list[i][1].evaluate(hp, x_t1, x_t2, **kwargs))
            
        return K
    
        
    def decomp_fn(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
        
        return stored_values

    
    def approx_logL(self, hp, x_l, x_t, R, stored_values, tol=1e-10, num_probes=80, lanczos_steps=30):

        if not stored_values:
            stored_values["key"] = jax.random.PRNGKey(42)
        
        N_l = x_l.shape[-1]
        N_t = x_t.shape[-1]

        K_inv_R, stored_values = self.solve(hp, x_l, x_t, R, stored_values, tol = tol)
        
        K_mult_fn = lambda r: self.K_mult_vec(hp, x_l, x_t, r.reshape((N_l, N_t)), {})[0].ravel()
        
        keys = jax.random.split(stored_values["key"], num_probes)
        probe_vals = jax.vmap(lambda k: slq_probe(k, K_mult_fn, N_l*N_t, lanczos_steps))(keys)
        stored_values["logdetK"] = jnp.mean(probe_vals)        
        stored_values["key"] = keys[-1]
        
        return -0.5 * (R * K_inv_R).sum() - 0.5 * stored_values["logdetK"] - 0.5 * N_l * N_t * jnp.log(2*jnp.pi), stored_values


    def generate_noise(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        size: Optional[int] = 1,
        stored_values: Optional[PyTree] = {},
    ) -> JAXArray:
        r"""Generate noise with the covariance matrix returned by this kernel using the input
        hyperparameters ``hp``.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s).
                May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l`` different wavelength/vertical
                regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s).
                May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal
                regression variables.
            size (int, optional): The number of different draws of noise to generate. Defaults to 1.
            stored_values (PyTree): Stored values from the decomposition of the covariance matrix. For
                :class:`GeneralKernel` this consists of the Cholesky factor and the log determinant
                of ``K``.
                
        Returns:
            JAXArray: Generate noise of shape ``(N_l, N_t)`` if ``size = 1`` or ``(N_l, N_t, size)``
            if size > 1.
        
        """
        
        
        # Get the length of each dimension
        N_l = x_l.shape[-1]
        N_t = x_t.shape[-1]

        # Generate a random normal vector
        Z = np.random.normal(size = (N_l, N_t))
        
        self.Sigma[0].cholesky_decomp(hp, x_l, x_l)
        self.Sigma[1].cholesky_decomp(hp, x_t, x_t)
        R_prime = self.Sigma[0].cho_mult(Z)
        R_draw = self.Sigma[1].cho_mult(R_prime.T).T

        for i in range(len(self.K_list)):
            # Eigendecomp is more numerically stable in general
            
            lam_l, Q_l = self.K_list[i][0].eigendecomp(hp, x_l, x_l)
            lam_t, Q_t = self.K_list[i][1].eigendecomp(hp, x_t, x_t)

            lam_sqrt_l = jnp.sqrt(jnp.abs(lam_l))
            lam_sqrt_t = jnp.sqrt(jnp.abs(lam_t))

            Z = np.random.normal(size = (N_l, N_t))
            Z = jnp.outer(lam_sqrt_l, lam_sqrt_t) * Z

            R_draw += kron_prod(Q_l, Q_t, Z)
        
        return R_draw
    
    
    def predict(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_l_pred: JAXArray,
        x_t: JAXArray,
        x_t_pred: JAXArray,
        R: JAXArray,
        M_s: JAXArray,
        wn: Optional[bool] = True,
        return_std_dev: Optional[bool] = True,
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
        
        raise Exception("Not Implemented yet!")
    

    def solve(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values,
        tol = 1e-10,
    ) -> JAXArray:
        r"""Calculates the product of the inverse of the covariance matrix with a vector, represented by
        a JAXArray of shape ``(N_l, N_t)``. Useful for testing for numerical stability.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            R (JAXArray): JAXArray of shape ``(N_l, N_t)`` representing the vector to multiply on the right by
                the inverse of the covariance matrix ``K``.
                
        Returns:
            JAXArray: The result of multiplying the inverse of the covariance matrix ``K`` by the vector ``R``.
        
        """

        K_mult_fn = lambda R: self.K_mult_vec(hp, x_l, x_t, R, {})[0]
        K_inv_R = jax.scipy.sparse.linalg.cg(K_mult_fn, R, tol=tol)[0]
        
        return K_inv_R, stored_values
    
    
    def K_mult_vec(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values,
        **kwargs,
    ) -> JAXArray:
        r"""Calculates the product of the covariance matrix with a vector, represented by a JAXArray of shape ``(N_l, N_t)`.
        Useful for testing for numerical stability.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            R (JAXArray): JAXArray of shape ``(N_l, N_t)`` representing the vector to multiply on the right by
                the covariance matrix ``K``.
                
        Returns:
            JAXArray: The result of multiplying the covariance matrix ``K`` by the vector ``R``.
        
        """

        R_prime = self.Sigma[0].left_mult(R, hp, x_l, x_l, **kwargs)
        Kr = self.Sigma[1].left_mult(R_prime.T, hp, x_t, x_t, **kwargs).T

        for i in range(len(self.K_list)):
            R_prime = self.K_list[i][0].left_mult(R, hp, x_l, x_l, **kwargs)
            Kr += self.K_list[i][1].left_mult(R_prime.T, hp, x_t, x_t, **kwargs).T
        
        return Kr, stored_values
