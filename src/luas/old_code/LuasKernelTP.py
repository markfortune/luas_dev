import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, hessian, vmap, custom_jvp, jit
from jax.flatten_util import ravel_pytree
from jax.scipy.special import gammaln

from copy import deepcopy
from tqdm import tqdm
from typing import Callable, Tuple, Union, Any, Optional
from functools import partial

from .luas_types import Kernel, PyTree, JAXArray, Scalar
from .kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable
from .jax_convenience_fns import array_to_pytree_2D


__all__ = [
    "LuasKernelTP",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class LuasKernelTP(Kernel):
    def __init__(
        self,
        Kl_fn: Callable,
        Kt_fn: Callable,
        Sl_fn: Callable,
        St_fn: Callable,
        use_stored_results: Optional[bool] = True,
    ):
        
        self.Kl = Kl_fn
        self.Kt = Kt_fn
        self.Sl = Sl_fn
        self.St = St_fn
           
        if use_stored_results:
            self.decomp_fn = self.eigendecomp_use_stored_results
        else:
            self.decomp_fn = self.eigendecomp_no_stored_results

        for fn in [self.Sl, self.St, self.Kl, self.Kt]:
            if hasattr(fn, "decomp"):
                if fn.decomp == "diag":
                    fn.decomp = diag_eigendecomp
            else:
                fn.decomp = jnp.linalg.eigh
                
        if self.Kl.decomp == diag_eigendecomp and not self.Sl.decomp == diag_eigendecomp:
            raise Warning("The transformation of Kl is set to be diagonal but the matrix Sl is not set to diagonal. This may be possible for example if Kl is a scalar times the identity matrix or Kl shares the same eigenvectors as Sl but it is not true if Kl is any general diagonal matrix. Alternatively perhaps Sl is also diagonal and you forgot to add Sl.decomp = 'diag'. Be careful to ensure the transformation of Kl is diagonal or else log likelihood values will be incorrect!")
        if self.Kt.decomp == diag_eigendecomp and not self.St.decomp == diag_eigendecomp:
            raise Warning("The transformation of Kt is set to be diagonal but the matrix St is not set to diagonal. This may be possible for example if Kt is a scalar times the identity matrix or Kt shares the same eigenvectors as St but it is not true if Kt is any general diagonal matrix. Alternatively perhaps St is also diagonal and you forgot to add St.decomp = 'diag'. Be careful to ensure the transformation of Kt is diagonal or else log likelihood values will be incorrect!")

        self.Sl_decomp_fn = lambda Sl: decomp_S(Sl, eigen_fn = self.Sl.decomp)
        self.St_decomp_fn = lambda St: decomp_S(St, eigen_fn = self.St.decomp)
        self.Kl_tilde_decomp_fn = lambda Kl, Sl_inv_sqrt: decomp_K_tilde(Kl, Sl_inv_sqrt, eigen_fn = self.Kl.decomp)
        self.Kt_tilde_decomp_fn = lambda Kt, St_inv_sqrt: decomp_K_tilde(Kt, St_inv_sqrt, eigen_fn = self.Kt.decomp)

    
    def eigendecomp_no_stored_results(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

            
        stored_values["Sl"] = self.Sl(hp, x_l, x_l)
        stored_values["lam_Sl"], stored_values["Sl_inv_sqrt"] = self.Sl_decomp_fn(stored_values["Sl"])

        stored_values["St"] = self.St(hp, x_t, x_t)
        stored_values["lam_St"], stored_values["St_inv_sqrt"] = self.St_decomp_fn(stored_values["St"])

        stored_values["Kl"] = self.Kl(hp, x_l, x_l)
        stored_values["lam_Kl_tilde"], stored_values["W_l"] = self.Kl_tilde_decomp_fn(stored_values["Kl"], stored_values["Sl_inv_sqrt"])
        
        stored_values["Kt"] = self.Kt(hp, x_t, x_t)
        stored_values["lam_Kt_tilde"], stored_values["W_t"] = self.Kt_tilde_decomp_fn(stored_values["Kt"], stored_values["St_inv_sqrt"])

        D = jnp.outer(stored_values["lam_Kl_tilde"], stored_values["lam_Kt_tilde"]) + 1.
        stored_values["D_inv"] = jnp.reciprocal(D)

        lam_S = jnp.outer(stored_values["lam_Sl"], stored_values["lam_St"])
        stored_values["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()
        
        return stored_values

    
    def eigendecomp_use_stored_results(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        stored_values: Optional[PyTree] = {},
        rtol: Optional[Scalar] = 1e-12,
        atol: Optional[Scalar] = 1e-12,
    ) -> PyTree:

        stored_values = deepcopy(stored_values)
        
        Sl = self.Sl(hp, x_l, x_l)
        St = self.St(hp, x_t, x_t)
        Kl = self.Kl(hp, x_l, x_l)
        Kt = self.Kt(hp, x_t, x_t)

        N_l = Sl.shape[0]
        N_t = St.shape[0]

        
        if stored_values: 
            Sl_diff = jax.lax.cond(jnp.allclose(Sl, stored_values["Sl"], rtol = rtol, atol = atol), lambda hp: False, lambda hp: True, hp)
            St_diff = jax.lax.cond(jnp.allclose(St, stored_values["St"], rtol = rtol, atol = atol), lambda hp: False, lambda hp: True, hp)
            Kl_diff = jax.lax.cond(jnp.allclose(Kl, stored_values["Kl"], rtol = rtol, atol = atol), lambda hp: Sl_diff, lambda hp: True, hp)
            Kt_diff = jax.lax.cond(jnp.allclose(Kt, stored_values["Kt"], rtol = rtol, atol = atol), lambda hp: St_diff, lambda hp: True, hp)
        else:
            Sl_diff = St_diff = Kl_diff = Kt_diff = True
            
            stored_values["lam_Sl"] = jnp.zeros(N_l)
            stored_values["Sl_inv_sqrt"] = jnp.zeros((N_l, N_l))
            stored_values["lam_St"] = jnp.zeros(N_t)
            stored_values["St_inv_sqrt"] = jnp.zeros((N_t, N_t))
            stored_values["lam_Kl_tilde"] = jnp.zeros(N_l)
            stored_values["W_l"] = jnp.zeros((N_l, N_l))
            stored_values["lam_Kt_tilde"] = jnp.zeros(N_t)
            stored_values["W_t"] = jnp.zeros((N_t, N_t))


        stored_values["lam_Sl"], stored_values["Sl_inv_sqrt"] = jax.lax.cond(Sl_diff, self.Sl_decomp_fn,
                                                                           lambda *args: (stored_values["lam_Sl"], stored_values["Sl_inv_sqrt"]), Sl)
        
        stored_values["lam_St"], stored_values["St_inv_sqrt"] = jax.lax.cond(St_diff, self.St_decomp_fn,
                                                                           lambda *args: (stored_values["lam_St"], stored_values["St_inv_sqrt"]), St)

        stored_values["lam_Kl_tilde"], stored_values["W_l"] = jax.lax.cond(Kl_diff, self.Kl_tilde_decomp_fn,
                                                                         lambda *args: (stored_values["lam_Kl_tilde"], stored_values["W_l"]), Kl, stored_values["Sl_inv_sqrt"])

        stored_values["lam_Kt_tilde"], stored_values["W_t"] = jax.lax.cond(Kt_diff, self.Kt_tilde_decomp_fn,
                                                                         lambda *args: (stored_values["lam_Kt_tilde"], stored_values["W_t"]), Kt, stored_values["St_inv_sqrt"])

        D = jnp.outer(stored_values["lam_Kl_tilde"], stored_values["lam_Kt_tilde"]) + 1.
        stored_values["D_inv"] = jnp.reciprocal(D)

        lam_S = jnp.outer(stored_values["lam_Sl"], stored_values["lam_St"])
        stored_values["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()

        stored_values["Sl"] = Sl
        stored_values["St"] = St
        stored_values["Kl"] = Kl
        stored_values["Kt"] = Kt

        return stored_values
    
    
    def logL(self, hp, x_l, x_t, R, stored_values):

        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        rKr = r_K_inv_r(R, stored_values)
        logdetK = logdetK_calc(stored_values)
        
        nu = jnp.power(10, hp["nu"])
        
        N = R.size
        other_terms = gammaln(0.5*(nu+N)) - gammaln(0.5*nu) - 0.5*N*jnp.log(jnp.pi*nu)
        
        logL = - 0.5 * (nu + N) * jnp.log(1 + rKr/nu) - 0.5 * logdetK + other_terms

        return  logL.sum(), stored_values

    
    def logL_hessianable(self, hp, x_l, x_t, R, stored_values):

        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        rKr = r_K_inv_r(R, stored_values)
        logdetK = logdetK_calc_hessianable(stored_values)
        
        nu = jnp.power(10, hp["nu"])
        
        N = R.size
        other_terms = gammaln(0.5*(nu+N)) - gammaln(0.5*nu) - 0.5*N*jnp.log(jnp.pi*nu)
        
        logL = - 0.5 * (nu + N) * jnp.log(1 + rKr/nu) - 0.5 * logdetK + other_terms

        return  logL.sum(), stored_values
        
    
    def K(self, hp, x_l, x_l_s, x_t, x_t_s, **kwargs):

        Kl = self.Kl(hp, x_l, x_l_s, **kwargs)
        Kt = self.Kt(hp, x_t, x_t_s, **kwargs)
        Sl = self.Sl(hp, x_l, x_l_s, **kwargs)
        St = self.St(hp, x_t, x_t_s, **kwargs)
        
        K = jnp.kron(Kl, Kt) + jnp.kron(Sl, St)
        
        return K
    
    
    def predict(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_l_s: JAXArray,
        x_t: JAXArray,
        x_t_s: JAXArray,
        R: JAXArray,
        M_s: JAXArray,
        stored_values: Optional[PyTree] = {},
        wn = True,
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
        
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        Kl_s = self.Kl(hp, x_l, x_l_s, wn = False)
        Kt_s = self.Kt(hp, x_t, x_t_s, wn = False)
        Sl_s = self.Sl(hp, x_l, x_l_s, wn = False)
        St_s = self.St(hp, x_t, x_t_s, wn = False)
        
        Kl_ss = self.Kl(hp, x_l_s, x_l_s, wn = wn)
        Kt_ss = self.Kt(hp, x_t_s, x_t_s, wn = wn)
        Sl_ss = self.Sl(hp, x_l_s, x_l_s, wn = wn)
        St_ss = self.St(hp, x_t_s, x_t_s, wn = wn)

        alpha = K_inv_vec(R, stored_values)

        gp_mean = M_s + kron_prod(Kl_s.T, Kt_s.T, alpha) + kron_prod(Sl_s.T, St_s.T, alpha)

        Y_l = Kl_s.T @ stored_values["W_l"]
        Y_t = Kt_s.T @ stored_values["W_t"]
        Z_l = Sl_s.T @ stored_values["W_l"]
        Z_t = St_s.T @ stored_values["W_t"]

        sigma_diag = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
        sigma_diag += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

        sigma_diag -= kron_prod(Y_l**2, Y_t**2, stored_values["D_inv"])
        sigma_diag -= kron_prod(Z_l**2, Z_t**2, stored_values["D_inv"])
        sigma_diag -= 2*kron_prod(Y_l * Z_l, Y_t * Z_t, stored_values["D_inv"])
        
        return gp_mean, sigma_diag

    
def diag_eigendecomp(K):
        return jnp.diag(K), jnp.eye(K.shape[0])


def decomp_S(S: JAXArray, eigen_fn: Optional[Callable] = jnp.linalg.eigh) -> Tuple[JAXArray, JAXArray]:

    lam_S, Q_S = eigen_fn(S)
    S_inv_sqrt = Q_S @ jnp.diag(jnp.sqrt(jnp.reciprocal(lam_S)))

    return lam_S, S_inv_sqrt
        
    
def decomp_K_tilde(K: JAXArray, S_inv_sqrt: JAXArray, eigen_fn: Optional[Callable] = jnp.linalg.eigh) -> Tuple[JAXArray, JAXArray]:
        
    K_tilde = S_inv_sqrt.T @ K @ S_inv_sqrt
    lam_K_tilde, Q_K_tilde = eigen_fn(K_tilde)
    W_K_tilde = S_inv_sqrt @ Q_K_tilde

    return lam_K_tilde, W_K_tilde
