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
from .kronecker_fns import make_vec, make_mat, kron_prod
from .jax_convenience_fns import array_to_pytree_2D

__all__ = ["GeneralKernel", "general_cholesky"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

    
def general_cholesky(K: JAXArray) -> Tuple[JAXArray, JAXArray]:
    """Takes an arbitrary covariance matrix K and returns the Cholesky decomposition
    as a lower triangular matrix as well as computing the log determinant of the covariance
    matrix.
    
    Args:
        K (JAXArray): Covariance matrix to decompose.
            
    """

    L_cho = JLA.cholesky(K)
    logdetK = 2*jnp.log(jnp.diag(L_cho)).sum()

    return L_cho, logdetK


def diag_cholesky(K: JAXArray) -> Tuple[JAXArray, JAXArray]:
    """Takes an arbitrary covariance matrix K and returns the Cholesky decomposition
    as a lower triangular matrix as well as computing the log determinant of the covariance
    matrix.
    
    Args:
        K (JAXArray): Covariance matrix to decompose.
            
    """

    K_diag = jnp.diag(K)
    L_cho = jnp.diag(jnp.sqrt(K_diag))
    logdetK = jnp.log(K_diag).sum()

    return L_cho, logdetK


class KronIdentityKernel(Kernel):
    """Kernel object which solves for the log likelihood for any general kernel function K.
    Can also generate noise from K and can be used to compute the GP predictive mean and 
    predictive variance conditioned on observed data.
    
    Args:
        K (Callable, optional): Function which returns the covariance matrix K.
        decomp_fn (Callable, optional): Function which given the computed matrix K
            computes the Cholesky decomposition and log determinant of K.
            Defaults to general_cholesky which performs Cholesky decomposition for
            any general covariance matrix.
            
    """
    
    def __init__(
        self,
        Kl: Optional[Callable] = None,
    ):
        
        self.Kl = Kl
        
        if self.Kl.decomp is None:
            self.Kl.decomp_fn = general_cholesky
        elif self.Kl.decomp == "diag":
            self.Kl.decomp_fn = diag_cholesky
        else:
            self.Kl.decomp_fn = self.Kl.decomp
            
        # alias to maintain consistency with LuasKernel which has a separate fn for calculating the hessian
        self.logL_hessianable = self.logL

    
        
    def decomp_fn(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray, 
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
    
        N_t = x_t.shape[-1]
        Kl = self.Kl(hp, x_l, x_l)
        
        stored_values["L_cho"], logdetKl = self.Kl.decomp_fn(Kl)
        stored_values["logdetK"] = N_t*logdetKl
        return stored_values
   
        
    def logL(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
            
        alpha = JLA.solve_triangular(stored_values["L_cho"], R, trans = 1)

        logL =  - 0.5 * stored_values["logdetK"] - (R.size/2.) * jnp.log(2*jnp.pi)  - 0.5 *  jnp.sum(jnp.square(alpha))

        return logL, stored_values
    

    def generate_noise(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        stored_values: Optional[PyTree] = {},
        size: Optional[int] = 1
    ) -> JAXArray:
        
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = {})
        
        z = np.random.normal(size = (x_l.shape[-1], x_t.shape[-1], size))
        r = jnp.einsum("ij,j...->i...", stored_values["L_cho"], z)
        R = make_mat(r, x_l.shape[-1], x_t.shape[-1])
                     
        return R
    
    
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
        return_std_dev = True,
    ) -> Tuple[JAXArray, JAXArray, JAXArray]:
        
        N_t = x_t.shape[-1]
        
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
            
        Kl_s = self.Kl(hp, x_l, x_l_s, wn = False)
        Kl_ss = self.Kl(hp, x_l_s, x_l_s, wn = wn)

        # Generate mean function and compute residuals
        alpha = JLA.solve_triangular(stored_values["L_cho"], R, trans = 1)
        K_inv_R = JLA.solve_triangular(stored_values["L_cho"], alpha, trans = 0)
        K_s_K_inv_R = kron_prod(Kl_s, jnp.eye(N_t), K_inv_R)

        gp_mean = M_s + make_mat(K_s_K_inv_R, x_l_s.shape[-1], x_t_s.shape[-1])

        if return_std_dev:
            sigma_diag = jnp.outer(jnp.diag(Kl_ss), jnp.ones(N_t))

            K_s_alpha = JLA.solve_triangular(stored_values["L_cho"], Kl_s, trans = 1)
            sigma_diag -= jnp.outer(jnp.diag(K_s_alpha.T @ K_s_alpha), jnp.ones(N_t))
            sigma_diag = make_mat(sigma_diag, x_l_s.shape[-1], x_t_s.shape[-1])
            pred_err = jnp.sqrt(sigma_diag)
        else:
            raise NotImplementedError
        
        return gp_mean, pred_err
    
