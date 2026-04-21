import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Callable, Tuple, Any, Union

import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

from luas.kernels.covtype import CovType
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import make_vec, make_mat
from luas.jax_convenience_fns import array_to_pytree_2D

__all__ = ["WhiteNoiseKernel"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

    

class WhiteNoiseKernel(CovType):
    """Kernel object which solves for the log likelihood for any general kernel function ``K``.
    Can also generate noise from ``K`` and can be used to compute the GP predictive mean and 
    predictive covariance matrix conditioned on observed data.
    
    Note:
        This method scales poorly in runtime and memory and is likely only appropriate for 
        small data sets. If the covariance matrix ``K`` possesses structure which can be exploited
        for matrix decomposition then specifying a ``decomp_fn`` which can more efficiently return
        a Cholesky factor and log determinant of ``K`` could lead to significant runtime savings.
        The :class:`LuasKernel` class should provide significant runtime savings if the covariance matrix
        has kronecker product structure in each dimension except in cases where one of the dimensions
        is very small or a sum of more than two kronecker products is needed.
        
    .. code-block:: python

        >>> from luas import GeneralKernel, kernels
        >>> def K_fn(hp, x_l1, x_l2, x_t1, x_t2, wn = True):
        >>> ... Kl = hp["h"]**2*kernels.squared_exp(x_l1, x_l2, hp["l_l"])
        >>> ... Kt = kernels.squared_exp(x_l1, x_l2, hp["l_t"])
        >>> ... K = jnp.kron(Kl, Kt)
        >>> ... return K
        >>> kernel = GeneralKernel(K = K_fn)
        ... )
    
    Args:
        K (Callable, optional): Function which returns the covariance matrix ``K``.
        decomp_fn (Callable, optional): Function which given the covariance matrix ``K``
            computes the Cholesky decomposition and log determinant of ``K``.
            Defaults to ``luas.GeneralKernel.general_cholesky`` which performs Cholesky decomposition for
            any general covariance matrix.
            
    """
    
    def __init__(
        self,
        diag: Optional[JAXArray] = 0.,
        wn_diag: Optional[JAXArray] = 0.,
        **kwargs,
    ):
        # Function used to build the covariance matrix K
        self.diag = diag
        self.wn_diag = wn_diag
        
        # alias to maintain consistency with LuasKernel which has a separate fn for calculating the hessian
        self.logL_hessianable = self.logL
        

    def evaluate(
        self,
        *X: JAXArray,
        wn = True,
        full = True,
        row_idx = None,
        col_idx = None,
        **kwargs,
    ) -> JAXArray:
        
        if not full or is_scalar(diag):
            raise Exception("Not implemented")
        
        diag_mat = jnp.diag(self.diag + wn * self.wn_diag)

        return diag_mat

    def decompose(
        self,
        *X: JAXArray,
        wn = True,
        **kwargs,
    ) -> JAXArray:

        self.D = self.diag + wn * self.wn_diag
        self.logdet = jnp.log(self.D).sum()
        
        return self, {"logdet":self.logdet}

    def matrix_sqrt(
        self,
        R: JAXArray,
        **kwargs,
    ) -> JAXArray:

        return R * jnp.sqrt(self.D)

    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        **kwargs,
    ) -> JAXArray:

        return R * jnp.sqrt(1/self.D)

    def inverse(
        self,
        R: JAXArray,
    ) -> JAXArray:

        return R * (1/self.D)

    def dot_solve(
        self,
        R: JAXArray,
    ) -> JAXArray:

        return jnp.sum(jnp.square(R)/self.D)

    def block_dot_solve(
        self,
        v1: JAXArray,
        v2: JAXArray,
        vec_dim: int,
    ) -> JAXArray:

        vec_prod = v1 * v2

        diag_vals = jnp.tensordot(1/self.D, vec_prod, axes=([vec_dim], [0]))

        return diag_vals

    def matmul(
        self,
        X1: Tuple,
        X2: Tuple,
        R: JAXArray,
        wn = True,
        full = True,
        row_idx = None,
        col_idx = None,
        **kwargs,
    ) -> JAXArray:
        
        diag = self.diag + wn * self.wn_diag

        if not full and not is_scalar(diag):
            raise Exception("Not implemented")

        return R * diag
        
    
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
        r"""Performs GP regression and computes the GP predictive mean and the GP predictive
        uncertainty as the standard devation at each location or else can return the full
        covariance matrix. Requires the input kernel function ``K`` to have a ``wn`` keyword
        argument that defines the kernel when white noise is included (``wn = True``) and
        when white noise isn't included (``wn = False``).
        
        Currently assumes the same input hyperparameters for both the observed and predicted
        locations. The predicted locations ``x_l_pred`` and ``x_t_pred`` may deviate from
        the observed locations ``x_l`` and ``x_t`` however.
        
        The GP predictive mean is defined as:
        
        .. math::

            \mathbb{E}[\vec{y}_*] = \vec{\mu}_* + \mathbf{K}_*^T \mathbf{K}^{-1} \vec{r}
            
        And the GP predictive covariance is given by:
        
        .. math::
            
            Var[\vec{y}_*] = \mathbf{K}_{**} - \mathbf{K}_*^T \mathbf{K}^{-1} \mathbf{K}_*
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrix ``K``. Will be
                unaffected if additional mean function parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_l_pred (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the prediction locations (which may be the same as the observed locations).
                May be of shape ``(N_l_pred,)`` or ``(d_l,N_l_pred)`` for ``d_l`` different
                wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            x_t_pred (JAXArray): Array containing time/horizontal dimension regression variable(s) for
                the prediction locations (which may be the same as the observed locations). May be of
                shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different time/horizontal regression variables.
            R (JAXArray): Residuals to be fit, equal to the observed data minus the deterministic
                mean function. Must have the same shape as the observed data ``(N_l, N_t)``.
            M_s (JAXArray): Mean function evaluated at the locations of the predictions ``x_l_pred``, ``x_t_pred``.
                Must have shape ``(N_l_pred, N_t_pred)`` where ``N_l_pred`` is the number of wavelength/vertical
                dimension predictions and ``N_t_pred`` the number of time/horizontal dimension predictions.
            wn (bool, optional): Whether to include white noise in the uncertainty at the predicted locations.
                Defaults to True.
            return_std_dev (bool, optional): If ``True`` will return the standard deviation of uncertainty at the predicted
                locations. Otherwise will return the full predictive covariance matrix. Defaults to True.
        
        Returns:
            (JAXArray, JAXArray): Returns a tuple of two elements, where the first element is
            the GP predictive mean at the prediction locations, the second element is either the
            standard deviation of the predictions if ``return_std_dev = True``, otherwise it will be
            the full covariance matrix of the predicted values.
        
        """
        # Calculate the covariance between predicted points with other predicted points
        if wn:
            K_ss = self.K(hp, x_l_pred, x_l_pred, x_t_pred, x_t_pred)
        else:
            K_ss = jnp.zeros_like(R)

        if return_std_dev:
            # Get diagonal of covariance of predicted locations with other predicted locations
            pred_err = K_ss.copy()
            
        else:
            # Compute full predictive covariance matrix
            pred_err = jnp.diag(K_ss.ravel())
        
        return jnp.zeros_like(R), pred_err
