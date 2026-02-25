import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Callable, Tuple, Any, Union
from math import prod

import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

from luas.kernels.covtype import CovType
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import make_vec, make_mat, kron_prod, tensor_mult
from luas.jax_convenience_fns import array_to_pytree_2D

__all__ = ["GeneralKernel"]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


class GeneralKernel(CovType):
    r"""Kernel object which solves for the log likelihood for any general kernel function ``K``.
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
        Sigma,
        *K_list,
        use_stored_values = False,
    ):
        # Function used to build the covariance matrix K
        if isinstance(Sigma, CovType):
            self.Sigma = (Sigma,) + K_list
            self.K_list = ()
        else:
            self.Sigma = Sigma
            self.K_list = K_list
        
        self.dim = len(self.Sigma)
        # alias to maintain consistency with LuasKernel which has a separate fn for calculating the hessian
        self.logL_hessianable = self.logL
    
    
    def evaluate(self, *X, **kwargs):

        dim = len(X)
        Sigma = self.Sigma[0].evaluate(X[0], X[0], **kwargs)
        
        for d in range(1, dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X[d], X[d], **kwargs))
        
        for i in range(len(self.K_list)):
            K = self.K_list[i][0].evaluate(X[0], X[0], **kwargs)
            for d in range(1, dim):
                K = jnp.kron(K, self.K_list[i][d].evaluate(X[d], X[d], **kwargs))
            
            Sigma += K
        
        return Sigma
    
        
    def decompose(
        self,
        *X,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
        
        # Simply builds the covariance matrix and decomposes it into a Cholesky factor L
        # and precomputes the log determinant of K for log likelihood calculations
        K = self.evaluate(*X)
        self.factor = JLA.cholesky(K, lower = True)
        stored_values["logdetK"] = 2*jnp.log(jnp.diag(self.factor)).sum()
        
        return self, stored_values

    
    def matrix_inv_sqrt(self, R, transpose = 0):
        
        R_shape = R.shape
        r = R.ravel("C")
        r_prime = JLA.solve_triangular(self.factor, r, lower = True, trans = transpose)
        return r_prime.reshape(R_shape)

    def matrix_sqrt(self, R, transpose = 0):
        
        R_shape = R.shape
        r = R.ravel("C")
        r_prime = self.factor @ r
        return r_prime.reshape(R_shape)

    
    def dot_solve(self, R):
        
        r = R.ravel("C")
        r_prime = JLA.solve_triangular(self.factor, r, lower = True, trans = 0)
        return jnp.square(r_prime).sum()

    
    def logL(self, R, stored_values, **kwargs):
        
        return - 0.5 * self.dot_solve(R) - 0.5 * stored_values["logdetK"] - 0.5 * R.size * jnp.log(2*jnp.pi)

    
    def matmul(
        self,
        X1,
        X2,
        R: JAXArray,
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

        R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

        for i in range(len(self.K_list)):
            R_prime += tensor_mult(self.K_list[i], X1, X2, R, **kwargs)
        
        return R_prime

        
    
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
        # Get the length of each prediction dimension
        N_l_pred = x_l_pred.shape[-1]
        N_t_pred = x_t_pred.shape[-1]
        
        # Calculate the decomposition of K for the required K^-1 calculations
        stored_values = self.decomp_fn(hp, x_l, x_t)
            
        # Calculate the covariance between the observed and predicted points
        K_s = self.kf(hp, x_l, x_l_pred, x_t, x_t_pred, wn = False)
        
        # Calculate the covariance between predicted points with other predicted points
        K_ss = self.kf(hp, x_l_pred, x_l_pred, x_t_pred, x_t_pred, wn = wn)

        # Flatten residuals vector
        r = R.ravel("C")
        
        # Use forward and backward substitution to solve K^-1 r using the Cholesky factor
        alpha = JLA.solve_triangular(stored_values["L_cho"], r, trans = 1)
        K_inv_R = JLA.solve_triangular(stored_values["L_cho"], alpha, trans = 0)
        
        # Computes the GP predictive mean
        gp_mean = K_s.T @ K_inv_R
        gp_mean = M_s + gp_mean.reshape(N_l_pred, N_t_pred)
        
        # Prepare to calculate K^-1 K_s in the predictive covariance calculation
        L_inv_K_s = JLA.solve_triangular(stored_values["L_cho"], K_s, trans = 1)
        
        if return_std_dev:
            # Get diagonal of covariance of predicted locations with other predicted locations
            pred_err = jnp.diag(K_ss)
            
            # Subtract off term related to covariance between predicted and observed locations
            # This method efficiently calculates only the diagonal of the term
            pred_err -= (L_inv_K_s**2).sum(0)
            
            # Convert shape to (N_l_pred, N_t_pred) to match observed data but at predicted locations
            pred_err = pred_err.reshape(N_l_pred, N_t_pred)
            
            # Convert from variance to std dev
            pred_err = jnp.sqrt(pred_err)
        else:
            # Directly compute the predictive covariance matrix
            pred_err = K_ss - L_inv_K_s.T @ L_inv_K_s
        
        return gp_mean, pred_err
    
