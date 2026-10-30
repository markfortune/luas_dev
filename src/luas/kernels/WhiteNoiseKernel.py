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


class WhiteNoiseKernel(CovType):
    r"""Diagonal covariance matrix with an independent variance for every data point.

    Models uncorrelated noise whose amplitude does not need to be separable in each
    dimension, i.e.

    .. math::

        k(\lambda, \lambda', t, t') = \sigma^2(\lambda, t)\,\delta_{\lambda\lambda'}\delta_{tt'},
        \qquad \mathbf{K} = \mathrm{diag}\left(\mathrm{vec}(\mathbf{D})\right),

    where :math:`\mathbf{D}` has the same shape as the data. If the variance can be
    written as :math:`\sigma^2(\lambda)\sigma^2(t)`, the same model can instead be
    built with :class:`luas.SeparableKernel` using ``Noise``/``KroneckerDelta``
    kernels, which can then be combined with correlated kernel terms.

    The variance is split into two parts, :math:`\mathbf{D} = \texttt{diag} +
    \texttt{wn\_diag}`. Both contribute to the log-likelihood, but ``wn_diag`` is
    treated as white noise and is dropped when ``wn = False`` (e.g. when
    :class:`luas.GP` computes a GP predictive mean), while ``diag`` is treated as
    part of the signal being fit.

    **Runtime scaling.** :math:`\mathcal{O}(N)` for :math:`N` total data points.

    **Selection.** :class:`luas.GP` uses ``WhiteNoiseKernel(wn_diag = 1.)`` by
    default when no kernel function is given, which gives an ordinary
    (unit-variance) least-squares log-likelihood. For parameter-dependent
    variances, have the kernel function return a ``WhiteNoiseKernel`` directly.

    Warning:
        ``diag`` and ``wn_diag`` should have the same shape as the data. Scalars or
        arrays which only broadcast against the data (e.g. shape ``(N_l, 1)``) are
        accepted by some methods but give an incorrect log determinant.

    Example:
        .. code-block:: python

            def kf(p, X):
                # Separate white noise amplitude for every data point
                return luas.WhiteNoiseKernel(wn_diag = jnp.exp(2*p["log_sigma_2D"]))

            gp = luas.GP(p, (x_l, x_t), kf = kf)

    Args:
        diag (JAXArray, optional): Variance at each data point which is retained
            when ``wn = False``. Defaults to ``0.``.
        wn_diag (JAXArray, optional): White noise variance at each data point,
            excluded when ``wn = False``. Defaults to ``0.``.
        **kwargs: Ignored.

    See Also:
        :class:`luas.SeparableKernel` for separable white noise and measurement
        covariance.
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
        self.opt_name = "WhiteNoiseKernel"
        

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
