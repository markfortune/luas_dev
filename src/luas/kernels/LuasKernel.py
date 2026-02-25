import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, hessian, vmap, custom_jvp, jit, lax
from jax.flatten_util import ravel_pytree
from copy import deepcopy
from tqdm import tqdm
from typing import Callable, Tuple, Union, Any, Optional
from functools import partial

from luas.kernels.covtype import CovType, Outer
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable, calc_total_size, cyclic_transpose, vmap_for_tensors, tensor_mult
from luas.jax_convenience_fns import array_to_pytree_2D, get_corr_mat
from luas.kernels.WhiteNoiseKernel import WhiteNoiseKernel

__all__ = [
    "LuasKernelNew",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)



class LuasKernel(CovType):
    r"""Kernel class which solves for the log likelihood for any covariance matrix which
    is the sum of two kronecker products of the covariance matrix in each of two dimensions
    i.e. the full covariance matrix K is given by:
    
    .. math::
        K = K_l \otimes K_t + S_l \otimes S_t
    
    although we can avoid calculating ``K`` for many calculations implemented here.
        
    The ``Kl`` and ``Sl`` functions should both return ``(N_l, N_l)`` matrices which will be the covariance
    matrices in the wavelength/vertical direction.
    
    The ``Kt`` and ``St`` functions should both return ``(N_t, N_t)`` matrices which will by the covariance
    matrices in the time/horizontal direction.
    
    .. code-block:: python

        >>> from luas import LuasKernel, kernels
        >>> def Kl_fn(hp, x_l1, x_l2, wn = True):
        >>> ... return hp["h"]**2*kernels.squared_exp(x_l1, x_l2, hp["l_l"])
        >>> def Kt_fn(hp, x_t1, x_t2, wn = True):
        >>> ... return kernels.squared_exp(x_t1, x_t2, hp["l_t"])
        >>> # ... And similarly for Sl_fn, St_fn
        >>> kernel = LuasKernel(Kl = Kl_fn, Kt = Kt_fn, Sl = Sl_fn, St = St_fn)
        ... )
    
    See https://luas.readthedocs.io/en/latest/tutorials.html for more detailed tutorials on how to use.
        
    Args:
        Kl (Callable): Function which returns the covariance matrix Kl, should be of the form
            ``Kl(hp, x_l1, x_l2, wn = True)``.
        Kt (Callable): Function which returns the covariance matrix Kt, should be of the form
            ``Kt(hp, x_t1, x_t2, wn = True)``.
        Sl (Callable): Function which returns the covariance matrix Sl, should be of the form
            ``Sl(hp, x_l1, x_l2, wn = True)``.
        St (Callable): Function which returns the covariance matrix St, should be of the form
            ``St(hp, x_t1, x_t2, wn = True)``.
        use_stored_values (bool, optional): Whether to perform checks if any of the component
            covariance matrices have changed and to make use of previously stored values for
            the decomposition of those matrices if they're the same. If ``False`` then will
            not perform these checks and will compute the eigendecomposition of all matrices
            for every calculation.
    
    """
    
    def __init__(
        self,
        Sigma,
        K,
        use_stored_values: Optional[bool] = False,
        eigen_dims = None,
    ):
        
        self.Sigma = Sigma
        self.K = K
        self.dim = len(Sigma)
        
        if eigen_dims is None:
            self.eigen_dims = range(self.dim)
        else:
            self.eigen_dims = eigen_dims
        
        self.K_list = [K]
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            self.decompose = self.eigendecomp_use_stored_values
        else:
            self.decompose = self.eigendecomp_no_stored_values

    
    def evaluate(self, *X, **kwargs):

        dim = len(X)
        Sigma = self.Sigma[0].evaluate(X[0], X[0], **kwargs)
        
        for d in range(1, dim):
            Sigma = jnp.kron(Sigma, self.Sigma[d].evaluate(X[d], X[d], **kwargs))
        
        K = self.K[0].evaluate(X[0], X[0], **kwargs)
        for d in range(1, dim):
            K = jnp.kron(K, self.K[d].evaluate(X[d], X[d], **kwargs))
        
        Sigma += K
        
        return Sigma
        
    
    def eigendecomp_no_stored_values(
        self,
        *X: Tuple,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

        if stored_values:
            R_shape = stored_values["R_shape"]
            total_size = jnp.prod(jnp.array(R_shape))
        else:
            total_size = calc_total_size(X)
        
        gp_dim = len(X)

        sigma_decomp_mats = ()
        eigen_decomp_mats = ()
        all_lam = jnp.ones(1)
        all_lam_shape = ()
        stored_values["logdetK"] = 0.

        for d in self.eigen_dims:
            Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d])

            if isinstance(self.K[d], Outer):
                K_d, _ = self.K[d].decompose(X[d])
            else:
                K_d = self.K[d]
            
            K_d_new = Sigma_d_new.inv_sqrt_transform(K_d)
            
            if gp_dim > 2:
                Sigma_d_new.matrix_sqrt = vmap_for_tensors(Sigma_d_new.matrix_sqrt)
                Sigma_d_new.matrix_inv_sqrt = vmap_for_tensors(Sigma_d_new.matrix_inv_sqrt)
                K_d_new.matrix_sqrt = vmap_for_tensors(K_d_new.matrix_sqrt)
                K_d_new.matrix_inv_sqrt = vmap_for_tensors(K_d_new.matrix_inv_sqrt)
            
            stored_values[f"lam_{d}"], stored_values[f"Q_{d}"] = K_d_new.eigendecomp(X[d])
            
            all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), stored_values[f"lam_{d}"])
            all_lam_shape = all_lam.shape
            
            sigma_decomp_mats += (Sigma_d_new,)
            eigen_decomp_mats += (stored_values[f"Q_{d}"],)
            stored_values["logdetK"] += (total_size/X[d].shape[-1])*stored_values_d["logdetK"]

        def transform_fn(R, transpose = 0):
            
            R_prime = cyclic_transpose(R, 2)

            if transpose:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        R_prime = sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 1)
                        R_prime = eigen_decomp_mats[d].T @ R_prime
                    R_prime = cyclic_transpose(R_prime, 1)
            else:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        R_prime = eigen_decomp_mats[d] @ R_prime
                        R_prime = sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 0)
                    R_prime = cyclic_transpose(R_prime, 1)

            R_prime = cyclic_transpose(R_prime, -2)
                
            return R_prime

        def inv_transform_fn(R, transpose = 0):
            R_prime = cyclic_transpose(R, 2)

            if transpose:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        R_prime = eigen_decomp_mats[d] @ R_prime
                        R_prime = sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 1)
                    R_prime = cyclic_transpose(R_prime, 1)
            else:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        R_prime = sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 0)
                        R_prime = eigen_decomp_mats[d].T @ R_prime
                    R_prime = cyclic_transpose(R_prime, 1)

            R_prime = cyclic_transpose(R_prime, -2)
            return R_prime

        self.transform_fn = transform_fn
        self.inv_transform_fn = inv_transform_fn

        K_tilde_diag = all_lam + 1
        
        self.kf_tilde = WhiteNoiseKernel(diag = K_tilde_diag, wn_diag = 0.)
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(*X)

        return self, stored_values


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
        

    def logL(
        self,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        
        R_prime = self.inv_transform_fn(R, transpose = 0)
        logL_tilde = self.kf_tilde.logL(R_prime, stored_values["kf_tilde_stored"])
        
        return logL_tilde - 0.5 * stored_values["logdetK"]


    def matmul(self, X1, X2, R, **kwargs):

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)
        K_R += tensor_mult(self.K, X1, X2, R, **kwargs)
        
        return K_R
    
    
    def logL_hessianable(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        """Computes the log likelihood using the method originally presented in Rakitsch et al. (2013)
        and also outlined in Fortune at al. (2024).
        
        Note:
            The hessian of this log likelihood function can be calculated using ``jax.hessian`` and
            should be more numerically stable for this than ``LuasKernel.logL``.
            However, this function is slower for calculating the gradients of the log likelihood so
            ``LuasKernel.logL`` is preferred unless the hessian is needed. Also returns stored values
            from the matrix decomposition.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
            x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
                for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
                different wavelength/vertical regression variables.
            x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
                observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
                time/horizontal regression variables.
            R (JAXArray): Residuals to be fit calculated from the observed data by subtracting the deterministic
                mean function. Must have the same shape as the observed data (N_l, N_t).
            stored_values (PyTree): Stored values from the decomposition of the covariance matrices. For
                :class:`LuasKernel` this consists of values computed using the eigendecomposition
                of each matrix and also the log determinant of ``K``.
                
        Returns:
            (Scalar, PyTree): A tuple where the first element is the value of the log likelihood.
            The second element is a PyTree which contains stored values from the decomposition of the
            covariance matrix.
        
        """
        
        # Calculate the decomposition of K
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values = stored_values)
        
        # Use functions with custom derivatives to accurately calculate the log
        # likelihood, its gradient and hessian
        rKr = r_K_inv_r(R, stored_values)
        logdetK = logdetK_calc_hessianable(stored_values)
        logL =  -0.5 * rKr - 0.5 * logdetK  - 0.5 * R.size * jnp.log(2*jnp.pi)

        return  logL, stored_values
        

    
    def predict(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_l_pred: JAXArray,
        x_t: JAXArray,
        x_t_pred: JAXArray,
        R: JAXArray,
        M_s: JAXArray,
        wn = True,
        return_std_dev = True,
    ) -> Tuple[JAXArray, JAXArray]:
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
        
        Note:
            The calculation of the full predictive covariance matrix when ``return_std_dev = False``
            is still experimental and may come with numerically stability issues. It is also very
            memory intensive and may cause code to crash. Future updates to luas may improve this.
        
        Args:
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
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
        # Calculate the decomposition of K
        stored_values = self.decomp_fn(hp, x_l, x_t)
        
        # Calculate the covariance between the observed and predicted points
        Kl_s = self.Kl.evaluate(hp, x_l, x_l_pred, wn = False)
        Kt_s = self.Kt.evaluate(hp, x_t, x_t_pred, wn = False)
        Sl_s = self.Sl.evaluate(hp, x_l, x_l_pred, wn = False)
        St_s = self.St.evaluate(hp, x_t, x_t_pred, wn = False)
        
        # Calculate the covariance between predicted points with other predicted points
        Kl_ss = self.Kl.evaluate(hp, x_l_pred, x_l_pred, wn = wn)
        Kt_ss = self.Kt.evaluate(hp, x_t_pred, x_t_pred, wn = wn)
        Sl_ss = self.Sl.evaluate(hp, x_l_pred, x_l_pred, wn = wn)
        St_ss = self.St.evaluate(hp, x_t_pred, x_t_pred, wn = wn)

        # Calculate K^-1 R
        K_inv_R, stored_values = self.solve(hp, x_l, x_t, R, stored_values)

        # Calculates the GP mean including the deterministic mean function at the prediction locations
        gp_mean = M_s + kron_prod(Kl_s.T, Kt_s.T, K_inv_R) + kron_prod(Sl_s.T, St_s.T, K_inv_R)

        # Prepare matrices for calculating the predictive covariance
        KW_l = Kl_s.T @ stored_values["W_l"]
        KW_t = Kt_s.T @ stored_values["W_t"]
        SW_l = Sl_s.T @ stored_values["W_l"]
        SW_t = St_s.T @ stored_values["W_t"]

        if return_std_dev:
            # Efficiently solves for the diagonal of the predictive covariance
            pred_err = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
            pred_err += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

            # K_s.T K^-1 K_s term can be broken into these three terms
            pred_err -= kron_prod(KW_l**2, KW_t**2, stored_values["D_inv"])
            pred_err -= kron_prod(SW_l**2, SW_t**2, stored_values["D_inv"])
            pred_err -= 2*kron_prod(KW_l * SW_l, KW_t * SW_t, stored_values["D_inv"])
            
            # Take the sqrt of the diagonal to get the std dev
            pred_err = jnp.sqrt(pred_err)
            
        else:
            # Note very memory intensive!
            K_W = jnp.kron(KW_l, KW_t) + jnp.kron(SW_l, SW_t)
            pred_err = -K_W @ jnp.diag(stored_values["D_inv"].ravel()) @ K_W.T
            
            # Add the K_ss term
            pred_err += jnp.kron(Kl_ss, Kt_ss) + jnp.kron(Sl_ss, St_ss)
        
        return gp_mean, pred_err
    
