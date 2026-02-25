import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, hessian, vmap, custom_jvp, jit
from jax.flatten_util import ravel_pytree
from copy import deepcopy
from tqdm import tqdm
from typing import Callable, Tuple, Union, Any, Optional
from functools import partial

from .luas_types import Kernel, PyTree, JAXArray, Scalar
from .kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable
from .jax_convenience_fns import array_to_pytree_2D, get_corr_mat


__all__ = [
    "LuasKernelNew",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)



class LuasKernel(Kernel):
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
    ):
        
        self.Sl, self.St = Sigma
        self.Kl, self.Kt = K

        # Define for consistency with other kernel objects
        self.Sigma = Sigma
        self.K_list = [K]
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        if use_stored_values:
            self.decompose = self.eigendecomp_use_stored_values
        else:
            self.decompose = self.eigendecomp_no_stored_values


    
    # def eigendecomp_use_stored_values(
    #     self,
    #     hp: PyTree,
    #     x_l: JAXArray,
    #     x_t: JAXArray, 
    #     stored_values: Optional[PyTree] = {},
    #     rtol: Optional[Scalar] = 1e-12,
    #     atol: Optional[Scalar] = 1e-12,
    # ) -> PyTree:
    #     r"""Required calculations for the decomposition of the overall matrix ``K`` where the previously
    #     stored decomposition of ``K`` may be used for the calculation of a new decomposition.
    #     This checking if any of the matrices have changed and if they are similar within the given
    #     tolerances a previously computed eigendecomposition can be used to avoid recalculating it.
    #     This can provide significant runtime savings if some hyperparameters are being kept fixed
    #     including if blocked Gibbs sampling is being used on groups of hyperparameters.
        
    #     We can decompose the inverse of ``K`` into the matrices:

    #     .. math::
        
    #         K^{-1} = [W_l \otimes W_t] D^{-1} [W_l^T \otimes W_t^T]
        
    #     Where this function will calculate ``W_l``, ``W_t`` and ``D_inv`` and stored them in the
    #     stored_values PyTree for future log likelihood calculations.
        
    #     Args:
    #         hp (Pytree): Hyperparameters needed to build the covariance matrices
    #             ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
    #             parameters are also included.
    #         x_l (JAXArray): Array containing wavelength/vertical dimension regression variable(s)
    #             for the observed locations. May be of shape ``(N_l,)`` or ``(d_l,N_l)`` for ``d_l``
    #             different wavelength/vertical regression variables.
    #         x_t (JAXArray): Array containing time/horizontal dimension regression variable(s) for the
    #             observed locations. May be of shape ``(N_t,)`` or ``(d_t,N_t)`` for ``d_t`` different
    #             time/horizontal regression variables.
    #         stored_values (PyTree): Stored values from the decomposition of the covariance matrices. For
    #             :class:`LuasKernel` this consists of values computed using the eigendecomposition
    #             of each matrix and also the log determinant of ``K``.
    #         rtol (Scalar): The relative tolerance value any of the component covariance matrices
    #             must be within in order for the matrix to be considered unchanged and stored values for
    #             its decomposition to be used.
    #         atol (Scalar): The absolute tolerance values any of the component covariance matrices
    #             must be within in order for the matrix to be considered unchanged and stored values for
    #             its decomposition to be used.
        
    #     Returns:
    #         PyTree: Stored values from the decomposition of the covariance matrices. For
    #         :class:`LuasKernel` this consists of values computed using the eigendecomposition
    #         of each matrix and also the log determinant of ``K``.
        
    #     """

    #     stored_values = deepcopy(stored_values)
        
    #     # Calculate each of the four component matrices
    #     Sl = self.Sl(hp, x_l, x_l)
    #     St = self.St(hp, x_t, x_t)
    #     Kl = self.Kl(hp, x_l, x_l)
    #     Kt = self.Kt(hp, x_t, x_t)
        
    #     if stored_values:
    #         # Check if any of the 4 component matrices have changed from their values in stored_values
            
    #         # Note JAX requires the two possible outputs of the conditional to be functions
    #         # so we use functions which just return True or False
    #         Sl_diff = jax.lax.cond(jnp.allclose(Sl, stored_values["Sl"], rtol = rtol, atol = atol),
    #                                lambda : False, lambda : True)
    #         St_diff = jax.lax.cond(jnp.allclose(St, stored_values["St"], rtol = rtol, atol = atol),
    #                                lambda : False, lambda : True)
            
    #         # Note that if Sl is different than Kl_tilde is also almost certainly different
    #         # so even if Kl hasn't changed we still need to recompute the decomposition of Kl_tilde and similarly for Kt
    #         Kl_tilde_diff = jax.lax.cond(jnp.allclose(Kl, stored_values["Kl"], rtol = rtol, atol = atol),
    #                                     lambda : Sl_diff, lambda : True)
    #         Kt_tilde_diff = jax.lax.cond(jnp.allclose(Kt, stored_values["Kt"], rtol = rtol, atol = atol),
    #                                     lambda : St_diff, lambda : True)
    #     else:
    #         Sl_diff = St_diff = Kl_tilde_diff = Kt_tilde_diff = True
            
    #         N_l = x_l.shape[-1]
    #         N_t = x_t.shape[-1]
            
    #         # JAX requires that the two outputs of any conditional statements have the same shape
    #         # so must define matrices of same shape as their actual values even though they will be overwritten
    #         stored_values["lam_Sl"] = jnp.zeros(N_l)
    #         stored_values["Sl_inv_sqrt"] = jnp.zeros((N_l, N_l))
    #         stored_values["lam_St"] = jnp.zeros(N_t)
    #         stored_values["St_inv_sqrt"] = jnp.zeros((N_t, N_t))
    #         stored_values["lam_Kl_tilde"] = jnp.zeros(N_l)
    #         stored_values["W_l"] = jnp.zeros((N_l, N_l))
    #         stored_values["lam_Kt_tilde"] = jnp.zeros(N_t)
    #         stored_values["W_t"] = jnp.zeros((N_t, N_t))


    #     # For each of the 4 component matrices conditionally decompose them if they have changed since the last calculation
    #     stored_values["lam_Sl"], stored_values["Sl_inv_sqrt"] = jax.lax.cond(Sl_diff,
    #                                                                        self.Sl_decomp_fn,
    #                                                                        lambda *args: (stored_values["lam_Sl"],
    #                                                                                       stored_values["Sl_inv_sqrt"]),
    #                                                                        Sl)
        
    #     stored_values["lam_St"], stored_values["St_inv_sqrt"] = jax.lax.cond(St_diff,
    #                                                                        self.St_decomp_fn,
    #                                                                        lambda *args: (stored_values["lam_St"], 
    #                                                                                       stored_values["St_inv_sqrt"]),
    #                                                                        St)

    #     stored_values["lam_Kl_tilde"], stored_values["W_l"] = jax.lax.cond(Kl_tilde_diff,
    #                                                                      self.Kl_tilde_decomp_fn,
    #                                                                      lambda *args: (stored_values["lam_Kl_tilde"],
    #                                                                                     stored_values["W_l"]),
    #                                                                      Kl, stored_values["Sl_inv_sqrt"])

    #     stored_values["lam_Kt_tilde"], stored_values["W_t"] = jax.lax.cond(Kt_tilde_diff,
    #                                                                      self.Kt_tilde_decomp_fn,
    #                                                                      lambda *args: (stored_values["lam_Kt_tilde"],
    #                                                                                     stored_values["W_t"]),
    #                                                                      Kt, stored_values["St_inv_sqrt"])

    #     # D is needed for calculation the log determinant of K
    #     D = jnp.outer(stored_values["lam_Kl_tilde"], stored_values["lam_Kt_tilde"]) + 1.
        
    #     # D^-1 is needed for calculating K^-1 r
    #     stored_values["D_inv"] = jnp.reciprocal(D)

    #     # Computes the log determinant of K
    #     lam_S = jnp.outer(stored_values["lam_Sl"], stored_values["lam_St"])
    #     stored_values["logdetK"] = jnp.log(jnp.multiply(D, lam_S)).sum()

    #     # Store in order to perform checks for the next call of this function
    #     stored_values["Sl"] = Sl
    #     stored_values["St"] = St
    #     stored_values["Kl"] = Kl
    #     stored_values["Kt"] = Kt

    #     return stored_values
    
    
    def eigendecomp_no_stored_values(
        self,
        x_l: JAXArray,
        x_t: JAXArray,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:
        r"""Required calculations for the decomposition of the overall matrix ``K`` where the previously
        stored decomposition of ``K`` cannot be used for the calculation of a new decomposition.
        This avoids checking if any of the matrices have changed but may result in performing the
        same eigendecomposition calculations multiple times.
        
        We can decompose the inverse of ``K`` into the matrices:

        .. math::
        
            K^{-1} = [W_l \otimes W_t] D^{-1} [W_l^T \otimes W_t^T]
        
        Where this function will calculate ``W_l``, ``W_t`` and ``D_inv`` and stored them in the
        ``stored_values`` PyTree for future log likelihood calculations.
        
        Note:
            Values still need to be stored for any log likelihood calculations so this method does
            not save memory over ``eigendecomp_use_stored_values``. It may however reduce runtimes
            by avoiding checking if matrices have changed so it could be beneficial if all hyperparameters
            are being varied simultaneously for each calculation.
            
        
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
            stored_values (PyTree): This may contain stored values from the decomposition of ``K`` but
                this method will not make use of it. This dictionary will simply be overwritten with
                new stored values from the decomposition of ``K``.
        
        Returns:
            PyTree: Stored values from the decomposition of the covariance matrices. For
            :class:`LuasKernel` this consists of values computed using the eigendecomposition
            of each matrix and also the log determinant of ``K``.
        
        """
        
        self.Sl, stored_values_Sl = self.Sl.decompose(x_l)
        self.St, stored_values_St = self.St.decompose(x_t)

        # Generate transformed objects, doesn' actually do transformation yet
        Kl_tilde = self.Sl.inv_sqrt_transform(self.Kl)
        Kt_tilde = self.St.inv_sqrt_transform(self.Kt) # Just returns Kt if St = Identity()

        # Evaluates transformation and does eigendecomp
        stored_values["lam_Kl_tilde"], stored_values["W_l"] = Kl_tilde.eigendecomp(x_l)
        stored_values["lam_Kt_tilde"], stored_values["W_t"] = Kt_tilde.eigendecomp(x_t)

        # D is needed for calculation the log determinant of K
        D = jnp.outer(stored_values["lam_Kl_tilde"], stored_values["lam_Kt_tilde"]) + 1.
        
        # D^-1 is needed for calculating K^-1 r
        stored_values["D_inv"] = jnp.reciprocal(D)

        # Computes the log determinant of K
        stored_values["logdetK"] = x_t.shape[-1]*stored_values_Sl["logdetK"] + x_l.shape[-1]*stored_values_St["logdetK"] + jnp.log(D).sum()

        return self, stored_values

    
    def solve(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        """Computes the log likelihood using the method originally presented in Rakitsch et al. (2013)
        and also outlined in Fortune at al. (2024). Also returns stored values from the matrix decomposition.
        
        Note:
            Calculating the hessian of this function with ``jax.hessian`` may not produce numerically stable
            results. ``LuasKernel.logL_hessianable`` is recommended is values of the hessian are needed.
            This method typically outperforms ``LuasKernel.logL_hessianable`` in runtime for gradient
            calculations however.
        
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
        # likelihood and its gradient

        R = self.Sl.cho_solve(R, transpose = 0)
        R = self.St.cho_solve(R.T, transpose = 0).T
        
        K_inv_R = K_inv_vec(R, stored_values)

        K_inv_R = self.Sl.cho_solve(K_inv_R, transpose = 1)
        K_inv_R = self.St.cho_solve(K_inv_R.T, transpose = 1).T
        
        return  K_inv_R, stored_values

    # Take a draw from K_inv
    def K_inv_draw(self, hp, x_l, x_t, stored_values, z = None):

        if z is not None:
            Z = z.copy()
            assert Z.shape == (x_l.shape[-1], x_t.shape[-1])
        else:
            Z = np.random.normal(size = (x_l.shape[-1], x_t.shape[-1]))
        
        stored_values = self.decomp_fn(hp, x_l, x_t, stored_values)
        
        Z *= jnp.sqrt(stored_values["D_inv"])
        Z = kron_prod(stored_values["W_l"], stored_values["W_t"], Z)
        Z = self.Sl.cho_solve(Z, transpose = 1)
        Z = self.St.cho_solve(Z.T, transpose = 1).T
    
        return Z, stored_values


    def K_mult_vec(self, hp, x_l, x_t, R, stored_values, **kwargs):

        R_prime = self.Sl.left_mult(R, hp, x_l, x_l, **kwargs)
        Kr = self.St.left_mult(R_prime.T, hp, x_t, x_t, **kwargs).T

        R_prime = self.Kl.left_mult(R, hp, x_l, x_l, **kwargs)
        Kr += self.Kt.left_mult(R_prime.T, hp, x_t, x_t, **kwargs).T
        
        return Kr
    
    
    def logL(
        self,
        R: JAXArray,
        stored_values: PyTree,
    ) -> Tuple[Scalar, PyTree]:
        """Computes the log likelihood using the method originally presented in Rakitsch et al. (2013)
        and also outlined in Fortune at al. (2024). Also returns stored values from the matrix decomposition.
        
        Note:
            Calculating the hessian of this function with ``jax.hessian`` may not produce numerically stable
            results. ``LuasKernel.logL_hessianable`` is recommended is values of the hessian are needed.
            This method typically outperforms ``LuasKernel.logL_hessianable`` in runtime for gradient
            calculations however.
        
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
        # stored_values = self.decompose(x_l, x_t, stored_values = stored_values)
        
        # Use functions with custom derivatives to accurately calculate the log
        # likelihood and its gradient
        
        R = self.Sl.matrix_inv_sqrt(R, transpose = 0)
        R = self.St.matrix_inv_sqrt(R.T, transpose = 0).T

        rKr = r_K_inv_r(R, stored_values)
        
        logdetK = logdetK_calc(stored_values)
        logL = - 0.5 * logdetK - 0.5 * R.size * jnp.log(2*jnp.pi) - 0.5 * rKr 

        return  logL

    
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
    

    def generate_noise(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        size: Optional[int] = 1,
        wn: Optional[bool] = True,
        z = None,
    ) -> JAXArray:
        r"""Generate noise with the covariance matrix returned by this kernel using the input
        hyperparameters ``hp``.
        
        Solves for the matrix square root of K and then multiplies this by a random normal vector.
        Doing it this way has numerical stability advantages over generating noise separately for
        each of the two kronecker products of K as they might not both be well-conditioned matrices.
        
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
            size (int, optional): The number of different draws of noise to generate. Defaults to 1.
            wn (bool, optional): Whether to include white noise when generating noise. Must have
                a `wn` keyword argument in all kernel functions ``Kl``, ``Kt``, ``Sl``, ``St``.
                
        Returns:
            JAXArray: If ``size = 1`` will generate noise of shape ``(N_l, N_t)``, otherwise if ``size > 1`` then
            generated noise will be of shape ``(N_l, N_t, size)``.
        
        """
        
        N_l = x_l.shape[-1]
        N_t = x_t.shape[-1]

        self.Sl.cholesky_decomp(hp, x_l)
        self.St.cholesky_decomp(hp, x_t)

        # Generate transformed objects, doesn' actually do transformation yet
        Kl_tilde = self.Sl.cholesky_transform(self.Kl)
        Kt_tilde = self.St.cholesky_transform(self.Kt) # Just returns Kt if St = Identity()

        # Evaluates transformation and does eigendecomp
        lam_Kl_tilde, Q_Kl_tilde = Kl_tilde.eigendecomp(hp, x_l)
        lam_Kt_tilde, Q_Kt_tilde = Kt_tilde.eigendecomp(hp, x_t)

        # D is needed for calculation the log determinant of K
        D = jnp.outer(lam_Kl_tilde, lam_Kt_tilde) + 1.
        D_sqrt = jnp.sqrt(D)
        D_sqrt = D_sqrt.reshape((N_l, N_t, 1))

        if z is None:
            z = np.random.normal(size = (N_l, N_t, size))
        else:
            assert z.shape[0] == N_l and z.shape[1] == N_t
        
        # Multiply by the matrix sqrt of K
        z = jnp.multiply(D_sqrt, z)

        kron_prod_vmap = jax.vmap(kron_prod, in_axes = (None, None, 2), out_axes = 2)

        z = kron_prod_vmap(Q_Kl_tilde, Q_Kt_tilde, z)

        Sl_cho_mult_vmap = jax.vmap(self.Sl.cho_mult, in_axes = 2, out_axes = 2)
        St_cho_mult_vmap = jax.vmap(self.St.cho_mult, in_axes = 2, out_axes = 2)

        z = Sl_cho_mult_vmap(z)
        z = jnp.swapaxes(St_cho_mult_vmap(jnp.swapaxes(z, 0, 1)), 0, 1)
        
        # If size = 1 then return as shape (N_l, N_t) instead of (N_l, N_t, 1)
        if size == 1:
            z = z.reshape((N_l, N_t))
            
        return z
