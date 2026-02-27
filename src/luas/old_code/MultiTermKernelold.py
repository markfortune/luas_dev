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
import tinygp
from tinygp.solvers.quasisep.core import LowerTriQSM, StrictLowerTriQSM, DiagQSM
import jax.scipy.linalg as JLA

from luas.kernels.covtype import Outer
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable
from luas.jax_convenience_fns import array_to_pytree_2D, get_corr_mat

cpu_device = jax.devices("cpu")[0]

__all__ = [
    "KinvR_block",
    "logL_block",
    "LuasLasrachKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


def orthonormal_nullspace_gen(A):
    
    N_l = A.shape[0]
    N_alpha = A.shape[1]

    V = jnp.zeros_like(A)
    U = jnp.zeros_like(A)
    Lam = jnp.zeros((N_alpha, N_alpha))

    for i in range(N_alpha):
        v_i = A[:, i]
        for j in range(i):
            Lam = Lam.at[j, i].set(jnp.dot(v_i, V[:, j]))
            v_i -= Lam[j, i] * V[:, j]
            
        Lam = Lam.at[i, i].set(jnp.linalg.norm(v_i))
        v_i /= Lam[i, i]
        V = V.at[:, i].set(v_i)
    
    # Householder vector
    for i in range(N_alpha):
        w_i = V[:, i]
        for j in range(i):
            w_i -= 2 * jnp.dot(U[:, j], w_i) * U[:, j]

        e_i = jnp.zeros(N_l)
        e_i = e_i.at[i].set(1)
        u_i = w_i - e_i
        u_i /= jnp.linalg.norm(u_i)

        U = U.at[:, i].set(u_i)
    
    return Lam, U



@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])


def faster_cel(kernel, X, y, diag=None):
    noise_model = tinygp.noise.Diagonal(diag=diag)
    matrix = kernel.to_symm_qsm(X)
    matrix += noise_model.to_qsm()
    factor = matrix.cholesky()
    
    return - 0.5 * (factor.solve(y)**2).sum() - jnp.sum(jnp.log(factor.diag.d)) - 0.5 * factor.shape[0] * jnp.log(2 * jnp.pi)
faster_cel_GP = jax.jit(faster_cel, device = cpu_device)

class MultiTermKernel(Kernel):
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
        *K_list,
        cel_dim = 1,
        use_stored_values: Optional[bool] = True,
    ):
        
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.cel_dim = cel_dim
        self.N_alpha = len(K_list)


        self.logL_hessianable = self.logL
        self.decomp_fn = self.decomp_no_stored_values
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        # if use_stored_values:
        #     self.decomp_fn = self.eigendecomp_use_stored_values
        # else:
        #     self.decomp_fn = self.eigendecomp_no_stored_values


    def decomp_no_stored_values(
        self,
        hp: PyTree,
        non_cel_vec: JAXArray,
        cel_vec: JAXArray,
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
        stored_values["A"] = jnp.stack([K_i[1-self.cel_dim].alpha for K_i in self.K_list], axis = 1)
        
        self.Sigma[1-self.cel_dim].cholesky_decomp(hp, non_cel_vec, non_cel_vec)

        # Generate transformed objects, doesn' actually do transformation yet
        A_tilde = self.Sigma[1-self.cel_dim].cho_solve(stored_values["A"], transpose = 0)

        # Evaluates transformation and does eigendecomp
        stored_values["J_dot"], stored_values["u_H_stack"] = orthonormal_nullspace_gen(A_tilde)

        # Computes the log determinant of K
        stored_values["logdetK"] = cel_vec.shape[-1]*self.Sigma[1-self.cel_dim].logdet
        
        return stored_values


    

# @jax.jit
# def radvel_evenfastergen(params, kernel_list, x_l, x_t, X, R):

#     N_alpha = params["A"].shape[1]
#     N_t = x_t.shape[-1]
#     N_l = x_l.shape[-1]
    
#     inv_sigma = 1/params["sigma"]

#     A = inv_sigma[:, jnp.newaxis] * params["A"]
    
#     Lam, U = orthonormal_nullspace_gen(A)

#     R_prime = inv_sigma[:, jnp.newaxis] * R

#     for i in range(N_alpha):
#         u_R_prime = U[:, i].T @ R_prime
#         R_prime -= jnp.outer(2*U[:, i], u_R_prime)

#         if i == 0:
#             kernel = Multiband(
#                 kernel=kernel_list[i],
#                 amplitudes=Lam[:, i],
#             )
#         else:
#             kernel += Multiband(
#                 kernel=kernel_list[i],
#                 amplitudes=Lam[:, i],
#             )

#     cel_logL = faster_cel_GP(kernel, X, R_prime[:N_alpha, :].ravel("F"), diag=jnp.ones(N_alpha*N_t))

#     return cel_logL - 0.5 * (R_prime[N_alpha:, :]**2).sum() - N_t * jnp.log(params["sigma"]).sum() - 0.5 * (N_l - N_alpha)*N_t * jnp.log(2*jnp.pi)


    
    
    def logL(self, hp, x_l, x_t, R, stored_values):

        non_cel_vec = (x_l, x_t)[1-self.cel_dim]
        cel_vec = (x_l, x_t)[self.cel_dim]

        stored_values = self.decomp_fn(hp, non_cel_vec, cel_vec, stored_values)
        
        if self.cel_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        R_prime = self.Sigma[1-self.cel_dim].cho_solve(R_prime, transpose = 0)

        for i in range(self.N_alpha):
            u_R_prime = stored_values["u_H_stack"][:, i].T @ R_prime
            R_prime -= jnp.outer(2*stored_values["u_H_stack"][:, i], u_R_prime)
    
            if i == 0:
                kernel = Multiband(
                    kernel=self.K_list[i][self.cel_dim].kf,
                    amplitudes=stored_values["J_dot"][:, i],
                )
            else:
                kernel += Multiband(
                    kernel=self.K_list[i][self.cel_dim].kf,
                    amplitudes=stored_values["J_dot"][:, i],
                )

        D_cel = (self.Sigma[self.cel_dim].diag + self.Sigma[self.cel_dim].wn_diag)*jnp.ones(cel_vec.shape[-1])
        D_cel_inv_sqrt = 1/jnp.sqrt(D_cel)
        diag_part = R_prime[self.N_alpha:, :] * D_cel_inv_sqrt

        x_t_long = jnp.kron(cel_vec, jnp.ones(self.N_alpha))
        x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(self.N_alpha))
        X = (x_t_long, x_l_long)
        
        cel_logL = faster_cel_GP(kernel, X, R_prime[:self.N_alpha, :].ravel("F"), diag=jnp.kron(D_cel, jnp.ones(self.N_alpha)))

        logL = cel_logL - 0.5 * (diag_part**2).sum() - 0.5 * stored_values["logdetK"] - 0.5 * (non_cel_vec.shape[-1] - self.N_alpha)*cel_vec.shape[-1] * jnp.log(2*jnp.pi)
        logL += -0.5 * (non_cel_vec.shape[-1] - self.N_alpha) * jnp.log(D_cel).sum()
    
        return logL, stored_values
        


    # This is terribly coded but a start
    def fast_LA(self, D_tensor, b_mat, stored_values):
        
        N_b = D_tensor.shape[0]
        N_l = D_tensor.shape[1]
        N_t = b_mat.shape[1]

        # Calc celerite block inverse times each time vector
        b_vecs = jnp.kron(b_mat, jnp.ones((N_l, 1, 1)))
        L_inv_b = calc_Linv_mat_vmap(*stored_values["cel_decomp"], b_vecs)

        # Multiply each pair of time vectors and sum over time dimension
        # Computes diagonal entries of diagonal matrices D_ij
        D_ij = L_inv_b[None, :, :, :] * L_inv_b[:, None, :, :]
        D_ij = D_ij.sum(3)
        
        # JAX/NumPy treats tensor matrix multiplication as if it is a stack of matrices stored in last two dimensions
        # Which works here as D_tensor is of shape (N_b, N_l, N_l)
        W_l_D = stored_values["W_l"].T @ D_tensor
        
        # Dj_W_l_D shape will initially be (N_b, N_b, N_l, N_l)
        Dj_W_l_D = D_ij[:, :, :, jnp.newaxis] * W_l_D[:, :, :]

        # We then sum over axis 1, summing over j
        Dj_W_l_D = Dj_W_l_D.sum(1)

        # Must be careful to transpose W_l_D just along the two wavelength axes
        # Broadcasting will result in N_b stacks of matrices multiplying together
        TKT = jnp.transpose(W_l_D, (0, 2, 1)) @ Dj_W_l_D

        # We then do our second sum over axis 0, summing over i
        TKT = TKT.sum(0)

        # Finally we invert T.T K^-1 T to get our covariance matrix
        Sigma_d = jnp.linalg.inv(TKT)
        logdetSigma_d = -jnp.linalg.slogdet(TKT)[1]
    
        return Sigma_d, logdetSigma_d


    
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
        
        raise Exception("Not yet implemented!")


    
    def solve(
        self,
        hp: PyTree,
        x_l: JAXArray,
        x_t: JAXArray,
        R: JAXArray,
        stored_values,
    ) -> JAXArray:
        r"""Calculates the product of the inverse of the covariance matrix with a vector, represented by
        a JAXArray of shape ``(N_l, N_t)``. Useful for testing for numerical stability.
        
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
            R (JAXArray): JAXArray of shape ``(N_l, N_t)`` representing the vector to multiply on the right by
                the inverse of the covariance matrix ``K``.
                
        Returns:
            JAXArray: The result of multiplying the inverse of the covariance matrix ``K`` by the vector ``R``.
        
        """
        
        raise Exception("Not implemented!")
    
    
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
            hp (Pytree): Hyperparameters needed to build the covariance matrices
                ``Kl``, ``Kt``, ``Sl``, ``St``. Will be unaffected if additional mean function
                parameters are also included.
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
        
        return Kr

