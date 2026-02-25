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

from .covtype import Outer
from .luas_types import Kernel, PyTree, JAXArray, Scalar
from .kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable, cyclic_transpose, calc_total_size
from .jax_convenience_fns import array_to_pytree_2D, get_corr_mat


__all__ = [
    "KinvR_block",
    "logL_block",
    "LuasLasrachKernel",
]

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


def KinvR_block(Kt_cel, Kt_diag, St_cel, St_diag, x_t, lam_Sl_tilde, R_prime):
    
    if Kt_cel is not None and St_cel is None:
        gp = tinygp.GaussianProcess(Kt_cel, x_t, diag = Kt_diag + lam_Sl_tilde*St_diag)
    elif Kt_cel is None and St_cel is not None:
        gp = tinygp.GaussianProcess(lam_Sl_tilde*St_cel, x_t, diag = Kt_diag + lam_Sl_tilde*St_diag)
    elif Kt_cel is not None and St_cel is not None:
        gp = tinygp.GaussianProcess(Kt_cel + lam_Sl_tilde*St_cel, x_t, diag = Kt_diag + lam_Sl_tilde*St_diag)

    R1 = gp.solver.solve_triangular(R_prime, transpose = 0)
    R2 = gp.solver.solve_triangular(R1, transpose = 1)
    return R2


# def logL_block(Kt_cel, Kt_diag, St_cel, St_diag, x_t, lam_Sl_tilde, R_prime):
    
#     if Kt_cel is not None and St_cel is None:
#         gp = tinygp.GaussianProcess(Kt_cel, x_t, diag = Kt_diag + lam_Sl_tilde*St_diag)
#     elif Kt_cel is None and St_cel is not None:
#         gp = tinygp.GaussianProcess(lam_Sl_tilde*St_cel, x_t, diag = Kt_diag + lam_Sl_tilde*St_diag)
#     elif Kt_cel is not None and St_cel is not None:
#         gp = tinygp.GaussianProcess(Kt_cel + lam_Sl_tilde*St_cel, x_t, diag = Kt_diag + lam_Sl_tilde*St_diag)
        
#     return gp.log_probability(R_prime)

KinvR_vmap = jax.vmap(KinvR_block, in_axes = (None, None, None, None, None, 0, 0))
# logL_vmap = jax.vmap(logL_block, in_axes = (None, None, None, None, None, 0, 0))

def gen_cho_solve(K, S, lam_K_tilde, r, hp, x, transpose = 0):
    K_block = lam_K_tilde * K
    K_block += S
    K_block.cholesky_decomp(hp, x, x)
    return K_block.cho_solve(r, transpose = transpose), K_block.logdet
general_cho_solve_vmap = jax.vmap(gen_cho_solve, in_axes = (None, None, 0, 0, None, None))
general_cho_solve_transpose_vmap = jax.vmap(lambda *args: gen_cho_solve(*args, transpose = 1), in_axes = (None, None, 0, 0, None, None))


def gen_cho_mult(K, S, lam_K_tilde, r, hp, x, transpose = 0):
    K_block = lam_K_tilde * K
    K_block += S
    K_block.cholesky_decomp(hp, x, x)
    return K_block.cho_mult(r, transpose = transpose)
general_cho_mult_vmap = jax.vmap(gen_cho_mult, in_axes = (None, None, 0, 0, None, None))

def block_celerite_decompose(Kt_cel, Kt_diag, St_cel, St_diag, x_t, lam_Sl_tilde):
    
    if Kt_cel is not None and St_cel is None:
        gp = tinygp.GaussianProcess(lam_Sl_tilde*Kt_cel, x_t, diag = lam_Sl_tilde*Kt_diag + St_diag)
    elif Kt_cel is None and St_cel is not None:
        gp = tinygp.GaussianProcess(St_cel, x_t, diag = lam_Sl_tilde*Kt_diag + St_diag)
    elif Kt_cel is not None and St_cel is not None:
        gp = tinygp.GaussianProcess(lam_Sl_tilde*Kt_cel + St_cel, x_t, diag = lam_Sl_tilde*Kt_diag + St_diag)
    elif Kt_cel is None and St_cel is None:
        raise Exception("Celerite dimension fully diagonal! Definitely a better optimisation option")

    d, p, q, a = gp.solver.factor.diag.d, gp.solver.factor.lower.p, gp.solver.factor.lower.q, gp.solver.factor.lower.a
    
    logdetK = 2*jnp.log(d).sum()
    return (d, p, q, a), logdetK
decomp_vmap = jax.vmap(block_celerite_decompose, in_axes = (None, None, None, None, None, 0))


def calc_Linv_vec(d, p, q, a, R):
    diag_qsm = DiagQSM(d)
    strict_lower_tri = StrictLowerTriQSM(p, q, a)
    solver = LowerTriQSM(diag_qsm, strict_lower_tri)

    # Maybe should be transpose? Need to check if right
    return solver.solve(R)
calc_Linv_vec_vmap = jax.vmap(calc_Linv_vec)
calc_Linv_mat_vmap = jax.vmap(calc_Linv_vec_vmap, in_axes = (None, None, None, None, 2))


def faster_cel_GP(kernel, X, y, diag=None):
    noise_model = tinygp.noise.Diagonal(diag=diag)
    matrix = kernel.to_symm_qsm(X)
    matrix += noise_model.to_qsm()
    factor = matrix.cholesky()
    
    return - 0.5 * (factor.solve(y)**2).sum() - jnp.sum(jnp.log(factor.diag.d)) - 0.5 * factor.shape[0] * jnp.log(2 * jnp.pi)



def KinvR_block(Kt_cel, Kt_diag, St_cel, St_diag, x_t, lam_Sl_tilde, R_prime):
    
    if Kt_cel is not None and St_cel is None:
        gp = tinygp.GaussianProcess(lam_Sl_tilde*Kt_cel, x_t, diag = lam_Sl_tilde*Kt_diag + St_diag)
    elif Kt_cel is None and St_cel is not None:
        gp = tinygp.GaussianProcess(St_cel, x_t, diag = lam_Sl_tilde*Kt_diag + St_diag)
    elif Kt_cel is not None and St_cel is not None:
        gp = tinygp.GaussianProcess(lam_Sl_tilde*Kt_cel + St_cel, x_t, diag = lam_Sl_tilde*Kt_diag + St_diag)

    R1 = gp.solver.solve_triangular(R_prime, transpose = 0)
    R2 = gp.solver.solve_triangular(R1, transpose = 1)
    return R2

def logL_block(Kt_cel, Kt_diag, St_cel, St_diag, x_t, lam_Sl_tilde, R_prime):
    
    if Kt_cel is not None and St_cel is None:
        logL = faster_cel_GP(Kt_cel, x_t, R_prime, diag = Kt_diag + lam_Sl_tilde*St_diag)
    elif Kt_cel is None and St_cel is not None:
        logL = faster_cel_GP(lam_Sl_tilde*St_cel, x_t, R_prime, diag = Kt_diag + lam_Sl_tilde*St_diag)
    elif Kt_cel is not None and St_cel is not None:
        logL = faster_cel_GP(Kt_cel + lam_Sl_tilde*St_cel, x_t, R_prime, diag = Kt_diag + lam_Sl_tilde*St_diag)
    else:
        raise Exception("Fully diagonal cel dim!")
        
    return logL

logL_vmap = jax.vmap(logL_block, in_axes = (None, None, None, None, None, 0, 0))


# Example usage
#stored_values["cel"] = decomp_vmap(time_kernel, np.zeros(N_t), None, St_diag, x_t, lam_Kl_tilde)
#L_inv_R = calc_Linv_vec_vmap(*stored_values["cel"], R)

def rotate_tuple(t, d):
    
    return t[d:] + t[:d]

def valid(*K_list):
    if len(K_list) > 2:
        return False
    else:
        return True
    
class LuasLasrachKernel(Kernel):
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
        eigen_dim = 0,
        use_stored_values: Optional[bool] = True,
    ):
        
        self.K_list = [K] # defined for consistency with other kernel objects

        # Ensure eigendecomposition dimension is first index
        self.Sigma = rotate_tuple(Sigma, eigen_dim)
        self.K = rotate_tuple(K, eigen_dim)
        self.eigen_dim = eigen_dim
        
        self.logL_hessianable = self.logL
        self.decompose = self.decomp_no_stored_values
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        # if use_stored_values:
        #     self.decomp_fn = self.eigendecomp_use_stored_values
        # else:
        #     self.decomp_fn = self.eigendecomp_no_stored_values


    # def decomp_no_stored_values(
    #     self,
    #     *X: JAXArray,
    #     stored_values: Optional[PyTree] = {},
    # ) -> PyTree:

    #     eigen_vec = X[self.eigen_dim]
    #     self.Sigma[self.eigen_dim].decomp(eigen_vec)

    #     # Generate transformed objects, doesn' actually do transformation yet
    #     K_tilde = self.Sigma[self.eigen_dim].decomp_transform(self.K[self.eigen_dim])

    #     # Evaluates transformation and does eigendecomp
    #     stored_values["lam_K_tilde"], stored_values["Q_K_tilde"] = K_tilde.eigendecomp(eigen_vec)
    #     stored_values["eigen_dim_rank"] = K_tilde.rank

    #     # Computes the log determinant of K
    #     stored_values["logdetK"] = self.Sigma[self.eigen_dim].logdet
        
    #     return stored_values

    def eigendecomp_no_stored_values(
        self,
        *X: Tuple,
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

        total_size = calc_total_size(X)
        gp_dim = len(X)

        sigma_decomp_mats = ()
        eigen_decomp_mats = ()
        all_lam = jnp.ones(1)
        all_lam_shape = ()
        stored_values["logdetK"] = 0.

        for d in jnp.arange(gp_dim):
            if d != self.eigen_dim:
                Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d])
                K_d_new = self.Sigma[d].inv_sqrt_transform(self.K[d])
                stored_values[f"lam_{d}"], stored_values[f"Q_{d}"] = K_d_new.eigendecomp(X[d])
                
                all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), stored_values[f"lam_{d}"])
                all_lam_shape = all_lam.shape
                
                sigma_decomp_mats += (Sigma_d_new,)
                eigen_decomp_mats += (stored_values[f"Q_{d}"],)
                stored_values["logdetK"] += (total_size/X[d].shape[-1])*stored_values_d["logdetK"]
            if d == self.eigen_dim:
                sigma_decomp_mats += (luas.covtype.Identity(),)
                eigen_decomp_mats += (luas.covtype.Identity(),)
                

        def transform_fn(R, transpose = 0):
            R_prime = cyclic_transpose(R, 2)

            if transpose:
                for d in jnp.arange(gp_dim):
                    R_prime = eigen_decomp_mats[d] @ R_prime
                    R_prime = sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 0)
                    R_prime = cyclic_transpose(R_prime, 1)
            else:
                for d in jnp.arange(gp_dim):
                    R_prime = sigma_decomp_mats[d].matrix_sqrt(R_prime, transpose = 1)
                    R_prime = eigen_decomp_mats[d].T @ R_prime
                    R_prime = cyclic_transpose(R_prime, 1)

            R_prime = cyclic_transpose(R_prime, -2)
                
            return R_prime

        def inv_transform_fn(R, transpose = 0):
            R_prime = cyclic_transpose(R, 2)

            if transpose:
                for d in jnp.arange(gp_dim):
                    R_prime = eigen_decomp_mats[d] @ R_prime
                    R_prime = sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 1)
                    R_prime = cyclic_transpose(R_prime, 1)
            else:
                for d in jnp.arange(gp_dim):
                    R_prime = sigma_decomp_mats[d].matrix_inv_sqrt(R_prime, transpose = 0)
                    R_prime = eigen_decomp_mats[d].T @ R_prime
                    R_prime = cyclic_transpose(R_prime, 1)

            R_prime = cyclic_transpose(R_prime, -2)
            return R_prime
        
        K_tilde_diag = all_lam
        self.kf_tilde = WhiteNoiseKernel(diag = K_tilde_diag)
        self.transform_fn = transform_fn
        self.inv_transform_fn = inv_transform_fn

        return self, stored_values
            
    def logL(self, *X, R, stored_values):

        eigen_vec, *noneigen_vec = rotate_tuple(X, eigen_dim)
        noneigen_size = calc_total_size(non_eigen_vec)

        stored_values = self.decompose(*X, stored_values = stored_values)
        eigen_dim_rank = stored_values["eigen_dim_rank"]
        
        R_prime = cyclic_transpose(R, self.eigen_dim)
        R_prime = self.Sigma[0].matrix_inv_sqrt(R_prime, transpose = 0)
        R_prime = stored_values["Q_K_tilde"].T @ R_prime
        
        # if rank is not specified it is assumed to be full rank and this separate calculation is not needed
        if eigen_dim_rank is not None:
            R_corr = R_prime[-eigen_dim_rank:]
            R_wn = R_prime[:-eigen_dim_rank]

            nullspace_Sigma = (luas.covtype.Identity(), self.Sigma[1:])
            build_kf_oneterm = lambda hp, x_l, x_t: nullspace_Sigma
            
            kf_singleterm = SingleTermKernel({}, eigen_vec[:-eigen_dim_rank], *noneigen_vec, kf = build_kf_oneterm)
            logL_nullspace = kf_singleterm.logL(R_wn)
            eigenvals = stored_values["lam_K_tilde"][-eigen_dim_rank:]
        else:
            R_corr = R_prime.copy()
            logL_nullspace = 0.
            eigenvals = stored_values["lam_K_tilde"].copy()

        R_corr = cyclic_transpose(R_corr, -self.eigen_dim)
        
        if isinstance(noneigen_vec, tuple):
            K_lower_dim = self.K[1-self.eigen_dim]
            Sigma_lower_dim = self.Sigma[1-self.eigen_dim]
            
            def build_kf_lowerdim(hp, *noneigen_vec):
                K_lower_dim[0] = hp["lam"] * K_lower_dim[0]
                return Sigma_lower_dim, K_lower_dim

            def logL_lowerdim(lam, R_corr):
                gp_lowerdim = GP({"lam":lam}, *noneigen_vec, kf = build_kf_lowerdim)
                return gp_lowerdim.logL(*noneigen_vec, R_corr)
        else:
            def logL_lowerdim(lam, R_corr):
                K_noneigen = self.Sigma[1-self.eigen_dim] + lam * self.K[1-self.eigen_dim]
                return K_noneigen.logL(noneigen_vec, R_corr)
        
        logL_vmap = jax.vmap(logL_lowerdim, in_axes = (0, self.eigen_dim))
        logL_lowerdim = logL_vmap(eigenvals, R_corr)
        
        return logL_lowerdim + logL_nullspace - 0.5 * noneigen_size * stored_values["logdetK"]

    
    def solve(self, hp, x_l, x_t, R, stored_values):
        non_cel_vec = (x_l, x_t)[1-self.cel_dim]
        cel_vec = (x_l, x_t)[self.cel_dim]

        stored_values = self.decomp_fn(hp, non_cel_vec, cel_vec, stored_values)
        
        if self.cel_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()
        
        R_prime = self.Sigma[1-self.cel_dim].cho_solve(R_prime, transpose = 0)
        R_prime = stored_values["Q_K_tilde"].T @ R_prime

        if self.Sigma[self.cel_dim].kf is None and type(self.K[1-self.cel_dim]) == Outer:
            # Single celerite block
            
            non_cel_rank = self.K[1-self.cel_dim].rank

            B_inv_R, logdetK_block = general_cho_solve_vmap(self.K[self.cel_dim], self.Sigma[self.cel_dim], stored_values["lam_K_tilde"][-non_cel_rank:],
                                   R_prime[-non_cel_rank:, :], hp, cel_vec)
            B_inv_R, _ = general_cho_solve_transpose_vmap(self.K[self.cel_dim], self.Sigma[self.cel_dim], stored_values["lam_K_tilde"][-non_cel_rank:],
                                   B_inv_R, hp, cel_vec)

            S_cel_diag = (self.Sigma[self.cel_dim].diag + self.Sigma[self.cel_dim].wn_diag)*jnp.ones(cel_vec.shape[-1])

            Dt_inv = 1/S_cel_diag
            D_inv_R = R_prime[:-non_cel_rank, :] * Dt_inv
            
            L_inv_R = jnp.zeros_like(R_prime)
            L_inv_R = L_inv_R.at[-non_cel_rank:, :].set(B_inv_R)
            L_inv_R = L_inv_R.at[:-non_cel_rank, :].set(D_inv_R)
            
        else:
            L_inv_R, logdetK_block = general_cho_solve_vmap(self.K[self.cel_dim], self.Sigma[self.cel_dim], stored_values["lam_K_tilde"],
                                               R_prime, hp, cel_vec)
            L_inv_R, _ = general_cho_solve_transpose_vmap(self.K[self.cel_dim], self.Sigma[self.cel_dim], stored_values["lam_K_tilde"],
                                               L_inv_R, hp, cel_vec)

        K_inv_R = stored_values["Q_K_tilde"] @ L_inv_R
        K_inv_R = self.Sigma[1-self.cel_dim].cho_solve(K_inv_R, transpose = 1)

        if self.cel_dim == 0:
            K_inv_R = K_inv_R.T
            
        return K_inv_R, stored_values

    
    
    # Take a draw from K_inv
    def K_inv_draw(self, hp, x_l, x_t, stored_values, z = None):
        non_cel_vec = (x_l, x_t)[1-self.cel_dim]
        cel_vec = (x_l, x_t)[self.cel_dim]

        if z is not None:
            Z = z.copy()
            assert Z.shape == (x_l.shape[-1], x_t.shape[-1])
        else:
            Z = np.random.normal(size = (x_l.shape[-1], x_t.shape[-1]))
        
        stored_values = self.decomp_fn(hp, non_cel_vec, cel_vec, stored_values)
        
        if self.cel_dim == 0:
            Z = Z.T

        L_inv_Z, _ = general_cho_solve_transpose_vmap(self.K[self.cel_dim], self.Sigma[self.cel_dim],
                                                      stored_values["lam_K_tilde"],
                                                      Z, hp, cel_vec)

        K_inv_draw = stored_values["Q_K_tilde"] @ L_inv_Z
        K_inv_draw = self.Sigma[1-self.cel_dim].cho_solve(K_inv_draw, transpose = 1)

        if self.cel_dim == 0:
            K_inv_draw = K_inv_draw.T
    
        return K_inv_draw, stored_values

    
    
    
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
        z = None,
        wn: Optional[bool] = True,
        stored_values = {},
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

        non_cel_vec = (x_l, x_t)[1-self.cel_dim]
        cel_vec = (x_l, x_t)[self.cel_dim]

        if z is not None:
            Z = z.reshape((x_l.shape[-1], x_t.shape[-1]))
        else:
            Z = np.random.normal(size = (x_l.shape[-1], x_t.shape[-1]))
        
        stored_values = self.decomp_fn(hp, non_cel_vec, cel_vec, stored_values)
        
        if self.cel_dim == 0:
            Z = Z.T

        L_Z = general_cho_mult_vmap(self.K[self.cel_dim], self.Sigma[self.cel_dim],
                                                      stored_values["lam_K_tilde"],
                                                      Z, hp, cel_vec)
        
        K_draw = stored_values["Q_K_tilde"] @ L_Z
        K_draw = self.Sigma[1-self.cel_dim].cho_mult(K_draw, transpose = 0)

        if self.cel_dim == 0:
            K_draw = K_draw.T
    
        return K_draw

    
    
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

        R_prime = self.K[0].left_mult(R, hp, x_l, x_l, **kwargs)
        Kr += self.K[1].left_mult(R_prime.T, hp, x_t, x_t, **kwargs).T
        
        return Kr
