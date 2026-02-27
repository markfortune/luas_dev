import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional
import tinygp

from luas import WhiteNoiseKernel, SingleKronTermKernel
from luas.kernels.covtype import Outer, Exp, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable, tensor_mult
from luas.kernels.tinygp_ext import Multiband

__all__ = [
    "KinvR_block",
    "logL_block",
    "LuasLasrachKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)



class GeneralQuasisep2D(CovType):
    def __init__(self, tinygp_kf, diag = 0., wn_diag = 0., params = None, cel_dim=1):
        self.tinygp_kf = tinygp_kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.cel_dim = cel_dim
        self.params = params

    def tinygp_coords(self, X):
        cel_vec = X[self.cel_dim]
        non_cel_vec = X[1-self.cel_dim]
        
        x_t_long = jnp.kron(cel_vec, jnp.ones(non_cel_vec.shape[-1]))
        x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(non_cel_vec.shape[-1]))
        
        return (x_t_long, x_l_long)

    def calc_diag(self, X, wn = True):
        
        return self.diag + wn*self.wn_diag

    def evaluate(self, x1, x2, wn = True, **kwargs):

        X1 = self.tinygp_coords(x1)
        X2 = self.tinygp_coords(x2)
        
        return self.tinygp_kf(X1, X2) + jnp.diag(self.calc_diag(x1, wn = wn))
    
    def decompose(self, x, wn = True, **kwargs):

        X = self.tinygp_coords(x)

        diag = self.calc_diag(x, wn = wn)

        noise_model = tinygp.noise.Diagonal(diag=diag)
        matrix = self.tinygp_kf.to_symm_qsm(X)
        matrix += noise_model.to_qsm()
        self.factor = matrix.cholesky()
        
        logdetK = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdetK":logdetK}
        
    def matrix_inv_sqrt(self, R, transpose=0):
        
        if self.cel_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        R_shape = R_prime.shape
        r = R_prime.ravel("F")
        
        if transpose:
            r = self.factor.transpose().solve(r)
        else:
            r = self.factor.solve(r)

        R_prime = r.reshape(R_shape, order="F")

        if self.cel_dim == 0:
            R_prime = R_prime.T
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if self.cel_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()
        
        R_shape = R_prime.shape
        r = R_prime.ravel("F")

        if transpose:
            L_r = self.factor.transpose() @ r
        else:
            L_r = self.factor @ r

        L_R = L_r.reshape(R_shape, order="F")

        if self.cel_dim == 0:
            L_R = L_R.T
            
        return L_R

    def scale(self, c):
        return GeneralQuasisep2D(self.tinygp_kf * c, diag = self.diag * c, wn_diag = self.wn_diag * c)

    def matmul(self, x1, x2, R, wn = True, **kwargs):

        X1 = self.tinygp_coords(x1)
        X2 = self.tinygp_coords(x2)
        
        if self.cel_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()
        
        R_shape = R_prime.shape
        r = R_prime.ravel("F")

        r_prime = self.tinygp_kf.matmul(X1, X2, r)

        diag = self.calc_diag(x1, wn = wn)
        noise_model = tinygp.noise.Diagonal(diag=diag)
        r_prime += noise_model @ r

        R_prime = r_prime.reshape(R_shape, order="F")

        if self.cel_dim == 0:
            R_prime = R_prime.T
            
        return R_prime
        
    def __add__(self, K):

        if isinstance(K, GeneralQuasisep2D):
            K_sum = GeneralQuasisep2D(K.tinygp_kf + self.tinygp_kf,
                                    diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

class Block2x2Kernel(CovType):
    def __init__(self, kf_A, kf_B = None, kf_D_CAB = None, kf_D = None, dim_split = 0, split_loc = 1):
        self.kf_A = kf_A
        self.kf_B_matmul = kf_B
        self.kf_D_CAB = kf_D_CAB
        self.kf_D = kf_D

        self.dim_split = dim_split
        self.split_loc = split_loc

        if self.kf_D is None and self.kf_B_matmul is None:
            self.kf_D = self.kf_D_CAB

    def x_split(self, x):
        x_A = x[self.dim_split][..., :self.split_loc]
        x_D = x[self.dim_split][..., self.split_loc:]

        if self.dim_split == 0:
            X_A = (x_A, x[1])
            X_D = (x_D, x[1])
        else:
            X_A = (x[0], x_A)
            X_D = (x[0], x_D)
            
        return X_A, X_D

    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        if self.kf_B_matmul is None:
            zeros_fn = lambda x1, x2, R, **kwargs: jnp.zeros(x1.shape[-1], x2.shape[-1])
            self.kf_B_matmul = zeros_fn

        A_mat = self.kf_A.evaluate(x1, x2)
        B_mat = self.kf_B_matmul(x1, x2, jnp.eye(x2.shape[-1]))

        if self.kf_D is None:
            C_A_inv_B = B_mat.T @ jnp.linalg.inv(A_mat) @ B_mat
            D_mat = self.kf_D_CAB.evaluate(x1, x2) - C_A_inv_B
        else:
            D_mat = self.kf_D.evaluate(x1, x2)

        if self.dim_split == 0:
            top_rows = jnp.concatenate([A_mat, B_mat], axis = 1)
            bottom_rows = jnp.concatenate([B_mat.T, D_mat], axis = 1)
        else:
            raise Exception("Not implemented")
        
        return jnp.concatenate([top_rows, bottom_rows], axis = 0)
    
    def decompose(self, X, **kwargs):

        X_A, X_D = self.x_split(X)

        self.kf_A, stored_values_A = self.kf_A.decompose(X_A, **kwargs)

        if self.kf_D_CAB is not None:
            self.kf_D_CAB, stored_values_D = self.kf_D_CAB.decompose(X_D, stored_values = stored_values_A)
        else:
            raise Exception("Not Implemented")

        if self.kf_B_matmul is not None:
            self.B_mult = lambda R, **kwargs: self.kf_B_matmul(X_A, X_D, R, **kwargs)
            self.C_mult = lambda R, **kwargs: self.kf_B_matmul(X_D, X_A, R, **kwargs)
        
        return self, {"logdetK":stored_values_A["logdetK"] + stored_values_D["logdetK"]}
        
    def matrix_inv_sqrt(self, R, transpose=0):

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]
        
        R_prime_A = self.kf_A.matrix_inv_sqrt(R_A, transpose = transpose)
        R_prime_D = self.kf_D_CAB.matrix_inv_sqrt(R_D, transpose = transpose)

        if self.kf_B_matmul is not None:
            if transpose == 0:
                R_C = self.kf_A.matrix_inv_sqrt(R_prime_A, transpose = 1)
                R_C = self.C_mult(R_C, wn = False)
                R_prime_D += -self.kf_D_CAB.matrix_inv_sqrt(R_C, transpose = 0)
            else:
                R_B = self.B_mult(R_prime_D, wn = False)
                R_prime_A += -self.kf_A.inverse(R_B)

        R_prime = jnp.concatenate([R_prime_A, R_prime_D], axis = self.dim_split)
            
        return R_prime
        
    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]
        
        R_prime_A = self.kf_A.matrix_sqrt(R_A, transpose = transpose)
        R_prime_D = self.kf_D_CAB.matrix_sqrt(R_D, transpose = transpose)

        if self.kf_B_matmul is not None:
            if transpose == 0:
                R_C = self.kf_A.matrix_inv_sqrt(R_A, transpose = 1)
                R_prime_D += self.C_mult(R_C, wn = False)
            else:
                R_B = self.B_mult(R_D, wn = False)
                R_prime_A += self.kf_A.matrix_inv_sqrt(R_B, transpose = 0)
        
        R_prime = jnp.concatenate([R_prime_A, R_prime_D], axis = self.dim_split)
        
        return R_prime

    def scale(self, c):
        raise Exception("Not implemented")

    def matmul(self, x1, x2, R, wn = True, **kwargs):

        if self.kf_D is None and self.kf_B_matmul is not None:
            raise Exception("Need to specify kernel function in block D for this method")

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]

        X1_A, X1_D = self.x_split(X1)
        X2_A, X2_D = self.x_split(X2)

        if self.kf_B_matmul is None:
            K_R_A = self.kf_A.matmul(X1_A, X2_A, R_A)
            K_R_D = self.kf_D.matmul(X1_D, X2_D, R_D)
        else:
            K_R_A = self.kf_A.matmul(X1_A, X2_A, R_A) + self.kf_B_matmul(X1_A, X2_B, R_D)
            K_R_D = self.kf_B_matmul(X1_B, X2_A, R_A) + self.kf_D.matmul(X1_D, X2_D, R_D)

        K_R = jnp.concatenate([K_R_A, K_R_D], axis = self.dim_split)
        
        return K_R

    
class MultiTermKernel(CovType):
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
        never_reduce_dim = False,
        use_stored_values: Optional[bool] = True,
    ):
        
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.cel_dim = cel_dim
        self.never_reduce_dim = never_reduce_dim
        self.N_alpha = len(K_list)

        self.logL_hessianable = self.logL
        self.decompose = self.decomp_no_stored_values
           
        # Have different decomposition functions depending on whether previous stored values
        # are to be used to avoid recalculating eigendecompositions
        # if use_stored_values:
        #     self.decomp_fn = self.eigendecomp_use_stored_values
        # else:
        #     self.decomp_fn = self.eigendecomp_no_stored_values

    def _rotate_to_cel_dim_wrapper(self, fn, cel_dim):

        def wrapped_fn(R, **kwargs):
            if cel_dim == 0:
                R_prime = R.T
            else:
                R_prime = R.copy()

            R_prime = fn(R_prime, **kwargs)

            if self.cel_dim == 0:
                R_prime = R_prime.T
                
            return R_prime

        return wrapped_fn



    def decomp_no_stored_values(
        self,
        *X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

        non_cel_vec = X[1-self.cel_dim]
        cel_vec = X[self.cel_dim]

        # Generate transformations
        self.Sigma_transf, stored_values_Sigma_transf = self.Sigma[1-self.cel_dim].decompose(non_cel_vec)

        # Define transformations which diagonalise Sigma matrix
        apply_sigma_matrix_sqrt = self._rotate_to_cel_dim_wrapper(self.Sigma_transf.matrix_sqrt, self.cel_dim)
        apply_sigma_inv_matrix_sqrt = self._rotate_to_cel_dim_wrapper(self.Sigma_transf.matrix_inv_sqrt, self.cel_dim)

        # Computes the log determinant of K
        stored_values["logdetK"] = cel_vec.shape[-1]*stored_values_Sigma_transf["logdetK"]
        
        stored_values["non_cel_rank"] = 0
        for K_i in self.K_list:
            stored_values["non_cel_rank"] += K_i[1-self.cel_dim].rank(non_cel_vec)
        
        stored_values["A"] = jnp.zeros((non_cel_vec.shape[-1], stored_values["non_cel_rank"]))
        stored_values["cel_kernel_order"] = []
        
        col_i = 0
        for (i, K_i) in enumerate(self.K_list):
            if isinstance(K_i[1-self.cel_dim], Outer):
                K_i_non_cel, _ = K_i[1-self.cel_dim].decompose(non_cel_vec)
                stored_values["A"] = stored_values["A"].at[:, col_i].set(K_i_non_cel.alpha)
                stored_values["cel_kernel_order"].append(K_i[self.cel_dim])
                col_i += 1
            else:
                K_i_non_cel, _ = K_i[1-self.cel_dim].decompose(non_cel_vec)
                
                for j in range(non_cel_vec.shape[-1]):
                    stored_values["A"] = stored_values["A"].at[:, col_i].set(K_i_non_cel.factor[:, j])
                    stored_values["cel_kernel_order"].append(K_i[self.cel_dim])
                    col_i += 1
        
        # Transform vectors 
        stored_values["J"] = apply_sigma_inv_matrix_sqrt(stored_values["A"], transpose = 0)

        # Handle Sigma mat in the cel_dim, likely just diagonal
        if type(self.Sigma[self.cel_dim]) in [GeneralQuasisep, Exp]:
            stored_values["J"] = jnp.stack([stored_values["J"], jnp.eye(non_cel_vec.shape[-1])], axis = 1)

            for j in range(non_cel_vec.shape[-1]):
                stored_values["cel_kernel_order"].append(self.Sigma[self.cel_dim])
                
            cel_diag = jnp.zeros(cel_vec.shape[-1])
        else:
            cel_diag = (self.Sigma[self.cel_dim].diag + self.Sigma[self.cel_dim].wn_diag)*jnp.ones(cel_vec.shape[-1])

        # If the total rank is less than the length of that dimension, we can reduce the dimension by exploiting sparsity
        reduce_dim = stored_values["non_cel_rank"] < non_cel_vec.shape[-1] and not self.never_reduce_dim

        if reduce_dim:
            stored_values["J"], householder_transform = orthonormal_nullspace_gen(stored_values["J"])
   
            def transform_fn(R, transpose = 0):
                
                if self.cel_dim == 0:
                    R_prime = R.T
                else:
                    R_prime = R.copy()

                if transpose:
                    R_prime = self.Sigma_transf.matrix_sqrt(R_prime, transpose = 1)
                    R_prime = householder_transform(R_prime, transpose = 0)
                else:
                    R_prime = householder_transform(R_prime, transpose = 1)
                    R_prime = self.Sigma_transf.matrix_sqrt(R_prime, transpose = 0)

                if self.cel_dim == 0:
                    R_prime = R_prime.T
                    
                return R_prime

            def inv_transform_fn(R, transpose = 0):
                
                if self.cel_dim == 0:
                    R_prime = R.T
                else:
                    R_prime = R.copy()

                if transpose:
                    R_prime = householder_transform(R_prime, transpose = 1)
                    R_prime = self.Sigma_transf.matrix_inv_sqrt(R_prime, transpose = 1)
                else:
                    R_prime = self.Sigma_transf.matrix_inv_sqrt(R_prime, transpose = 0)
                    R_prime = householder_transform(R_prime, transpose = 0)

                if self.cel_dim == 0:
                    R_prime = R_prime.T
                    
                return R_prime
                
            self.transform_fn = transform_fn
            self.inv_transform_fn = inv_transform_fn
            
            # x_t_long = jnp.kron(cel_vec, jnp.ones(stored_values["non_cel_rank"]))
            # x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(stored_values["non_cel_rank"]))

            total_cel_diag = jnp.kron(cel_diag, jnp.ones(stored_values["non_cel_rank"]))

        else:

            self.transform_fn = apply_sigma_matrix_sqrt
            self.inv_transform_fn = apply_sigma_inv_matrix_sqrt
            
            # x_t_long = jnp.kron(cel_vec, jnp.ones(non_cel_vec.shape[-1]))
            # x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(non_cel_vec.shape[-1]))

            total_cel_diag = jnp.kron(cel_diag, jnp.ones(non_cel_vec.shape[-1]))
                

        # Build quasiseparable kernel with tinygp
        for i in range(stored_values["non_cel_rank"]):
            # if stored_values["cel_kernel_order"][i].diag or stored_values["cel_kernel_order"][i].wn_diag:
            #     raise Exception("Adding diagonal terms to this celerite term not yet supported with this optimisation")
            
            if i == 0:
                kernel_tinygp = Multiband(
                    kernel=stored_values["cel_kernel_order"][i].tinygp_kf,
                    amplitudes=stored_values["J"][:, i],
                )
            else:
                kernel_tinygp += Multiband(
                    kernel=stored_values["cel_kernel_order"][i].tinygp_kf,
                    amplitudes=stored_values["J"][:, i],
                )
        
        kf_quasi2D = GeneralQuasisep2D(kernel_tinygp, diag = total_cel_diag, cel_dim=self.cel_dim)

        if reduce_dim:
            if isinstance(self.Sigma[self.cel_dim], (Identity, ScaledIdentity, Diagonal)):
                null_space_rank = non_cel_vec.shape[-1] - stored_values["non_cel_rank"]
                if self.cel_dim == 0:
                    rest_cel_diag = jnp.outer(cel_diag, jnp.ones(null_space_rank))
                else:
                    rest_cel_diag = jnp.outer(jnp.ones(null_space_rank), cel_diag)
                    
                kf_D = WhiteNoiseKernel(diag = rest_cel_diag)
            else:
                if self.cel_dim == 0:
                    kf_D = SingleKronTermKernel((self.Sigma[self.cel_dim], Identity()))
                else:
                    kf_D = SingleKronTermKernel((Identity(), self.Sigma[self.cel_dim]))

            print("kf_A", kf_quasi2D, "kf_D_CAB", kf_D, "dim_split", 1-self.cel_dim, "split_loc", stored_values["non_cel_rank"], "cel_dim",self.cel_dim)
            self.kf_tilde = Block2x2Kernel(kf_quasi2D, kf_D_CAB = kf_D, dim_split = 1-self.cel_dim, split_loc = stored_values["non_cel_rank"])
        else:
            self.kf_tilde = kf_quasi2D
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X)

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

        for K in self.K_list:
            K_R += tensor_mult(K, X1, X2, R, **kwargs)
        
        return K_R

    
    # def logL(self, R, stored_values):

    #     # non_cel_vec = (x_l, x_t)[1-self.cel_dim]
    #     # cel_vec = (x_l, x_t)[self.cel_dim]

    #     # stored_values = self.decomp_fn(hp, non_cel_vec, cel_vec, stored_values)
        
    #     if self.cel_dim == 0:
    #         R_prime = R.T
    #     else:
    #         R_prime = R.copy()

    #     R_prime = self.Sigma[1-self.cel_dim].cho_solve(R_prime, transpose = 0)

    #     for i in range(stored_values["non_cel_rank"]):

    #         # print(i, stored_values["cel_kernel_order"])
    #         # cel_kernel = self.total_K_list[stored_values["cel_kernel_order"][i]][self.cel_dim].kf
    #         cel_kernel = stored_values["cel_kernel_order"][i].kf

    #         if stored_values["cel_kernel_order"][i].diag or stored_values["cel_kernel_order"][i].wn_diag:
    #             # If diagonal terms added to these celerite kernels then probably need to add some
    #             # New terms for that too
    #             raise Exception("Adding diagonal terms to this celerite term not yet supported with this optimisation")
            
    #         if i == 0:
    #             kernel = Multiband(
    #                 kernel=cel_kernel,
    #                 amplitudes=stored_values["J"][:, i],
    #             )
    #         else:
    #             kernel += Multiband(
    #                 kernel=cel_kernel,
    #                 amplitudes=stored_values["J"][:, i],
    #             )

    #     D_cel = (self.Sigma[self.cel_dim].diag + self.Sigma[self.cel_dim].wn_diag)*jnp.ones(cel_vec.shape[-1])

    #     x_t_long = jnp.kron(cel_vec, jnp.ones(non_cel_vec.shape[-1]))
    #     x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(non_cel_vec.shape[-1]))
    #     X = (x_t_long, x_l_long)
        
    #     cel_logL = faster_cel_GP(kernel, X, R_prime.ravel("F"),
    #                              diag=jnp.kron(D_cel, jnp.ones(non_cel_vec.shape[-1])))

    #     logL = cel_logL - 0.5 * stored_values["logdetK"]
    
    #     return logL, stored_values
        


    # def matmul(
    #     self,
    #     X1: Tuple[JAXArray],
    #     X2: Tuple[JAXArray],
    #     R: JAXArray,
    #     **kwargs,
    # ) -> JAXArray:
    #     r"""Calculates the product of the covariance matrix with a vector, represented by a JAXArray of shape ``(N_l, N_t)`.
    #     Useful for testing for numerical stability.
        
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
    #         R (JAXArray): JAXArray of shape ``(N_l, N_t)`` representing the vector to multiply on the right by
    #             the covariance matrix ``K``.
                
    #     Returns:
    #         JAXArray: The result of multiplying the covariance matrix ``K`` by the vector ``R``.
        
    #     """
        
    #     R_prime = self.Sigma[0].matmul(X1, x2, **kwargs)
    #     Kr = self.Sigma[1].left_mult(R_prime.T, hp, x_t, x_t, **kwargs).T

    #     for i in range(len(self.K_list)):
    #         R_prime = self.K_list[i][0].left_mult(R, hp, x_l, x_l, **kwargs)
    #         Kr += self.K_list[i][1].left_mult(R_prime.T, hp, x_t, x_t, **kwargs).T
        
    #     return Kr


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

    def householder_transform(R, transpose = 0):
        R_prime = R.copy()

        if transpose:
            for i in range(N_alpha):
                u_R_prime = U[:, -i-1].T @ R_prime
                R_prime -= jnp.outer(2*U[:, -i-1], u_R_prime)
        else:
            for i in range(N_alpha):
                u_R_prime = U[:, i].T @ R_prime
                R_prime -= jnp.outer(2*U[:, i], u_R_prime)

        return R_prime
    
    return Lam, householder_transform

