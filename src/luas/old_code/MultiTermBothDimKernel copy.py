import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional
import tinygp

from luas import WhiteNoiseKernel, SingleKronTermKernel
from luas.kernels.covtype import Outer, Exp, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import kron_prod, logdetK_calc, r_K_inv_r, K_inv_vec, logdetK_calc_hessianable, tensor_mult
from luas.kernels.tinygp_ext import ScaledKernel, LowRankProduct
from luas.kernels.BlockKernel import Block2x2Kernel
from luas.kernels.MixingMatQuasisep import MixingMatQuasisep, orthonormal_nullspace_gen
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM

__all__ = [
    "KinvR_block",
    "logL_block",
    "LuasLasrachKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)


# class MixingMatQuasisep(CovType):
#     def __init__(self, mixing_mat, kernel_list, P_list = [], Q_list = [], add_QSM = None,
#                  diag = 0., wn_diag = 0., params = None, cel_dim = 1, use_block = True):
#         self.mixing_mat = mixing_mat # shape [N, N_alpha]
#         self.kernel_list = kernel_list # N_alpha list of tinygp kernel functions
#         self.P_list = P_list
#         self.Q_list = Q_list
#         self.diag = diag
#         self.wn_diag = wn_diag
#         self.rank = mixing_mat.shape[1]
#         self.add_QSM = add_QSM
#         self.params = params
#         self.cel_dim = cel_dim
#         self.use_block = use_block

#     def _evaluate_at_ind(self, X, row_idx, col_idx, flip_kron = False):

#         for i in range(self.rank):
#             K_i_eval = self.kernel_list[i](X[self.cel_dim][row_idx[self.cel_dim]],
#                                            X[self.cel_dim][col_idx[self.cel_dim]],
#                                            row_idx = row_idx, col_idx = col_idx, full = False)
#             # Need to add diagonal term if included!
#             # K_i_eval += (self.kernel_list[i].diag + self.kernel_list[i].wn_diag)

#             if self.P_list:
#                 K_i_eval += self.P_list[i][row_idx[self.cel_dim]] @ self.Q_list[i][col_idx[self.cel_dim]].T

#             mixing_eval = self.mixing_mat[row_idx[1-self.cel_dim], i:i+1] @ self.mixing_mat[col_idx[1-self.cel_dim], i:i+1].T

#             if flip_kron:
#                 if self.cel_dim == 0:
#                     K_kron = jnp.kron(mixing_eval, K_i_eval)
#                 else:
#                     K_kron = jnp.kron(K_i_eval, mixing_eval)
#             else:
#                 if self.cel_dim == 0:
#                     K_kron = jnp.kron(K_i_eval, mixing_eval)
#                 else:
#                     K_kron = jnp.kron(mixing_eval, K_i_eval)

#             if i == 0:
#                 K_eval = K_kron.copy()
#             else:
#                 K_eval += K_kron

#         if self.add_QSM is not None:
#             raise Exception("Not implemented")

#         return K_eval

#     def _tinygp_coords(self, X):
#         cel_vec = X[self.cel_dim]
#         non_cel_vec = X[1-self.cel_dim]
        
#         x_t_long = jnp.kron(cel_vec, jnp.ones(non_cel_vec.shape[-1]))
#         x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(non_cel_vec.shape[-1]))
        
#         return (x_t_long, x_l_long)

#     def _to_symm_qsm(self, X, wn = True):
        
#         for i in range(self.rank):
#             if i == 0:
#                 self.kf_2D = ScaledKernel(
#                     kernel=self.kernel_list[0].tinygp_kf,
#                     amplitudes=self.mixing_mat[:, 0],
#                 )
#             else:
#                 new_term = ScaledKernel(
#                     kernel=self.kernel_list[i].tinygp_kf,
#                     amplitudes=self.mixing_mat[:, i],
#                 )
#                 self.kf_2D = tinygp.kernels.quasisep.Sum(self.kf_2D, new_term, use_block = self.use_block)

#         tiny_coords = self._tinygp_coords(X)
#         quasisep_cov = self.kf_2D.to_symm_qsm(tiny_coords)

#         diag = (self.diag + wn*self.wn_diag)*jnp.ones(tiny_coords[0].shape[-1])
#         quasisep_cov += tinygp.noise.Diagonal(diag).to_qsm()
        
#         for i in range(len(self.P_list)):
#             P_long = jnp.kron(self.P_list[i], self.mixing_mat[:, i:i+1])
#             Q_long = jnp.kron(self.Q_list[i], self.mixing_mat[:, i:i+1])

#             quasisep_cov += LowRankProduct(P_long, Q_long).to_qsm()

#         if self.add_QSM is not None:
#             quasisep_cov += self.add_QSM
    
#         return quasisep_cov

#     def _flatten_R(self, R):
                
#         if self.cel_dim == 0:
#             return R.ravel("C")
#         else:
#             return R.ravel("F")

#     def _reshape_R(self, r, R_shape):
        
#         if self.cel_dim == 0:
#             return r.reshape(R_shape, order='C')
#         else:
#             return r.reshape(R_shape, order='F')

#     def _householder_transf(self, X, u):

#         new_P_list = []
#         new_Q_list = []
        
#         for i in range(self.rank):
#             w_i = self.kernel_list[i].matmul(X[self.cel_dim], X[self.cel_dim], u, full = True)

#             if self.P_list:
#                 w_i += LowRankProduct(self.P_list[i], self.Q_list[i]) @ u

#             u_dot_w = jnp.dot(u, w_i)
#             p1 = -2 * u
#             q1 = (w_i - 2 * u_dot_w * u)

#             p2 = w_i
#             q2 = p1.copy()

#             P_add = jnp.stack([p1, p2], axis = 1)
#             Q_add = jnp.stack([q1, q2], axis = 1)

#             if self.P_list:
#                 new_P_list.append(jnp.concatenate([self.P_list[i], P_add], axis = 1))
#                 new_Q_list.append(jnp.concatenate([self.Q_list[i], Q_add], axis = 1))
#             else:
#                 new_P_list.append(P_add)
#                 new_Q_list.append(Q_add)
        
#         return MixingMatQuasisep(self.mixing_mat, self.kernel_list, diag = self.diag, wn_diag = self.wn_diag,
#                                  P_list = new_P_list, Q_list = new_Q_list, cel_dim = self.cel_dim, add_QSM = self.add_QSM)
        
#     def decompose(self, X, wn = True, **kwargs):

#         self.quasi_mat = self._to_symm_qsm(X, wn = wn)
#         self.factor = self.quasi_mat.cholesky()
#         self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
#         return self, {"logdet":self.logdet}
        
#     def evaluate(self, x1, x2, wn = True, **kwargs):
        
#         return self._to_symm_qsm(x1, wn = wn).to_dense()
        
#     def matrix_inv_sqrt(self, R, transpose=0):
        
#         R_shape = R.shape
#         r = self._flatten_R(R)

#         if transpose:
#             r_prime = self.factor.transpose().solve(r)
#         else:
#             r_prime = self.factor.solve(r)

#         R_prime = self._reshape_R(r_prime, R_shape)
            
#         return R_prime

#     def matrix_sqrt(self, R, transpose=0, **kwargs):
        
#         R_shape = R.shape
#         r = self._flatten_R(R)

#         if transpose:
#             r_prime = self.factor.transpose() @ r
#         else:
#             r_prime = self.factor @ r

#         R_prime = self._reshape_R(r_prime, R_shape)
            
#         return R_prime

#     def inv_sqrt_transform(self, K):

#         raise Exception("Not implemented!")

#     def eigendecomp(self, x, **kwargs):

#         K_eval = self.evaluate(x, x, **kwargs)
        
#         return jnp.linalg.eigh(K_eval)

#     def scale(self, c):
        
#         if self.QSM_to_add is not None:
#             new_QSM_to_add = self.QSM_to_add * c
#         else:
#             new_QSM_to_add = None
        
#         return MixingMatQuasisep(self.mixing_mat * c, self.kernel_list, diag = self.diag * c, wn_diag = self.wn_diag * c,
#                                  P_list = self.P_list * c, Q_list = self.Q_list, cel_dim = self.cel_dim, add_QSM = self.add_QSM)

#     def matmul(self, x1, x2, R, wn = True, **kwargs):

#         R_shape = R.shape
#         r = self._flatten_R(R)

#         self.quasi_mat = self._to_symm_qsm(x1, wn = wn)
#         r_prime = self.quasi_mat @ r
        
#         R_prime = self._reshape_R(r_prime, R_shape)
            
#         return R_prime


#     def __add__(self, K):

#         if isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
#             K_sum = MixingMatQuasisep(self.mixing_mat, self.kernel_list, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
#                                       P_list = self.P_list, Q_list = self.Q_list, cel_dim = self.cel_dim, add_QSM = self.add_QSM)

#         elif isinstance(K, MixingMatQuasisep):
            
#             new_P_list = self.P_list + K.P_list
#             new_Q_list = self.Q_list + K.Q_list

#             new_mixing_mat = jnp.concatenate([self.mixing_mat, K.mixing_mat], axis = 1)
#             new_kernel_list = self.kernel_list + K.kernel_list
                
#             K_sum = MixingMatQuasisep(new_mixing_mat, new_kernel_list, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
#                                       P_list = new_P_list, Q_list = new_Q_list, cel_dim = self.cel_dim, add_QSM = self.add_QSM)
            
#         else:
#             raise Exception(f"{type(K)} not recognised or addition not supported")
            
#         return K_sum


    
class MultiTermBothDimKernel(CovType):
    
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

    
    def _truncated_basis_outer_tensor(self, N_alpha, N_beta, N_t):

        e_alpha = jnp.eye(N_alpha)              # (N_alpha, N_alpha)
        e_t = jax.nn.one_hot(jnp.arange(N_beta), N_t)       # (N_beta, N_t)
    
        T = jnp.einsum("ia,jt->ijat", e_alpha, e_t)
        return T.reshape(N_alpha * N_beta, N_alpha, N_t)
        

    def decomp_no_stored_values(
        self,
        *X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

        self.N_l = X[0].shape[-1]
        self.N_t = X[1].shape[-1]

        # Generate transformations
        Sigma_transf0, stored_values_Sigma0 = self.Sigma[0].decompose(X[0])
        Sigma_transf1, stored_values_Sigma1 = self.Sigma[1].decompose(X[1])

        self.Sigma_transf = (Sigma_transf0, Sigma_transf1)

        # Computes the log determinant of Sigma_l \otimes Sigma_t
        stored_values["logdet"] = X[0].shape[-1]*stored_values_Sigma1["logdet"]
        stored_values["logdet"] += X[1].shape[-1]*stored_values_Sigma0["logdet"]

        self.N_alpha = 0
        self.N_beta = 0

        self.K_transf_list = ()
        for K_i in self.K_list:
            K0 = self.Sigma_transf[0].inv_sqrt_transform(K_i[0])
            K1 = self.Sigma_transf[1].inv_sqrt_transform(K_i[1])
            self.K_transf_list += ((K0, K1),)

            if isinstance(K0, Outer) and not isinstance(K1, Outer):
                self.N_alpha += 1
            elif not isinstance(K0, Outer) and isinstance(K1, Outer):
                self.N_beta += 1
            else:
                raise Exception("Not implemented!")
                
        stored_values["A"] = jnp.zeros((X[0].shape[-1], self.N_alpha))
        stored_values["B"] = jnp.zeros((X[1].shape[-1], self.N_beta))
        stored_values["A_kernel_order"] = []
        stored_values["B_kernel_order"] = []
        
        col_i = 0
        col_j = 0
        for (i, K_i) in enumerate(self.K_transf_list):
            if isinstance(K_i[0], Outer):
                K0_outer, _ = K_i[0].decompose(X[0])
                stored_values["A"] = stored_values["A"].at[:, col_i].set(K0_outer.alpha)
                stored_values["A_kernel_order"].append(K_i[1])
                col_i += 1
            elif isinstance(K_i[1], Outer):
                K1_outer, _ = K_i[1].decompose(X[1])
                stored_values["B"] = stored_values["B"].at[:, col_j].set(K1_outer.alpha)
                stored_values["B_kernel_order"].append(K_i[0])
                col_j += 1
        

        reduce_dim = self.N_alpha < X[0].shape[-1] and self.N_beta < X[1].shape[-1]
        assert reduce_dim

        stored_values["J_A"], stored_values["U_A"], self.householder_transform_A = orthonormal_nullspace_gen(stored_values["A"])
        stored_values["J_B"], stored_values["U_B"], self.householder_transform_B = orthonormal_nullspace_gen(stored_values["B"])

        kf_A = MixingMatQuasisep(stored_values["J_A"], stored_values["A_kernel_order"], diag = 1., cel_dim = 1)
        kf_B = MixingMatQuasisep(stored_values["J_B"], stored_values["B_kernel_order"], diag = 1., cel_dim = 0)

        for i in range(self.N_alpha):
            kf_B = kf_B.householder_transform(X, stored_values["U_A"][:, i])
            
        for j in range(self.N_beta):
            kf_A = kf_A.householder_transform(X, stored_values["U_B"][:, j])

        # Calculate component of K_l \otimes \beta \beta^T which falls within block A (concentrated in top left corner of matrix)
        top_corner_ind = (jnp.arange(self.N_alpha), jnp.arange(self.N_beta))
        top_corner_eval = kf_B._evaluate_at_ind(X, top_corner_ind, top_corner_ind, flip_kron = True)

        banded_term = as_banded(top_corner_eval, zero_pad_to_len = self.N_alpha * X[1].shape[-1])

        kf_A = MixingMatQuasisep(kf_A.mixing_mat, kf_A.kernel_list,
                                 cel_dim = 1, add_QSM = banded_term, diag = 1.)

        non_zero_B_ind = (jnp.arange(self.N_alpha, X[0].shape[-1]), jnp.arange(self.N_beta))
        self.B_rows = kf_B._evaluate_at_ind(X, top_corner_ind, non_zero_B_ind)

        self.kf_B = kf_B

        x_A = X[0][..., :self.N_alpha]
        # x_D = X[0][..., self.N_alpha:]
        X_A = (x_A, X[1])
        # X_D = (x_D, X[1])

        # Calc C A^-1 B for D block
        kf_A, stored_vals_A = kf_A.decompose(X_A)

        A_inv_loc = self._truncated_basis_outer_tensor(self.N_alpha, self.N_beta, self.N_t)

        L_A_inv_loc = jax.vmap(kf_A.matrix_inv_sqrt)(A_inv_loc)
        K_A_inv_alphabeta = jnp.einsum('iat,jat->ij', L_A_inv_loc, L_A_inv_loc)

        K_A_inv_B = K_A_inv_alphabeta @ self.B_rows
        C_A_inv_B = LowRankProduct(self.B_rows.T, K_A_inv_B.T).to_qsm()
        self.C_A_inv_B = C_A_inv_B.to_dense()

        stored_values["kf_B_P_list"] = kf_B.P_list
        stored_values["kf_B_Q_list"] = kf_B.Q_list
        
        # P and Q components within D block
        P_list_trunc = [P[self.N_alpha:, :] for P in kf_B.P_list]
        Q_list_trunc = [Q[self.N_alpha:, :] for Q in kf_B.Q_list]

        kf_D_corr = MixingMatQuasisep(kf_B.mixing_mat, kf_B.kernel_list, P_list = P_list_trunc, Q_list = Q_list_trunc,
                                 cel_dim = 0, diag = 1., add_QSM = -C_A_inv_B) 

        self.kf_D = MixingMatQuasisep(kf_B.mixing_mat, kf_B.kernel_list, P_list = P_list_trunc, Q_list = Q_list_trunc,
                                 cel_dim = 0, diag = 1.)

        kf_D_wn = WhiteNoiseKernel(diag = 1.)
        K_D_CAB = Block2x2Kernel(kf_D_corr, kf_D_CAB = kf_D_wn,
                             dim_split = 1, split_loc = self.N_beta)

        self.kf_tilde = Block2x2Kernel(kf_A, kf_B = self.K_B_matmul, kf_D_CAB = K_D_CAB, dim_split = 0, split_loc = self.N_alpha)
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X)

        self.P_list = kf_B.P_list
        self.Q_list = kf_B.Q_list

        stored_values["logdet"] += stored_values["kf_tilde_stored"]["logdet"]
        self.logdet = stored_values["logdet"]

        return self, stored_values

    
    def transform_fn(self, R, transpose = 0):
        
        if transpose:
            R_prime = self.Sigma_transf[0].matrix_sqrt(R, transpose = 1)
            R_prime = self.householder_transform_A(R_prime, transpose = 0)
        else:
            R_prime = self.householder_transform_A(R, transpose = 1)
            R_prime = self.Sigma_transf[0].matrix_sqrt(R_prime, transpose = 0)

        R_prime = R_prime.T

        if transpose:
            R_prime = self.Sigma_transf[1].matrix_sqrt(R_prime, transpose = 1)
            R_prime = self.householder_transform_B(R_prime, transpose = 0)
        else:
            R_prime = self.householder_transform_B(R_prime, transpose = 1)
            R_prime = self.Sigma_transf[1].matrix_sqrt(R_prime, transpose = 0)
            
        return R_prime.T


    def inv_transform_fn(self, R, transpose = 0):

        if transpose:
            R_prime = self.householder_transform_A(R, transpose = 1)
            R_prime = self.Sigma_transf[0].matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_transf[0].matrix_inv_sqrt(R, transpose = 0)
            R_prime = self.householder_transform_A(R_prime, transpose = 0)

        R_prime = R_prime.T

        if transpose:
            R_prime = self.householder_transform_B(R_prime, transpose = 1)
            R_prime = self.Sigma_transf[1].matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_transf[1].matrix_inv_sqrt(R_prime, transpose = 0)
            R_prime = self.householder_transform_B(R_prime, transpose = 0)

        return R_prime.T

    
    def K_B_matmul(self, X1, X2, R_B, transpose = 0, **kwargs):

        if transpose:
            B_mat = self.B_rows.T
            ravel_style = "C"
        else:
            B_mat = self.B_rows.copy()
            ravel_style = "C"
            
        R_sparse = R_B[:, :self.N_beta]

        r_sparse = R_sparse.ravel(ravel_style)
        sparse_B_r = B_mat @ r_sparse

        if transpose:
            sparse_B_R = sparse_B_r.reshape((-1, self.N_beta), order = ravel_style)
            B_R = jnp.concatenate([sparse_B_R,
                                   jnp.zeros((sparse_B_R.shape[0], self.N_t - sparse_B_R.shape[1])),
                                  ], axis = 1)
        else:
            sparse_B_R = sparse_B_r.reshape((self.N_alpha, self.N_beta), order = ravel_style)
            
            B_R = jnp.concatenate([sparse_B_R,
                                   jnp.zeros((sparse_B_R.shape[0], self.N_t - sparse_B_R.shape[1])),
                                  ], axis = 1)
            
        return B_R

        
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
       
    def matmul(self, X1, X2, R, **kwargs):

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

        for K in self.K_list:
            K_R += tensor_mult(K, X1, X2, R, **kwargs)
        
        return K_R


def as_banded(A, zero_pad_to_len = None):
    # Assumes matrix is symmetric

    N = A.shape[1]

    if zero_pad_to_len is not None:
        zero_padding = zero_pad_to_len - N
    else:
        zero_padding = 0
    
    diag = jnp.concatenate([jnp.diag(A), jnp.zeros(zero_padding)])
    off_diags = jnp.zeros((N + zero_padding, N-1))
    
    for i in range(1, A.shape[1]):
        off_diags = off_diags.at[:N-i,i-1].set(jnp.diag(A, k = i))

    return tinygp.noise.Banded(diag, off_diags).to_qsm()

