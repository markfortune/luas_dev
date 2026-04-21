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
from luas.kernels.tinygp_ext import Multiband

from tinygp.noise import Noise
from tinygp.solvers.quasisep.core import DiagQSM, StrictLowerTriQSM, SymmQSM, SquareQSM, StrictUpperTriQSM

__all__ = [
    "KinvR_block",
    "logL_block",
    "LuasLasrachKernel",
]

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

class QSM_Kernel(CovType):
    def __init__(self, QSM, cel_dim = 1):
        self.QSM = QSM
        self.cel_dim = cel_dim

    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        return self.QSM.to_dense()
    
    def decompose(self, x, wn = True, **kwargs):

        self.factor = self.QSM.cholesky()
        
        logdetK = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdetK":logdetK}
        
    def _flatten_R(self, R):
                
        if self.cel_dim == 0:
            return R.ravel("C")
        else:
            return R.ravel("F")

    def _reshape_R(self, r, R_shape):
        
        if self.cel_dim == 0:
            return r.reshape(R_shape, order='C')
        else:
            return r.reshape(R_shape, order='F')

        
    def matrix_inv_sqrt(self, R, transpose=0):
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose().solve(r)
        else:
            r_prime = self.factor.solve(r)

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose() @ r
        else:
            r_prime = self.factor @ r

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def inv_sqrt_transform(self, K):

        raise Exception("Not implemented!")


    def scale(self, c):
        return QSM_Kernel(self.QSM)

        
    # def __add__(self, K):

    #     if isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
    #         K_sum = GeneralQuasisep(self.tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
    #     elif isinstance(K, Exp) or isinstance(K, GeneralQuasisep):
    #         K_sum = GeneralQuasisep(K.tinygp_kf + self.tinygp_kf,
    #                                 diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
    #     elif isinstance(K, Outer) or isinstance(K, General) or isinstance(K, OuterPlusScaledIdentity):
    #         # Should update as don't lose quasiseparability
    #         K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))
    #     else:
    #         raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
    #     return K_sum
        

class LowRankProduct(Noise):

    P: JAXArray
    Q: JAXArray

    def __check_init__(self) -> None:
        if jnp.ndim(self.P) != 2:
            raise ValueError(
                "Shape must be (n_data, n_vec)"
            )
        if jnp.ndim(self.Q) != 2:
            raise ValueError(
                "Shape must be (n_data, n_vec)"
            )
        if self.P.shape != self.Q.shape:
            raise ValueError(
                "Shape of P and Q must be the same"
            )

    def diagonal(self) -> JAXArray:
        
        return (self.P * self.Q).sum(1)
    
    def _add(self, other):
        P_comb = jnp.concatenate([self.P, other.P], axis = 1)
        Q_comb = jnp.concatenate([self.Q, other.Q], axis = 1)
        return LowRankProduct(P_comb, Q_comb)
    
    def __add__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __radd__(self, other: JAXArray) -> JAXArray:
        return self._add(other)

    def __mul__(self, other):
        if is_scalar(other):
            return self.scale(other)
        else:
            raise Exception("Not implemented!")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other: JAXArray) -> JAXArray:
        H_T_other = self.Q.T @ other
        return self.P @ H_T_other

    def scale(self, c):
        return LowRankProduct(self.P*c, self.Q)
    
    def to_qsm(self):

        diag_term = DiagQSM(d=self.diagonal())
        lower_term = StrictLowerTriQSM(p=self.P, q=self.Q, a=jnp.kron(jnp.ones((self.P.shape[0], 1, 1)), jnp.eye(self.P.shape[1])))
        
        return SymmQSM(diag=diag_term, lower=lower_term)


class MixingMatQuasisep(CovType):
    def __init__(self, mixing_mat, kernel_list, outer_QSM = None, add_QSM = None,
                 diag = 0., wn_diag = 0., params = None, cel_dim = 1):
        self.mixing_mat = mixing_mat # shape [N, N_alpha]
        self.kernel_list = kernel_list # N_alpha list of tinygp kernel functions
        self.outer_QSM = outer_QSM
        self.diag = diag
        self.wn_diag = wn_diag
        self.rank = mixing_mat.shape[1]
        self.add_QSM = add_QSM
        self.params = params
        self.cel_dim = cel_dim

    def _evaluate_at_ind(self, X, row_ind, col_ind, flip_kron = False):

        cel_vec_at_ind1 = X[self.cel_dim][row_ind[self.cel_dim]]
        cel_vec_at_ind2 = X[self.cel_dim][col_ind[self.cel_dim]]

        for i in range(self.rank):
            K_i_eval = self.kernel_list[i].tinygp_kf(X[self.cel_dim][row_ind[self.cel_dim]], X[self.cel_dim][col_ind[self.cel_dim]])
            # Need to add diagonal term if included!
            # K_i_eval += (self.kernel_list[i].diag + self.kernel_list[i].wn_diag)

            if self.P_list:
                K_i_eval += self.P_list[i][row_ind[self.cel_dim]] @ self.Q_list[i][col_ind[self.cel_dim]].T

            mixing_eval = self.mixing_mat[row_ind[1-self.cel_dim], i:i+1] @ self.mixing_mat[col_ind[1-self.cel_dim], i:i+1].T


            if flip_kron:
                if self.cel_dim == 0:
                    K_kron = jnp.kron(mixing_eval, K_i_eval)
                else:
                    K_kron = jnp.kron(K_i_eval, mixing_eval)
            else:
                if self.cel_dim == 0:
                    K_kron = jnp.kron(K_i_eval, mixing_eval)
                else:
                    K_kron = jnp.kron(mixing_eval, K_i_eval)

            if i == 0:
                K_eval = K_kron.copy()
            else:
                K_eval += K_kron

        if self.add_QSM is not None:
            raise Exception("Not implemented")

        return K_eval

    def _tinygp_coords(self, X):
        cel_vec = X[self.cel_dim]
        non_cel_vec = X[1-self.cel_dim]
        
        x_t_long = jnp.kron(cel_vec, jnp.ones(non_cel_vec.shape[-1]))
        x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(non_cel_vec.shape[-1]))
        
        return (x_t_long, x_l_long)

    def QSM_kron_identity(self, outer_QSM, rank):
        a_w_zero = outer_QSM.lower.a.copy()
        a_w_zero = a_w_zero.at[0, ...].set(0.)
        p_w_zero = outer_QSM.lower.p.copy()
        p_w_zero = p_w_zero.at[0, ...].set(0.)
        
        d_long = jnp.kron(jnp.ones(rank), outer_QSM.diag.d)
        a_long = jnp.concatenate([outer_QSM.lower.a, jnp.kron(jnp.ones((rank-1, 1, 1)), a_w_zero)], axis = 0)
        p_long = jnp.concatenate([outer_QSM.lower.p, jnp.kron(jnp.ones((rank-1, 1)), p_w_zero)], axis = 0)
        q_long = jnp.kron(jnp.ones((rank, 1)), outer_QSM.lower.q)
        
        diag_term = DiagQSM(d=d_long)
        lower_term = StrictLowerTriQSM(p=p_long, q=q_long, a=a_long)
        
        if hasattr(outer_QSM, "upper"):
            a_upper_w_zero = outer_QSM.upper.a.copy()
            a_upper_w_zero = a_upper_w_zero.at[0, ...].set(0.)
            p_upper_w_zero = outer_QSM.upper.p.copy()
            p_upper_w_zero = p_upper_w_zero.at[0, ...].set(0.)
            
            a_upper_long = jnp.concatenate([outer_QSM.upper.a, jnp.kron(jnp.ones((rank-1, 1, 1)), a_upper_w_zero)], axis = 0)
            p_upper_long = jnp.concatenate([outer_QSM.upper.p, jnp.kron(jnp.ones((rank-1, 1)), p_upper_w_zero)], axis = 0)
            q_upper_long = jnp.kron(jnp.ones((rank, 1)), outer_QSM.upper.q)

            upper_term = StrictUpperTriQSM(p=p_long, q=q_long, a=a_long)
        else:
            upper_term = lower_term.T
            
        outer_QSM_kron = SquareQSM(diag=diag_term, lower=lower_term, upper = upper_term)

        return outer_QSM_kron
        

    def to_symm_QSM(self, X, wn = True):
        
        for i in range(self.rank):
            if i == 0:
                self.kf_2D = Multiband(
                    kernel=self.kernel_list[0].tinygp_kf,
                    amplitudes=self.mixing_mat[:, 0],
                )
            else:
                self.kf_2D += Multiband(
                    kernel=self.kernel_list[i].tinygp_kf,
                    amplitudes=self.mixing_mat[:, i],
                )

        tiny_coords = self._tinygp_coords(X)
        quasisep_cov = self.kf_2D.to_symm_qsm(tiny_coords)

        def unblock_fn(Block_obj):
            return Block_obj.to_dense()
    
        a_unblocked = jax.vmap(unblock_fn)(quasisep_cov.lower.a)
        lower_unblocked = StrictLowerTriQSM(p = quasisep_cov.lower.p, q = quasisep_cov.lower.q, a = a_unblocked)
        quasisep_cov = SymmQSM(diag = quasisep_cov.diag, lower = lower_unblocked)

        self.quasisep_cov = quasisep_cov

        plt.imshow(quasisep_cov.to_dense())
        plt.show()


        if self.outer_QSM is not None:

            outer_QSM_kron = self.QSM_kron_identity(self.outer_QSM, self.rank)
            
            plt.imshow(outer_QSM_kron.to_dense())
            plt.show()
            
            quasisep_cov_square = outer_QSM_kron @ quasisep_cov @ outer_QSM_kron.T
            quasisep_cov = SymmQSM(diag = quasisep_cov_square.diag, lower = quasisep_cov_square.lower)

        diag = (self.diag + wn*self.wn_diag)*jnp.ones(tiny_coords[0].shape[-1])
        quasisep_cov += tinygp.noise.Diagonal(diag).to_qsm()

        if self.add_QSM is not None:
            quasisep_cov += self.add_QSM
    
        return quasisep_cov

    def _flatten_R(self, R):
                
        if self.cel_dim == 0:
            return R.ravel("C")
        else:
            return R.ravel("F")

    def _reshape_R(self, r, R_shape):
        
        if self.cel_dim == 0:
            return r.reshape(R_shape, order='C')
        else:
            return r.reshape(R_shape, order='F')

        
    def decompose(self, X, wn = True, **kwargs):

        self.quasi_mat = self.to_symm_QSM(X, wn = wn)
        self.factor = self.quasi_mat.cholesky()
        
        logdetK = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdetK":logdetK}
        
    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        return self.to_symm_QSM(x1, wn = wn).to_dense()
        
    def matrix_inv_sqrt(self, R, transpose=0):
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose().solve(r)
        else:
            r_prime = self.factor.solve(r)

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):
        
        R_shape = R.shape
        r = self._flatten_R(R)

        if transpose:
            r_prime = self.factor.transpose() @ r
        else:
            r_prime = self.factor @ r

        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

    def inv_sqrt_transform(self, K):

        raise Exception("Not implemented!")

    
    def eigendecomp(self, x, **kwargs):

        K_eval = self.evaluate(x, x, **kwargs)
        
        return jnp.linalg.eigh(K_eval)


    def scale(self, c):
        
        if self.QSM_to_add is not None:
            new_QSM_to_add = self.QSM_to_add * c
        else:
            new_QSM_to_add = None
        
        return MixingMatQuasisep(self.mixing_mat * c, self.kernel_list, diag = self.diag * c, wn_diag = self.wn_diag * c,
                                 P_list = self.P_list * c, Q_list = self.Q_list, cel_dim = self.cel_dim, add_QSM = self.add_QSM)

    def matmul(self, x1, x2, R, wn = True, **kwargs):

        R_shape = R.shape
        r = self._flatten_R(R)

        self.quasi_mat = self.to_symm_QSM(x1, wn = wn)
        r_prime = self.quasi_mat @ r
        
        R_prime = self._reshape_R(r_prime, R_shape)
            
        return R_prime

        
    def __add__(self, K):

        if isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
            K_sum = MixingMatQuasisep(self.mixing_mat, self.kernel_list, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                      P_list = self.P_list, Q_list = self.Q_list, cel_dim = self.cel_dim, add_QSM = self.add_QSM)

        elif isinstance(K, MixingMatQuasisep):
            
            new_P_list = self.P_list + K.P_list
            new_Q_list = self.Q_list + K.Q_list

            new_mixing_mat = jnp.concatenate([self.mixing_mat, K.mixing_mat], axis = 1)
            new_kernel_list = self.kernel_list + K.kernel_list
                
            K_sum = MixingMatQuasisep(new_mixing_mat, new_kernel_list, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag,
                                      P_list = new_P_list, Q_list = new_Q_list, cel_dim = self.cel_dim, add_QSM = self.add_QSM)
            
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported")
            
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
            self.C_mult = lambda R, **kwargs: self.kf_B_matmul(X_A, X_D, R, transpose = 1, **kwargs)
        
        return self, {"logdetK":stored_values_A["logdetK"] + stored_values_D["logdetK"]}
        
    def matrix_inv_sqrt(self, R, transpose=0):

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]

        # print(R_A.shape, self.dim_split)

        if self.kf_B_matmul is not None:
            if transpose == 0:
                R_prime_A = self.kf_A.matrix_inv_sqrt(R_A, transpose = transpose)
                
                R_C = self.kf_A.matrix_inv_sqrt(R_prime_A, transpose = 1)
                R_C = self.C_mult(R_C, wn = False)
                R_prime_D = self.kf_D_CAB.matrix_inv_sqrt(R_D - R_C, transpose = 0)
            else:
                R_prime_D = self.kf_D_CAB.matrix_inv_sqrt(R_D, transpose = 1)
                
                R_B = self.B_mult(R_prime_D, wn = False)
                R_B = self.kf_A.matrix_inv_sqrt(R_B, transpose = 0)
                R_prime_A = self.kf_A.matrix_inv_sqrt(R_A - R_B, transpose = 1)
        else:
            R_prime_A = self.kf_A.matrix_inv_sqrt(R_A, transpose = transpose)
            R_prime_D = self.kf_D_CAB.matrix_inv_sqrt(R_D, transpose = transpose)

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

    
class MultiTermBothDimKernel2(CovType):
    
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

    def _truncated_basis_outer_tensor2(self, N_alpha, N_beta, N_l):

        e_beta = jnp.eye(N_beta)              # (N_alpha, N_alpha)
        e_l = jax.nn.one_hot(jnp.arange(N_alpha), N_l)       # (N_beta, N_t)
    
        T = jnp.einsum("ia,jt->ijat", e_l, e_beta)
        return T.reshape(N_alpha * N_beta, N_l, N_beta)
        

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
        stored_values["logdetK"] = X[0].shape[-1]*stored_values_Sigma1["logdetK"]
        stored_values["logdetK"] += X[1].shape[-1]*stored_values_Sigma0["logdetK"]

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

        stored_values["J_A"], householder_QSM_A, self.householder_transform_A = orthonormal_nullspace_gen(stored_values["A"])
        stored_values["J_B"], householder_QSM_B, self.householder_transform_B = orthonormal_nullspace_gen(stored_values["B"])

        kf_A = MixingMatQuasisep(stored_values["J_A"], stored_values["A_kernel_order"], outer_QSM = householder_QSM_B.T,
                                 diag = 1., cel_dim = 1)
        kf_B = MixingMatQuasisep(stored_values["J_B"], stored_values["B_kernel_order"], outer_QSM = householder_QSM_A.T,
                                 diag = 1., cel_dim = 0)
        self.kf_B = kf_B
        X_B = (X[0], X[1][:self.N_beta])

        self.houseA = householder_QSM_A

        x_A = X[0][..., :self.N_alpha]
        X_A = (x_A, X[1])

        self.QSM_B = kf_B.to_symm_QSM(X_B)

        A_inv_loc = self._truncated_basis_outer_tensor2(self.N_alpha, self.N_beta, self.N_l)
        B_all = jax.vmap(lambda R: self.QSM_B @ R.ravel("C"))(A_inv_loc)

        banded_block = B_all[:, :self.N_alpha * self.N_beta]

        def reorder_tensor(T, N_alpha, N_beta):
            return (
                T.reshape(N_alpha, N_beta, N_alpha, N_beta)
                 .transpose(1, 0, 3, 2)
                 .reshape(N_alpha * N_beta, N_alpha * N_beta)
            )
            
        banded_block = reorder_tensor(banded_block, self.N_alpha, self.N_beta)

        print("new", banded_block)
        banded_term = as_banded(banded_block, zero_pad_to_len = self.N_alpha * X[1].shape[-1])

        kf_A = MixingMatQuasisep(kf_A.mixing_mat, kf_A.kernel_list, outer_QSM = kf_A.outer_QSM,
                                 cel_dim = 1, add_QSM = banded_term, diag = 1.)

        # Calc C A^-1 B for D block
        self.B_rows = B_all[:, self.N_alpha * self.N_beta:]


        kf_A, stored_vals_A = kf_A.decompose(X_A)

        A_inv_loc = self._truncated_basis_outer_tensor(self.N_alpha, self.N_beta, self.N_t)
        L_A_inv_loc = jax.vmap(kf_A.matrix_inv_sqrt)(A_inv_loc)
        K_A_inv_alphabeta = jnp.einsum('iat,jat->ij', L_A_inv_loc, L_A_inv_loc)

        K_A_inv_B = K_A_inv_alphabeta @ self.B_rows
        C_A_inv_B = LowRankProduct(self.B_rows.T, K_A_inv_B.T).to_qsm()

        # Build D block QSM
        diag_trunc = DiagQSM(d = self.QSM_B.diag.d[self.N_alpha * self.N_beta:])
        p_trunc = self.QSM_B.lower.p[self.N_alpha * self.N_beta:, :]
        q_trunc = self.QSM_B.lower.q[self.N_alpha * self.N_beta:, :]
        a_trunc = self.QSM_B.lower.a[self.N_alpha * self.N_beta:, :, :]
        lower_trunc = StrictLowerTriQSM(p = p_trunc, q = q_trunc, a = a_trunc)
        QSM_D_block = SymmQSM(diag = diag_trunc, lower = lower_trunc)

        print(QSM_D_block, C_A_inv_B, self.B_rows.shape)
        QSM_D_min_C_A_inv_B = QSM_D_block - C_A_inv_B

        kf_D_corr = QSM_Kernel(QSM_D_min_C_A_inv_B, cel_dim = 0)

        kf_D_wn = WhiteNoiseKernel(diag = 1.)
        K_D_CAB = Block2x2Kernel(kf_D_corr, kf_D_CAB = kf_D_wn,
                             dim_split = 1, split_loc = self.N_beta)

        self.kf_tilde = Block2x2Kernel(kf_A, kf_B = self.K_B_matmul, kf_D_CAB = K_D_CAB, dim_split = 0, split_loc = self.N_alpha)
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X)

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

        diag_term = DiagQSM(d=1. - 2 * u_i**2)
        lower_term = StrictLowerTriQSM(p=-2*u_i.reshape((N_l, 1)), q=u_i.reshape((N_l, 1)), a=jnp.ones((N_l, 1, 1)))
        householder_i = SquareQSM(diag=diag_term, lower=lower_term, upper = lower_term.T)
        
        if i == 0:
            householder_QSM = householder_i
        else:
            householder_QSM = householder_QSM @ householder_i

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
    
    return Lam, householder_QSM, householder_transform


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


# class ScaledQuasisep(GeneralQuasisep):
#     def __init__(self, base_kf, amplitudes, diag = 0., wn_diag = 0., params = None, cel_dim=1):
#         self.base_kf = base_kf
#         self.diag = diag
#         self.wn_diag = wn_diag
#         self.amplitudes = amplitudes
#         self.params = params

#     def tinygp_coords(self, X):
        
#         return (X, jnp.arange(X.shape[-1]))

#     def calc_diag(self, X, wn = True):
        
#         return self.diag + wn*self.wn_diag

#     def evaluate(self, x1, x2, wn = True, **kwargs):

#         X1 = self.tinygp_coords(x1)
#         X2 = self.tinygp_coords(x2)

#         self.tinygp_kf = Multiband(self.base_kf, jnp.array(self.amplitudes))
        
#         return self.tinygp_kf(X1, X2) + jnp.diag(self.calc_diag(x1, wn = wn))
    
#     def decompose(self, x, wn = True, **kwargs):

#         diag = self.calc_diag(x, wn = wn)
#         noise_model = tinygp.noise.Diagonal(diag=diag)
        
#         X = self.tinygp_coords(x)
#         self.tinygp_kf = Multiband(self.base_kf, jnp.array(self.amplitudes))
#         matrix = self.tinygp_kf.to_symm_qsm(X)
        
#         matrix += noise_model.to_qsm()
#         self.factor = matrix.cholesky()
        
#         logdetK = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
#         return self, {"logdetK":logdetK}
    

#     def scale(self, c):
#         return ScaledQuasisep(self.base_kf * c, self.amplitudes * c, diag = self.diag * c, wn_diag = self.wn_diag * c)

#     def matmul(self, x1, x2, r, wn = True, **kwargs):

#         X1 = self.tinygp_coords(x1)
#         X2 = self.tinygp_coords(x2)

#         self.tinygp_kf = Multiband(self.base_kf, jnp.array(self.amplitudes))

#         r_prime = self.tinygp_kf.matmul(X1, X2, r)

#         diag = self.calc_diag(x1, wn = wn)
#         noise_model = tinygp.noise.Diagonal(diag=diag)
#         r_prime += noise_model @ r

#         return r_prime
        
#     def __add__(self, K):

#         if isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
#             K_sum = ScaledQuasisep(self.tinygp_kf, self.amplitudes,
#                                     diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
#         else:
#             raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
#         return K_sum

# class MixingMat(Noise):

#     H: JAXArray

#     def __check_init__(self) -> None:
#         if jnp.ndim(self.H) != 2:
#             raise ValueError(
#                 "Shape must be (n_data, n_vec)"
#             )

#     def diagonal(self) -> JAXArray:
#         if jnp.ndim(self.H) == 1:
#             return self.H**2
#         else:
#             return (self.H**2).sum(1)
    
#     def _add(self, other: JAXArray) -> JAXArray:
#         return MixingMat(jnp.concatenate([self.H, other.H], axis = 1))
    
#     def __add__(self, other: JAXArray) -> JAXArray:
#         return self._add(other)

#     def __radd__(self, other: JAXArray) -> JAXArray:
#         return self._add(other)

#     def __mul__(self, other):
#         if is_scalar(other):
#             return self.scale(other)
#         else:
#             raise Exception("Not implemented!")

#     def __rmul__(self, other):
#         return self.__mul__(other)

#     def __matmul__(self, other: JAXArray) -> JAXArray:
#         H_T_other = self.H.T @ other
#         return self.H @ H_T_other

#     def scale(self, c):
#         return MixingMat(self.H*jnp.sqrt(c))
    
#     def to_qsm(self):

#         diag_term = DiagQSM(d=self.diagonal())
#         lower_term = StrictLowerTriQSM(p=self.H, q=self.H, a=jnp.kron(jnp.ones((self.H.shape[0], 1, 1)), jnp.eye(self.H.shape[1])))
        
#         return SymmQSM(diag=diag_term, lower=lower_term)

# class GeneralQuasisep2D(CovType):
#     def __init__(self, tinygp_kf, diag = 0., wn_diag = 0., params = None, cel_dim=1):
#         self.tinygp_kf = tinygp_kf
#         self.diag = diag
#         self.wn_diag = wn_diag
#         self.cel_dim = cel_dim
#         self.params = params

#     def tinygp_coords(self, X):
#         cel_vec = X[self.cel_dim]
#         non_cel_vec = X[1-self.cel_dim]
        
#         x_t_long = jnp.kron(cel_vec, jnp.ones(non_cel_vec.shape[-1]))
#         x_l_long = jnp.kron(jnp.ones(cel_vec.shape[-1], dtype = int), jnp.arange(non_cel_vec.shape[-1]))
        
#         return (x_t_long, x_l_long)

#     def calc_diag(self, X, wn = True):
        
#         return self.diag + wn*self.wn_diag

#     def evaluate(self, x1, x2, wn = True, **kwargs):

#         X1 = self.tinygp_coords(x1)
#         X2 = self.tinygp_coords(x2)
        
#         return self.tinygp_kf(X1, X2) + jnp.diag(self.calc_diag(x1, wn = wn))
    
#     def decompose(self, x, wn = True, **kwargs):

#         X = self.tinygp_coords(x)

#         diag = self.calc_diag(x, wn = wn)

#         noise_model = tinygp.noise.Diagonal(diag=diag)
#         matrix = self.tinygp_kf.to_symm_qsm(X)
#         matrix += noise_model.to_qsm()
#         self.factor = matrix.cholesky()
        
#         logdetK = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
#         return self, {"logdetK":logdetK}
        
#     def matrix_inv_sqrt(self, R, transpose=0):
        
#         if self.cel_dim == 0:
#             R_prime = R.T
#         else:
#             R_prime = R.copy()

#         R_shape = R_prime.shape
#         r = R_prime.ravel("F")
        
#         if transpose:
#             r = self.factor.transpose().solve(r)
#         else:
#             r = self.factor.solve(r)

#         R_prime = r.reshape(R_shape, order="F")

#         if self.cel_dim == 0:
#             R_prime = R_prime.T
            
#         return R_prime

#     def matrix_sqrt(self, R, transpose=0, **kwargs):

#         if self.cel_dim == 0:
#             R_prime = R.T
#         else:
#             R_prime = R.copy()
        
#         R_shape = R_prime.shape
#         r = R_prime.ravel("F")

#         if transpose:
#             L_r = self.factor.transpose() @ r
#         else:
#             L_r = self.factor @ r

#         L_R = L_r.reshape(R_shape, order="F")

#         if self.cel_dim == 0:
#             L_R = L_R.T
            
#         return L_R

#     def scale(self, c):
#         return GeneralQuasisep2D(self.tinygp_kf * c, diag = self.diag * c, wn_diag = self.wn_diag * c)

#     def matmul(self, x1, x2, R, wn = True, **kwargs):

#         X1 = self.tinygp_coords(x1)
#         X2 = self.tinygp_coords(x2)
        
#         if self.cel_dim == 0:
#             R_prime = R.T
#         else:
#             R_prime = R.copy()
        
#         R_shape = R_prime.shape
#         r = R_prime.ravel("F")

#         r_prime = self.tinygp_kf.matmul(X1, X2, r)

#         diag = self.calc_diag(x1, wn = wn)
#         noise_model = tinygp.noise.Diagonal(diag=diag)
#         r_prime += noise_model @ r

#         R_prime = r_prime.reshape(R_shape, order="F")

#         if self.cel_dim == 0:
#             R_prime = R_prime.T
            
#         return R_prime
        
#     def __add__(self, K):

#         if isinstance(K, GeneralQuasisep2D):
#             K_sum = GeneralQuasisep2D(K.tinygp_kf + self.tinygp_kf,
#                                     diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
#         else:
#             raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
#         return K_sum
