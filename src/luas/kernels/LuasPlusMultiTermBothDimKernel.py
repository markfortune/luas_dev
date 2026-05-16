import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional
import matplotlib.pyplot as plt

from luas.kernels.covtype import Outer, Exp, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal, General
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import tensor_mult, vmap_for_tensors, cyclic_transpose
from luas.kernels.householder import orthonormal_nullspace_gen
from luas.kernels.BlockKernel import Block2x2Kernel
from luas.kernels.GeneralKernel import GeneralKernel
from luas.kernels.LuasPlusMultiTermKernel import LuasPlusMultiTermKernel
import luas.kernels

__all__ = [
    "LuasPlusMultiTermBothDimKernel",
]


def read_K_list(K_list, X):

    # Initialise for loop reading K_list
    dense_kron = None
    alpha_list = [] # np.zeros((X[self.fast_dim].shape[-1], self.N_alpha))
    beta_list = []

    low_rank_kernels_dim0 = []
    low_rank_kernels_dim1 = []
    for K_i in K_list:
        K_0 = K_i[0]
        K_1 = K_i[1]

        # rank_0 = K_0.rank(X[0])
        # rank_1 = K_1.rank(X[1])

        if isinstance(K_0, Outer) and isinstance(K_1, CovType):
            K_0, _ = K_0.decompose(X[0])
            alpha_list.append(K_0.alpha)
            low_rank_kernels_dim1.append(K_1)

        elif isinstance(K_0, CovType) and isinstance(K_1, Outer):
            K_1, _ = K_1.decompose(X[1])
            beta_list.append(K_1.alpha)
            low_rank_kernels_dim0.append(K_0)

        elif isinstance(K_0, CovType) and isinstance(K_1, CovType) and dense_kron is None:
            dense_kron = [K_0, K_1]

        elif isinstance(dense_kron, tuple):
            raise Exception("Can only have one dense kronecker term")
        
        else:
            raise Exception("All kernel terms must be valid CovType objects")
        
    if alpha_list:
        alpha_mat = jnp.stack(alpha_list, axis = 1)
        alpha_terms = (alpha_mat, low_rank_kernels_dim1)
    else:
        alpha_terms = None

    if beta_list:
        beta_mat = jnp.stack(beta_list, axis = 1)
        beta_terms = (beta_mat, low_rank_kernels_dim0)
    else:
        beta_terms = None

    return dense_kron, alpha_terms, beta_terms

class LuasPlusMultiTermBothDimKernel(CovType):
    
    def __init__(
        self,
        Sigma,
        *K_list,
        use_stored_values: Optional[bool] = True,
        **kwargs,
    ):
        
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list

        self.logL_hessianable = self.logL
        self.decompose = self.decomp_no_stored_values
    
    

    def decomp_no_stored_values(
        self,
        X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = {},
    ) -> PyTree:

        self.N_l = X[0].shape[-1]
        self.N_t = X[1].shape[-1]

        dense_kron, (alpha_mat, alpha_kernel_list), (beta_mat, beta_kernel_list) = read_K_list(self.K_list, X)

        self.N_alpha = alpha_mat.shape[1]
        self.N_beta = beta_mat.shape[1]

        # Required for this optimisation to work
        assert 0 < self.N_alpha < self.N_l and 0 < self.N_beta < self.N_t 

        ######### Do transforms in both dims #########
        # Decompose slow Sigma mat and add its contribution to the log determinant
        self.Sigma_transf = [None, None]
        self.Sigma_transf[0], stored_values_Sigma_l = self.Sigma[0].decompose(X[0])
        stored_values["logdet"] = self.N_t*stored_values_Sigma_l["logdet"]

        self.Sigma_transf[1], stored_values_Sigma_t = self.Sigma[1].decompose(X[1])
        stored_values["logdet"] += self.N_l*stored_values_Sigma_t["logdet"]

        alpha_mat = self.Sigma_transf[0].matrix_inv_sqrt(alpha_mat)
        beta_mat = self.Sigma_transf[1].matrix_inv_sqrt(beta_mat)
        
        # Get Householder transform of all Outer covariances
        self.J_A, stored_values["U_A"], self.householder_transform_A = orthonormal_nullspace_gen(alpha_mat, reverse = True)
        self.J_B, stored_values["U_B"], self.householder_transform_B = orthonormal_nullspace_gen(beta_mat, reverse = True)
        
        ####### Transform kernels #######

        self.K_transf = [None, None]
        self.K_transf[0] = self.Sigma_transf[0].inv_sqrt_transform(dense_kron[0], X[0])
        for i in range(self.N_alpha):
            self.K_transf[0] = self.K_transf[0].householder_transform(X[0], stored_values["U_A"][:, i])

            # Transform alpha terms
            alpha_kernel_list[i] = self.Sigma_transf[1].inv_sqrt_transform(alpha_kernel_list[i], X[1])
            for j in range(self.N_beta):
                alpha_kernel_list[i] = alpha_kernel_list[i].householder_transform(X[1], stored_values["U_B"][:, j])


        K_transf0_eval = self.K_transf[0].evaluate(X[0], X[0], full = True)
        self.K_transf0_A = K_transf0_eval[:-self.N_alpha, :-self.N_alpha]
        self.K_transf0_B = K_transf0_eval[:-self.N_alpha, -self.N_alpha:]
        self.K_transf0_D = K_transf0_eval[-self.N_alpha:, -self.N_alpha:]
        
        self.lam_K_transf0_A, self.Q_K_transf0_A = jnp.linalg.eigh(self.K_transf0_A)

        self.K_beta_A = jnp.zeros((self.N_beta, self.N_l - self.N_alpha, self.N_l - self.N_alpha))
        self.K_beta_B = jnp.zeros((self.N_beta, self.N_l - self.N_alpha, self.N_alpha))
        self.K_beta_D = jnp.zeros((self.N_beta, self.N_alpha, self.N_alpha))

        self.K_transf[1] = self.Sigma_transf[1].inv_sqrt_transform(dense_kron[1], X[1])
        for j in range(self.N_beta):
            self.K_transf[1] = self.K_transf[1].householder_transform(X[1], stored_values["U_B"][:, j])

            # Transform beta kernel terms
            beta_kernel_list[j] = self.Sigma_transf[0].inv_sqrt_transform(beta_kernel_list[j], X[0])
            for i in range(self.N_alpha):
                beta_kernel_list[j] = beta_kernel_list[j].householder_transform(X[0], stored_values["U_A"][:, i])

            K_beta_j_eval = beta_kernel_list[j].evaluate(X[0], X[0], full = True)
            
            self.K_beta_A = self.K_beta_A.at[j, :, :].set(K_beta_j_eval[:-self.N_alpha, :-self.N_alpha])
            self.K_beta_B = self.K_beta_B.at[j, :, :].set(K_beta_j_eval[:-self.N_alpha, -self.N_alpha:])
            self.K_beta_D = self.K_beta_D.at[j, :, :].set(K_beta_j_eval[-self.N_alpha:, -self.N_alpha:])


        ######### Build A Block #########


        def K_A_transform_fn(R, transpose = 0):
            if transpose:
                R_prime = self.Q_K_transf0_A.T @ R
            else:
                R_prime = self.Q_K_transf0_A @ R
            return R_prime
        
        def K_A_inv_transform_fn(R, transpose = 0):
            if transpose:
                R_prime = self.Q_K_transf0_A @ R
            else:
                R_prime = self.Q_K_transf0_A.T @ R
            return R_prime

        K_A_kernel = ((Identity(), Identity()),
                      (Diagonal(diag=self.lam_K_transf0_A), self.K_transf[1]),
                     )
        
        for j in range(self.N_beta):
            K_A_kernel += ((luas.kernels.Fixed(self.Q_K_transf0_A.T @ self.K_beta_A[j] @ self.Q_K_transf0_A, ignore_idx = True), Outer(self.J_B[:, j])),)

        kf_A = LuasPlusMultiTermKernel(*K_A_kernel, eigen_both = True, fast_dim=1,
                                        transform=False, transform_fn = K_A_transform_fn, inv_transform_fn = K_A_inv_transform_fn)
        kf_A, stored_values_A = kf_A.decompose((X[0][:-self.N_alpha], X[1]))

        # C_A_inv_B calc

        C_A_inv_B = self._calc_C_A_inv_B(kf_A, X)

        
        ######### Build D block #########

        kernel_D = ((Identity(), Identity()),)

        for i in range(self.N_alpha):
            kernel_D += ((Outer(self.J_A[:, i]), alpha_kernel_list[i]),)

        K_t_eval = self.K_transf[1].evaluate(X[1], X[1], full = True)
        K_D_dense = jnp.kron(self.K_transf0_D, K_t_eval)
        for j in range(self.N_beta):
            beta_vec = jnp.zeros(self.N_t)
            beta_vec = beta_vec.at[-self.N_beta:].set(self.J_B[:, j])
            K_D_dense += jnp.kron(self.K_beta_D[j], jnp.outer(beta_vec, beta_vec))
        

        kf_D_CAB = GeneralKernel(*kernel_D, dense_mat = K_D_dense - C_A_inv_B)
        kf_D = GeneralKernel(*kernel_D, dense_mat = K_D_dense)


        ######### Define full 2x2 Block kernel and decompose #########
        self.kf_tilde = Block2x2Kernel(kf_A = kf_A, kf_B = self.K_B_matmul, kf_D_CAB = kf_D_CAB, kf_D = kf_D,
                                       dim_split = 0, split_loc = self.N_l-self.N_alpha,
                                       split_idx = True, kf_B_eval = self.kf_B_eval)
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X, full = True)

        stored_values["logdet"] += stored_values["kf_tilde_stored"]["logdet"]
        self.logdet = stored_values["logdet"]

        return self, stored_values


    def K_B_matmul(
        self,
        X1,
        X2,
        R: JAXArray,
        transpose = 0,
        **kwargs,
    ) -> JAXArray:

        if transpose:
            R_prime = self.K_transf0_B.T @ R
            R_prime = self.K_transf[1].matmul(X1[1], X2[1], R_prime.T, full = True).T

            R_C_corr = jnp.zeros((R_prime.shape[0], self.N_beta))
            for j in range(self.N_beta):
                R_prime_j = self.K_beta_B[j].T @ R
                R_prime_j = R_prime_j[:, -self.N_beta:] @ self.J_B[:, j]
                R_C_corr += jnp.outer(R_prime_j, self.J_B[:, j])

            R_prime = R_prime.at[:, -self.N_beta:].set(R_prime[:, -self.N_beta:] + R_C_corr)
        else:
            R_prime = self.K_transf0_B @ R
            R_prime = self.K_transf[1].matmul(X1[1], X2[1], R_prime.T, full = True).T

            R_C = R[:, -self.N_beta:]
            R_C_corr = jnp.zeros((R_prime.shape[0], self.N_beta))
            for j in range(self.N_beta):
                R_prime_j = R_C @ self.J_B[:, j]
                R_prime_j = jnp.outer(R_prime_j, self.J_B[:, j])
                R_C_corr += self.K_beta_B[j] @ R_prime_j
            R_prime = R_prime.at[:, -self.N_beta:].set(R_prime[:, -self.N_beta:] + R_C_corr)
        
        return R_prime
    

    def _calc_C_A_inv_B(self, kf_A, X):
        """
        This calculation is pretty cursed and could probably be optimised or cleaned up quite a lot 
        if this optimisation actually gets used for something
        """

        K_tilde_B = self.K_transf[1].evaluate(X[1], X[1], full = True)

        self.lam_K_transf1_A, self.Q_K_transf1_A = kf_A.lam_fast_A, kf_A.Q_fast_A
        self.K_transf1_B, self.K_transf1_D = kf_A.V_K_B, kf_A.V_K_D
        block_AD_matrix_inv_sqrt = kf_A.kf_tilde.kf_D_CAB.matrix_inv_sqrt

        # Calc R_A_K_inv_R_A

        Q_K_transf0_B = self.Q_K_transf0_A.T @ self.K_transf0_B

        R_A = jnp.kron(jnp.ones((self.N_t - self.N_beta, 1, 1)), Q_K_transf0_B) # shape (N_t - N_beta, N_l - N_alpha, N_alpha)
        lam_A = 1 + jnp.outer(self.lam_K_transf1_A, self.lam_K_transf0_A) # shape (N_t - N_beta, N_l - N_alpha)
        inv_lam_A = 1/lam_A

        Lam_Q_R_A = jnp.einsum("ij,ijm->ijm", inv_lam_A, R_A)
        R_A_K_A_inv_R_A = jnp.einsum("ijm,ijn->imn", R_A, Lam_Q_R_A)

        n = jnp.arange(self.N_t - self.N_beta)
        R_A_K_A_inv_R_A_dense = jnp.zeros((self.N_alpha, self.N_t - self.N_beta, self.N_alpha, self.N_t - self.N_beta))
        R_A_K_A_inv_R_A_dense = R_A_K_A_inv_R_A_dense.at[:, n, :, n].add(R_A_K_A_inv_R_A)

        # vmapped function to calculate fast D_inv_sqrt for tensors of shape (N_beta, N_beta_vecs, N_l - N_alpha, N_alpha_vecs)
        # returns matrix of shape (N_beta * N_l - N_alpha, N_alpha_vecs, N_beta_vecs)
        def D_inv_sqrt_R(D_inv_sqrt, R):
            return D_inv_sqrt(R).ravel()
        D_inv_sqrt_K = jax.vmap(jax.vmap(D_inv_sqrt_R, in_axes = (None, 1), out_axes = 1), in_axes = (None, 3), out_axes = 1)

        # Calc L_D_inv_C_A_inv_R_A
        C_Q_R_A = jnp.einsum("ij,jkm->ijkm", self.K_transf1_B.T @ self.Q_K_transf1_A, Lam_Q_R_A)
        C_A_inv_R_A = jnp.einsum("k,ijkl->ijkl", self.lam_K_transf0_A, C_Q_R_A)
        L_D_inv_C_A_inv_R_A = D_inv_sqrt_K(block_AD_matrix_inv_sqrt, C_A_inv_R_A)
        
        # Calc L_D_inv_R_D
        K_1_D = jnp.eye(self.N_beta)
        K_0 = self.Q_K_transf0_A.T @ self.K_transf0_B
        total_K_cov = K_1_D[:, :, None, None] * K_0[None, None, :, :]
        L_D_inv_R_D = D_inv_sqrt_K(block_AD_matrix_inv_sqrt, total_K_cov)

         # Calc L_D_inv_R_D2
        total_beta_cov = jnp.zeros((self.N_beta, self.N_beta, self.N_l - self.N_alpha, self.N_alpha))
        for k in range(self.N_beta):
            K_1_k = jnp.outer(self.J_B[:, k], self.J_B[:, k])
            K_0_k = self.Q_K_transf0_A.T @ self.K_beta_B[k]
            total_beta_cov += K_1_k[:, :, None, None] * K_0_k[None, None, :, :]

        L_D_inv_R_D2 = D_inv_sqrt_K(block_AD_matrix_inv_sqrt, total_beta_cov)


        # Calculate dot products of matrix inv sqrts for R_A, R_D and R_D2
        R_A_B_D_inv_C_R_A = jnp.einsum("ijk,imn->jkmn", L_D_inv_C_A_inv_R_A, L_D_inv_C_A_inv_R_A)
        R_D_K_D_inv_R_D = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D, L_D_inv_R_D)
        R_D_K_D_inv_R_A = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D, L_D_inv_C_A_inv_R_A)

        R_D2_K_D_inv_R_D = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D2, L_D_inv_R_D)
        R_D2_K_D_inv_R_A = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D2, L_D_inv_C_A_inv_R_A)
        R_D2_K_D_inv_R_D2 = jnp.einsum("ijk,imn->jkmn", L_D_inv_R_D2, L_D_inv_R_D2)

        # Combine into full matrix (N_alpha * N_t, N_alpha * N_t) block by block
        full_A = jnp.zeros((self.N_alpha, self.N_t, self.N_alpha, self.N_t))
        full_A = full_A.at[:, -self.N_beta:, :, :-self.N_beta].set(-R_D_K_D_inv_R_A)
        full_A = full_A.at[:, :-self.N_beta, :, :-self.N_beta].set(R_A_K_A_inv_R_A_dense + R_A_B_D_inv_C_R_A)
        full_A = full_A.at[:, :-self.N_beta, :, -self.N_beta:].set(jnp.transpose(-R_D_K_D_inv_R_A, [2, 3, 0, 1]))
        full_A = full_A.at[:, -self.N_beta:, :, -self.N_beta:].set(R_D_K_D_inv_R_D)

        # Do transformations for I \otimes K_tilde1 factored out of B term so far
        N_total = self.N_alpha * self.N_t
        full_A_dense = full_A.reshape(N_total, N_total)

        I_kron_K_transf1_B = jnp.kron(jnp.eye(self.N_alpha), K_tilde_B)
        Lam_QKT = I_kron_K_transf1_B @ jnp.kron(jnp.eye(self.N_alpha), jax.scipy.linalg.block_diag(self.Q_K_transf1_A, jnp.eye(self.N_beta)))
        C_A_inv_B = Lam_QKT @ full_A_dense @ Lam_QKT.T

        # Add additional terms corresponding to beta_kernels in inner D block of outer B block
        full_A2 = jnp.zeros((self.N_alpha, self.N_t, self.N_alpha, self.N_t))
        full_A2 = full_A2.at[:, -self.N_beta:, :, :-self.N_beta].set(-R_D2_K_D_inv_R_A)
        full_A2 = full_A2.at[:, -self.N_beta:, :, -self.N_beta:].set(R_D2_K_D_inv_R_D)

        full_A2_dense = full_A2.reshape(N_total, N_total)
        A2_term = full_A2_dense @ Lam_QKT.T
        C_A_inv_B2 = A2_term + A2_term.T

        C_A_inv_B2_reshape = C_A_inv_B2.reshape(self.N_alpha, self.N_t, self.N_alpha, self.N_t)
        current_D_block = C_A_inv_B2_reshape[:, -self.N_beta:, :, -self.N_beta:]
        C_A_inv_B2_reshape = C_A_inv_B2_reshape.at[:, -self.N_beta:, :, -self.N_beta:].set(current_D_block + R_D2_K_D_inv_R_D2)
        C_A_inv_B2 = C_A_inv_B2_reshape.reshape(N_total, N_total)

        return C_A_inv_B + C_A_inv_B2
    

    def _calc_C_A_inv_B_unopt(self, kf_A, X):
        B_mat_dense = self.kf_B_eval(X, X)
        L_A_inv_B = jnp.zeros_like(B_mat_dense)
        for i in range(B_mat_dense.shape[1]):
            B_vec = B_mat_dense[:, i].reshape((self.N_l - self.N_alpha, self.N_t))
            L_A_inv_B = L_A_inv_B.at[:, i].set(kf_A.matrix_inv_sqrt(B_vec).ravel())
        C_A_inv_B = L_A_inv_B.T @ L_A_inv_B

        return C_A_inv_B
    
    def kf_B_eval(self, X1, X2):
        K_B = jnp.kron(self.K_transf0_B, self.K_transf[1].evaluate(X1[1], X2[1])) 
        
        for j in range(self.N_beta):
            beta_vec = jnp.zeros(self.N_t)
            beta_vec = beta_vec.at[-self.N_beta:].set(self.J_B[:, j])
            K_B += jnp.kron(self.K_beta_B[j], jnp.outer(beta_vec, beta_vec))

        return K_B
    

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
