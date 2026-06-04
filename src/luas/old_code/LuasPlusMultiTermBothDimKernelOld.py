import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import Outer, Exp, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal, General
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import tensor_mult, vmap_for_tensors, cyclic_transpose
from luas.kernels.householder import orthonormal_nullspace_gen
from luas.kernels.BlockKernel import Block2x2Kernel, BlockKernel
from luas.kernels.GeneralKernel import GeneralKernel
from luas.src.luas.kernels.LuasKernelND import LuasKernel
from luas.kernels.MultiTermKernel import MultiTermKernel
from luas.kernels.LuasPlusMultiTermKernel import LuasPlusMultiTermKernel
from luas.kernel_selector import read_K_list
import luas.kernels

__all__ = [
    "LuasPlusMultiTermBothDimKernel",
]

class LuasPlusMultiTermBothDimKernelOld(CovType):
    
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
    

    def _calc_C_A_inv_B(self, kf_A_matrix_inv_sqrt, lam_K_tilde, V_K_B, V_Sigma_B):

        # self.V_K_B of shape (N_fast - N_alpha, N_alpha)
        # K_B_nonzero of shape (N_slow, N_alpha, N_fast - N_alpha)
        K_B_nonzero = jnp.kron(lam_K_tilde.reshape((lam_K_tilde.size, 1, 1)), V_K_B.T)
        K_B_nonzero += jnp.kron(jnp.ones((lam_K_tilde.size, 1, 1)), V_Sigma_B.T)

        # Reshape to (N_alpha, N_fast - N_alpha, N_slow) for easy vmap
        K_B_nonzero = cyclic_transpose(K_B_nonzero, 1)

        # L_A_inv_K_B should still be (N_alpha, N_fast - N_alpha, N_slow)
        A_inv_sqrt_fn = vmap_for_tensors(kf_A_matrix_inv_sqrt)
        L_A_inv_K_B = A_inv_sqrt_fn(K_B_nonzero, transpose = 0)

        # C_A_inv_B will be of shape (N_alpha, N_alpha, N_slow)
        C_A_inv_B = jnp.einsum('mij,nij->mnj', L_A_inv_K_B, L_A_inv_K_B)
        C_A_inv_B = cyclic_transpose(C_A_inv_B, -1)

        n = jnp.arange(self.N_slow)
        N_total = self.N_slow * self.N_alpha

        C_A_inv_B_dense = jnp.zeros((N_total, N_total))
        C_A_inv_B_dense = C_A_inv_B_dense.reshape(self.N_alpha, self.N_slow, self.N_alpha, self.N_slow).at[:, n, :, n].add(C_A_inv_B).reshape(N_total, N_total)

        return C_A_inv_B_dense
    

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
        # Assumes data residuals pre-rotated to ensure fast dim is second dim

        K_A_kernel = ((Identity(), Identity()),
                      (luas.kernels.Fixed(self.K_transf0_A, ignore_idx = True), self.K_transf[1]),
                     )
        
        # Definitely a better way to do this! Avoids stupid Householder matrix issue
        beta_vec = jnp.ones(self.N_t)
        K_A_kernel += ((luas.kernels.Fixed(0. * self.K_beta_A[j], ignore_idx = True), Outer(beta_vec)),)

        for j in range(self.N_beta):
            # Issue with e_-1 and householder
            beta_vec = jnp.zeros(self.N_t)
            beta_vec = beta_vec.at[-self.N_beta:].set(self.J_B[:, j])
            K_A_kernel += ((luas.kernels.Fixed(self.K_beta_A[j], ignore_idx = True), Outer(beta_vec)),)

        kf_A = LuasPlusMultiTermKernel(*K_A_kernel, eigen_both = True, fast_dim=1)
        # kf_A = GeneralKernel(*K_A_kernel)
        kf_A, stored_values_A = kf_A.decompose((X[0][:-self.N_alpha], X[1]))

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
        
        # Calculates B.T @ A_inv @ B for inverse of D block

        # kf_Ajnp.kron(self.K_transf0_B, jnp.eye(self.N_beta)
        # C_A_inv_B = self._calc_C_A_inv_B(kf_A.matrix_inv_sqrt, self.lam_K_tilde, self.V_K_B, self.V_Sigma_B)

        B_mat_dense = self.kf_B_eval(X, X)
        L_A_inv_B = jnp.zeros_like(B_mat_dense)
        for i in range(B_mat_dense.shape[1]):
            B_vec = B_mat_dense[:, i].reshape((self.N_l - self.N_alpha, self.N_t))
            L_A_inv_B = L_A_inv_B.at[:, i].set(kf_A.matrix_inv_sqrt(B_vec).ravel())
        C_A_inv_B = L_A_inv_B.T @ L_A_inv_B

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
