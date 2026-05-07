import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import CovType, Identity, ScaledIdentity, Diagonal
from luas.kronecker_fns import tensor_mult, calc_total_size

class BlockKernel(CovType):
    def __init__(
        self,
        Sigma,
        *K_list,
        block_dim = 1,
        non_block_dim_size = None,
        use_stored_values: Optional[bool] = True,
    ):
        
        self.Sigma = Sigma
        self.K_list = K_list
        self.block_dim = block_dim
        self.non_block_dim_size = non_block_dim_size

    def evaluate(self, X1, X2, full = True, row_idx = (None, None), col_idx = (None, None), **kwargs):
        assert full == True

        for i in range(self.non_block_dim_size):
            Sigma_eval = self.block_kernels[0].evaluate(X1[self.block_dim], X2[self.block_dim],
                                                        full = full,
                                                        row_idx = row_idx[self.block_dim],
                                                        col_idx = col_idx[self.block_dim],
                                                        **kwargs)

            if self.block_dim == 1:
                K_eval = jnp.kron(jnp.diag(self.lam_arr[:, 0]), Sigma_eval)
            else:
                K_eval = jnp.kron(Sigma_eval, jnp.diag(self.lam_arr[:, 0]))
                                
            for i in range(1, len(self.block_kernels)):
                K_i_eval = self.block_kernels[i].evaluate(X1[self.block_dim], X2[self.block_dim],
                                                          row_idx = row_idx[self.block_dim],
                                                          col_idx = col_idx[self.block_dim],
                                                          full = full, **kwargs)

                if self.block_dim == 1:
                    K_eval += jnp.kron(jnp.diag(self.lam_arr[:, i]), K_i_eval)
                else:
                    K_eval += jnp.kron(K_i_eval, jnp.diag(self.lam_arr[:, i]))

        return K_eval
            
    
    def decompose(self, X, full = True, idx = (None, None), **kwargs):

        self.block_kernels = (self.Sigma[self.block_dim],)

        self.lam_arr = jnp.zeros((self.non_block_dim_size, 1 + len(self.K_list)))
        diag_term_0 = self.Sigma[1-self.block_dim].diag + self.Sigma[1-self.block_dim].wn_diag
        self.lam_arr = self.lam_arr.at[:, 0].set(diag_term_0)

        for i in range(len(self.K_list)):
            self.block_kernels += (self.K_list[i][self.block_dim],)

            diag_term_i = self.K_list[i][1-self.block_dim].diag + self.K_list[i][1-self.block_dim].wn_diag
            self.lam_arr = self.lam_arr.at[:, i+1].set(diag_term_i)

        def matrix_inv_sqrt_calc(r, lam_arr, transpose):

            kf_block = lam_arr[0] * self.block_kernels[0]
            for i in range(1, len(self.block_kernels)):
                kf_block += lam_arr[i] * self.block_kernels[i]

            kf_block, stored_values = kf_block.decompose(X[self.block_dim], idx = idx[self.block_dim])
            return kf_block.matrix_inv_sqrt(r, transpose = transpose), stored_values["logdet"]
        
        def matrix_sqrt_calc(r, lam_arr, transpose):

            kf_block = lam_arr[0] * self.block_kernels[0]
            for i in range(1, len(self.block_kernels)):
                kf_block += lam_arr[i] * self.block_kernels[i]

            kf_block, stored_values = kf_block.decompose(X[self.block_dim], idx = idx[self.block_dim])
            return kf_block.matrix_sqrt(r, transpose = transpose), stored_values["logdet"]
        
        def logL_calc(r, lam_arr):

            kf_block = lam_arr[0] * self.block_kernels[0]
            for i in range(1, len(self.block_kernels)):
                kf_block += lam_arr[i] * self.block_kernels[i]

            kf_block, _ = kf_block.decompose(X[self.block_dim], idx = idx[self.block_dim])
            return kf_block.logL(r)

        self.matrix_inv_sqrt_calc_vmap = jax.vmap(matrix_inv_sqrt_calc, in_axes = (1-self.block_dim, 0, None),
                                                  out_axes = (1-self.block_dim, 0))
        self.matrix_sqrt_calc_vmap = jax.vmap(matrix_sqrt_calc, in_axes = (1-self.block_dim, 0, None),
                                              out_axes = (1-self.block_dim, 0))

        self.logL_vmap = jax.vmap(logL_calc, in_axes = (1-self.block_dim, 0))
        
        # Really need to improve
        R_zero = jnp.zeros((self.non_block_dim_size, X[self.block_dim].shape[-1]))

        if self.block_dim == 0:
            R_zero = R_zero.T
        
        R_prime, logdet_arr = self.matrix_sqrt_calc_vmap(R_zero, self.lam_arr, 0)
        self.logdet = logdet_arr.sum()

        return self, {"logdet":self.logdet}
    
    def matrix_sqrt(self, R, transpose = 0):
        R_prime, logdet_arr = self.matrix_sqrt_calc_vmap(R, self.lam_arr, transpose)

        # self.logdet = logdet_arr.sum()
        return R_prime
    
    def matrix_inv_sqrt(self, R, transpose = 0):
        R_prime, logdet_arr = self.matrix_inv_sqrt_calc_vmap(R, self.lam_arr, transpose)

        # self.logdet = logdet_ar.sum()
        return R_prime
        
    def scale(self, c):
        raise Exception("Not implemented")

    # def matmul(self, x1, x2, R, wn = True, **kwargs):

    #     R_prime = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

    #     for i in range(len(self.K_list)):
    #         R_prime += tensor_mult(self.K_list[i], X1, X2, R, **kwargs)
        
    #     return R_prime
    

class Block2x2Kernel(CovType):
    def __init__(self, kf_A, kf_B = None, kf_D_CAB = None, kf_D = None, A_full = True, D_full = True,
                 dim_split = 0, split_loc = None, split_idx = False, kf_B_eval = None):
        self.kf_A = kf_A
        self.kf_B_matmul = kf_B
        self.kf_D_CAB = kf_D_CAB
        self.kf_D = kf_D
        self.kf_B_eval = kf_B_eval

        self.dim_split = dim_split
        self.split_loc = split_loc
        self.split_idx = split_idx
        self.A_full = A_full
        self.D_full = D_full

        if self.kf_D is None and self.kf_B_matmul is None:
            self.kf_D = self.kf_D_CAB


    def evaluate(self, x1, x2, wn = True, full = True, row_idx = None, col_idx = None, **kwargs):

        x1_A, x1_D, row_idx_A, row_idx_D = self.x_split(x1, idx = row_idx)
        x2_A, x2_D, col_idx_A, col_idx_D = self.x_split(x2, idx = col_idx)
        
        if self.kf_B_matmul is None:
            zeros_fn = lambda x1, x2, R, **kwargs: jnp.zeros(x1_A.shape[-1], x2_D.shape[-1])
            self.kf_B_matmul = zeros_fn
        else:
            assert self.kf_B_eval is not None

        A_mat = self.kf_A.evaluate(x1_A, x2_A)

        if self.kf_B_eval is not None:
            B_mat = self.kf_B_eval(x1_A, x2_D)

        C_A_inv_B = B_mat.T @ jnp.linalg.inv(A_mat) @ B_mat
        if self.kf_D is not None:
            D_mat = self.kf_D.evaluate(x1_D, x2_D, full = self.D_full)
        else:
            D_mat = self.kf_D_CAB.evaluate(x1_D, x2_D, full = self.D_full) + C_A_inv_B

        calc_C_A_inv_B = self.kf_D.evaluate(x1_D, x2_D, full = self.D_full)
        calc_C_A_inv_B -= self.kf_D_CAB.evaluate(x1_D, x2_D, full = self.D_full)
        print("C_A_inv_B:", calc_C_A_inv_B, C_A_inv_B, (calc_C_A_inv_B - C_A_inv_B).std())
        self.dense_C_A_inv_B = C_A_inv_B

        if self.dim_split == 0 or self.dim_split == 1:
            top_rows = jnp.concatenate([A_mat, B_mat], axis = 1)
            bottom_rows = jnp.concatenate([B_mat.T, D_mat], axis = 1)
        # else:

        #     raise Exception("Not implemented")
        
        return jnp.concatenate([top_rows, bottom_rows], axis = 0)
    

    def x_split(self, X, idx = None):
        x_A = X[self.dim_split][..., :self.split_loc]
        x_D = X[self.dim_split][..., self.split_loc:]

        if self.dim_split == 0:
            X_A = (x_A, X[1])
            X_D = (x_D, X[1])

            if idx is None:
                idx_A = (jnp.arange(self.split_loc), jnp.arange(X[1].shape[-1]))
                idx_D = (jnp.arange(self.split_loc, X[0].shape[-1]), jnp.arange(X[1].shape[-1]))
            else:
                idx_A = (idx[0][:self.split_loc], idx[1])
                idx_D = (idx[0][self.split_loc:], idx[1])
        else:
            X_A = (X[0], x_A)
            X_D = (X[0], x_D)

            if idx is None:
                idx_A = (jnp.arange(X[0].shape[-1]), jnp.arange(self.split_loc))
                idx_D = (jnp.arange(X[0].shape[-1]), jnp.arange(self.split_loc, X[1].shape[-1]))
            else:
                idx_A = (idx[0], idx[1][:self.split_loc])
                idx_D = (idx[0], idx[1][self.split_loc:])

        if not self.split_idx:
            idx_A = None
            idx_D = None
        
        return X_A, X_D, idx_A, idx_D


    def decompose(self, X, idx = None, full = True, **kwargs):

        X_A, X_D, idx_A, idx_D = self.x_split(X, idx = idx)

        self.kf_A, stored_values_A = self.kf_A.decompose(X_A, idx = idx_A, full = self.A_full, **kwargs)

        if self.kf_D_CAB is not None:
            self.kf_D_CAB, stored_values_D = self.kf_D_CAB.decompose(X_D, idx = idx_D, full = self.D_full)
        else:
            raise Exception("Not Implemented")

        if self.kf_B_matmul is not None:
            self.B_mult = lambda R, **kwargs: self.kf_B_matmul(X_A, X_D, R, **kwargs)
            self.C_mult = lambda R, **kwargs: self.kf_B_matmul(X_A, X_D, R, transpose = 1, **kwargs)

        self.logdet = stored_values_A["logdet"] + stored_values_D["logdet"]
        
        return self, {"logdet":self.logdet}
        
    def matrix_inv_sqrt(self, R, transpose=0):

        if self.dim_split == 0:
            R_A = R[:self.split_loc, :]
            R_D = R[self.split_loc:, :]
        elif self.dim_split == 1:
            R_A = R[:, :self.split_loc]
            R_D = R[:, self.split_loc:]

        if self.kf_B_matmul is not None:
            if transpose == 0:
                R_prime_A = self.kf_A.matrix_inv_sqrt(R_A, transpose = 0)
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

        X1_A, X1_D, idx_D1 = self.x_split(X1)
        X2_A, X2_D, idx_D2 = self.x_split(X2)

        if self.kf_B_matmul is None:
            K_R_A = self.kf_A.matmul(X1_A, X2_A, R_A)
            K_R_D = self.kf_D.matmul(X1_D, X2_D, R_D)
        else:
            K_R_A = self.kf_A.matmul(X1_A, X2_A, R_A) + self.kf_B_matmul(X1_A, X2_D, R_D)
            K_R_D = self.kf_B_matmul(X1_D, X2_A, R_A) + self.kf_D.matmul(X1_D, X2_D, R_D,
                                                                         row_idx = idx_D1, col_idx = idx_D2)

        K_R = jnp.concatenate([K_R_A, K_R_D], axis = self.dim_split)
        
        return K_R
    

