import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional

from luas.kernels.covtype import Outer, Exp, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal, General
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, is_scalar
from luas.kronecker_fns import tensor_mult, vmap_for_tensors, cyclic_transpose
from luas.kernels.householder import orthonormal_nullspace_gen
from luas.kernels.BlockKernel import Block2x2Kernel, BlockKernel
from luas.kernels.GeneralKernel import GeneralKernel
from luas.kernels.LuasKernel import LuasKernel

__all__ = [
    "LuasPlusMultiTermKernel",
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

class LuasPlusMultiTermKernel(CovType):
    
    def __init__(
        self,
        Sigma,
        *K_list,
        fast_dim = 1,
        never_reduce_dim = False,
        use_stored_values: Optional[bool] = True,
        transform = True,
        transform_fn = None,
        inv_transform_fn = None,
        eigen_both = False,
        use_pmap = False,
        **kwargs,
    ):
        
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.fast_dim = fast_dim
        self.eigen_both = eigen_both
        self.transform = transform
        self.use_pmap = use_pmap

        if transform_fn is not None:
            self.transform_fn = transform_fn
        else:
            self.transform_fn = self.default_transform_fn

        if inv_transform_fn is not None:
            self.inv_transform_fn = inv_transform_fn
        else:
            self.inv_transform_fn = self.default_inv_transform_fn

        self.logL_hessianable = self.logL
        self.decompose = self.decomp_no_stored_values


    def _rotate_to_fast_dim_wrapper(self, fn, fast_dim):

        def wrapped_fn(R, **kwargs):
            if fast_dim == 1:
                R_prime = R.T
            else:
                R_prime = R.copy()

            R_prime = fn(R_prime, **kwargs)
            if fast_dim == 1:
                R_prime = R_prime.T
            return R_prime

        return wrapped_fn
    

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
        full = True,
        idx = (None, None),
    ) -> PyTree:

        self.N_slow = X[1-self.fast_dim].shape[-1]
        self.N_fast = X[self.fast_dim].shape[-1]
        self.N_alpha = len(self.K_list) - 1

        dense_kron, *low_rank_terms = read_K_list(self.K_list, X)
        alpha_mat, kernel_list = low_rank_terms[self.fast_dim]

        # Required for this optimisation to work
        reduce_dim = self.N_alpha < X[self.fast_dim].shape[-1]
        assert reduce_dim

        # Check low rank terms have correct number of terms and dimensions
        assert alpha_mat.shape[-1] == len(kernel_list) == self.N_alpha

        if self.transform:
            assert alpha_mat.shape[0] == self.N_fast
        else:
            assert alpha_mat.shape[0] == self.N_alpha

        # Check only have low rank terms in one dimension
        assert low_rank_terms[1-self.fast_dim] is None


        ######### Do transforms in slow_dim #########

        if self.transform:
            # Decompose slow Sigma mat and add its contribution to the log determinant
            self.Sigma_slow, stored_values_Sigma_slow = self.Sigma[1-self.fast_dim].decompose(X[1-self.fast_dim])
            stored_values["logdet"] = self.N_fast*stored_values_Sigma_slow["logdet"]
            
            dense_kron[1-self.fast_dim] = self.Sigma_slow.inv_sqrt_transform(dense_kron[1-self.fast_dim],
                                                                            X[1-self.fast_dim])

            # Take eigendecomp of slow_dim K_tilde
            self.lam_K_tilde, self.Q_K_slow_tilde = dense_kron[1-self.fast_dim].eigendecomp(X[1-self.fast_dim],
                                                                                            full = full,
                                                                                            idx = idx[1-self.fast_dim])

            # Transform all kernels in kernel_list with simultaneous diagonalisation transforms in slow_dim
            for i in range(self.N_alpha):
                kernel_list[i] = self.Sigma_slow.inv_sqrt_transform(kernel_list[i], X[1-self.fast_dim])
                kernel_list[i] = kernel_list[i].general_transf(self.Q_K_slow_tilde.T, X[1-self.fast_dim])


            ######### Do transforms in fast_dim #########
            # Get Householder transform of all Outer covariances
            stored_values["J_A"], stored_values["U_A"], self.householder_transform = orthonormal_nullspace_gen(alpha_mat,
                                                                                                            reverse = True)
            
            # Now must transform fast covariance matrices with Householder transform i.e. H K H.T, H Sigma H.T
            K_fast_tilde = dense_kron[self.fast_dim]
            Sigma_fast_tilde = self.Sigma[self.fast_dim]
            for i in range(self.N_alpha):
                K_fast_tilde = K_fast_tilde.householder_transform(X[self.fast_dim], stored_values["U_A"][:, i])
                Sigma_fast_tilde = Sigma_fast_tilde.householder_transform(X[self.fast_dim], stored_values["U_A"][:, i])
        
        else:
            # Pre-transformed, e.g. called from LuasPlusMultiTermBothDimKernel
            # Should have inputs (Identity, Sigma_fast_tilde), (lam_K_tilde, K_fast_tilde), *(J_alpha_i, kernel_list[i])
            stored_values["logdet"] = 0.
            stored_values["J_A"] = alpha_mat
            self.lam_K_tilde = dense_kron[1-self.fast_dim].diag
            Sigma_fast_tilde = self.Sigma[self.fast_dim] # Should just be the identity
            K_fast_tilde = dense_kron[self.fast_dim]
            

        ######### Build A Block #########
        # Assumes data residuals pre-rotated to ensure fast dim is second dim
        K_A_kernel = ((Sigma_fast_tilde, Identity()),
                    (K_fast_tilde, Diagonal(diag = self.lam_K_tilde))
                    )

        # A block consists of a Block Kernel which could be handled through LuasLasrachKernel or LuasKernel
        if self.eigen_both:
            kf_A = LuasKernel(*K_A_kernel)
                
            kf_A, kf_A_stored_values = kf_A.decompose((X[self.fast_dim][:-self.N_alpha], X[1-self.fast_dim]),
                                                            full = False, idx = (jnp.arange(self.N_fast - self.N_alpha), jnp.arange(self.N_slow))
            )
            # Use for LuasPlusMultiTermBothDimKernel
            self.lam_fast_A, self.Q_fast_A = kf_A_stored_values["lam_0"], kf_A_stored_values["Q_0"]
        else:
            kf_A = BlockKernel(*K_A_kernel, non_block_dim_size = self.N_slow, block_dim=0, use_pmap = self.use_pmap)
            
            kf_A, kf_A_stored_values = kf_A.decompose((X[self.fast_dim][:-self.N_alpha], X[1-self.fast_dim]),
                                                            full = False, 
                                                            idx = (jnp.arange(self.N_fast - self.N_alpha), jnp.arange(self.N_slow)))



        ######### Build B and B.T block #########
        # Evaluate these fast covariance matrices within the B, B.T and D blocks
        V_K = K_fast_tilde.evaluate(X[self.fast_dim], X[self.fast_dim][-self.N_alpha:], full = False,
                                    row_idx = jnp.arange(self.N_fast),
                                    col_idx = jnp.arange(self.N_fast-self.N_alpha, self.N_fast))
        V_Sigma = Sigma_fast_tilde.evaluate(X[self.fast_dim], X[self.fast_dim][-self.N_alpha:], full = False,
                                    row_idx = jnp.arange(self.N_fast),
                                    col_idx = jnp.arange(self.N_fast-self.N_alpha, self.N_fast))
        
        # Split these matrices between the B and D blocks (D block is only the last N_alpha rows and columns)
        self.V_K_B, self.V_K_D = V_K[:-self.N_alpha, :], V_K[-self.N_alpha:, :]
        self.V_Sigma_B, self.V_Sigma_D = V_Sigma[:-self.N_alpha, :], V_Sigma[-self.N_alpha:, :]
        

        ######### Build D block #########
        K_alpha = ()
        for i in range(self.N_alpha):
            K_alpha += ((Outer(stored_values["J_A"][:, i]), kernel_list[i]),)
        
        K_D_dense = jnp.kron(self.V_Sigma_D, jnp.eye(self.N_slow))
        K_D_dense += jnp.kron(self.V_K_D, jnp.diag(self.lam_K_tilde))
        
        # Calculates B.T @ A_inv @ B for inverse of D block
        self.C_A_inv_B = self._calc_C_A_inv_B(kf_A.matrix_inv_sqrt, self.lam_K_tilde, self.V_K_B, self.V_Sigma_B)

        kf_D_CAB = GeneralKernel(*K_alpha, dense_mat = K_D_dense - self.C_A_inv_B)
        kf_D = GeneralKernel(*K_alpha, dense_mat = K_D_dense)


        ######### Define full 2x2 Block kernel and decompose #########
        self.kf_tilde = Block2x2Kernel(kf_A = kf_A, kf_B = self.K_B_matmul, kf_D_CAB = kf_D_CAB, kf_D = kf_D,
                                       dim_split = 0, split_loc = self.N_fast-self.N_alpha,
                                       split_idx = True, kf_B_eval = self.kf_B_eval, A_full = False, D_full = True)
        
        self.kf_tilde.matrix_sqrt = self._rotate_to_fast_dim_wrapper(self.kf_tilde.matrix_sqrt, self.fast_dim)
        self.kf_tilde.matrix_inv_sqrt = self._rotate_to_fast_dim_wrapper(self.kf_tilde.matrix_inv_sqrt, self.fast_dim)
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose((X[self.fast_dim], X[1-self.fast_dim]),
                                                                                  full = False)

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
            R_prime = self.V_K_B.T @ R 
            R_prime = R_prime * self.lam_K_tilde
            R_prime += self.V_Sigma_B.T @ R
        else:
            R_prime = self.V_K_B @ R 
            R_prime = R_prime * self.lam_K_tilde
            R_prime += self.V_Sigma_B @ R
        
        return R_prime
    
    def kf_B_eval(self, X1, X2):
        return jnp.kron(self.V_K_B, jnp.diag(self.lam_K_tilde)) + jnp.kron(self.V_Sigma_B, jnp.eye(self.N_slow))
    

    def default_transform_fn(self, R, transpose = 0):

        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()
        
        if transpose:
            R_prime = self.Sigma_slow.matrix_sqrt(R_prime, transpose = 1)
            R_prime = self.Q_K_slow_tilde.T @ R_prime
        else:
            R_prime = self.Q_K_slow_tilde @ R_prime
            R_prime = self.Sigma_slow.matrix_sqrt(R_prime, transpose = 0)

        R_prime = R_prime.T
        R_prime = self.householder_transform(R_prime, transpose = 1-transpose)

        if self.fast_dim == 1:
            R_prime = R_prime.T
        
        return R_prime


    def default_inv_transform_fn(self, R, transpose = 0):

        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()
        
        if transpose:
            R_prime = self.Q_K_slow_tilde @ R_prime
            R_prime = self.Sigma_slow.matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_slow.matrix_inv_sqrt(R_prime, transpose = 0)
            R_prime = self.Q_K_slow_tilde.T @ R_prime

        R_prime = R_prime.T
        R_prime = self.householder_transform(R_prime, transpose = transpose)

        if self.fast_dim == 1:
            R_prime = R_prime.T
        
        return R_prime


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
