import numpy as np
from copy import deepcopy
from typing import Optional, Callable, Tuple, Any, Union
import scipy
import warnings

import jax
from jax import grad, value_and_grad, hessian, vmap, tree_util
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

import tinygp

from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, CovType, is_scalar
from luas.kronecker_fns import make_vec, make_mat, cyclic_transpose, vmap_for_tensors
from luas.jax_convenience_fns import array_to_pytree_2D

    
class CovType():
    K_list = []
    params = None

    def evaluate(self, x1, x2, **kwargs):
        raise Exception("Not implemented")
    
    def decompose(self, x, **kwargs):
        raise Exception("Not implemented")

    def matrix_inv_sqrt(self, R, transpose=0):
        raise Exception("Not implemented")

    def matrix_sqrt(self, R, transpose=0):
        raise Exception("Not implemented")
    
    def eigendecomp(self, x, **kwargs):
        
        K = self.evaluate(x, x, **kwargs)
        return jnp.linalg.eigh(K)
    
    def dot_solve(self, R):
        
        L_inv_R = self.matrix_inv_sqrt(R, transpose = 0)
        return jnp.square(L_inv_R).sum()

    
    def dot_solve_w_inverse(self, R):

        K_inv_R = self.inverse(R)
            
        return (R * K_inv_R).sum()

        
    def logL(self, R, stored_values, **kwargs):
        
        return - 0.5 * self.dot_solve(R) - 0.5 * stored_values["logdetK"] - 0.5 * R.size * jnp.log(2*jnp.pi)

    
    def inverse(self, R):

        R_prime = self.matrix_inv_sqrt(R, transpose=0)
        K_inv_R = self.matrix_inv_sqrt(R_prime, transpose=1)

        return K_inv_R

    def inv_sqrt_transform(self, K):
        
        if isinstance(K, Outer):
            alpha_tilde = self.matrix_inv_sqrt(K.alpha, transpose=0)
            K_tilde = Outer(alpha = alpha_tilde)
        else:
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(hp, x1, x2, **kwargs):
                K_eval = K.evaluate(x1, x2, **kwargs)
                K_prime = self.matrix_inv_sqrt(K_eval, transpose=0)
                K_tilde = self.matrix_inv_sqrt(K_prime.T, transpose=0)
                
                return K_tilde

            K_tilde = General(kf_transf)

        return K_tilde
    
    def scale(self, c):
        return General(kf = lambda hp, x1, x2, **kwargs: c*self.evaluate(x1, x2, **kwargs))
        

    def matmul(self, x1, x2, other, **kwargs):
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) @ other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return self.evaluate(x1, x2, **kwargs) @ other
        else:
            raise Exception("Not implemented")
    
    def rank(self, x):
        return x.shape[-1]
        
    
    def __add__(self, other):
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray) or is_scalar(other):
            return General(kf = lambda hp, x1, x2, **kwargs: other + other.evaluate(x1, x2, **kwargs))
        else:
            raise Exception("Not implemented")

    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        elif is_scalar(other):
            return self.scale(other)
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other)
        else:
            raise Exception("Not implemented")

    def __radd__(self, other):
        return self.__add__(other)
        
    def __rmul__(self, other):
        return self.__mul__(other)



class General(CovType):
    def __init__(self, kf, hp = {}, diag = 0., wn_diag = 0., params = None):
        self.kf = kf
        self.hp = hp
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params
        
    def evaluate(self, x1, x2, wn = True, **kwargs):
        if x1.shape == x2.shape:
            return self.kf(self.hp, x1, x2, wn = wn, **kwargs) + jnp.diag((self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1]))
        else:
            return self.kf(self.hp, x1, x2, wn = wn, **kwargs)

    def decompose(self, x, **kwargs):
        # By default returns the lower triangular Cholesky factor
        # i.e. K = L @ L.T
        # Matches with tinygp default but not scipy or jax.scipy default
        
        K = self.evaluate(x, x, **kwargs)
        
        self.factor = JLA.cholesky(K, lower=True)
        logdetK = 2*jnp.log(jnp.diag(self.factor)).sum()
        
        return self, {"logdetK":logdetK}

    def matrix_inv_sqrt(self, R, transpose=0, **kwargs):

        R_prime = jax.scipy.linalg.solve_triangular(self.factor, R, trans=transpose, lower=True)

        return R_prime

    
    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R



class Identity(CovType):

    def __init__(self):
        self.diag = 1.
        self.wn_diag = 0.
        self.params = []
    
    def evaluate(self, x1, x2, **kwargs):
        return jnp.diag(1.*jnp.isclose(x1, x2))
    
    def decompose(self, x, **kwargs):
        return self, {"logdetK":0.}

    def matrix_inv_sqrt(self, R, transpose=0):
        return R
        
    def matrix_sqrt(self, R, **kwargs):
        return R
    
    def eigendecomp(self, x, **kwargs):

        return jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])
    
    def inverse(self, R, **kwargs):
        return R
        
    def dot_solve(self, R):

        return jnp.square(R).sum()
        
    def inv_sqrt_transform(self, K, **kwargs):

        return K

    def scale(self, c):
        return ScaledIdentity(diag = c)

    def matmul(self, x1, x2, R, **kwargs):

        return R
    
    def __add__(self, K):

        if isinstance(K, Exp) or isinstance(K, GeneralQuasisep):
            K_sum = GeneralQuasisep(K.tinygp_kf, diag = K.diag + 1., wn_diag = K.wn_diag)
        elif isinstance(K, Outer):
            K_sum = OuterPlusScaledIdentity(K.alpha_init, diag = 1.)
        elif isinstance(K, OuterPlusScaledIdentity):
            K_sum = OuterPlusScaledIdentity(K.alpha, diag = 1. + K.diag, wn_diag = K.wn_diag)
        elif isinstance(K, Identity):
            K_sum = ScaledIdentity(diag = 2.)
        elif isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
            K_sum = type(K)(diag = K.diag + 1., wn_diag = K.wn_diag)
        elif isinstance(K, General):
            K_sum = General(lambda hp, x1, x2, **kwargs: K.evaluate(x1, x2, **kwargs), diag = 1.)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

        
        
        
class ScaledIdentity(CovType):
    def __init__(self, diag = 0., wn_diag = 0., params = None):
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params

        assert is_scalar(self.diag)
        assert is_scalar(self.wn_diag)
        
    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        D = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        
        return jnp.diag(D)

    def decompose(self, x, wn = True, **kwargs):

        self.D = self.diag + wn*self.wn_diag
        self.factor = jnp.sqrt(self.D)
        
        return self, {"logdetK":x.shape[-1]*jnp.log(self.D)}

    def matrix_inv_sqrt(self, R, **kwargs):

        return R/self.factor

    def matrix_sqrt(self, R, **kwargs):

        return self.factor * R
        
    def eigendecomp(self, x, wn = True, **kwargs):

        const_diag = self.diag + wn*self.wn_diag
        
        return const_diag*jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])

    def inv_sqrt_transform(self, K):

        if isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
            K_tilde = type(K)(diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)

        elif isinstance(K, Identity):
            K_tilde = ScaledIdentity(diag = 1/self.D)

        elif isinstance(K, Outer):
            K_tilde = Outer(alpha = K.alpha_init/self.factor)
            
        elif isinstance(K, GeneralQuasisep):
            K_tilde = GeneralQuasisep(K.tinygp_kf * (1/self.D), diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)
            
        elif isinstance(K, Exp):
            K_tilde = Exp(K.l, sigma = (1/self.factor) * K.sigma)
        else:
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(hp, x1, x2, **kwargs):
                
                return (1/self.D) * K.evaluate(x1, x2, **kwargs)

            K_tilde = General(kf_transf)
            
        return K_tilde

    def matmul(self, x1, x2, R, wn = True, **kwargs):

        const = self.diag + wn*self.wn_diag

        return const * R

    def scale(self, c):
        
        return ScaledIdentity(diag = self.diag * c, wn_diag = self.wn_diag * c)

    def __add__(self, K):

        if isinstance(K, Exp) or isinstance(K, GeneralQuasisep):
            K_sum = GeneralQuasisep(K.tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, Identity) or isinstance(K, ScaledIdentity):
            K_sum = ScaledIdentity(diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, Diagonal):
            K_sum = Diagonal(diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, Outer):
            K_sum = OuterPlusScaledIdentity(K.alpha_init, diag = self.diag, wn_diag = self.wn_diag)
        elif isinstance(K, OuterPlusScaledIdentity):
            K_sum = OuterPlusScaledIdentity(K.alpha, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, General):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs), diag = self.diag, wn_diag = self.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum


class Diagonal(CovType):
    def __init__(self, diag = 0., wn_diag = 0.):
        self.diag = diag
        self.wn_diag = wn_diag

        assert is_scalar(self.diag) or self.diag.ndim < 2
        assert is_scalar(self.wn_diag) or self.wn_diag.ndim < 2
    
    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        D = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        
        return jnp.diag(D)
    
    def decompose(self, x, wn = True, **kwargs):

        self.D = (self.diag + wn*self.wn_diag)*jnp.ones(x.shape[-1])
        self.factor = jnp.sqrt(self.D)
        logdetK = jnp.log(self.D).sum()
        stored_values = {"logdetK":logdetK}
        return self, stored_values
    
    def matrix_sqrt(self, R, **kwargs):
        
        return (self.factor * R.T).T
        
    def matrix_inv_sqrt(self, R, transpose=0):
        
        D_inv_sqrt = 1/self.factor

        return (D_inv_sqrt * R.T).T
    
    def eigendecomp(self, x, wn = True, **kwargs):

        D = (self.diag + wn*self.wn_diag)*jnp.ones(x.shape[-1])
        
        return D, jnp.eye(x.shape[-1])
    
    
    def inv_sqrt_transform(self, K):

        if isinstance(K, Diagonal):
            K_tilde = Diagonal(diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)

        elif isinstance(K, Identity):
            K_tilde = Diagonal(diag = 1/self.D)

        elif isinstance(K, Outer):
            K_tilde = Outer(alpha = K.alpha_init/jnp.sqrt(self.D))

        else:
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(hp, x1, x2, **kwargs):
                
                D_inv_sqrt = 1/jnp.sqrt(self.D)
                return jnp.outer(D_inv_sqrt, D_inv_sqrt) * K.evaluate(x1, x2, **kwargs)

            K_tilde = General(kf_transf)
            
        return K_tilde

    
    def inverse(self, R, transpose=0):
        
        D_inv = 1/self.D
        
        return (D_inv * R.T).T


    def matmul(self, x1, x2, R, wn = True, **kwargs):

        D = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        return (D * R.T).T

    def scale(self, c):
        return Diagonal(diag = self.diag * c, wn_diag = self.wn_diag * c)


    def __add__(self, K):

        if isinstance(K, Exp) or isinstance(K, GeneralQuasisep):
            K_sum = GeneralQuasisep(K.tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
            K_sum = Diagonal(diag = K.diag + self.diag, wn_diag = K.wn_diag + self.wn_diag)
        elif isinstance(K, Outer) or isinstance(K, OuterPlusScaledIdentity) or isinstance(K, General): # or isinstance(K, OuterPlusScaledIdentity):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum


class GeneralQuasisep(CovType):
    def __init__(self, tinygp_kf, diag = 0., wn_diag = 0., params = None):
        self.tinygp_kf = tinygp_kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params

    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        return self.tinygp_kf(x1, x2) + jnp.diag((self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1]))
    
    def decompose(self, x, wn = True, **kwargs):

        diag = (self.diag + wn*self.wn_diag)*jnp.ones(x.shape[-1])
        noise_model = tinygp.noise.Diagonal(diag=diag)
        matrix = self.tinygp_kf.to_symm_qsm(x)
        matrix += noise_model.to_qsm()
        self.factor = matrix.cholesky()
        
        logdetK = 2*jnp.sum(jnp.log(self.factor.diag.d))
        
        return self, {"logdetK":logdetK}
        
    def matrix_inv_sqrt(self, R, transpose=0):

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime

    def matrix_sqrt(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R

        
    def logL(self, R, stored_values, **kwargs):
        
        return -0.5 * self.dot_solve(R) - 0.5 * stored_values["logdetK"] - 0.5 * R.size * jnp.log(2*jnp.pi)
        

    def eigendecomp(self, x, **kwargs):

        K_eval = self.evaluate(x, x, **kwargs)
        
        return jnp.linalg.eigh(K_eval)


    def scale(self, c):
        return GeneralQuasisep(self.tinygp_kf * c, diag = self.diag * c, wn_diag = self.wn_diag * c)

    def matmul(self, x1, x2, R, wn = True, **kwargs):

        # assert x1 == x2
        
        R_prime = self.tinygp_kf.matmul(x1, x2, R)

        diag = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        noise_model = tinygp.noise.Diagonal(diag=diag)
        R_prime += noise_model @ R
        
        return R_prime
        
    def __add__(self, K):

        if isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
            K_sum = GeneralQuasisep(self.tinygp_kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, Exp) or isinstance(K, GeneralQuasisep):
            K_sum = GeneralQuasisep(K.tinygp_kf + self.tinygp_kf,
                                    diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, Outer) or isinstance(K, General) or isinstance(K, OuterPlusScaledIdentity):
            # Should update as don't lose quasiseparability
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum
        
    # Should implement multiplying celerite kernels together but diagonal is tricky!
    # def __mul__(self, other) -> Kernel:
    #     if type(other) in [Exp, Quasisep]:
    #         K_sum = GeneralQuasisep(self.tinygp_kf * K.tinygp_kf, diag = self.diag * K.diag, wn_diag = self.wn_diag + K.wn_diag)
    #     elif isinstance(other, CovType):
        
    #     return self.scale(other)
        
    
class Exp(CovType):
    def __init__(self, l, sigma = 1., params = None):
        self.l = l
        self.sigma = sigma
        self.tinygp_kf = tinygp.kernels.quasisep.Exp(scale = self.l, sigma = self.sigma)
        self.diag = 0.
        self.wn_diag = 0.
        self.params = params
    
    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        return self.tinygp_kf(x1, x2)
    
    def decompose(self, x, **kwargs):
        
        matrix = self.tinygp_kf.to_symm_qsm(x)
        self.factor = matrix.cholesky()
        logdetK = 2*jnp.sum(jnp.log(self.factor.diag.d))
        return self, {"logdetK":logdetK}
        
    def matrix_sqrt(self, R, transpose = 0):

        if transpose:
            return self.transpose().factor @ R
        else:
            return self.factor @ R

    def matrix_inv_sqrt(self, R, transpose=0):

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime
        
    def eigendecomp(self, x, **kwargs):
        
        N = x.shape[-1]
        result_shape = (jax.ShapeDtypeStruct((N,), x.dtype), jax.ShapeDtypeStruct((N, N), x.dtype))
        lam, Q = jax.pure_callback(self.fast_exp_eigh_scipy, result_shape, x, self.l)
        return self.sigma**2 * lam, Q
        
    def fast_exp_eigh_scipy(self, x1, scale):
    
        if x1.ndim == 2:
            x1 = x1[0, :]
            
        r = np.exp(-np.diff(x1/scale))
        e = 1/(1/r - r)
        
        arr_diag = np.zeros(x1.shape[-1])
        arr_diag[:-1] = 1 + r*e
        arr_diag[1:] += r*e
        arr_diag[-1] += 1
        
        lam, Q = scipy.linalg.eigh_tridiagonal(arr_diag, -e)
        
        return 1/lam, Q
    
    def matmul(self, x1, x2, R, **kwargs):

        return self.tinygp_kf.matmul(x1, x2, R)

    def scale(self, c):
        return Exp(self.l, sigma = self.sigma * jnp.sqrt(jnp.abs(c)))
        
    def __add__(self, K):

        if isinstance(K, Identity) or isinstance(K, ScaledIdentity) or isinstance(K, Diagonal):
            K_sum = GeneralQuasisep(self.tinygp_kf, diag = K.diag, wn_diag = K.wn_diag)
        elif isinstance(K, Exp) or isinstance(K, GeneralQuasisep):
            K_sum = GeneralQuasisep(K.tinygp_kf + self.tinygp_kf, diag = K.diag, wn_diag = K.wn_diag)

        # This could be improved
        elif isinstance(K, Outer):
            K_sum = General(lambda *args, **kwargs: self.kf(*args) + jnp.outer(self.alpha, self.alpha))
        elif isinstance(K, General) or isinstance(K, OuterPlusScaledIdentity):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2) + K.evaluate(x1, x2, **kwargs))
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    # def __mul__(self, other) -> Kernel:
    #     if isinstance(other, CovType) or jnp.ndim(other) != 0:
    #         raise ValueError(
    #             "Can't multiply Identity kernel with other kernels"
    #         )
    #     return self.scale(other)



class Outer(CovType):
    def __init__(self, alpha, params = None):
        self.alpha_init = alpha
        self.diag = 0.
        self.wn_diag = 0.
        self.params = params

        assert is_scalar(self.alpha_init) or isinstance(self.alpha_init, jax.Array) or isinstance(self.alpha_init, np.ndarray)

    def rank(self, x):
        return 1
        
    def evaluate(self, x1, x2, **kwargs):
        return jnp.outer(self.alpha_init * jnp.ones(x1.shape[-1]), self.alpha_init * jnp.ones(x2.shape[-1]))

    def decompose(self, x, **kwargs):
        # Matrix is decomposed by definition
        # Except need to handle ConstantKernel where self.alpha_init is a float

        self.alpha = self.alpha_init * jnp.ones(x.shape[-1])
        return self, {"logdetK":-jnp.inf}
    
    def matrix_sqrt(self, R, **kwargs):

        if R.ndim == 1:
            return jnp.kron(self.alpha, self.alpha @ R)
        elif R.ndim == 2:
            return jnp.outer(self.alpha, self.alpha @ R)
        else:
            raise Exception("Not implemented")

    def matrix_inv_sqrt(self, R, **kwargs):

        return Exception("This matrix is not invertible!")
        
    def dot_solve(self, R):
        
        return Exception("This matrix is not invertible!")
        
    def logL(self, R, stored_values, **kwargs):
        
        return Exception("This matrix is not invertible!")

    def inv_sqrt_transform(self, K):

        return Exception("This matrix is not invertible!")

    def eigendecomp(self, x, i = -1, **kwargs):
        # Calculates the Householder transformation of alpha for the eigenvector matrix
        # And the matrix's sole non-zero eigenvalue which is the squared norm of alpha

        H = HouseholderTransform(self.alpha_init * jnp.ones(x.shape[-1]), i = i)
        
        return H.lam, H
    
    def matmul(self, x1, x2, other, **kwargs):
        self.vec1 = self.alpha_init * jnp.ones(x1.shape[-1])
        self.vec2 = self.alpha_init * jnp.ones(x2.shape[-1])
        
        if other.ndim == 1:
            return jnp.kron(self.vec1, self.vec2 @ other)
        elif other.ndim == 2:
            return jnp.outer(self.vec1, self.vec2 @ other)
        else:
            raise Exception("Not implemented")
            
    def scale(self, c):
        return Outer(self.alpha_init * jnp.sqrt(c))

        
    def __add__(self, K):

        if isinstance(K, Identity) or isinstance(K, ScaledIdentity):
            K_sum = OuterPlusScaledIdentity(self.alpha_init, diag = K.diag, wn_diag = K.wn_diag)
        elif isinstance(K, General) or isinstance(K, Outer):
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))
        elif isinstance(K, GeneralQuasisep) or isinstance(K, Exp) or isinstance(K, Diagonal) or isinstance(K, OuterPlusScaledIdentity):
            # Should update as you can always add outer products without losing quasiseparability
            K_sum = General(lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    # Could implement an efficient Quasisep multiplication
    # def __mul__(self, other) -> Kernel:
    #     if isinstance(other, CovType) or jnp.ndim(other) != 0:
    #         raise ValueError(
    #             "Can't multiply kernel with other kernels"
    #         )
    #     return self.scale(other)


class OuterPlusScaledIdentity(CovType):
    def __init__(self, alpha, diag = 0., wn_diag = 0., params = None):
        self.alpha = alpha
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params

        assert isinstance(self.alpha, jax.Array) or isinstance(self.alpha, np.ndarray)
        assert is_scalar(self.diag)
        assert is_scalar(self.wn_diag)
    
    def evaluate(self, x1, x2, wn = True, **kwargs):
        if x1.shape == x2.shape:
            D = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        else:
            raise Exception("Not implemented")
            
        rank1_mat = jnp.outer(self.alpha * jnp.ones(x1.shape[-1]), self.alpha * jnp.ones(x2.shape[-1]))
        
        return rank1_mat + jnp.diag(D)

    def decompose(self, x, wn = True, i = -1, **kwargs):

        self.H = HouseholderTransform(self.alpha, i = i)
        self.lam = self.H.lam + (self.diag + wn * self.wn_diag)
        
        logdetK = jnp.log(self.lam).sum()
        return self, {"logdetK":logdetK}

        
    def matrix_inv_sqrt(self, R, transpose=0, **kwargs):

        lam_inv_sqrt = 1/jnp.sqrt(self.lam)

        if transpose:
            R_prime = (lam_inv_sqrt * R.T).T
            R_prime = self.H @ R_prime
        else:
            R_prime = self.H @ R
            R_prime = (lam_inv_sqrt * R_prime.T).T
    
        return R_prime

    
    def matrix_sqrt(self, R, transpose=0, **kwargs):

        lam_sqrt = jnp.sqrt(self.lam)

        if transpose:
            R_prime = self.H @ R
            R_prime = (lam_sqrt * R_prime.T).T
    
        else:
            R_prime = (lam_sqrt * R.T).T
            R_prime = self.H @ R_prime

        return R_prime

    def eigendecompose(self, x, wn = True, i = -1, **kwargs):

        self.H = HouseholderTransform(self.alpha, i = i)
        self.lam = self.H.lam + (self.diag + wn * self.wn_diag)
        
        logdetK = jnp.log(self.lam).sum()
            
        return self.lam, self.H

    def matmul(self, x1, x2, other, wn = True, **kwargs):
        self.vec1 = self.alpha * jnp.ones(x1.shape[-1])
        self.vec2 = self.alpha * jnp.ones(x2.shape[-1])

        D = self.diag + wn * self.wn_diag
        
        if other.ndim == 1:
            return D * other + jnp.kron(self.vec1, self.vec2 @ other)
        elif other.ndim == 2:
            return D * other + jnp.outer(self.vec1, self.vec2 @ other)
        else:
            raise Exception("Not implemented")

    def __add__(self, K):
        if isinstance(K, Identity) or isinstance(K, ScaledIdentity):
            K_sum = OuterPlusScaledIdentity(self.alpha, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        else:
            K_sum = General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + K.evaluate(x1, x2, **kwargs))
        return K_sum
        


@tree_util.register_pytree_node_class
class HouseholderTransform:
    def __init__(self, vec, i = -1):
        self.vec = vec
        self.T = self
        basis_vec = jnp.zeros_like(vec)
        basis_vec = basis_vec.at[i].set(1)

        norm_alpha = jnp.linalg.norm(self.vec)

        alpha_normalised = self.vec/norm_alpha
        u = alpha_normalised - basis_vec
        self.u_hat = u/jnp.linalg.norm(u)
        self.lam = norm_alpha**2 * basis_vec

    def __matmul__(self, other):
        if other.ndim == 1:
            return other - 2 * jnp.kron(self.u_hat, self.u_hat @ other)
        elif other.ndim == 2:
            return other - 2 * jnp.outer(self.u_hat, self.u_hat @ other)
        elif other.ndim > 2:
            return vmap_for_tensors(self.__matmul__)(other)
        else:
            raise Exception("Not implemented")

    def __rmatmul__(self, other):
        if other.ndim == 1:
            return other - 2 * jnp.kron(other @ self.u_hat, self.u_hat)
        elif other.ndim == 2:
            return other - 2 * jnp.outer(other @ self.u_hat, self.u_hat @ other)
        else:
            raise Exception("Not implemented")

    def __repr__(self):
        dense_mat = self.to_dense()
        return f"HouseholderTransform({dense_mat})"

    def to_dense(self):
        self.dense = jnp.eye(self.u_hat.size) - 2 * jnp.outer(self.u_hat, self.u_hat)
        return self.dense
        
    # Required for JAX
    def tree_flatten(self):
        return (self.__repr__(),), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class Toeplitz(CovType):
    def __init__(self, kf):
        self.kf = kf


class Banded(CovType):
    def __init__(self, kf):
        self.kf = kf

class Lowrank(CovType):
    def __init__(self, kf):
        self.kf = kf

class Periodic(CovType):
    def __init__(self, kf):
        self.kf = kf


class General2D(CovType):
    def __init__(self, kf):
        self.kf = kf

class Block2D(CovType):
    def __init__(self, kf):
        self.kf = kf

class Diagonal2D(CovType):
    def __init__(self, diag):
        self.diag = diag

class Quasisep2D(CovType):
    def __init__(self, kf):
        self.kf = kf
        