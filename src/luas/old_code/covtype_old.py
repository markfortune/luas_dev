import numpy as np
from copy import deepcopy
from typing import Optional, Callable, Tuple, Any, Union
import scipy
import warnings

import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

import tinygp
import george

from .luas_types import Kernel, PyTree, JAXArray, Scalar, CovType
from .kronecker_fns import make_vec, make_mat
from .jax_convenience_fns import array_to_pytree_2D


class Identity(CovType):

    def __init__(self):
        self.logdet = 0.
        self.kf = None
        self.diag = 1.
        self.wn_diag = 0.

    
    def scale(self, c):
        return ScaledIdentity(diag = c)

    
    def __add__(self, K):

        if type(K) == Exp:
            K_sum = GeneralCelerite(K.kf, diag = 1.)
        elif type(K) == Identity:
            K_sum = Diagonal(diag = 2.)
        elif type(K) == Outer:
            K_sum = Diag_and_Outer(K.alpha, diag = 1.)
        elif type(K) == Diagonal:
            K_sum = Diagonal(diag = K.diag + 1., wn_diag = K.wn_diag)
        elif type(K) in [GeneralCelerite, General]:
            K_sum = type(K)(K.kf, diag = K.diag + 1., wn_diag = K.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    
    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply Identity kernel with other kernels"
            )
        return self.scale(other)
        
    
    def evaluate(self, hp, x1, x2, **kwargs):
        return jnp.diag(1.*jnp.isclose(x1, x2))

    
    def cholesky_decomp(self, hp, x, **kwargs):
        self.factor = jnp.ones(x.shape[-1])

    def cho_solve(self, R, **kwargs):

        return R

    def cho_mult(self, R, **kwargs):

        return R

    def cholesky_transform(self, K):

        return K

    def eigendecomp(self, hp, x, **kwargs):

        return jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])

    def left_mult(self, R, *args, **kwargs):

        return R

    def right_mult(self, R, *args, **kwargs):

        return R

        
class ScaledIdentity(CovType):
    def __init__(self, diag = 0., wn_diag = 0.):
        self.diag = diag
        self.wn_diag = wn_diag
        self.kf = None

    
    def evaluate(self, hp, x1, x2, wn = True, **kwargs):
        
        D = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        
        return jnp.diag(D)
        

    def __add__(self, K):

        if type(K) == Exp:
            K_sum = GeneralCelerite(K.kf, diag = self.diag, wn_diag = self.wn_diag)
        elif type(K) in [Identity, ScaledIdentity, Diagonal]:
            K_sum = Diagonal(diag = K.diag + self.diag, wn_diag = K.wn_diag + self.wn_diag)
        elif type(K) == Outer:
            K_sum = Diag_and_Outer(K.alpha, diag = self.diag, wn_diag = self.wn_diag)
        elif type(K) in [GeneralCelerite, General]:
            K_sum = type(K)(K.kf, diag = K.diag + self.diag, wn_diag = K.wn_diag + self.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    
    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply Identity kernel with other kernels"
            )
        return self.scale(other)
    
    def cholesky_decomp(self, hp, x, **kwargs):

        self.D = self.diag + self.wn_diag
        self.factor = jnp.sqrt(self.D)
        self.logdet = x.shape[-1]*jnp.log(self.D)

    
    def cholesky_transform(self, K):

        if type(K) == Diagonal:
            K_tilde = Diagonal(diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)

        elif type(K) == Identity:
            K_tilde = ScaledIdentity(diag = 1/self.D)
            
        elif type(K) == ScaledIdentity:
            K_tilde = ScaledIdentity(diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)

        elif type(K) == Outer:
            K_tilde = Outer(alpha = K.alpha/self.factor)

        else:
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(*args, **kwargs):
                
                return (1/self.D) * K.evaluate(*args, **kwargs)

            K_tilde = General(kf_transf)
            
        return K_tilde

    def cho_solve(self, R, **kwargs):

        return R_prime/self.factor

    
    def cho_mult(self, R, **kwargs):

        return self.factor * R
        

    def eigendecomp(self, hp, x, wn = True, **kwargs):

        const_diag = self.diag + wn*self.wn_diag
        
        return const_diag*jnp.ones(x.shape[-1]), jnp.eye(x.shape[-1])

    def left_mult(self, R, hp, x1, x2, wn = True, **kwargs):

        const = self.diag + wn*self.wn_diag

        return const * R

    def scale(self, c):
        return Diagonal(diag = self.diag * c, wn_diag = self.wn_diag * c)

    # def rank(self, hp, x1, x2, wn = True):
        
    #     D = (self.diag + self.wn_diag)*jnp.ones(x1.shape[-1])
        
    #     return (D > 0.).sum()


class HODLR(CovType):
    def __init__(self, kf, diag = 0., wn_diag = 0., sigma = 1.):
        self.kf = kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.sigma = sigma

        self.gp_hodlr = george.GP(kf, white_noise = self.diag, solver=george.HODLRSolver, tol=1e-10)
        
    def scale(self, c):
        
        return HODLR(self.kf, diag = self.diag, wn_diag = self.wn_diag, sigma = c * self.sigma)

    
    def __add__(self, K):

        # if type(K) in [HODLR]:
        #     K_sum = HODLR(self.kf + K.kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        if type(K) in [Diagonal, ScaledIdentity]:
            K_sum = HODLR(self.kf, diag = self.diag + K.diag/self.sigma, wn_diag = self.wn_diag + K.wn_diag/self.sigma, sigma = self.sigma)
        elif type(K) in [Identity]:
            K_sum = HODLR(self.kf, diag = self.diag + 1./self.sigma, wn_diag = self.wn_diag, sigma = self.sigma)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply kernel with other kernels"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)
        

    def evaluate(self, hp, x1, x2, wn = True, **kwargs):
        
        return self.scale * (self.kf.get_matrix(x1, x2) + jnp.diag((self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])))

    
    def cholesky_decomp(self, hp, x, wn = True, **kwargs):
        
        self.gp_hodlr.compute(x)
        self.logdet = x.shape[-1] * jnp.log(self.sigma) + self.gp_hodlr.solver.log_determinant

    
    def cholesky_transform(self, K):
        raise Exception("Not implemented")
        

    def cho_solve(self, R, transpose=0):
        
        self.logdet = self.gp_hodlr.solver.log_determinant + (1/self.sigma) * self.gp_hodlr.solver.dot_solve(R)
            
        return 0.*R

    def cho_mult(self, R, transpose=0, **kwargs):
         raise Exception("Not implemented")

    
    def eigendecomp(self, hp, x, wn = True, **kwargs):
        raise Exception("Not implemented")

    
    # There's a faster way of doing this I've yet to implement
    def left_mult(self, R, hp, x1, x2, wn = True, **kwargs):
        raise Exception("Not implemented")



class Outer(CovType):
    def __init__(self, alpha):
        self.alpha = alpha
        self.kf = lambda *args, **kwargs: jnp.outer(self.alpha, self.alpha)
        self.rank = 1
        self.diag = 0.
        self.wn_diag = 0.

    def __add__(self, K):

        if type(K) in [Diagonal, Identity]:
            K_sum = Diag_and_Outer(self.alpha, diag = K.diag, wn_diag = K.wn_diag)
        elif type(K) in [General]:
            K_sum = General(lambda *args, **kwargs: K.kf(*args, **kwargs) + jnp.outer(self.alpha, self.alpha),
                            diag = K.diag, wn_diag = K.wn_diag)
        elif type(K) in [GeneralCelerite, Exp]:
            # Should update as you can always add outer products without losing celeriteness
            K_sum = General(lambda *args, **kwargs: K.evaluate(*args, **kwargs) + jnp.outer(self.alpha, self.alpha))
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum
        

    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply kernel with other kernels"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def scale(self, c):
        return Outer(self.alpha * jnp.sqrt(c))

    def evaluate(self, hp, x1, x2, **kwargs):
        return jnp.outer(self.alpha * jnp.ones(x1.shape[-1]), self.alpha * jnp.ones(x2.shape[-1]))

    def cholesky_decomp(self, hp, x, **kwargs):
        pass

    def cho_solve(self, R, **kwargs):

        return Exception("This matrix is not invertible!")

    def cho_mult(self, R, **kwargs):

        return jnp.outer(self.alpha, R[0, :])

    def cholesky_transform(self, K):

        return Exception("This matrix is not invertible!")

    def eigendecomp(self, hp, x, **kwargs):

        e_min1 = jnp.zeros(x.shape[-1])
        e_min1 = e_min1.at[-1].set(1)

        norm_alpha = jnp.linalg.norm(self.alpha * jnp.ones(x.shape[-1]))

        alpha_normalised = self.alpha/norm_alpha
        u1 = alpha_normalised - e_min1
        u1 /= jnp.linalg.norm(u1)
        
        return norm_alpha**2*e_min1, jnp.eye(x.shape[-1]) - 2*jnp.outer(u1, u1)

    def left_mult(self, R, hp, *args, **kwargs):

        return jnp.outer(self.alpha, self.alpha.T @ R)

    # def right_mult(self, R, *args):

    #     return jnp.outer(hp[self.param_name], hp[self.param_name].T @ R)



class General(CovType):
    def __init__(self, kf, diag = 0., wn_diag = 0., params = None):
        self.kf = kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.params = params
        
    def evaluate(self, hp, x1, x2, wn = True, **kwargs):
        if x1.shape == x2.shape:
            return self.kf(hp, x1, x2, wn = wn, **kwargs) + jnp.diag((self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1]))
        else:
            return self.kf(hp, x1, x2, wn = wn, **kwargs)

    def __add__(self, K):

        if type(K) in [Diagonal, Identity]:
            K_sum = General(self.kf, diag = K.diag + self.diag, wn_diag = K.wn_diag + self.wn_diag)
        elif type(K) == Outer:
            K_sum = General(lambda *args, **kwargs: self.kf(*args, **kwargs) + jnp.outer(K.alpha, K.alpha),
                            diag = self.diag, wn_diag = self.wn_diag)
        elif type(K) in [General, GeneralCelerite, Exp]:
            K_sum = General(lambda *args, **kwargs: self.kf(*args, **kwargs) + K.evaluate(*args, **kwargs),
                        diag = self.diag, wn_diag = self.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    # def __mul__(self, other) -> Kernel:
    #     if isinstance(other, CovType) or jnp.ndim(other) != 0:
    #         raise ValueError(
    #             "Can't multiply kernel with other kernels"
    #         )
    #     return self.scale(other)

    # def __rmul__(self, other) -> Kernel:
    #     if isinstance(other, CovType) or jnp.ndim(other) != 0:
    #         raise ValueError(
    #             "Can't multiply kernel with other kernels"
    #         )
    #     return self.scale(other)

    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply kernel with other kernels"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def scale(self, c):
        return General(lambda *args, **kwargs: self.kf(*args, **kwargs) * c, diag = self.diag * c, wn_diag = self.wn_diag * c)
    
    def cholesky_decomp(self, hp, x, **kwargs):
        # By default returns the lower triangular Cholesky factor
        # i.e. K = L @ L.T
        # Matches with Celerite default but not scipy or jax.scipy default
        
        K = self.evaluate(hp, x, x, **kwargs)

        self.factor = JLA.cholesky(K, lower=True)
        self.logdet = 2*jnp.log(jnp.diag(self.factor)).sum()


    def cho_solve(self, R, transpose=0):

        R_prime = jax.scipy.linalg.solve_triangular(self.factor, R, trans=transpose, lower=True)

        return R_prime

    
    def cho_mult(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R


    def cholesky_transform(self, K):

        if type(K) == Outer:
            alpha_tilde = jax.scipy.linalg.solve_triangular(self.factor, K.alpha, trans=0, lower=True)
            K_tilde = Outer(alpha = alpha_tilde)
            
        else:
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(*args, **kwargs):
                K_eval = K.evaluate(*args, **kwargs)
                K_prime = jax.scipy.linalg.solve_triangular(self.factor, K_eval, trans=0,lower=True)
                K_tilde = jax.scipy.linalg.solve_triangular(self.factor, K_prime.T, trans=0,lower=True)
                
                return K_tilde

            K_tilde = General(kf_transf)

        return K_tilde
        
    
    def eigendecomp(self, hp, x, **kwargs):
        
        K = self.evaluate(hp, x, x, **kwargs)

        return jnp.linalg.eigh(K)

    def left_mult(self, R, hp, x1, x2, **kwargs):
        K = self.evaluate(hp, x1, x2, **kwargs)

        return K @ R

    def right_mult(self, R, hp, x1, x2, **kwargs):
        K = self.evaluate(hp, x1, x2, **kwargs)

        return R @ K


class GeneralCelerite(CovType):
    def __init__(self, kf, diag = 0., wn_diag = 0.):
        self.kf = kf
        self.diag = diag
        self.wn_diag = wn_diag

    def scale(self, c):
        return GeneralCelerite(self.kf * c, diag = self.diag * c, wn_diag = self.wn_diag * c)

    def __add__(self, K):

        if type(K) in [Identity, Diagonal]:
            K_sum = GeneralCelerite(self.kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif type(K) in [Exp, GeneralCelerite]:
            K_sum = GeneralCelerite(K.kf + self.kf, diag = self.diag + K.diag, wn_diag = self.wn_diag + K.wn_diag)
        elif type(K) == Outer:
            # Can update as don't lose celeriteness
            K_sum = General(lambda *args, **kwargs: self.evaluate(*args, **kwargs) + jnp.outer(self.alpha, self.alpha))
        elif type(K) == General:
            K_sum = General(lambda *args, **kwargs: self.evaluate(*args, **kwargs) + K.kf(*args, **kwargs), diag = K.diag, wn_diag = K.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    
    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply kernel with other kernels"
            )
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)
        

    def evaluate(self, hp, x1, x2, wn = True, **kwargs):
        
        return self.kf(x1, x2) + jnp.diag((self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1]))

    
    def cholesky_decomp(self, hp, x, wn = True, **kwargs):

        diag = (self.diag + wn*self.wn_diag)*jnp.ones(x.shape[-1])
        noise_model = tinygp.noise.Diagonal(diag=diag)
        matrix = self.kf.to_symm_qsm(x)
        matrix += noise_model.to_qsm()
        self.factor = matrix.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))

    
    def cholesky_transform(self, K):

        if type(K) == Outer:
            K_tilde = Outer(alpha = self.factor.solve(K.alpha))

        else:
            def kf_transf(*args, **kwargs):
                K_eval = K.evaluate(*args, **kwargs)
                K_prime = self.factor.solve(K_eval)
                K_prime = self.factor.solve(K_prime.T)
                
                return K_prime
            K_tilde = General(kf_transf)

        return K_tilde
        

    def cho_solve(self, R, transpose=0):

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime

    def cho_mult(self, R, transpose=0, **kwargs):

        if transpose:
            return self.factor.transpose() @ R
        else:
            return self.factor @ R

    
    def eigendecomp(self, hp, x, wn = True, **kwargs):

        diag = (self.diag + wn*self.wn_diag)*jnp.ones(x.shape[-1])
        K_eval = self.evaluate(hp, x, x) + jnp.diag(diag)
        
        return jnp.linalg.eigh(K_eval)

    
    # There's a faster way of doing this I've yet to implement
    def left_mult(self, R, hp, x1, x2, wn = True, **kwargs):

        diag = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        noise_model = tinygp.noise.Diagonal(diag=diag)
        matrix = self.kf.to_symm_qsm(x1)
        matrix += noise_model.to_qsm()

        return matrix @ R



def faster_cel_GP(kernel, X, y, diag=None):
    noise_model = tinygp.noise.Diagonal(diag=diag)
    matrix = kernel.to_symm_qsm(X)
    matrix += noise_model.to_qsm()
    factor = matrix.cholesky()
    
    return - 0.5 * (factor.solve(y)**2).sum() - jnp.sum(jnp.log(factor.diag.d)) - 0.5 * factor.shape[0] * jnp.log(2 * jnp.pi)


def exp_eigh(x1, scale, sigma):
    
    if x1.ndim == 2:
        x1 = x1[0, :]
        
    r = np.exp(-np.diff(x1/scale))
    e = 1/(1/r - r)
    
    arr_diag = np.zeros(x1.shape[-1])
    arr_diag[:-1] = 1 + r*e
    arr_diag[1:] += r*e
    arr_diag[-1] += 1
    
    lam, Q = scipy.linalg.eigh_tridiagonal(arr_diag, -e)
    
    return sigma**2/lam, Q

    
class Exp(CovType):
    def __init__(self, l, sigma = 1.):
        self.l = l
        self.sigma = sigma
        self.kf = tinygp.kernels.quasisep.Exp(scale = self.l, sigma = self.sigma)
        self.diag = 0.
        self.wn_diag = 0.

        
    def __add__(self, K):

        if type(K) in [Identity, Diagonal]:
            K_sum = GeneralCelerite(self.kf, diag = K.diag, wn_diag = K.wn_diag)
        elif type(K) in [Exp, GeneralCelerite]:
            K_sum = GeneralCelerite(K.kf + self.kf, diag = K.diag, wn_diag = K.wn_diag)
        elif type(K) == Outer:
            K_sum = General(lambda *args, **kwargs: self.kf(*args) + jnp.outer(self.alpha, self.alpha))
        elif type(K) in [General]:
            K_sum = General(lambda hp, x1, x2, **kwargs: self.kf(x1, x2) + K.kf(hp, x1, x2, **kwargs),
                        diag = K.diag, wn_diag = K.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    
    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply Identity kernel with other kernels"
            )
        return self.scale(other)
        
    
    def evaluate(self, hp, x1, x2, wn = True, **kwargs):
        
        return self.kf(x1, x2)

    
    def cholesky_decomp(self, hp, x, **kwargs):
        
        matrix = self.kf.to_symm_qsm(x)
        self.factor = matrix.cholesky()
        self.logdet = 2*jnp.sum(jnp.log(self.factor.diag.d))

    
    def cholesky_transform(self, K):
        
        def kf_transf(*args, **kwargs):
            K_eval = K.evaluate(*args, **kwargs)
            K_prime = self.factor.solve(K_eval)
            K_prime = self.factor.solve(K_prime.T)
            
            return K_tilde

        return General(kf_transf)
        

    def cho_solve(self, R, transpose=0):

        if transpose:
            R_prime = self.factor.transpose().solve(R)
        else:
            R_prime = self.factor.solve(R)
            
        return R_prime

    def cho_mult(self, R, **kwargs):

        return self.factor @ R

    
    def eigendecomp(self, hp, x, wn = True, **kwargs):
        
        N = x.shape[-1]
        result_shape = (jax.ShapeDtypeStruct((N,), x.dtype), jax.ShapeDtypeStruct((N, N), x.dtype))
        return jax.pure_callback(exp_eigh, result_shape, x, self.l, self.sigma)

    
    # There's a faster way of doing this I've yet to implement
    def left_mult(self, R, hp, x1, x2, **kwargs):

        matrix = self.kf.to_symm_qsm(x1)

        return matrix @ R

    
    def scale(self, c):
        return Exp(self.l, sigma = self.sigma * jnp.sqrt(jnp.abs(c)))

    

    # def rank(self, hp, x1, x2, wn = True):
        
    #     return x1.shape[-1]


class Diagonal(CovType):
    def __init__(self, diag = 0., wn_diag = 0.):
        self.diag = diag
        self.wn_diag = wn_diag
        self.kf = None

        assert type(self.diag) or self.diag.ndim < 2
        assert type(self.wn_diag) or self.wn_diag.ndim < 2

    
    def evaluate(self, hp, x1, x2, wn = True, **kwargs):
        
        D = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])
        
        return jnp.diag(D)
        

    def __add__(self, K):

        if type(K) == Exp:
            K_sum = GeneralCelerite(K.kf, diag = self.diag, wn_diag = self.wn_diag)
        elif type(K) in [Identity, Diagonal]:
            K_sum = Diagonal(diag = K.diag + self.diag, wn_diag = K.wn_diag + self.wn_diag)
        elif type(K) == Outer:
            K_sum = Diag_and_Outer(K.alpha, diag = self.diag, wn_diag = self.wn_diag)
        elif type(K) in [GeneralCelerite, General]:
            K_sum = type(K)(K.kf, diag = K.diag + self.diag, wn_diag = K.wn_diag + self.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum


    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply kernel with other kernels"
            )
        return self.scale(other)
        
    
    def cholesky_decomp(self, hp, x, **kwargs):

        self.D = (self.diag + self.wn_diag)*jnp.ones(x.shape[-1])
        self.factor = jnp.sqrt(self.D)
        self.logdet = jnp.log(self.D).sum()

    
    def cholesky_transform(self, K):

        if type(K) == Diagonal:
            K_tilde = Diagonal(diag = K.diag/self.D, wn_diag = K.wn_diag/self.D)

        elif type(K) == Identity:
            K_tilde = Diagonal(diag = 1/(self.D))

        elif type(K) == Outer:
            K_tilde = Outer(alpha = K.alpha/jnp.sqrt(self.D))

        else:
            # Define a new General covtype for K where it's kernel function transforms it
            def kf_transf(*args, **kwargs):
                
                D_inv_sqrt = 1/jnp.sqrt(self.D)
                return jnp.outer(D_inv_sqrt, D_inv_sqrt) * K.evaluate(*args, **kwargs)

            K_tilde = General(kf_transf)
            
        return K_tilde

    def cho_solve(self, R, transpose=0):
        
        D_inv_sqrt = 1/self.factor

        R_prime = (D_inv_sqrt * R.T).T

        return R_prime

    
    def cho_mult(self, R, **kwargs):

        return (self.factor * R.T).T
        

    def eigendecomp(self, hp, x, wn = True, **kwargs):

        D = (self.diag + wn*self.wn_diag)*jnp.ones(x.shape[-1])
        
        return D, jnp.eye(x.shape[-1])

    def left_mult(self, R, hp, x1, x2, wn = True, **kwargs):

        D = (self.diag + wn*self.wn_diag)*jnp.ones(x1.shape[-1])

        return (D * R.T).T

    def scale(self, c):
        return Diagonal(diag = self.diag * c, wn_diag = self.wn_diag * c)

    # def rank(self, hp, x1, x2, wn = True):
        
    #     D = (self.diag + self.wn_diag)*jnp.ones(x1.shape[-1])
        
    #     return (D > 0.).sum()


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


def Diag_and_Outer(alpha, **kwargs):
    # Placeholder until actual implementation
    
    return General(kf = lambda *args, **kwargs: jnp.outer(alpha, alpha), **kwargs)
    

# class Diag_and_Outer(CovType):
#     def __init__(self, kf, diag, wn_diag = 0.):
#         self.kf = kf
#         self.diag = diag
#         self.wn_diag = wn_diag
#         Diag_and_Outer(alpha = K.alpha, diag = 1.)

class General2D(CovType):
    def __init__(self, kf):
        self.kf = kf

class Block2D(CovType):
    def __init__(self, kf):
        self.kf = kf

class Diagonal2D(CovType):
    def __init__(self, diag):
        self.diag = diag

class Celerite2D(CovType):
    def __init__(self, kf):
        self.kf = kf
        