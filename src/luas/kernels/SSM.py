import numpy as np
from typing import Callable
from luas.luas_types import Scalar, PyTree, is_scalar, Kernel
from luas.kernels.covtype import CovType, Diagonal, Identity, ScaledIdentity
import smolgp
import jax.numpy as jnp

__all__ = [
    "SSM",
]


class SSM(CovType):

    def __init__(self, kf, X = (), y_aux = (), diag = 1e-14, wn_diag = 0., params = None):
        self.kf_SSM = kf
        self.diag = diag
        self.wn_diag = wn_diag
        self.tol = tol
        self.params = params
        self.X = X


    def evaluate(self, x1, x2, wn = True, **kwargs):
        
        return self.gp_SSM.kf_SSM(x1, x2) + np.diag((self.diag + wn*self.wn_diag)*np.ones(x1.shape[-1]))
    
    def decompose(self, x, wn = True, **kwargs):
        
        self.gp_SSM = smolgp.gp.GaussianProcess(kf, self.X + x, diag = self.diag + self.wn_diag)
        
        return self, {"logdetK": 0.}

    def matrix_sqrt(self, R, **kwargs):
         raise Exception("Not implemented")

    def matrix_sqrt(self, R, **kwargs):
         raise Exception("Not implemented")

    def dot_solve(self, R):
        raise Exception("Not implemented")

    def logL(self, R, stored_values, **kwargs):

        return self.gp_SSM.log_probability(R)

    def inverse(self, R, **kwargs):
        raise Exception("Not implemented")

    def scale(self, c):
        return SSM(self.kf_SSM * c, X = self.X, diag = self.diag * c, wn_diag = self.wn_diag * c)

    def __add__(self, K):

        if isinstance(K, SSM):
            K_sum = SSM(self.kf_SSM + K.kf_SSM, X = self.X, diag = self.diag + K.diag,
                          wn_diag = self.wn_diag + K.wn_diag)
        elif isinstance(K, (Diagonal, Identity, ScaledIdentity)):
            K_sum = SSM(self.kf_SSM, X = self.X, diag = self.diag + K.diag,
                          wn_diag = self.wn_diag + K.wn_diag)
        else:
            raise Exception(f"{type(K)} not recognised or addition not supported yet")
            
        return K_sum

    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType) or jnp.ndim(other) != 0:
            raise ValueError(
                "Can't multiply kernel with other kernels"
            )
        return self.scale(other)

