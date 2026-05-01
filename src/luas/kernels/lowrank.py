import numpy as np
import scipy
import jax
import jax.numpy as jnp
from luas.kernels.base import evaluate_kernel, squared_exp_calc, cosine_calc
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar, CovType, is_scalar
from luas.kernels.covtype import Outer, General
from luas.kernels.householder import HouseholderProduct
import luas.kernels.tinygp_ext

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

from luas.kernels.diagonal import KroneckerDelta, Noise

def _eigsh_worker(shm_name, shape, dtype, k):
    from scipy.sparse.linalg import eigsh
    import numpy as np
    from multiprocessing import shared_memory
    
    # Attach to shared memory — no copy
    shm = shared_memory.SharedMemory(name=shm_name)
    A_np = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    vals, vecs = eigsh(A_np, k=k)
    result = (vals.astype(dtype), vecs.astype(dtype))
    shm.close()
    return result


_pool = ProcessPoolExecutor(max_workers=1)
class LowRank(CovType):
    # This function has the most cursed behaviour I've seen, but it seems to run fast up to around N ~ 1000 - 1500
    # But somewhere above 1500 it only seems to be able to run once and then hangs afterwards
    # Probably something to do with multithreading from scipy playing badly with JIT-compilation I have no idea
    def __init__(self, kf, hp = {}, rank = None, params = None, use_shared_memory = True):
        self.kf = kf
        self.hp = hp
        self.fixed_rank = rank
        self.params = params
        self.use_shared_memory = use_shared_memory

        assert rank is not None

    def eigendecomp(self, x, wn = True, **kwargs):

        K = self.evaluate(x, x, full = True, wn = True)

        result_shape = (jax.ShapeDtypeStruct((self.fixed_rank,), K.dtype), jax.ShapeDtypeStruct((K.shape[0], self.fixed_rank), K.dtype))

        if self.use_shared_memory:
            lam_sparse, Q_sparse = jax.pure_callback(self.sparse_eigh_shared_mem, result_shape, K, self.fixed_rank)
        else:
            lam_sparse, Q_sparse = jax.pure_callback(self.sparse_eigh, result_shape, K, self.fixed_rank)

        Q = HouseholderProduct(Q_sparse)
        lam = jnp.concatenate([jnp.zeros(x.shape[-1] - self.fixed_rank), lam_sparse])
        
        return lam, Q
    
    def sparse_eigh(self, K, rank):
        K_np = np.asarray(K)
        lam, Q = _pool.submit(scipy.sparse.linalg.eigsh, K_np, k=rank).result()
        return lam, Q


    def sparse_eigh_shared_mem(self, K, rank):
        K_np = np.asarray(K)
    
        # Write matrix into shared memory
        shm = shared_memory.SharedMemory(create=True, size=K_np.nbytes)
        shared_A = np.ndarray(K_np.shape, dtype=K_np.dtype, buffer=shm.buf)
        shared_A[:] = K_np
        
        try:
            lam, Q = _pool.submit(
                _eigsh_worker, shm.name, K_np.shape, K_np.dtype, rank
            ).result()
        finally:
            shm.close()
            shm.unlink()
        
        return lam, Q

    def rank(self, x):
        return self.fixed_rank

    def scale(self, c):
        return LowRank(kf = lambda hp, x1, x2, **kwargs: c*self.evaluate(x1, x2, **kwargs), rank = self.fixed_rank, use_shared_memory=self.use_shared_memory)
    
    def __add__(self, other):
        if isinstance(other, LowRank):
            return LowRank(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs),
                           rank = self.fixed_rank + other.fixed_rank, use_shared_memory=self.use_shared_memory and other.use_shared_memory)
        if isinstance(other, Outer):
            return LowRank(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs),
                           rank = self.fixed_rank + 1, use_shared_memory=self.use_shared_memory)
        # elif isinstance(other, CovType):
        #     return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other.evaluate(x1, x2, **kwargs))
        elif isinstance(other, (jax.Array, np.ndarray)) or is_scalar(other):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) + other)
        else:
            raise Exception("Addition not implemented")
        
    def transform_with_inv_sqrt(self, matrix_sqrt_fn):
        def kf_transf(hp, x1, x2, **kwargs):
                K_eval = self.evaluate(x1, x2, **kwargs)
                K_prime = matrix_sqrt_fn(K_eval, transpose=0)
                K_tilde = matrix_sqrt_fn(K_prime.T, transpose=0)
                
                return K_tilde

        return LowRank(kf_transf, rank = self.fixed_rank, use_shared_memory=self.use_shared_memory)


    def evaluate(self, x1, x2, **kwargs):
        return self.kf(self.hp, x1, x2, **kwargs)
    
    def matmul(self, x1, x2, R, **kwargs):
        return self.evaluate(x1, x2, **kwargs) @ R

    # def decompose(self, x, full = True, **kwargs):
    #     # By default returns the lower triangular Cholesky factor
    #     # i.e. K = L @ L.T
    #     # Matches with tinygp default but not scipy or jax.scipy default
        
    #     K = self.evaluate(x, x, full = full, **kwargs)
        
    #     self.factor = JLA.cholesky(K, lower=True)
    #     self.logdet = 2*jnp.log(jnp.diag(self.factor)).sum()
        
    #     return self, {"logdet":self.logdet, "factor":self.factor, "hp":self.hp}

    # def matrix_inv_sqrt(self, R, transpose=0, **kwargs):

    #     R_prime = jax.scipy.linalg.solve_triangular(self.factor, R, trans=transpose, lower=True)

    #     return R_prime
    
    # def matrix_sqrt(self, R, transpose=0, **kwargs):

    #     if transpose:
    #         return self.factor.transpose() @ R
    #     else:
    #         return self.factor @ R


    def __mul__(self, other) -> Kernel:
        if isinstance(other, CovType):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other.evaluate(x1, x2, **kwargs))
        elif is_scalar(other):
            return self.scale(other)
        elif isinstance(other, jax.Array) or isinstance(other, np.ndarray):
            return General(kf = lambda hp, x1, x2, **kwargs: self.evaluate(x1, x2, **kwargs) * other)
        else:
            raise Exception("Not implemented")



def SquaredExp(scale: Scalar, axes: JAXArray | None = None, rank = None, **kwargs) -> JAXArray:
    r"""Squared exponential kernel function, also known as the radial basis function,
    used with ``luas.kernels.evaluate_kernel`` to build a covariance matrix.
    
    .. math::

        k(x, y) = \exp\Bigg( -\frac{|x - y|^2}{2L^2}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    return LowRank(lambda hp, x, y, **kwargs: evaluate_kernel(squared_exp_calc, x, y, scale), rank = rank, **kwargs)
ExpSquared = SquaredExp


def Cosine(period: Scalar, **kwargs) -> JAXArray:
    r"""Cosine kernel, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices which have periodic covariance.
    
    .. math::

        k(x, y) = \cos\Bigg(\frac{2\pi|x - y|}{P}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        P (Scalar): Period
        
    Returns:
        JAXArray: Covariance between two input vectors
        
    """

    return LowRank(lambda hp, x, y, **kwargs: evaluate_kernel(cosine_calc, x, y, period), rank = 2, **kwargs)



def Linear(alpha: JAXArray, **kwargs) -> JAXArray:
    r"""Matern 3/2 kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{3} \frac{|x - y|}{L}\Bigg) \exp\Bigg( -\sqrt{3} \frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    return Outer(alpha = alpha)


def Constant(const: JAXArray, **kwargs) -> JAXArray:
    r"""Matern 3/2 kernel function, used with ``luas.kernels.evaluate_kernel``
    to build covariance matrices.
    
    .. math::

        k(x, y) = \Bigg(1 + \sqrt{3} \frac{|x - y|}{L}\Bigg) \exp\Bigg( -\sqrt{3} \frac{|x - y|}{L}\Bigg)
    
    Args:
        x (JAXArray): Input vector 1
        y (JAXArray): Input vector 2
        L (Scalar): Length scale
        
    Returns:
        Scalar: Covariance between two input vectors
        
    """
    
    return Outer(alpha = const)


def ExpSineSquaredApprox(gamma: Scalar, period: Scalar, order: int, sigma: Scalar = 1., **kwargs):


    # if order is None:
    #     periodic_scale = jnp.sqrt(2/gamma)
    #     max_order = jax.lax.cond(periodic_scale < 1/6, lambda _: 16, lambda _: (jnp.floor(4. * periodic_scale**-0.8)).astype("int"), periodic_scale)
    #     order = jax.lax.cond(periodic_scale > 1., lambda _: 4, lambda _: max_order, periodic_scale) - 1

    kernel_quasi = luas.kernels.tinygp_ext.ExpSineSquaredApprox(gamma = gamma, period = period, sigma = sigma, order = order)
    rank = 1 + 2 * kernel_quasi.order
    
    return LowRank(lambda hp, x1, x2, **kwargs: kernel_quasi(x1, x2), rank = rank, **kwargs)
