import tinygp
from tinygp.kernels.quasisep import Quasisep
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from luas.luas_types import JAXArray


@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])


class MaternNuHalf(Quasisep):
    nu: int = eqx.field(static=True)
    scale: JAXArray | float
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))

    def _f(self):
        return jnp.sqrt(self.nu - ((self.nu - 1) % 2)) / self.scale

    def design_matrix(self):
        f = self._f()
        dtype = jnp.result_type(f, self.sigma)
        C = companion_C((self.nu - 1)//2, dtype)
        return f * C

    def stationary_covariance(self):
        dtype = jnp.result_type(self.scale, self.sigma)
        return stationary_covariance_C((self.nu - 1)//2, dtype)

    def observation_model(self, X):
        del X
        h = jnp.zeros(((self.nu + 1)//2,), dtype=jnp.result_type(self.sigma))
        return h.at[0].set(self.sigma)

    def transition_matrix(self, X1, X2):
        dt = X2 - X1
        tau = dt * self._f()
        dtype = jnp.result_type(tau, self.sigma)
        C = companion_C((self.nu - 1)//2, dtype)

        # print(dt, self.p, self.scale, self.sigma)

        # optional safety cutoff for huge tau
        tau_max = 80.0
        return lax.cond(
            tau > tau_max,
            lambda _: jnp.zeros_like(C),
            lambda _: jsp.linalg.expm(C.T * tau),
            operand=None,
        )

def solve_continuous_lyapunov_kron(F, Q):
    n = F.shape[0]
    I = jnp.eye(n, dtype=F.dtype)
    A = jnp.kron(I, F) + jnp.kron(F, I)
    b = -Q.reshape(-1)
    P = jnp.linalg.solve(A, b).reshape(n, n)
    return 0.5 * (P + P.T)

def binom_row(n, dtype):
    # Returns [C(n,0), C(n,1), ..., C(n,n-1)]
    ks = jnp.arange(n, dtype=dtype)

    def step(c, k):
        c_next = c * (n - k) / (k + 1)
        return c_next, c

    _, coeffs = lax.scan(step, jnp.array(1.0, dtype), ks)
    return coeffs

def companion_C(p, dtype):
    n = p + 1
    coeffs = binom_row(n, dtype)
    C = jnp.eye(n, k=1, dtype=dtype)
    return C.at[-1, :].set(-coeffs)

def stationary_covariance_C(p, dtype):
    C = companion_C(p, dtype)
    n = p + 1
    E = jnp.zeros((n, n), dtype=dtype).at[-1, -1].set(1.0)
    P0 = solve_continuous_lyapunov_kron(C, E)
    return P0 / P0[0, 0]  # normalized, independent of scale
