"""Taylor-spectrum quasisep approximation to the squared exponential kernel.

Trainable/JIT-friendly design with closed-form transition propagation:
- Root solving/polynomial algebra is done once at construction (NumPy).
- Scale is trainable; AR coefficients scale analytically with ``scale``.
- Transition matrix uses a precomputed real modal decomposition at reference
  scale plus closed-form 1x1 and 2x2 block exponentials (no jax.scipy.expm).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import tinygp
from tinygp.kernels.quasisep import Quasisep

from luas.luas_types import JAXArray


def _inverse_psd_taylor_poly(scale: float, order: int) -> np.ndarray:
    if order < 1:
        raise ValueError("order must be >= 1")
    a = 0.5 * float(scale) ** 2
    coeffs = np.zeros(2 * order + 1, dtype=np.float64)
    for k in range(order + 1):
        idx = 2 * order - 2 * k
        coeffs[idx] = ((-1) ** k) * (a**k) / math.factorial(k)
    return coeffs


def _stable_ar_coeffs(scale: float, order: int) -> np.ndarray:
    poly = _inverse_psd_taylor_poly(scale=scale, order=order)
    roots = np.roots(poly)

    tol = 1e-10
    stable = roots[np.real(roots) < -tol]
    if stable.size != order:
        idx = np.argsort(np.real(roots))
        stable = roots[idx[:order]]

    monic_desc = np.poly(stable)  # [1, a_{p-1}, ..., a0]
    monic_desc = np.real_if_close(monic_desc, tol=1e4).astype(np.float64)
    return monic_desc[1:][::-1]  # [a0, ..., a_{p-1}]


def _companion_from_ar_np(ar: np.ndarray) -> np.ndarray:
    n = ar.shape[0]
    F = np.eye(n, k=1, dtype=np.float64)
    F[-1, :] = -ar
    return F


def _solve_continuous_lyapunov_kron(F: JAXArray, Q: JAXArray) -> JAXArray:
    n = F.shape[0]
    I = jnp.eye(n, dtype=F.dtype)
    A = jnp.kron(I, F) + jnp.kron(F, I)
    b = -Q.reshape(-1)
    P = jnp.linalg.solve(A, b).reshape(n, n)
    return 0.5 * (P + P.T)


def _build_modal_blocks_for_reference(order: int) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    np.ndarray,
    tuple[int, ...],
    np.ndarray,
    np.ndarray,
]:
    """Build real modal transform for F_ref^T with F_ref at scale=1.

    Returns:
      T, Tinv,
      real_starts, real_decay,
      complex_starts, complex_decay, complex_freq
    such that exp(F_ref^T * tau) = T @ exp(B*tau) @ Tinv, where B is block
    diagonal with 1x1 real and 2x2 damped-rotation blocks.
    """
    ar_ref = _stable_ar_coeffs(scale=1.0, order=order)
    F_ref_t = _companion_from_ar_np(ar_ref).T

    evals, evecs = np.linalg.eig(F_ref_t)

    used = np.zeros(order, dtype=bool)
    cols: list[np.ndarray] = []

    real_starts: list[int] = []
    real_decay: list[float] = []

    complex_starts: list[int] = []
    complex_decay: list[float] = []
    complex_freq: list[float] = []

    tol = 1e-10
    for i in range(order):
        if used[i]:
            continue

        lam = evals[i]
        if abs(lam.imag) < tol:
            used[i] = True
            v = np.real(evecs[:, i])
            cols.append(v)
            real_starts.append(len(cols) - 1)
            real_decay.append(float(lam.real))
        else:
            if lam.imag < 0:
                continue
            conj_target = np.conjugate(lam)
            cands = np.where(~used)[0]
            j = cands[np.argmin(np.abs(evals[cands] - conj_target))]
            used[i] = True
            used[j] = True

            v = evecs[:, i]
            cols.append(np.real(v))
            cols.append(np.imag(v))
            complex_starts.append(len(cols) - 2)
            complex_decay.append(float(lam.real))
            complex_freq.append(float(lam.imag))

    T = np.column_stack(cols)
    Tinv = np.linalg.inv(T)

    return (
        T,
        Tinv,
        tuple(real_starts),
        np.asarray(real_decay, dtype=np.float64),
        tuple(complex_starts),
        np.asarray(complex_decay, dtype=np.float64),
        np.asarray(complex_freq, dtype=np.float64),
    )


@tinygp.helpers.dataclass
class SETaylorQuasisep(Quasisep):
    """SE approximation using Taylor PSD + quasiseparable state space."""

    base_ar: JAXArray
    modal_T: JAXArray
    modal_Tinv: JAXArray
    real_decay: JAXArray
    complex_decay: JAXArray
    complex_freq: JAXArray
    sigma: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    scale: JAXArray | float = eqx.field(default_factory=lambda: jnp.ones(()))
    order: int = eqx.field(default=5, static=True)
    real_starts: tuple[int, ...] = eqx.field(default_factory=tuple, static=True)
    complex_starts: tuple[int, ...] = eqx.field(default_factory=tuple, static=True)

    def _scaled_ar(self) -> JAXArray:
        n = self.base_ar.shape[0]
        exponents = jnp.arange(n, 0, -1, dtype=jnp.result_type(self.scale))
        return self.base_ar / (self.scale**exponents)

    def design_matrix(self) -> JAXArray:
        ar = self._scaled_ar()
        n = ar.shape[0]
        F = jnp.eye(n, k=1, dtype=ar.dtype)
        return F.at[-1, :].set(-ar)

    def stationary_covariance(self) -> JAXArray:
        F = self.design_matrix()
        n = F.shape[0]
        E = jnp.zeros((n, n), dtype=F.dtype).at[-1, -1].set(1.0)
        P = _solve_continuous_lyapunov_kron(F, E)
        return P / P[0, 0]

    def observation_model(self, X: JAXArray) -> JAXArray:
        del X
        n = int(self.order)
        h = jnp.zeros((n,), dtype=jnp.result_type(self.sigma, self.scale))
        return h.at[0].set(self.sigma)

    def _transition_ref(self, tau: JAXArray) -> JAXArray:
        n = int(self.order)
        dtype = jnp.result_type(self.scale, self.sigma)
        Bexp = jnp.zeros((n, n), dtype=dtype)

        for i, s in enumerate(self.real_starts):
            lam = self.real_decay[i]
            Bexp = Bexp.at[s, s].set(jnp.exp(lam * tau))

        for i, s in enumerate(self.complex_starts):
            a = self.complex_decay[i]
            b = self.complex_freq[i]
            e = jnp.exp(a * tau)
            c = jnp.cos(b * tau)
            sn = jnp.sin(b * tau)
            block = e * jnp.array([[c, sn], [-sn, c]], dtype=dtype)
            Bexp = Bexp.at[s : s + 2, s : s + 2].set(block)

        return self.modal_T @ Bexp @ self.modal_Tinv

    def transition_matrix(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dt = X2 - X1
        tau = dt / self.scale

        # F(scale)^T = D @ (F_ref^T / scale) @ D^{-1}, D=diag(scale^k)
        A_ref = self._transition_ref(tau)
        powers = jnp.arange(int(self.order), dtype=jnp.result_type(self.scale))
        d = self.scale**powers
        return (d[:, None] * A_ref) / d[None, :]


def se_taylor_quasisep(
    scale: float | JAXArray,
    sigma: float | JAXArray = 1.0,
    order: int = 5,
) -> SETaylorQuasisep:
    # if order > 5:
    #     raise ValueError("order > 5 can be numerically unstable with this implementation")
    if order < 1:
        raise ValueError("order must be >= 1")

    base_ar_np = _stable_ar_coeffs(scale=1.0, order=int(order))
    T_np, Tinv_np, rs, rd, cs, cd, cf = _build_modal_blocks_for_reference(int(order))

    return SETaylorQuasisep(
        base_ar=jnp.asarray(base_ar_np),
        modal_T=jnp.asarray(T_np),
        modal_Tinv=jnp.asarray(Tinv_np),
        real_starts=rs,
        real_decay=jnp.asarray(rd),
        complex_starts=cs,
        complex_decay=jnp.asarray(cd),
        complex_freq=jnp.asarray(cf),
        sigma=jnp.asarray(sigma),
        scale=jnp.asarray(scale),
        order=int(order),
    )


def build_se_taylor_quasisep(
    scale: float | JAXArray,
    sigma: float | JAXArray = 1.0,
    order: int = 5,
) -> SETaylorQuasisep:
    return se_taylor_quasisep(scale=scale, sigma=sigma, order=order)




class SquaredExpApprox(Quasisep):
    scale: JAXArray | float
    order: int
    _h_coeffs: JAXArray
    T: JAXArray
    total_kernel: tinygp.kernels.quasisep.Quasisep
    sigma: JAXArray | float = 1.
    
    def __init__(self, scale, order = 5, sigma = 1):
        self.scale = scale
        self.order = order
        
        power_spec_coeffs = self._sq_exp_coeffs()
        stable = get_stable_roots(power_spec_coeffs, self.order)
        self._h_coeffs = stoc_diff_coeffs_from_roots(stable)

        F_ref = jnp.diag(jnp.ones(self.order-1), k = 1)
        F_ref = F_ref.at[-1, :].set(-self._h_coeffs)
        lam, eigenvecs = jnp.linalg.eig(F_ref.T)

        tol = 1e-10

        real_roots_ind = jnp.abs(jnp.imag(lam)) < tol
        real_roots = jnp.real(lam[real_roots_ind])
        real_eigenvecs = jnp.real(eigenvecs[:, real_roots_ind])

        complex_roots_real = jnp.real(lam[~real_roots_ind])
        complex_roots_imag = jnp.imag(lam[~real_roots_ind])
        complex_sort_ind = jnp.argsort(jnp.abs(complex_roots_imag))
        complex_sort_ind_firsts = complex_sort_ind[::2]
        complex_roots_real_sorted = complex_roots_real[complex_sort_ind_firsts]
        complex_roots_imag_sorted = complex_roots_imag[complex_sort_ind_firsts]
        complex_eigenvecs_real = jnp.real(eigenvecs[:, ~real_roots_ind][:, complex_sort_ind_firsts])
        complex_eigenvecs_imag = jnp.imag(eigenvecs[:, ~real_roots_ind][:, complex_sort_ind_firsts])

        self.T = jnp.concatenate([real_eigenvecs, complex_eigenvecs_real, complex_eigenvecs_imag], axis = 1)
        all_real_parts = jnp.concatenate([real_roots, complex_roots_real_sorted])
        imag_parts = complex_roots_imag_sorted.copy()

        for i in range(all_real_parts.size):
            kernel_term = tinygp.kernels.quasisep.Exp(all_real_parts[i])
            if i >= real_roots.size:
                kernel_term *= tinygp.kernels.quasisep.Cosine(2 * jnp.pi/imag_parts[i - real_roots.size])

            if i == 0:
                self.total_kernel = kernel_term
            else:
                self.total_kernel += kernel_term


    def to_symm_qsm(self, X):

        Pinf = self.stationary_covariance()
        a = jax.vmap(self.total_kernel.transition_matrix)(
            jax.tree_util.tree_map(lambda y: jnp.append(y[0], y[:-1]), X), X
        )
        h = jax.vmap(self.observation_model)(X)

        q = h @ jnp.linalg.inv(self.T)
        p = (h @ Pinf) @ self.T
        d = jnp.sum(p * q, axis=1)
        print(d, jnp.sum((h @ Pinf) * h, axis = 1))
        
        p = jax.vmap(lambda x, y: x @ y)(p, a)
        return SymmQSM(diag=DiagQSM(d=d), lower=StrictLowerTriQSM(p=p, q=q, a=a))
    
    
    def _sq_exp_coeffs(self):
        m = jnp.arange(self.order, -1, -1)
        
        log_coeffs = gammaln(self.order + 1) - gammaln(m + 1) + (self.order - m) * jnp.log(2)
        all_coeffs = jnp.zeros(2*self.order + 1)
        all_coeffs = all_coeffs.at[::2].set(jnp.power(-1., m) * jnp.exp(log_coeffs))
        
        return all_coeffs

    
    def stationary_covariance(self):

        F = self.design_matrix()
        LqL_T = jnp.zeros((self.order, self.order)).at[-1, -1].set(1.)
        
        P_inf = scipy.linalg.solve_continuous_lyapunov(F, -LqL_T)
        
        # Scaling of power spec arbitrary based on sigma, so just normalise P_inf
        return P_inf/P_inf[0, 0]


    def design_matrix(self):
        exponents = jnp.arange(self.order, 0, -1)
        h_coeffs_scaled = self._h_coeffs/(self.scale ** exponents)

        print("Diff eq. coeffs scaled (h_0, .., h_order-1):", h_coeffs_scaled)

        F = jnp.diag(np.ones(self.order-1), k = 1)
        F = F.at[-1, :].set(-h_coeffs_scaled)
        return F
        
    def transition_matrix(self, X1, X2):

        A_diag = self.total_kernel.transition_matrix(X1, X2)
        return self.T @ A_diag @ jnp.linalg.inv(self.T)

    def observation_model(self, X):
        
        H = jnp.zeros(self.order)
        H = H.at[0].set(self.sigma)

        return H