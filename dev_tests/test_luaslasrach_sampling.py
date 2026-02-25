"""Validate LuasLasrach sampling via matrix_sqrt.

This script checks that LuasLasrachKernelNDJIT is consistent with LuasKernel for:
1) decomposition + logL,
2) matrix_sqrt / matrix_inv_sqrt round-trips,
3) empirical covariance from random draws.

Run:
    python3 dev_tests/test_luaslasrach_sampling.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from luas.kernels.LuasKernel import LuasKernel
from luas.kernels.LuasLasrachKernel import LuasLasrachKernelNDJIT
from luas.kernels.base import SquaredExp, Matern32, Noise

jax.config.update("jax_enable_x64", True)


def _flatten_last(sample_2d):
    """Flatten (N_l, N_t) -> (N_l*N_t,) in C-order, matching kron/evaluate ordering."""
    return np.asarray(sample_2d).reshape(-1)


def main(seed=0, n_draws=600, atol_roundtrip=1e-8, atol_logl=1e-7):
    key = jax.random.PRNGKey(seed)

    # Keep dimensions moderate for the Monte-Carlo covariance check.
    n_l, n_t = 6, 18
    x_l = jnp.linspace(0.0, 1.0, n_l)
    x_t = jnp.linspace(0.0, 3.0, n_t)

    Sigma = (
        Noise(0.07) + 0.20 * SquaredExp(0.35),
        Noise(0.03),
    )
    K = (
        0.45 * SquaredExp(0.25),
        0.80 * Matern32(0.30),
    )

    lk = LuasKernel(Sigma=Sigma, K=K, eigen_dims=(0, 1))
    llk = LuasLasrachKernelNDJIT(Sigma=Sigma, K=K, blackbox_dim=1)

    # Decompose both kernels.
    lk, sv_lk = lk.decompose(x_l, x_t, stored_values={"R_shape": (n_l, n_t)})
    llk, sv_llk = llk.decompose(x_l, x_t, stored_values={"R_shape": (n_l, n_t)})

    # Consistency check on logL.
    key, subkey = jax.random.split(key)
    R = jax.random.normal(subkey, (n_l, n_t))
    logl_lk = float(lk.logL(R, sv_lk))
    logl_llk = float(llk.logL(R, sv_llk))
    logl_err = abs(logl_lk - logl_llk)
    print(f"logL(LuasKernel)    = {logl_lk:.12f}")
    print(f"logL(LuasLasrach)   = {logl_llk:.12f}")
    print(f"|ΔlogL|             = {logl_err:.3e}")
    assert logl_err < atol_logl, f"logL mismatch too large: {logl_err}"

    # matrix_sqrt / matrix_inv_sqrt round-trip checks.
    key, subkey = jax.random.split(key)
    Z = jax.random.normal(subkey, (n_l, n_t))

    Y = llk.matrix_sqrt(Z, transpose=0)
    Z_back = llk.matrix_inv_sqrt(Y, transpose=0)
    err0 = float(jnp.max(jnp.abs(Z - Z_back)))

    YT = llk.matrix_sqrt(Z, transpose=1)
    Z_back_T = llk.matrix_inv_sqrt(YT, transpose=1)
    err1 = float(jnp.max(jnp.abs(Z - Z_back_T)))

    print(f"max|Z - K^-1/2(K^1/2 Z)| (transpose=0): {err0:.3e}")
    print(f"max|Z - K^-T/2(K^T/2 Z)| (transpose=1): {err1:.3e}")
    assert err0 < atol_roundtrip, f"round-trip error (transpose=0) too large: {err0}"
    assert err1 < atol_roundtrip, f"round-trip error (transpose=1) too large: {err1}"

    # Monte-Carlo: sample y = K^{1/2} z, then compare empirical covariance to dense evaluate.
    cov_true = np.asarray(llk.evaluate(x_l, x_t))
    samples = np.zeros((n_draws, n_l * n_t), dtype=np.float64)

    for i in range(n_draws):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n_l, n_t))
        y = llk.matrix_sqrt(z)
        samples[i] = _flatten_last(y)

    emp_cov = np.cov(samples, rowvar=False, bias=False)

    frob_rel = np.linalg.norm(emp_cov - cov_true, ord="fro") / np.linalg.norm(cov_true, ord="fro")
    diag_rel = np.linalg.norm(np.diag(emp_cov) - np.diag(cov_true)) / np.linalg.norm(np.diag(cov_true))

    print(f"relative Frobenius covariance error: {frob_rel:.3e}")
    print(f"relative diagonal covariance error : {diag_rel:.3e}")

    # Sampling error shrinks ~1/sqrt(n_draws). This threshold is intentionally modest.
    assert frob_rel < 0.20, f"Empirical covariance mismatch too large: {frob_rel}"
    assert diag_rel < 0.12, f"Empirical covariance diagonal mismatch too large: {diag_rel}"

    print("\nPASS: LuasLasrach matrix_sqrt produces consistent kernel samples.")


if __name__ == "__main__":
    main()
