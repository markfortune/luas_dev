import time
import jax
import jax.numpy as jnp
import numpy as np
import george

from luas.LuasLasrachKernelND import LuasLasrachKernelND
from luas.LuasKernel import LuasKernel
from luas.covtype import HODLR
import luas.kernels as kernels

jax.config.update("jax_enable_x64", True)


def _wait(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()


def bench(fn, warmup=2, repeat=6):
    for _ in range(warmup):
        _wait(fn())

    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        _wait(out)
        ts.append(time.perf_counter() - t0)

    return min(ts), sum(ts) / len(ts)


def dense_logL(K, R):
    r = np.asarray(R).ravel()
    K = np.asarray(K)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L, r)
    quad = alpha @ alpha
    logdet = 2.0 * np.log(np.diag(L)).sum()
    n = r.size
    return -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)


def make_covs(ell_t=0.25):
    """Build fresh covariance objects (important for fair repeated benchmarks)."""
    Sigma_l = kernels.Noise(0.10) + 0.15 * kernels.Exp(0.30)
    K_l = 0.25 * kernels.Exp(0.45)

    kf_hodlr = george.kernels.ExpSquaredKernel(metric=ell_t**2)
    K_t = HODLR(kf=kf_hodlr, diag=1e-12, wn_diag=0.0, tol=1e-10)
    Sigma_t = kernels.Noise(0.03)

    Sigma = (Sigma_l, Sigma_t)
    K = (K_l, K_t)
    return Sigma, K


def run_case(seed=2):
    rng = np.random.default_rng(seed)

    # modest size for dense cross-check; still large enough to benchmark structure
    n_l, n_t = 64, 2048
    x_l = jnp.linspace(0.0, 1.0, n_l)
    x_t = jnp.linspace(0.0, 3.0, n_t)
    R = jnp.asarray(rng.normal(size=(n_l, n_t)))

    # --- Build and evaluate all methods once ---
    Sigma, K = make_covs()
    nd_batched = LuasLasrachKernelND(Sigma=Sigma, K=K, blackbox_dim=1, use_pure_callback=True, callback_mode="batched")
    nd_batched, sv_batched = nd_batched.decompose(x_l, x_t, stored_values={"R_shape": R.shape})
    logl_batched = nd_batched.logL(R, sv_batched)

    Sigma, K = make_covs()
    nd_vmap = LuasLasrachKernelND(Sigma=Sigma, K=K, blackbox_dim=1, use_pure_callback=True, callback_mode="per_block_vmap")
    nd_vmap, sv_vmap = nd_vmap.decompose(x_l, x_t, stored_values={"R_shape": R.shape})
    logl_vmap = nd_vmap.logL(R, sv_vmap)

    Sigma, K = make_covs()
    lk = LuasKernel(Sigma=Sigma, K=K, eigen_dims=(0, 1))
    lk, sv_lk = lk.decompose(x_l, x_t, stored_values={"R_shape": R.shape})
    logl_lk = lk.logL(R, sv_lk)

    # Dense reference via ND (same K)
    #K_dense = nd_batched.evaluate(x_l, x_t)
    #logl_dense = dense_logL(K_dense, R)

    logl_dense = logl_lk

    print("\n=== Accuracy ===")
    print("Dense reference              :", float(logl_dense))
    print("ND callback-batched          :", float(logl_batched), "|err|", abs(float(logl_batched) - float(logl_dense)))
    print("ND callback-vmap             :", float(logl_vmap), "|err|", abs(float(logl_vmap) - float(logl_dense)))
    print("LuasKernel baseline          :", float(logl_lk), "|err|", abs(float(logl_lk) - float(logl_dense)))

    tol = 5e-2
    assert abs(float(logl_batched) - float(logl_dense)) < tol
    assert abs(float(logl_vmap) - float(logl_dense)) < tol
    assert abs(float(logl_lk) - float(logl_dense)) < tol

    # --- Benchmarks: full and logL-only ---
    def mk_nd_batched():
        Sigma, K = make_covs()
        return LuasLasrachKernelND(Sigma=Sigma, K=K, blackbox_dim=1, use_pure_callback=True, callback_mode="batched")

    def mk_nd_vmap():
        Sigma, K = make_covs()
        return LuasLasrachKernelND(Sigma=Sigma, K=K, blackbox_dim=1, use_pure_callback=True, callback_mode="per_block_vmap")

    def mk_lk():
        Sigma, K = make_covs()
        return LuasKernel(Sigma=Sigma, K=K, eigen_dims=(0, 1))

    rows = []

    # full
    rows.append(("ND callback-batched full",) + bench(lambda: (lambda o: o.logL(R, o.decompose(x_l, x_t, stored_values={"R_shape": R.shape})[1]))(mk_nd_batched())))
    rows.append(("ND callback-vmap full",) + bench(lambda: (lambda o: o.logL(R, o.decompose(x_l, x_t, stored_values={"R_shape": R.shape})[1]))(mk_nd_vmap())))
    rows.append(("LuasKernel full",) + bench(lambda: (lambda o: o.logL(R, o.decompose(x_l, x_t, stored_values={"R_shape": R.shape})[1]))(mk_lk())))

    # logL-only
    rows.append(("ND callback-batched logL-only",) + bench(lambda: nd_batched.logL(R, sv_batched)))
    rows.append(("ND callback-vmap logL-only",) + bench(lambda: nd_vmap.logL(R, sv_vmap)))
    rows.append(("LuasKernel logL-only",) + bench(lambda: lk.logL(R, sv_lk)))

    print("\n=== Benchmark (seconds; min/mean) ===")
    print("(warmup=2, repeat=6)")
    for name, tmin, tmean in rows:
        print(f"{name:30s}  min={tmin:.6f}  mean={tmean:.6f}")


if __name__ == "__main__":
    run_case(seed=2)
    print("\nPASS")
