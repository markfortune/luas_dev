import time
import jax
import jax.numpy as jnp

from luas.LuasKernel import LuasKernel
from luas.LuasLasrachKernelND import LuasLasrachKernelND, LuasLasrachKernelNDJIT
import luas.kernels as kernels

jax.config.update("jax_enable_x64", True)


def _wait(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()


def bench(fn, warmup=2, repeat=8):
    for _ in range(warmup):
        _wait(fn())

    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        y = fn()
        _wait(y)
        ts.append(time.perf_counter() - t0)

    return min(ts), sum(ts) / len(ts)


def profile_method(name, make_obj, x_l, x_t, R, repeat=8):
    obj = make_obj()
    obj, sv = obj.decompose(x_l, x_t, stored_values={"R_shape": R.shape})
    logl = obj.logL(R, sv)

    t_decomp = bench(lambda: make_obj().decompose(x_l, x_t, stored_values={"R_shape": R.shape})[1]["logdetK"], repeat=repeat)
    t_logl = bench(lambda: obj.logL(R, sv), repeat=repeat)
    t_full = bench(lambda: (lambda o: o.logL(R, o.decompose(x_l, x_t, stored_values={"R_shape": R.shape})[1]))(make_obj()), repeat=repeat)

    return {
        "name": name,
        "logL": float(logl),
        "decompose": t_decomp,
        "logL_only": t_logl,
        "full": t_full,
    }


def run_case(seed=2):
    key = jax.random.PRNGKey(seed)

    n_l, n_t = 24, 1200
    x_l = jnp.linspace(0.0, 1.0, n_l)
    x_t = jnp.linspace(0.0, 5.0, n_t)

    Sigma = (
        kernels.Noise(0.08) + 0.12 * kernels.SquaredExp(0.25),
        kernels.Noise(0.04),
    )
    K = (
        0.30 * kernels.SquaredExp(0.35),
        0.75 * kernels.Matern32(0.20),
    )

    R = jax.random.normal(key, (n_l, n_t))

    # Baseline reference only (not heavily benchmarked)
    lk = LuasKernel(Sigma=Sigma, K=K, eigen_dims=(0, 1))
    lk, sv_lk = lk.decompose(x_l, x_t, stored_values={"R_shape": R.shape})
    logl_ref = lk.logL(R, sv_lk)

    makers = [
        ("NDJIT", lambda: LuasLasrachKernelNDJIT(Sigma=Sigma, K=K, blackbox_dim=1)),
        ("ND callback-batched", lambda: LuasLasrachKernelND(Sigma=Sigma, K=K, blackbox_dim=1, use_pure_callback=True, callback_mode="batched")),
        ("ND callback-vmap", lambda: LuasLasrachKernelND(Sigma=Sigma, K=K, blackbox_dim=1, use_pure_callback=True, callback_mode="per_block_vmap")),
    ]

    results = []
    for name, mk in makers:
        results.append(profile_method(name, mk, x_l, x_t, R, repeat=8))

    print("\n=== 2D Accuracy vs LuasKernel ===")
    print("LuasKernel baseline:", float(logl_ref))
    for r in results:
        err = abs(r["logL"] - float(logl_ref))
        print(f"{r['name']:20s} logL={r['logL']:.12f} |err|={err:.3e}")
        assert err < 1e-6, f"{r['name']} mismatch too large: {err}"

    print("\n=== 2D Isolated Benchmark (seconds; min/mean) ===")
    print("(warmup=2, repeat=8 per method; methods run in isolation blocks)")
    for r in results:
        dmin, dmean = r["decompose"]
        lmin, lmean = r["logL_only"]
        fmin, fmean = r["full"]
        print(f"{r['name']}")
        print(f"  decompose-only   min={dmin:.6f} mean={dmean:.6f}")
        print(f"  logL-only        min={lmin:.6f} mean={lmean:.6f}")
        print(f"  full             min={fmin:.6f} mean={fmean:.6f}")


if __name__ == "__main__":
    run_case(seed=2)
    print("\nPASS")
