"""LuasLasrach-style ND kernels with one designated blackbox dimension.

This module provides two closely-related implementations:

- :class:`LuasLasrachKernelND`
  Callback-friendly implementation designed for blackbox ``CovType`` objects
  that may contain non-JAX code (e.g. george/HODLR). It supports
  ``jax.pure_callback`` execution paths while preserving the blackbox solver
  used inside each block.

- :class:`LuasLasrachKernelNDJIT`
  Fully JAX/JIT-friendly implementation that assumes all operations inside the
  blackbox ``CovType`` are JAX-traceable. It uses ``jax.vmap`` over blocks with
  no Python callbacks in the core path.

Both classes follow the same high-level decomposition strategy:

1. Pick one dimension as the *blackbox* dimension.
2. For all other dimensions, whiten by ``Sigma_d`` and eigendecompose the
   transformed ``K_d``.
3. Combine eigenvalues into a Kronecker-product vector ``lambda``.
4. Solve many independent blackbox blocks
   ``B_i = lambda_i * K_bb + Sigma_bb``.

This is a practical ND generalisation of the LuasLasrach idea from the draft
paper while keeping the ``CovType`` object model used throughout ``luas``.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from luas.kernels.covtype import CovType
from luas.luas_types import PyTree, JAXArray, Scalar
from luas.kronecker_fns import calc_total_size, cyclic_transpose, vmap_for_tensors

__all__ = [
    "LuasLasrachKernelND",          # callback-friendly (keeps CovType blackbox optimisations)
    "LuasLasrachKernelNDJIT",       # fully JIT-friendly (dense blackbox blocks)
]

jax.config.update("jax_enable_x64", True)


class BlackBoxTildeKernelCallback(CovType):
    """Blockwise transformed kernel with callback support for non-JAX blackboxes.

    This helper kernel operates on the transformed system where non-blackbox
    dimensions have already been diagonalised. For each block index ``i``:

    .. math::
        B_i = \\lambda_i K_{bb} + \\Sigma_{bb}

    Parameters
    ----------
    Sigma_bb, K_bb
        ``CovType`` objects for the designated blackbox dimension.
    lambdas
        Flattened Kronecker eigenvalue vector from all non-blackbox dimensions.
    bb_dim
        Integer axis index of the blackbox dimension in the original tensor.
    use_pure_callback
        If ``True``, block operations are executed via ``jax.pure_callback``
        so non-JAX Python/C++ code can run inside an otherwise JAX pipeline.
    callback_mode
        Callback execution style:

        - ``"per_block_vmap"``: vmapped callback per block
        - ``"batched"``: one callback handling all blocks

    Notes
    -----
    This class intentionally avoids storing a per-block object list so that
    it can interoperate cleanly with vmapped callback paths.
    """

    def __init__(
        self,
        Sigma_bb: CovType,
        K_bb: CovType,
        lambdas: JAXArray,
        bb_dim: int,
        use_pure_callback: bool = True,
        callback_mode: str = "per_block_vmap",
        cache_blocks: bool = True,
    ):
        self.Sigma_bb = Sigma_bb
        self.K_bb = K_bb
        self.lambdas = jnp.asarray(lambdas).ravel()
        self.bb_dim = bb_dim
        self.use_pure_callback = use_pure_callback
        self.callback_mode = callback_mode
        self.cache_blocks = cache_blocks
        self.bb_size = None
        self.x_bb = None

        # Optional Python-side cache for callback-batched path.
        self._cached_blocks = None
        self._cached_logdets = None
        self._cached_lambdas = None

    def decompose(self, *X: Tuple, **kwargs):
        self.x_bb = X[self.bb_dim]
        self.bb_size = self.x_bb.shape[-1]

        if self.use_pure_callback:
            if self.callback_mode == "batched":
                if self.cache_blocks:
                    # Precompute/cache decomposed blocks once so repeated logL-only
                    # calls avoid re-decomposing each block.
                    self._build_block_cache_py(np.asarray(self.lambdas))
                    block_logdets = jnp.asarray(self._cached_logdets, dtype=self.lambdas.dtype)
                else:
                    out_spec = jax.ShapeDtypeStruct((self.lambdas.shape[0],), self.lambdas.dtype)
                    block_logdets = jax.pure_callback(self._block_logdets_py, out_spec, self.lambdas)
            else:
                out_spec = jax.ShapeDtypeStruct((), self.lambdas.dtype)
                block_logdets = jax.vmap(
                    lambda lam: jax.pure_callback(
                        self._single_logdet_py,
                        out_spec,
                        lam,
                        vmap_method="sequential",
                    )
                )(self.lambdas)
            return self, {"logdetK": jnp.sum(block_logdets)}

        block_logdets = jax.vmap(self._single_logdet_jax)(self.lambdas)
        return self, {"logdetK": jnp.sum(block_logdets)}

    def _reshape_to_blocks(self, R: JAXArray):
        R_perm = jnp.moveaxis(R, self.bb_dim, -1)
        shape_perm = R_perm.shape
        R_blocks = R_perm.reshape((-1, self.bb_size))
        return R_blocks, shape_perm

    def _reshape_from_blocks(self, R_blocks: JAXArray, shape_perm: Tuple[int, ...]):
        R_perm = R_blocks.reshape(shape_perm)
        return jnp.moveaxis(R_perm, -1, self.bb_dim)

    def _build_block_cache_py(self, lambdas_np: np.ndarray):
        """Build Python-side cache of decomposed blackbox blocks.

        This is used by callback-batched mode to accelerate repeated logL-only
        calls at fixed decomposition.
        """
        self._cached_blocks = []
        self._cached_logdets = np.zeros((len(lambdas_np),), dtype=np.float64)
        self._cached_lambdas = np.asarray(lambdas_np, dtype=np.float64)

        for i, lam in enumerate(self._cached_lambdas):
            block = float(lam) * self.K_bb + self.Sigma_bb
            block, sv = block.decompose(self.x_bb)
            self._cached_blocks.append(block)
            self._cached_logdets[i] = float(sv["logdetK"])

    def _single_logdet_jax(self, lam):
        block = lam * self.K_bb + self.Sigma_bb
        _, sv = block.decompose(self.x_bb)
        return sv["logdetK"]

    def _single_dot_jax(self, lam, r):
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return block.dot_solve(r)

    def _single_inv_jax(self, lam, r):
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return block.inverse(r)

    def _single_sqrt_jax(self, lam, r, transpose=0):
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return block.matrix_sqrt(r, transpose=transpose)

    def _single_inv_sqrt_jax(self, lam, r, transpose=0):
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return block.matrix_inv_sqrt(r, transpose=transpose)

    # ---- pure_callback helpers (Python side) ----
    def _single_logdet_py(self, lam_np):
        lam = float(np.asarray(lam_np))
        block = lam * self.K_bb + self.Sigma_bb
        _, sv = block.decompose(self.x_bb)
        return np.asarray(sv["logdetK"], dtype=np.float64)

    def _single_dot_py(self, lam_np, r_np):
        lam = float(np.asarray(lam_np))
        r = np.asarray(r_np)
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return np.asarray(block.dot_solve(r), dtype=np.float64)

    def _single_inv_py(self, lam_np, r_np):
        lam = float(np.asarray(lam_np))
        r = np.asarray(r_np)
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return np.asarray(block.inverse(r), dtype=np.float64)

    def _single_sqrt_py(self, lam_np, r_np, transpose_np):
        lam = float(np.asarray(lam_np))
        r = np.asarray(r_np)
        transpose = int(np.asarray(transpose_np))
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return np.asarray(block.matrix_sqrt(r, transpose=transpose), dtype=np.float64)

    def _single_inv_sqrt_py(self, lam_np, r_np, transpose_np):
        lam = float(np.asarray(lam_np))
        r = np.asarray(r_np)
        transpose = int(np.asarray(transpose_np))
        block = lam * self.K_bb + self.Sigma_bb
        block, _ = block.decompose(self.x_bb)
        return np.asarray(block.matrix_inv_sqrt(r, transpose=transpose), dtype=np.float64)

    def _block_logdets_py(self, lambdas_np):
        lambdas_np = np.asarray(lambdas_np)
        if self.cache_blocks and self._cached_blocks is not None and len(self._cached_blocks) == len(lambdas_np):
            return np.asarray(self._cached_logdets, dtype=np.float64)

        out = np.zeros((len(lambdas_np),), dtype=np.float64)
        for i, lam in enumerate(lambdas_np):
            out[i] = self._single_logdet_py(lam)
        return out

    def _block_dots_py(self, lambdas_np, r_blocks_np):
        lambdas_np = np.asarray(lambdas_np)
        r_blocks_np = np.asarray(r_blocks_np)
        out = np.zeros((len(lambdas_np),), dtype=np.float64)

        use_cache = self.cache_blocks and self._cached_blocks is not None and len(self._cached_blocks) == len(lambdas_np)
        if use_cache:
            for i, r in enumerate(r_blocks_np):
                out[i] = float(self._cached_blocks[i].dot_solve(r))
            return out

        for i, (lam, r) in enumerate(zip(lambdas_np, r_blocks_np)):
            out[i] = self._single_dot_py(lam, r)
        return out

    def _block_invs_py(self, lambdas_np, r_blocks_np):
        lambdas_np = np.asarray(lambdas_np)
        r_blocks_np = np.asarray(r_blocks_np)
        out = np.zeros_like(r_blocks_np, dtype=np.float64)

        use_cache = self.cache_blocks and self._cached_blocks is not None and len(self._cached_blocks) == len(lambdas_np)
        if use_cache:
            for i, r in enumerate(r_blocks_np):
                out[i] = np.asarray(self._cached_blocks[i].inverse(r), dtype=np.float64)
            return out

        for i, (lam, r) in enumerate(zip(lambdas_np, r_blocks_np)):
            out[i] = self._single_inv_py(lam, r)
        return out

    def _block_sqrts_py(self, lambdas_np, r_blocks_np, transpose_np):
        lambdas_np = np.asarray(lambdas_np)
        r_blocks_np = np.asarray(r_blocks_np)
        transpose = int(np.asarray(transpose_np))
        out = np.zeros_like(r_blocks_np, dtype=np.float64)

        use_cache = self.cache_blocks and self._cached_blocks is not None and len(self._cached_blocks) == len(lambdas_np)
        if use_cache:
            for i, r in enumerate(r_blocks_np):
                out[i] = np.asarray(self._cached_blocks[i].matrix_sqrt(r, transpose=transpose), dtype=np.float64)
            return out

        for i, (lam, r) in enumerate(zip(lambdas_np, r_blocks_np)):
            out[i] = self._single_sqrt_py(lam, r, transpose)
        return out

    def _block_inv_sqrts_py(self, lambdas_np, r_blocks_np, transpose_np):
        lambdas_np = np.asarray(lambdas_np)
        r_blocks_np = np.asarray(r_blocks_np)
        transpose = int(np.asarray(transpose_np))
        out = np.zeros_like(r_blocks_np, dtype=np.float64)

        use_cache = self.cache_blocks and self._cached_blocks is not None and len(self._cached_blocks) == len(lambdas_np)
        if use_cache:
            for i, r in enumerate(r_blocks_np):
                out[i] = np.asarray(self._cached_blocks[i].matrix_inv_sqrt(r, transpose=transpose), dtype=np.float64)
            return out

        for i, (lam, r) in enumerate(zip(lambdas_np, r_blocks_np)):
            out[i] = self._single_inv_sqrt_py(lam, r, transpose)
        return out

    def matrix_sqrt(self, R: JAXArray, transpose=0) -> JAXArray:
        R_blocks, shape_perm = self._reshape_to_blocks(R)

        if self.use_pure_callback:
            transpose_arr = jnp.asarray(transpose, dtype=jnp.int32)
            if self.callback_mode == "batched":
                out_spec = jax.ShapeDtypeStruct(R_blocks.shape, R_blocks.dtype)
                out_blocks = jax.pure_callback(self._block_sqrts_py, out_spec, self.lambdas, R_blocks, transpose_arr)
            else:
                out_spec = jax.ShapeDtypeStruct((self.bb_size,), R_blocks.dtype)
                out_blocks = jax.vmap(
                    lambda lam, r: jax.pure_callback(
                        self._single_sqrt_py,
                        out_spec,
                        lam,
                        r,
                        transpose_arr,
                        vmap_method="sequential",
                    )
                )(self.lambdas, R_blocks)
        else:
            out_blocks = jax.vmap(lambda lam, r: self._single_sqrt_jax(lam, r, transpose=transpose))(self.lambdas, R_blocks)

        return self._reshape_from_blocks(out_blocks, shape_perm)

    def matrix_inv_sqrt(self, R: JAXArray, transpose=0) -> JAXArray:
        R_blocks, shape_perm = self._reshape_to_blocks(R)

        if self.use_pure_callback:
            transpose_arr = jnp.asarray(transpose, dtype=jnp.int32)
            if self.callback_mode == "batched":
                out_spec = jax.ShapeDtypeStruct(R_blocks.shape, R_blocks.dtype)
                out_blocks = jax.pure_callback(self._block_inv_sqrts_py, out_spec, self.lambdas, R_blocks, transpose_arr)
            else:
                out_spec = jax.ShapeDtypeStruct((self.bb_size,), R_blocks.dtype)
                out_blocks = jax.vmap(
                    lambda lam, r: jax.pure_callback(
                        self._single_inv_sqrt_py,
                        out_spec,
                        lam,
                        r,
                        transpose_arr,
                        vmap_method="sequential",
                    )
                )(self.lambdas, R_blocks)
        else:
            out_blocks = jax.vmap(lambda lam, r: self._single_inv_sqrt_jax(lam, r, transpose=transpose))(self.lambdas, R_blocks)

        return self._reshape_from_blocks(out_blocks, shape_perm)

    def inverse(self, R: JAXArray) -> JAXArray:
        R_blocks, shape_perm = self._reshape_to_blocks(R)

        if self.use_pure_callback:
            if self.callback_mode == "batched":
                out_spec = jax.ShapeDtypeStruct(R_blocks.shape, R_blocks.dtype)
                out_blocks = jax.pure_callback(self._block_invs_py, out_spec, self.lambdas, R_blocks)
            else:
                out_spec = jax.ShapeDtypeStruct((self.bb_size,), R_blocks.dtype)
                out_blocks = jax.vmap(
                    lambda lam, r: jax.pure_callback(
                        self._single_inv_py,
                        out_spec,
                        lam,
                        r,
                        vmap_method="sequential",
                    )
                )(self.lambdas, R_blocks)
        else:
            out_blocks = jax.vmap(self._single_inv_jax)(self.lambdas, R_blocks)

        return self._reshape_from_blocks(out_blocks, shape_perm)

    def dot_solve(self, R: JAXArray) -> Scalar:
        R_blocks, _ = self._reshape_to_blocks(R)

        if self.use_pure_callback:
            if self.callback_mode == "batched":
                out_spec = jax.ShapeDtypeStruct((self.lambdas.shape[0],), R.dtype)
                dots = jax.pure_callback(self._block_dots_py, out_spec, self.lambdas, R_blocks)
            else:
                out_spec = jax.ShapeDtypeStruct((), R.dtype)
                dots = jax.vmap(
                    lambda lam, r: jax.pure_callback(
                        self._single_dot_py,
                        out_spec,
                        lam,
                        r,
                        vmap_method="sequential",
                    )
                )(self.lambdas, R_blocks)
        else:
            dots = jax.vmap(self._single_dot_jax)(self.lambdas, R_blocks)

        return jnp.sum(dots)

    def logL(self, R: JAXArray, stored_values: PyTree, **kwargs) -> Scalar:
        return -0.5 * self.dot_solve(R) - 0.5 * stored_values["logdetK"] - 0.5 * R.size * jnp.log(2 * jnp.pi)


class BlackBoxTildeKernelJITVmap(CovType):
    """Fully JIT-friendly block kernel for the transformed blackbox dimension.

    This class mirrors :class:`BlackBoxTildeKernelCallback` but removes callback
    usage entirely. It assumes the supplied ``CovType`` objects are fully
    traceable under JAX (decompose/solve/logdet), so per-block operations can be
    vmapped directly.

    Notes
    -----
    Runtime is typically dominated by the blackbox solver's own complexity.
    For quasisep-style kernels this can be very fast and scales much better than
    dense fallback approaches.
    """

    def __init__(self, Sigma_bb: CovType, K_bb: CovType, lambdas: JAXArray, bb_dim: int):
        self.Sigma_bb = Sigma_bb
        self.K_bb = K_bb
        self.lambdas = jnp.asarray(lambdas).ravel()
        self.bb_dim = bb_dim
        self.bb_size = None
        self.x_bb = None

    def decompose(self, *X: Tuple, **kwargs):
        self.x_bb = X[self.bb_dim]
        self.bb_size = self.x_bb.shape[-1]

        def block_logdet(lam):
            block = lam * self.K_bb + self.Sigma_bb
            _, sv = block.decompose(self.x_bb)
            return sv["logdetK"]

        block_logdets = jax.vmap(block_logdet)(self.lambdas)
        return self, {"logdetK": jnp.sum(block_logdets)}

    def _reshape_to_blocks(self, R: JAXArray):
        R_perm = jnp.moveaxis(R, self.bb_dim, -1)
        shape_perm = R_perm.shape
        R_blocks = R_perm.reshape((-1, self.bb_size))
        return R_blocks, shape_perm

    def _reshape_from_blocks(self, R_blocks: JAXArray, shape_perm: Tuple[int, ...]):
        R_perm = R_blocks.reshape(shape_perm)
        return jnp.moveaxis(R_perm, -1, self.bb_dim)

    def matrix_sqrt(self, R: JAXArray, transpose=0) -> JAXArray:
        R_blocks, shape_perm = self._reshape_to_blocks(R)

        def block_sqrt(lam, r):
            block = lam * self.K_bb + self.Sigma_bb
            block, _ = block.decompose(self.x_bb)
            return block.matrix_sqrt(r, transpose=transpose)

        out_blocks = jax.vmap(block_sqrt)(self.lambdas, R_blocks)
        return self._reshape_from_blocks(out_blocks, shape_perm)

    def matrix_inv_sqrt(self, R: JAXArray, transpose=0) -> JAXArray:
        R_blocks, shape_perm = self._reshape_to_blocks(R)

        def block_inv_sqrt(lam, r):
            block = lam * self.K_bb + self.Sigma_bb
            block, _ = block.decompose(self.x_bb)
            return block.matrix_inv_sqrt(r, transpose=transpose)

        out_blocks = jax.vmap(block_inv_sqrt)(self.lambdas, R_blocks)
        return self._reshape_from_blocks(out_blocks, shape_perm)

    def inverse(self, R: JAXArray) -> JAXArray:
        R_blocks, shape_perm = self._reshape_to_blocks(R)

        def block_inv(lam, r):
            block = lam * self.K_bb + self.Sigma_bb
            block, _ = block.decompose(self.x_bb)
            return block.inverse(r)

        out_blocks = jax.vmap(block_inv)(self.lambdas, R_blocks)
        return self._reshape_from_blocks(out_blocks, shape_perm)

    def dot_solve(self, R: JAXArray) -> Scalar:
        R_blocks, _ = self._reshape_to_blocks(R)

        def block_dot(lam, r):
            block = lam * self.K_bb + self.Sigma_bb
            block, _ = block.decompose(self.x_bb)
            return block.dot_solve(r)

        return jnp.sum(jax.vmap(block_dot)(self.lambdas, R_blocks))

    def logL(self, R: JAXArray, stored_values: PyTree, **kwargs) -> Scalar:
        return -0.5 * self.dot_solve(R) - 0.5 * stored_values["logdetK"] - 0.5 * R.size * jnp.log(2 * jnp.pi)


class _LuasLasrachKernelNDBase(CovType):
    """Shared ND LuasLasrach decomposition logic.

    This base class handles:

    - Kronecker assembly for dense ``evaluate``
    - decomposition of non-blackbox dimensions
    - construction of transform/inverse-transform closures
    - bookkeeping of determinant contributions

    Subclasses only need to implement ``_build_kf_tilde`` to choose how the
    transformed blackbox blocks are solved (callback vs pure JAX).
    """

    def __init__(
        self,
        Sigma,
        K,
        blackbox_dim: int = -1,
        eigen_dims=None,
        use_stored_values: Optional[bool] = False,
    ):
        self.Sigma = Sigma
        self.K = K
        self.dim = len(Sigma)
        self.blackbox_dim = blackbox_dim % self.dim

        if eigen_dims is None:
            self.eigen_dims = tuple(d for d in range(self.dim) if d != self.blackbox_dim)
        else:
            self.eigen_dims = tuple(d for d in eigen_dims if d != self.blackbox_dim)

        self.K_list = [K]
        self.decompose = self.eigendecomp_use_stored_values if use_stored_values else self.eigendecomp_no_stored_values

    def _build_kf_tilde(self, lambdas: JAXArray):
        raise NotImplementedError

    def evaluate(self, *X, **kwargs):
        """Construct the full dense covariance matrix.

        Primarily useful for diagnostics and small-scale validation since this
        explicitly forms Kronecker products in dense form.
        """
        dim = len(X)
        Sigma_full = self.Sigma[0].evaluate(X[0], X[0], **kwargs)
        for d in range(1, dim):
            Sigma_full = jnp.kron(Sigma_full, self.Sigma[d].evaluate(X[d], X[d], **kwargs))

        K_full = self.K[0].evaluate(X[0], X[0], **kwargs)
        for d in range(1, dim):
            K_full = jnp.kron(K_full, self.K[d].evaluate(X[d], X[d], **kwargs))

        return Sigma_full + K_full

    def eigendecomp_no_stored_values(self, *X: Tuple, stored_values: Optional[PyTree] = None):
        """Decompose kernel without attempting reuse of previous stored values.

        Steps
        -----
        1. For each non-blackbox dimension ``d``:
           - decompose ``Sigma[d]``
           - form transformed ``K_tilde[d] = Sigma[d]^(-1/2) K[d] Sigma[d]^(-1/2)``
           - eigendecompose ``K_tilde[d]``
        2. Kronecker-combine eigenvalues into ``all_lam``.
        3. Build transformation closures to map residual tensors into/out of the
           transformed block basis.
        4. Delegate transformed block decomposition to subclass-provided
           ``kf_tilde`` object.
        """
        stored_values = {} if stored_values is None else stored_values

        if stored_values:
            R_shape = stored_values["R_shape"]
            total_size = jnp.prod(jnp.array(R_shape))
        else:
            total_size = calc_total_size(X)

        gp_dim = len(X)
        sigma_decomp_mats = []
        eigen_decomp_mats = []
        all_lam = jnp.ones(1)
        all_lam_shape = ()
        stored_values["logdetK"] = 0.0

        for d in range(gp_dim):
            if d in self.eigen_dims:
                Sigma_d_new, stored_values_d = self.Sigma[d].decompose(X[d])
                K_d_new = Sigma_d_new.inv_sqrt_transform(self.K[d])

                # Important: do NOT overwrite methods in-place here.
                # Re-wrapping on every decompose call causes nested vmaps and severe slowdown/hangs.

                stored_values[f"lam_{d}"], stored_values[f"Q_{d}"] = K_d_new.eigendecomp(X[d])
                all_lam = jnp.kron(all_lam.reshape(all_lam_shape + (1,)), stored_values[f"lam_{d}"])
                all_lam_shape = all_lam.shape

                sigma_decomp_mats.append(Sigma_d_new)
                eigen_decomp_mats.append(stored_values[f"Q_{d}"])
                stored_values["logdetK"] += (total_size / X[d].shape[-1]) * stored_values_d["logdetK"]
            else:
                sigma_decomp_mats.append(None)
                eigen_decomp_mats.append(None)

        def transform_fn(R, transpose=0):
            R_prime = cyclic_transpose(R, 2)
            if transpose:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        ms = sigma_decomp_mats[d].matrix_sqrt
                        ms_use = vmap_for_tensors(ms) if gp_dim > 2 else ms
                        R_prime = ms_use(R_prime, transpose=1)
                        R_prime = eigen_decomp_mats[d].T @ R_prime
                    R_prime = cyclic_transpose(R_prime, 1)
            else:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        R_prime = eigen_decomp_mats[d] @ R_prime
                        ms = sigma_decomp_mats[d].matrix_sqrt
                        ms_use = vmap_for_tensors(ms) if gp_dim > 2 else ms
                        R_prime = ms_use(R_prime, transpose=0)
                    R_prime = cyclic_transpose(R_prime, 1)
            return cyclic_transpose(R_prime, -2)

        def inv_transform_fn(R, transpose=0):
            R_prime = cyclic_transpose(R, 2)
            if transpose:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        R_prime = eigen_decomp_mats[d] @ R_prime
                        mis = sigma_decomp_mats[d].matrix_inv_sqrt
                        mis_use = vmap_for_tensors(mis) if gp_dim > 2 else mis
                        R_prime = mis_use(R_prime, transpose=1)
                    R_prime = cyclic_transpose(R_prime, 1)
            else:
                for d in range(gp_dim):
                    if d in self.eigen_dims:
                        mis = sigma_decomp_mats[d].matrix_inv_sqrt
                        mis_use = vmap_for_tensors(mis) if gp_dim > 2 else mis
                        R_prime = mis_use(R_prime, transpose=0)
                        R_prime = eigen_decomp_mats[d].T @ R_prime
                    R_prime = cyclic_transpose(R_prime, 1)
            return cyclic_transpose(R_prime, -2)

        self.transform_fn = transform_fn
        self.inv_transform_fn = inv_transform_fn

        stored_values["all_lam"] = all_lam.ravel()
        self.kf_tilde = self._build_kf_tilde(stored_values["all_lam"])
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(*X)

        return self, stored_values

    def eigendecomp_use_stored_values(self, *X: Tuple, stored_values: Optional[PyTree] = None):
        return self.eigendecomp_no_stored_values(*X, stored_values=stored_values)

    def matrix_sqrt(self, R: JAXArray, transpose=0) -> JAXArray:
        if transpose:
            R_prime = self.transform_fn(R, transpose=1)
            R_prime = self.kf_tilde.matrix_sqrt(R_prime, transpose=1)
        else:
            R_prime = self.kf_tilde.matrix_sqrt(R, transpose=0)
            R_prime = self.transform_fn(R_prime, transpose=0)
        return R_prime

    def matrix_inv_sqrt(self, R: JAXArray, transpose=0) -> JAXArray:
        if transpose:
            R_prime = self.kf_tilde.matrix_inv_sqrt(R, transpose=1)
            R_prime = self.inv_transform_fn(R_prime, transpose=1)
        else:
            R_prime = self.inv_transform_fn(R, transpose=0)
            R_prime = self.kf_tilde.matrix_inv_sqrt(R_prime, transpose=0)
        return R_prime

    def inverse(self, R: JAXArray) -> JAXArray:
        R_prime = self.inv_transform_fn(R, transpose=0)
        R_prime = self.kf_tilde.inverse(R_prime)
        return self.inv_transform_fn(R_prime, transpose=1)

    def dot_solve(self, R: JAXArray) -> Scalar:
        K_inv_R = self.inverse(R)
        return jnp.sum(R * K_inv_R)

    def logL(self, R: JAXArray, stored_values: PyTree) -> Scalar:
        """Compute log-likelihood in transformed coordinates.

        The transformed log-likelihood from ``kf_tilde`` is corrected by the
        determinant contribution from non-blackbox whitening transforms.
        """
        R_prime = self.inv_transform_fn(R, transpose=0)
        logL_tilde = self.kf_tilde.logL(R_prime, stored_values["kf_tilde_stored"])
        return logL_tilde - 0.5 * stored_values["logdetK"]


class LuasLasrachKernelND(_LuasLasrachKernelNDBase):
    """ND LuasLasrach kernel with optional callback-based blackbox execution.

    Use this class when your blackbox ``CovType`` includes non-JAX code or
    external libraries (e.g. george/HODLR). The non-blackbox dimensions still
    use the same Kronecker/eigen transform machinery as the JIT class.

    Parameters
    ----------
    Sigma, K
        Tuples/lists of per-dimension ``CovType`` objects such that
        ``K_total = kron(Sigma_d) + kron(K_d)``.
    blackbox_dim
        Dimension index to be treated as blockwise blackbox solve dimension.
    eigen_dims
        Optional iterable of dimensions to eigendecompose. By default all
        dimensions except ``blackbox_dim`` are used.
    use_stored_values
        Placeholder compatibility flag; currently recomputes decomposition.
    use_pure_callback
        If ``True`` run blackbox operations through ``jax.pure_callback``.
    callback_mode
        ``"per_block_vmap"`` or ``"batched"`` callback strategy.
    cache_blocks
        If ``True`` and ``callback_mode='batched'``, decomposed block objects are
        cached at ``decompose`` time and reused during repeated ``logL`` calls.
        This substantially accelerates logL-only workloads for non-JAX backends.
    """

    def __init__(self, *args, use_pure_callback: bool = True, callback_mode: str = "per_block_vmap", cache_blocks: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_pure_callback = use_pure_callback
        self.callback_mode = callback_mode
        self.cache_blocks = cache_blocks

    def _build_kf_tilde(self, lambdas: JAXArray):
        return BlackBoxTildeKernelCallback(
            Sigma_bb=self.Sigma[self.blackbox_dim],
            K_bb=self.K[self.blackbox_dim],
            lambdas=lambdas,
            bb_dim=self.blackbox_dim,
            use_pure_callback=self.use_pure_callback,
            callback_mode=self.callback_mode,
            cache_blocks=self.cache_blocks,
        )


class LuasLasrachKernelNDJIT(_LuasLasrachKernelNDBase):
    """Fully JAX/JIT-compatible ND LuasLasrach kernel.

    This variant assumes all operations in the blackbox ``CovType`` are JAX
    traceable, allowing vmapped block decomposition and solves without host
    callbacks.

    In practice this is the preferred high-performance path for quasisep-like
    blackbox kernels implemented natively in JAX.
    """

    def _build_kf_tilde(self, lambdas: JAXArray):
        return BlackBoxTildeKernelJITVmap(
            Sigma_bb=self.Sigma[self.blackbox_dim],
            K_bb=self.K[self.blackbox_dim],
            lambdas=lambdas,
            bb_dim=self.blackbox_dim,
        )

def LuasLasrachKernel(jit = True, **kwargs):
    if jit:
        return LuasLasrachKernelNDJIT(**kwargs)
    else:
        return LuasLasrachKernelND(**kwargs)

