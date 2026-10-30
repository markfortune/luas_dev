import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from typing import Callable, Tuple, Union, Any, Optional
import tinygp

from luas import WhiteNoiseKernel, SeparableKernel
from luas.kernels.covtype import Outer, GeneralQuasisepPlusNoise, GeneralQuasisep, CovType, Identity, ScaledIdentity, Diagonal
from luas.kernels.householder import HouseholderProduct, orthonormal_nullspace_gen
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kronecker_fns import tensor_mult
from luas.kernels.tinygp_ext import ScaledKernel
from luas.kernels.BlockKernel import Block2x2Kernel, DifferentBlocksKernel
from luas.kernels.MixingMatQuasisep import MixingMatQuasisep
from luas.kernels.MixingMatGeneral import MixingMatGeneral

__all__ = [
    "MultiLowRankKernel",
]

class MultiLowRankKernel(CovType):
    r"""GP optimisation for a separable noise term plus a sum of low-rank kernel
    terms, with all low-rank vectors along the same dimension.

    With the low-rank vectors along wavelength (``fast_dim = 1``), this handles
    kernel functions of the form

    .. math::

        k(\lambda, \lambda', t, t') = s_\lambda(\lambda, \lambda')\, s_t(t, t')
        + \sum_{i=1}^{N_\alpha} \alpha_i(\lambda)\alpha_i(\lambda')\, k_{t; i}(t, t'),

    which on a rectangular grid gives the covariance matrix

    .. math::

        \mathbf{K} = \mathbf{\Sigma}_\lambda \otimes \mathbf{\Sigma}_t
        + \sum_{i=1}^{N_\alpha} \boldsymbol{\alpha}_{\lambda; i}
        \boldsymbol{\alpha}_{\lambda; i}^T \otimes \mathbf{K}_{t; i}.

    The dimensions can be flipped (``fast_dim = 0``) to instead model terms
    :math:`\mathbf{K}_{\lambda; j} \otimes \boldsymbol{\beta}_{t; j}
    \boldsymbol{\beta}_{t; j}^T`, e.g. to analytically marginalise over a
    wavelength-dependent amplitude of a fixed time-dependent ramp or baseline
    model. ``fast_dim`` is always the dimension holding the full kernels
    :math:`k_{t; i}`, and the low-rank vectors lie along the other dimension.

    Each low-rank term models a systematic with a fixed shape along one dimension
    (e.g. common-mode systematics or stellar granulation with a wavelength-dependent
    amplitude). The model is closely related to PCA/SysRem with a GP prior on each
    component, and to the linear model of coregionalisation.

    **Method.** The noise term is whitened by
    :math:`\mathbf{\Sigma}_\lambda^{-\frac{1}{2}}`, and the whitened vectors
    :math:`\boldsymbol{\alpha}'_{\lambda; i} = \mathbf{\Sigma}_\lambda^{-\frac{1}{2}}
    \boldsymbol{\alpha}_{\lambda; i}` are QR decomposed using :math:`N_\alpha`
    Householder transformations. This reduces the problem to a dense
    :math:`(N_\alpha N_t, N_\alpha N_t)` block, decomposed with the quasiseparable
    method of Gordon et al. (2020) when the time kernels are quasiseparable, plus a
    diagonal block. For :math:`N_\alpha = 1` this is identical to
    :class:`luas.LuasLasrachKernel`.

    **Runtime scaling** (low-rank vectors along :math:`\lambda`, quasiseparable
    :math:`k_{t; i}` with mean rank :math:`\bar{J}_{K_t}`, diagonal
    :math:`\mathbf{\Sigma}_t`):

    - Diagonal :math:`\mathbf{\Sigma}_\lambda`:
      :math:`\mathcal{O}(N_\alpha^3 N_t \bar{J}_{K_t}^2 + N_\alpha N_\lambda N_t)`.
    - Plus :math:`\mathcal{O}(N_\lambda J_{\Sigma_\lambda}^2 + N_\lambda N_t
      J_{\Sigma_\lambda})` for quasiseparable :math:`\mathbf{\Sigma}_\lambda`, or
      :math:`\mathcal{O}(N_\lambda^3 + N_\lambda^2 N_t)` for arbitrary
      :math:`\mathbf{\Sigma}_\lambda`.
    - With ``orthogonalise = True``:
      :math:`\mathcal{O}(N_\alpha N_t \bar{J}_{K_t}^2 + N_\alpha N_\lambda N_t)`
      plus the same :math:`\mathbf{\Sigma}_\lambda` costs.
    - Non-quasiseparable :math:`k_{t; i}` (``use_quasi = False``):
      :math:`\mathcal{O}(N_\alpha^3 N_t^3 + N_\alpha N_\lambda N_t)`.

    **Selection.** :class:`luas.GP` selects this optimisation automatically when the
    kernel function returns the noise term followed only by low-rank terms, all with
    their low-rank vectors along the same dimension. Rank-1 terms are recognised
    from kernels returning an outer product, i.e. ``luas.kernels.lowrank.Constant``,
    ``luas.kernels.lowrank.Amplitude``, ``luas.kernels.Constant`` and
    ``luas.kernels.Linear``. ``fast_dim`` is set automatically, and ``use_quasi``
    is set to whether all kernels along ``fast_dim`` are quasiseparable or diagonal.

    Warning:
        :math:`\mathbf{\Sigma}_\lambda` must be positive definite and the noise term
        must be the **first** tuple returned by the kernel function.
        :math:`\mathbf{\Sigma}_t` is normally diagonal (e.g. ``KroneckerDelta``). A
        non-diagonal :math:`\mathbf{\Sigma}_t` is also accepted, but is handled as
        additional components, increasing the runtime.

    Example:
        Common-mode systematics plus extra systematics in a single light curve, with
        readout noise (from the ``luas`` tutorial):

        .. code-block:: python

            def kf(p, X):
                s_l = luas.kernels.quasisep.Noise(jnp.exp(p["log_sigma"]))
                s_l += jnp.exp(2*p["log_h_readout"][0]) * luas.kernels.quasisep.Exp(jnp.exp(p["log_l_readout"][0]))
                s_t = luas.kernels.quasisep.KroneckerDelta()

                # Common-mode systematics: same shape and amplitude at every wavelength
                alpha_l1 = luas.kernels.lowrank.Constant(jnp.exp(p["log_h_const"][0]))
                k_t1 = luas.kernels.quasisep.Matern32(jnp.exp(p["log_l_t1"][0]))

                # Extra systematics only in the 10th light curve
                e_10 = jnp.zeros(X[0].shape[-1]).at[9].set(jnp.exp(p["log_h10"][0]))
                alpha_l2 = luas.kernels.lowrank.Amplitude(e_10)
                k_t2 = luas.kernels.quasisep.SquaredExpApprox(jnp.exp(p["log_l_t2"][0]))

                return (s_l, s_t), (alpha_l1, k_t1), (alpha_l2, k_t2)

            gp = luas.GP(p, (x_l, x_t), kf = kf)  # Uses MultiLowRankKernel with fast_dim = 1

    Args:
        Sigma (tuple[CovType, CovType]): The covariance objects
            :math:`(\mathbf{\Sigma}_\lambda, \mathbf{\Sigma}_t)` for the noise term.
        *K_list (tuple[CovType, CovType]): Each low-rank term, given as a tuple with
            one covariance object per dimension. The entry along the low-rank
            dimension (``1 - fast_dim``) should be a rank-1 outer product (see above).
            Other low-rank covariance objects which define ``rank`` (e.g.
            ``luas.kernels.lowrank.SquaredExp(..., rank = R)``) are also accepted when
            this kernel is forced with ``use_kernel``; they are eigendecomposed and
            split into ``R`` rank-1 components sharing the same kernel along
            ``fast_dim``.
        fast_dim (int): **Required.** The dimension holding the full kernels
            :math:`k_{t; i}` (``1`` if the low-rank vectors are along dimension 0,
            ``0`` if they are along dimension 1). Must be passed as a keyword
            argument to :class:`luas.GP` when using ``use_kernel =
            luas.MultiLowRankKernel``.
        QR_transform (bool, optional): Whether to use the Householder QR
            transformation to reduce the dense part of the problem to an
            :math:`(N_\alpha N_t, N_\alpha N_t)` block. Only has an effect when the
            total rank is less than the length of the low-rank dimension. Defaults to
            ``True``; setting ``False`` is mainly useful for testing, and currently
            fails if :math:`\mathbf{\Sigma}_t` is not diagonal.
        use_stored_values (bool, optional): Currently unused.
        use_quasi (bool, optional): If ``True``, decompose the dense block using
            quasiseparable methods, which requires every kernel along ``fast_dim`` to
            be quasiseparable or diagonal. If ``False``, use a general Cholesky
            decomposition. Defaults to ``True``.
        orthogonalise (bool, optional): If ``True``, assume the whitened vectors
            :math:`\boldsymbol{\alpha}'_{\lambda; i}` are mutually orthogonal, so the
            problem splits into :math:`N_\alpha` independent 1D GPs (scaling linearly
            in :math:`N_\alpha`). Requires ``QR_transform = True`` and a total rank no
            larger than the length of the low-rank dimension. Defaults to ``False``.
        **kwargs: Ignored, accepted so that all Kronecker kernels share a common
            constructor interface.

    Warning:
        ``orthogonalise = True`` discards the off-diagonal terms of the QR
        decomposition, so the log-likelihood is only correct if the whitened
        vectors really are orthogonal. This is not checked.

    See Also:
        :class:`luas.LuasLasrachKernel`, the :math:`N_\alpha = 1` special case.
        :class:`luas.DoubleMultiLowRankKernel` for low-rank terms along both
        dimensions.
        :class:`luas.LuasPlusMultiLowRankKernel` for also including a full-rank 2D
        kernel term.
    """
    def __init__(
        self,
        Sigma,
        *K_list,
        fast_dim = None,
        QR_transform = True,
        use_stored_values: Optional[bool] = False,
        use_quasi = True,
        orthogonalise = False,
        **kwargs,
    ):
        assert fast_dim is not None # Must specify the fast dimension
        self.Sigma = Sigma[0], Sigma[1]
        self.K_list = K_list
        self.fast_dim = fast_dim
        self.QR_transform = QR_transform
        self.N_alpha = len(K_list)
        self.use_quasi = use_quasi

        if self.N_alpha == 1:
            # For one term converges to same optimisation as LuasLasrachKernel
            self.opt_name = "LuasLasrachKernel/MultiLowRankKernel"
        else:
            self.opt_name = "MultiLowRankKernel"

        self.orthogonalise = orthogonalise

        if orthogonalise:
            print("Will orthogonalise components, not this will ignore any non-orthogonal component of the whitened alpha vectors!")

        if not QR_transform and orthogonalise:
            raise Exception("Orthogonalisation requires a QR transformation! Either set QR_transform = False or orthogonalise = False")

        if use_stored_values:
            raise NotImplementedError("Use of pre-calculated stored values not yet implemented.")
        else:
            self.decompose = self.decompose_no_stored_values


    def _rotate_to_fast_dim_wrapper(self, fn, fast_dim):

        def wrapped_fn(R, **kwargs):
            if fast_dim == 0:
                R_prime = R.T
            else:
                R_prime = R.copy()

            R_prime = fn(R_prime, **kwargs)

            if self.fast_dim == 0:
                R_prime = R_prime.T
                
            return R_prime

        return wrapped_fn
    

    def decompose_no_stored_values(
        self,
        X: Tuple[JAXArray],
        stored_values: Optional[PyTree] = None,
        full = True,
        idx = None,
    ) -> PyTree:

        # should to something about idx and full
        assert full == True

        stored_values = {} if stored_values is None else stored_values
        
        non_cel_vec = X[1-self.fast_dim]
        cel_vec = X[self.fast_dim]

        # Generate transformations
        self.Sigma_transf, stored_values_Sigma_transf = self.Sigma[1-self.fast_dim].decompose(non_cel_vec, full = True)

        # Computes the log determinant of K
        stored_values["logdet"] = cel_vec.shape[-1]*stored_values_Sigma_transf["logdet"]
        
        stored_values["non_cel_rank"] = 0
        for K_i in self.K_list:
            stored_values["non_cel_rank"] += K_i[1-self.fast_dim].rank(non_cel_vec)
        
        stored_values["A"] = jnp.zeros((non_cel_vec.shape[-1], stored_values["non_cel_rank"]))
        cel_kernel_order = []
        
        col_i = 0
        for (i, K_i) in enumerate(self.K_list):
            if isinstance(K_i[1-self.fast_dim], Outer):
                K_i_non_cel, _ = K_i[1-self.fast_dim].decompose(non_cel_vec)
                stored_values["A"] = stored_values["A"].at[:, col_i].set(K_i_non_cel.alpha)
                cel_kernel_order.append(K_i[self.fast_dim])
                col_i += 1
            else:
                lam, Q = K_i[1-self.fast_dim].eigendecomp(non_cel_vec)

                rank_i = K_i[1-self.fast_dim].rank(non_cel_vec)
                lam_nonzero = lam[-rank_i:]

                if isinstance(Q, HouseholderProduct):
                    alpha_eigen = Q.V[:, -rank_i:] @ jnp.diag(jnp.sqrt(lam_nonzero))
                else:
                    alpha_eigen = Q[:, -rank_i:] @ jnp.diag(jnp.sqrt(lam_nonzero))
                
                for j in range(rank_i):
                    stored_values["A"] = stored_values["A"].at[:, col_i].set(alpha_eigen[:, j])
                    
                    # Could accept a list of kernels in K_i[self.fast_dim]
                    cel_kernel_order.append(K_i[self.fast_dim])
                    col_i += 1
        if stored_values["A"].shape[1] > stored_values["A"].shape[0] and self.orthogonalise:
            raise Exception(f"Too many components to orthogonalise! Rank of low-rank dimension = {stored_values['A'].shape[1]},"\
                            f"which is greater than the length of the low-rank dimension {non_cel_vec.shape[-1]}")
        
        # Transform vectors 
        stored_values["J"] = self.Sigma_transf.matrix_inv_sqrt(stored_values["A"], transpose = 0)

        # If the total rank is less than the length of that dimension, we can reduce the dimension by exploiting sparsity
        reduce_dim1 = stored_values["non_cel_rank"] < non_cel_vec.shape[-1] and self.QR_transform

        # Even if the total rank is equal to the length of the dimension, orthogonalising the vectors still gives a speed boost
        reduce_dim2 = stored_values["non_cel_rank"] <= non_cel_vec.shape[-1] and self.orthogonalise

        self.reduce_dim = reduce_dim1 or reduce_dim2

        if self.reduce_dim:
            stored_values["J"], U, self.householder_transform = orthonormal_nullspace_gen(stored_values["J"])
        
        # Handle Sigma mat in the fast_dim, likely just diagonal
        if not isinstance(self.Sigma[self.fast_dim], (Identity, ScaledIdentity, Diagonal)):
            add_basis = jnp.eye(stored_values["non_cel_rank"])
            stored_values["J"] = jnp.concatenate([stored_values["J"], add_basis], axis = 1)

            for j in range(stored_values["non_cel_rank"]):
                cel_kernel_order.append(self.Sigma[self.fast_dim])

            if hasattr(self.Sigma[self.fast_dim], "noise_model") and self.Sigma[self.fast_dim].noise_model is not None:
                assert self.use_quasi == False # Not implemented otherwise
            
            if self.use_quasi:
                cel_diag = (self.Sigma[self.fast_dim].diag + self.Sigma[self.fast_dim].wn_diag)*jnp.ones(cel_vec.shape[-1])
            else:
                cel_diag = 0.*jnp.ones(cel_vec.shape[-1])
            
        else:
            cel_diag = (self.Sigma[self.fast_dim].diag + self.Sigma[self.fast_dim].wn_diag)*jnp.ones(cel_vec.shape[-1])

        if self.reduce_dim:
            total_cel_diag = jnp.kron(cel_diag, jnp.ones(stored_values["non_cel_rank"]))
        else:
            total_cel_diag = jnp.kron(cel_diag, jnp.ones(non_cel_vec.shape[-1]))

        # print(stored_values["J"], jnp.diag(jnp.diag(stored_values["J"])))
        
        if self.orthogonalise:
            if self.fast_dim:
                D_mat = total_cel_diag.reshape((stored_values["non_cel_rank"], -1), order = "C")
            else:
                D_mat = total_cel_diag.reshape((stored_values["non_cel_rank"], -1), order = "F")
            # print(D_mat, jnp.diag(stored_values["J"]), cel_kernel_order)
            kf_quasi2D = DifferentBlocksKernel(D_mat, jnp.diag(stored_values["J"])**2, cel_kernel_order, block_dim = self.fast_dim)
        elif self.use_quasi:
            # kf_quasi2D = MixingMatQuasisep(mixing_mat = jnp.diag(jnp.diag(stored_values["J"])), kernel_list = cel_kernel_order,
            #                                 diag = total_cel_diag, fast_dim = self.fast_dim)
            kf_quasi2D = MixingMatQuasisep(mixing_mat = stored_values["J"], kernel_list = cel_kernel_order,
                                            diag = total_cel_diag, fast_dim = self.fast_dim)
        else:
            kf_quasi2D = MixingMatGeneral(mixing_mat = stored_values["J"], kernel_list = cel_kernel_order,
                                            diag = total_cel_diag, fast_dim = self.fast_dim)

        if self.reduce_dim:
            if isinstance(self.Sigma[self.fast_dim], (Identity, ScaledIdentity, Diagonal)):
                null_space_rank = non_cel_vec.shape[-1] - stored_values["non_cel_rank"]
                if self.fast_dim == 0:
                    rest_cel_diag = jnp.outer(cel_diag, jnp.ones(null_space_rank))
                else:
                    rest_cel_diag = jnp.outer(jnp.ones(null_space_rank), cel_diag)
                    
                kf_D = WhiteNoiseKernel(diag = rest_cel_diag)
            else:
                if self.fast_dim == 0:
                    kf_D = SeparableKernel(self.Sigma[self.fast_dim], Identity())
                else:
                    kf_D = SeparableKernel(Identity(), self.Sigma[self.fast_dim])

            self.kf_tilde = Block2x2Kernel(kf_A = kf_quasi2D, kf_D_CAB = kf_D,
                                           dim_split = 1-self.fast_dim, A_full = True, D_full = True,
                                           split_loc = stored_values["non_cel_rank"])
        else:
            self.kf_tilde = kf_quasi2D
        
        self.kf_tilde, stored_values["kf_tilde_stored"] = self.kf_tilde.decompose(X, full = True)
        stored_values["logdet"] += stored_values["kf_tilde_stored"]["logdet"]

        self.logdet = stored_values["logdet"]

        return self, stored_values
        
    
    def transform_fn(self, R, transpose = 0):
        
        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        if transpose:
            R_prime = self.Sigma_transf.matrix_sqrt(R_prime, transpose = 1)

            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 0)
        else:
            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 1)
            R_prime = self.Sigma_transf.matrix_sqrt(R_prime, transpose = 0)

        if self.fast_dim == 0:
            R_prime = R_prime.T
            
        return R_prime


    def inv_transform_fn(self, R, transpose = 0):
        
        if self.fast_dim == 0:
            R_prime = R.T
        else:
            R_prime = R.copy()

        if transpose:
            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 1)
            R_prime = self.Sigma_transf.matrix_inv_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.Sigma_transf.matrix_inv_sqrt(R_prime, transpose = 0)
            if self.reduce_dim:
                R_prime = self.householder_transform(R_prime, transpose = 0)

        if self.fast_dim == 0:
            R_prime = R_prime.T
            
        return R_prime
    

    def matrix_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:

        if transpose:
            R_prime = self.transform_fn(R, transpose = 1)
            R_prime = self.kf_tilde.matrix_sqrt(R_prime, transpose = 1)
        else:
            R_prime = self.kf_tilde.matrix_sqrt(R, transpose = 0)
            R_prime = self.transform_fn(R_prime, transpose = 0)
        
        return R_prime


    def matrix_inv_sqrt(
        self,
        R: JAXArray,
        transpose = 0,
    ) -> JAXArray:

        if transpose:
            R_prime = self.kf_tilde.matrix_inv_sqrt(R, transpose = 1)
            R_prime = self.inv_transform_fn(R_prime, transpose = 1)
        else:
            R_prime = self.inv_transform_fn(R, transpose = 0)
            R_prime = self.kf_tilde.matrix_inv_sqrt(R_prime, transpose = 0)
        
        return R_prime


    def matmul(self, X1, X2, R, **kwargs):

        K_R = tensor_mult(self.Sigma, X1, X2, R, **kwargs)

        for K in self.K_list:
            K_R += tensor_mult(K, X1, X2, R, **kwargs)
        
        return K_R



