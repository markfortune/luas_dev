import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Callable, Tuple, Any, Union

import jax
from jax import grad, value_and_grad, hessian, vmap
import jax.numpy as jnp
import jax.scipy.linalg as JLA
from jax.flatten_util import ravel_pytree

import luas.kernels.covtype as covtype
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas.kernels import (
    SingleKronTermKernel,
    LuasKernel,
    LuasLasrachKernel,
    MultiTermKernel,
    GeneralKernel,
)

cov_types_1D = [covtype.General, covtype.GeneralQuasisep, covtype.Diagonal, covtype.Exp,
                covtype.Toeplitz, covtype.Banded, covtype.Lowrank, covtype.Periodic,
                covtype.Identity, covtype.OuterPlusScaledIdentity, covtype.Outer]

cov_type_cel_compat = [covtype.GeneralQuasisep, covtype.Exp, covtype.Diagonal,
                       covtype.Banded, covtype.Identity]
cov_type_cel = [covtype.GeneralQuasisep, covtype.Exp]

cov_types_2D = [covtype.General2D, covtype.Block2D, covtype.Diagonal2D]
Sigma2D = [covtype.Diagonal2D, covtype.Block2D, covtype.Quasisep2D, covtype.General2D]

def kernel_selector(
        regressors,
        Sigma,
        *args,
        verbose = False,
        max_gen_cholesky_blocks = 10,
    ):

    dim = len(regressors)
    print_str = ""

    # Ensure dimensions are correct
    assert (len(Sigma) <= dim)
    for arg in args:
        assert len(arg) == dim

    nonkron_Sigma = (type(Sigma[0]) in Sigma2D)
    if not nonkron_Sigma:
        assert len(Sigma) == dim

    if dim == 1:
        kernel = OneDimKernel
        kernel_kwargs = {}
    elif dim == 2:
        N_l = regressors[0].shape[-1]
        N_t = regressors[1].shape[-1]
        if N_l > N_t:
            longest_dim = 0
        else:
            longest_dim = 1
    else:
        print(f"Haven't implemented {dim}D GPs yet! Can only have one or two dimensions, first argument should be a tuple of regression variables in each dimension.")

    data_shape = jnp.array([regress.shape[-1] for regress in regressors])
    # longest_dim = data_shape.argmax()
    num_terms = 1 + len(args)

    if dim == 1:
        if type(Sigma[0]) in cov_types_1D:
            print_str += "1D GP, potentially with optimisations\n"
        else:
            print_str += "Error! Only regression variable(s) for a single dimension but kernel terms not consistent with a single dimension.\n"
    elif dim == 2:
        N_l = regressors[0].shape[-1]
        N_t = regressors[1].shape[-1]
        data_shape = (N_l, N_t)

        # Check whether there is a valid "Celerite dimension", will do nothing if args is None
        longest_dim_celerite_compat = True
        shortest_dim_celerite_compat = True
        for arg in args:
            longest_dim_celerite_compat = (type(arg[longest_dim]) in cov_type_cel_compat) and longest_dim_celerite_compat
            shortest_dim_celerite_compat = (type(arg[1 - longest_dim]) in cov_type_cel_compat) and shortest_dim_celerite_compat

        if longest_dim_celerite_compat:
            cel_dim = longest_dim
        elif shortest_dim_celerite_compat:
            cel_dim = 1 - longest_dim
        else:
            cel_dim = None
        
        if nonkron_Sigma:
            # Optimisations where Sigma is not a Kronecker product
            
            if num_terms == 1:
                if type(Sigma[0]) == covtype.Diagonal2D:
                    print_str += "White noise, could maybe include with 1D GP case\n"
                elif type(Sigma[0]) == covtype.Block2D:
                    print_str += "sum of 1D GPs\n"
                elif type(Sigma[0]) == covtype.Celerite2D:
                    print_str += "Sortable 2D celerite opt\n"
                elif type(Sigma[0]) == covtype.General2D:
                    print_str += "Do general Cholesky"
                else:
                    print_str += "Shouldn't be able to see this!\n"
                    
            if num_terms > 1:
                if cel_dim is not None:
                    if type(Sigma[0]) == covtype.Diagonal2D:
                        print_str += "Gordon optimisation\n"
                    elif type(Sigma[0]) == covtype.Block2D:
                        print_str += "Eigendecomp Blocks to be general diagonal, then Gordon opt\n"
                    elif type(Sigma[0]) in [covtype.Celerite2D, covtype.General2D]:
                        print_str += "Error, can't combine Celerite2D or General2D with other terms\n"
                    else:
                        print_str += "Shouldn't be able to see this!"
                else:
                    print_str += "Need one of the dimensions to be Celerite compatible if Sigma is not a Kronecker product!\n"
                
        else:

            longest_dim_contains_cel = (type(Sigma[longest_dim]) in cov_type_cel)
            for arg in args:
                longest_dim_contains_cel = longest_dim_contains_cel or (type(arg[longest_dim]) in cov_type_cel)
                        
            if longest_dim_contains_cel and longest_dim_celerite_compat and type(Sigma[longest_dim]) in cov_type_cel_compat:
                fully_cel_dim = longest_dim
            elif shortest_dim_celerite_compat and type(Sigma[longest_dim-1]) in cov_type_cel_compat:
                fully_cel_dim = 1 - longest_dim
            else:
                fully_cel_dim = None

            
            # Determine whether Sigma is a kronecker product of diagonal matrices
            kron_diagonal = (type(Sigma[0]) == covtype.Diagonal) and (type(Sigma[1]) == covtype.Diagonal)

            # Optimisations where Sigma is a kronecker product
            if num_terms == 1:
                if type(Sigma[0]) in cov_types_1D and type(Sigma[1]) in cov_types_1D:
                    print_str += "Cholesky both\n"
            
            elif num_terms == 2:
                if kron_diagonal and (type(args[0][0]) == covtype.Exp) and (type(args[0][1]) == covtype.Diag_and_Outer):
                        print_str += "Do opt 5, cel_dim = 0\n"
                    
                elif kron_diagonal and (type(args[0][0]) == covtype.Diag_and_Outer) and (type(args[0][1]) == covtype.Exp):
                        print_str += "Do opt 5, cel_dim = 1\n"
                        
                elif fully_cel_dim == longest_dim:
                    print_str += "Using LuasLasrach with Celerite decomposing longest dimension\n"
                    kernel = LuasLasrachKernel
                    kernel_kwargs = {"cel_dim":fully_cel_dim}

                elif fully_cel_dim == 1 - longest_dim:
                    print_str += """Using Celerite on shorter dimension. This will still give the correct answer but in some cases LuasKernel may be faster, 
particularly when the shorter dimension is significantly shorter than the longer dimension and for complex Celerite kernels.
Consider setting use_kernel = LuasKernel to check if this is faster for your dataset size.\n"""
                    kernel = LuasLasrachKernel
                    kernel_kwargs = {"cel_dim":fully_cel_dim}
            
                elif data_shape[1-longest_dim] < max_gen_cholesky_blocks or type(args[0][1-longest_dim]) == covtype.Outer:
                    print_str += "Using LuasLasrach with general cholesky decomposition on the longest dimension\n"
                    if not type(args[0][1-longest_dim]) == covtype.Outer:
                        print_str += """Using LuasKernel may be faster here it depends on dataset size/kernel choice.
You can trying switching to it by setting use_kernel = LuasKernel\n"""
                    kernel = LuasLasrachKernel
                    kernel_kwargs = {"cel_dim":longest_dim}

                elif fully_cel_dim is None and (type(Sigma[0]) in cov_types_1D) and (type(Sigma[1]) in cov_types_1D) \
                     and (type(args[0][0]) in cov_types_1D) and (type(args[0][1]) in cov_types_1D):
                    print_str += "Do rakitsch\n"
                    print_str += "There is a periodic scenario here used in PSR_celery which might work here\n"
                    kernel = LuasKernel
                    kernel_kwargs = {}

                else:
                    print_str += "Make sure all terms are valid luas.cov_types!\n"

            elif num_terms > 2:
                if fully_cel_dim is not None:
                    rank_noncel_dim = 0
                    
                    for arg in args:
                        # rank_noncel_dim += arg[1-cel_dim].rank(regressors[1-cel_dim])
                        rank_noncel_dim += arg[1-cel_dim].rank
                        
                    if rank_noncel_dim < data_shape[1-cel_dim]:
                        print_str += "Do reduced Gordon opt\n"
                        kernel = MultiTermKernel
                        kernel_kwargs = {"cel_dim":longest_dim}
                    else:
                        print_str += "Do Gordon opt, also a periodic possibility I'm ignoring here!\n"
                        kernel = GordonKernel
                else:
                    print_str += "Can only optimise >2 terms if one of the dimensions is Celerite compatible!\n"
                    print_str += "Try use_kernel = GeneralKernel if you would like to run without a GP optimisation, this could be very computationally expensive though!"
            
            else:
                # num_terms somehow negative
                print_str += "Definitely shouldn't be possible to see this!\n"

    if verbose:
        print(print_str)
    return kernel, kernel_kwargs