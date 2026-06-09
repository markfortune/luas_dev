import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from copy import deepcopy
from datetime import datetime
import pickle
import os

import pymc as pm
from pymc.step_methods.hmc.quadpotential import QuadPotentialFullAdapt
from pymc.pytensorf import floatX

import arviz as az
import corner
import pandas as pd

from ..GP import GP
from ..pymc_ext import LuasPyMC
from ..numpyro_ext import LuasNumPyro
from ..integrators import BouncingLeapfrog

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.initialization import init_to_value
from jax import random

from ..jax_convenience_fns import order_list, order_dict, pytree_to_array_2D, array_to_pytree_2D, varying_params_wrapper

# # Ensure we are using double precision floats as JAX uses single precision by default
# jax.config.update("jax_enable_x64", True)

az.rcParams["plot.max_subplots"] = 100

def save_params(dict_to_save, filename):
    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(dict_to_save, f)
        
    df = pd.DataFrame()
    for (k, v) in dict_to_save.items():
        df.loc[k, np.arange(v.size)] = v
    df.to_csv(filename + ".csv")


def save_dict(dict_to_save, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict_to_save, f)


def load_dict(filename):
    with open(filename, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def GPload(kernel, mf = None, logPrior = None, save_folder = None, filename_suffix = None, idata_filename = None, run_logP = True):
            
        vars_folder = os.path.join(save_folder, "vars")
        
        p = load_dict(os.path.join(vars_folder, f"p_{filename_suffix}.pkl"))
        param_bounds = load_dict(os.path.join(vars_folder, f"param_bounds_{filename_suffix}.pkl"))
            
        x_l = np.load(os.path.join(vars_folder, f"x_l_{filename_suffix}.npy"))
        x_t = np.load(os.path.join(vars_folder, f"x_t_{filename_suffix}.npy"))
        Y = np.load(os.path.join(vars_folder, f"Y_{filename_suffix}.npy"))
        
        cov_mat = {}
        try:
            cov_mat = load_dict(os.path.join(vars_folder, f"cov_dict_{filename_suffix}.pkl"))
            print("Covariance matrix dictionary found.")
        except:
            print("No covariance matrix dictionary found.")
        
        # Prioritise dictionary cov_mat
        if not cov_mat:
            try:
                cov_mat = np.load(os.path.join(vars_folder, f"cov_mat_{filename_suffix}.npy"))
                print("Covariance matrix array found.")
            except:
                print("No covariance matrix array found.")
        
        if idata_filename:
            idata = az.from_json(os.path.join(vars_folder, idata_filename))
        else:
            idata = None
        
        gp = GPremium(kernel, x_l, x_t, mf = mf, logPrior = logPrior, save_folder = save_folder + "2", filename_suffix = filename_suffix,
                     cov_mat = cov_mat, idata = idata, param_bounds = param_bounds)
        
        if run_logP:
            print(gp.logP(p, Y))
        
        return gp, p, Y


class GPremium(GP):
    def __init__(self, *args, param_bounds = {}, cov_mat = {}, idata = None, save_folder = "analyses", filename_suffix = "test",
                 mf = None, logPrior = None, jit = True, **kwargs):
        
        self.param_bounds = param_bounds
        self.save_folder = save_folder
        self.filename_suffix = filename_suffix
        self.idata = idata
        self.cov_mat = cov_mat

        # if jit:
        #     self.solve = jax.jit(self.solve)
        
        # if mf is None:
        #     print("Mean function not specified. Defaulting to zeros.")
        # if logPrior is None:
        #     print("LogPrior function not specified. Defaulting to zero.")
            
        super().__init__(*args, mf = mf, logPrior = logPrior, jit = jit, **kwargs)

    
    def solve(
        self,
        p,
        Y,
    ):
        """Computes the log likelihood without returning any stored values from the
        decomposition of the covariance matrix.
        
        Args:
            p (PyTree): Pytree of hyperparameters used to calculate the covariance matrix
                in addition to any mean function parameters which may be needed to calculate the mean function.
            Y (JAXArray): Observed data to fit, must be of shape ``(N_l, N_t)``.
        
        Returns:
            Scalar: The value of the log likelihood.
            
        """
        
        # Calculate the residuals after subtraction of the deterministic mean function
        R = Y - self.mf(p, self.x_l, self.x_t)
        
        # Use the specific log likelihood calculation of the chosen Kernel object
        # to compute the log likelihood and any stored values from the decomposition
        # are also returned by default but not returned by this method
        K_inv_R, stored_values = self.kf.solve(p, self.x_l, self.x_t, R, {})
        
        return K_inv_R
        

    def make_numpyro_model(self, p, Y, param_bounds = {}, vars = None, fixed_vars = None):
        
        if not param_bounds:
            param_bounds = self.param_bounds
            
        p_fit, make_p_dict = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)
        
        def model(p_fixed, Y_obs):
            var_dict = deepcopy(p_fixed)

            for var in p_fit.keys():
                if var in param_bounds:
                    var_dict[var] = numpyro.sample(var, dist.Uniform(low = param_bounds[var][0],
                                                                     high = param_bounds[var][1]))
                else:
                    var_dict[var] = numpyro.sample(var, dist.ImproperUniform(dist.constraints.real, (),
                                                                             event_shape = p[var].shape))

            numpyro.sample("log_like", LuasNumPyro(gp = self, var_dict = var_dict), obs = Y_obs)
        
        return model
        
        
    def make_pymc_model(self, p, Y, param_bounds = {}, vars = None, fixed_vars = None):
        
        if not param_bounds:
            param_bounds = self.param_bounds
        
        p_fit, make_p_dict = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)
        
        with pm.Model() as model:

            var_dict = deepcopy(p)
            
            for var in p_fit.keys():
                if var in param_bounds.keys():
                    var_dict[var] = pm.Uniform(var, lower=param_bounds[var][0],
                                               upper=param_bounds[var][1], shape=p[var].shape)
                else:
                    var_dict[var] = pm.Flat(var, shape=p[var].shape)

            LuasPyMC("log_like", gp = self, var_dict = var_dict, Y = Y)
        
        return model, var_dict
    
    
    def optimise(self, *args, backend = "pymc", **kwargs):
        if backend == "pymc":
            return self.pymc_optimise(*args, **kwargs)
        elif backend == "numpyro":
            return self.numpyro_optimise(*args, **kwargs)
        else:
            raise Exception("Backend must be either 'pymc' or 'numpyro'!")
            
            
    def mcmc(self, *args, backend = "pymc", **kwargs):
        if backend == "pymc":
            return self.pymc_mcmc(*args, **kwargs)
        elif backend == "numpyro":
            return self.numpyro_mcmc(*args, **kwargs)
        else:
            raise Exception("Backend must be either 'pymc' or 'numpyro'!")
        
        
    def pymc_optimise(self, p, Y, param_bounds = {}, vars = None, fixed_vars = None,
                      maxeval = 30000, include_transformed = False, **kwargs):
        
        if not param_bounds:
            param_bounds = self.param_bounds

        for par in param_bounds.keys():
            if par in p.keys():
                if np.any(param_bounds[par][0] == p[par]):
                    p[par] += 1e-6
                if np.any(param_bounds[par][1] == p[par]):
                    p[par] -= 1e-6
            
        p_fit, make_p_dict = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)

        model, var_dict = self.make_pymc_model(p, Y, vars = vars, fixed_vars = fixed_vars, param_bounds = param_bounds)

        map_estimate = pm.find_MAP(model = model, start = p_fit, include_transformed = include_transformed,
                                   maxeval = maxeval, **kwargs)
        p_opt = deepcopy(p)
        p_opt.update(map_estimate)

        return p_opt
    
    
    def numpyro_optimise(self, p, Y, param_bounds = {}, vars = None, fixed_vars = None,
                         maxeval = 5000, step_size = 1e-3, **kwargs):
        
        if not param_bounds:
            param_bounds = self.param_bounds
            
        p_fit, make_p_dict = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)

        model = self.make_numpyro_model(p, Y, vars = vars, fixed_vars = fixed_vars, param_bounds = param_bounds)

        optimizer = numpyro.optim.Adam(step_size=step_size)
        guide = AutoLaplaceApproximation(model, init_loc_fn = init_to_value(values=p))
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        svi_result = svi.run(random.PRNGKey(0), maxeval, p, Y)
        params = svi_result.params
        p_fit = guide.median(params)

        p_opt = deepcopy(p)
        p_opt.update(p_fit)
            
        return p_opt
            
        
    def save_trace(self, trace, draw):
        self.trace = trace
        
        if os.path.exists(f"crash{self.crashcode}"):
            raise Exception(f"Crashcode {self.crashcode} found!")
            
        if len(trace) % 100 == 0:
            timediff = (datetime.now() - self.start_time).total_seconds()
            split = (datetime.now() - self.last_time).total_seconds()
            self.last_time = datetime.now()
            print(f"{datetime.now().strftime('%H:%M:%S')}, draw: {len(trace)}, split: {round(split/60., 1)}, total minutes eclipsed: {round(timediff/60., 1)}")
        elif len(trace) == 1:
            timediff = (datetime.now() - self.start_time).total_seconds()
            self.last_time = datetime.now()
            print(f"{datetime.now().strftime('%H:%M:%S')}, draw: {len(trace)}, minutes eclipsed: {round(timediff/60., 1)}")
        
    
    def pymc_mcmc(self, p, Y, draws = 1000, tune = 1000, chains = 2, cores = 1, vars = None, fixed_vars = None,
                  vars_LA = None, fixed_vars_LA = None, save = False,
             slice_sample = [], param_bounds = {}, cov_mat = None, non_cov_vars = [], non_cov_vars_dense = [],
                  NUTS_kwargs = {}, regularise = True, large_block_size = 50, bouncing_bounds = None,
               regularise_const = 100., large = False, LA_NUTS_param = None, modify_cov_mat = None, **other_sample_kwargs):
               
        self.crashcode = np.random.randint(10000)
        print("Crashcode:", self.crashcode)
        
        for par in p.keys():
            p[par] = np.array(p[par])
        
        if not param_bounds:
            param_bounds = self.param_bounds
            
        if non_cov_vars:
            if type(non_cov_vars[0]) == list:
                all_non_cov_vars = []
                for non_cov_list in non_cov_vars:
                    all_non_cov_vars += non_cov_list
            else:
                all_non_cov_vars = non_cov_vars.copy()
        else:
            all_non_cov_vars = []
            
        if non_cov_vars_dense:
            if type(non_cov_vars_dense[0]) == list:
                for non_cov_vars_dense_list in non_cov_vars_dense:
                    all_non_cov_vars += non_cov_vars_dense_list
            else:
                all_non_cov_vars += non_cov_vars_dense
          
        all_non_cov_vars = list(set(all_non_cov_vars))
            
            
        if vars_LA is None and fixed_vars_LA is None:
            if vars is not None:
                vars_LA = [var for var in vars if var not in slice_sample + all_non_cov_vars]
            elif fixed_vars is not None:
                fixed_vars_LA = fixed_vars + slice_sample + all_non_cov_vars
            else:
                raise Exception("Both vars and fixed_vars cannot be None!")

        p_fit, make_p_dict = varying_params_wrapper(p, vars = vars, fixed_vars = fixed_vars)
        model, var_dict = self.make_pymc_model(p, Y, vars = vars, fixed_vars = fixed_vars, param_bounds = param_bounds)

        steps = []
        
        if cov_mat is None:
            self.cov_mat, ordered_param_list = self.laplace_approx_with_bounds(p, Y, param_bounds,
                                                                               large_block_size = large_block_size,
                                                                        vars = vars_LA, fixed_vars = fixed_vars_LA,
                                                                        return_array = False, regularise = regularise,
                                                                        large = large, regularise_const = regularise_const)
            
            if modify_cov_mat:
                self.cov_mat = modify_cov_mat(self.cov_mat)
            
            if LA_NUTS_param is None:
                LA_NUTS_param_list = [var for var in ordered_param_list if var not in slice_sample + all_non_cov_vars]
                
                if LA_NUTS_param_list:
                    NUTS_model_vars = [var_dict[var] for var in LA_NUTS_param_list]
                    cov_mat_NUTS = pytree_to_array_2D(p, self.cov_mat, param_order = LA_NUTS_param_list)
                    steps.append(pm.NUTS(NUTS_model_vars, scaling = cov_mat_NUTS, is_cov = True, model = model, **NUTS_kwargs))
                
            elif type(LA_NUTS_param[0]) == list:
                for LA_NUTS_param_list in LA_NUTS_param:
                    NUTS_model_vars = [var_dict[var] for var in LA_NUTS_param_list]
                    cov_mat_NUTS = pytree_to_array_2D(p, self.cov_mat, param_order = LA_NUTS_param_list)
                    steps.append(pm.NUTS(NUTS_model_vars, scaling = cov_mat_NUTS, is_cov = True, model = model, **NUTS_kwargs))
                    
            else:
                raise Exception("LA_NUTS_param keyword argument must be a list of lists")
            
            
        elif type(cov_mat) == dict:
            
            if LA_NUTS_param is None:
                LA_NUTS_param_list = [var for var in ordered_param_list if var not in slice_sample + all_non_cov_vars]
                
                if LA_NUTS_param_list:
                    NUTS_model_vars = [var_dict[var] for var in LA_NUTS_param_list]
                    cov_mat_NUTS = pytree_to_array_2D(p, cov_mat, param_order = LA_NUTS_param_list)
                    steps.append(pm.NUTS(NUTS_model_vars, scaling = cov_mat_NUTS, is_cov = True, model = model, **NUTS_kwargs))
                
            elif type(LA_NUTS_param[0]) == list:
                for LA_NUTS_param_list in LA_NUTS_param:
                    NUTS_model_vars = [var_dict[var] for var in LA_NUTS_param_list]
                    cov_mat_NUTS = pytree_to_array_2D(p, cov_mat, param_order = LA_NUTS_param_list)
                    steps.append(pm.NUTS(NUTS_model_vars, scaling = cov_mat_NUTS, is_cov = True, model = model, **NUTS_kwargs))
                    
            else:
                raise Exception("LA_NUTS_param keyword argument must be a list of lists")

        # elif type(cov_mat) == dict:
        #     NUTS_param_list = [var_dict[var] for var in order_list(list(p_fit.keys())) if var not in slice_sample + all_non_cov_vars]
            
        #     if NUTS_param_list:
        #         cov_mat_NUTS = pytree_to_array_2D(p, cov_mat, param_order = NUTS_param_list)
        #         steps.append(pm.NUTS(NUTS_param_list, scaling = cov_mat, is_cov = True, model = model, **NUTS_kwargs))
        
        else:
            NUTS_param_list = [var_dict[var] for var in order_list(list(p_fit.keys())) if var not in slice_sample + all_non_cov_vars]
            
            if NUTS_param_list:
                steps.append(pm.NUTS(NUTS_param_list, scaling = cov_mat, is_cov = True, model = model, **NUTS_kwargs))

        if len(steps) == 1 and bouncing_bounds is not None:
            steps[0].integrator = BouncingLeapfrog(step, lower = bouncing_bounds[0], upper = bouncing_bounds[1])
                
        if non_cov_vars:
            if type(non_cov_vars[0]) == list:
                for non_cov_list in non_cov_vars:
                    steps.append(pm.NUTS([var_dict[var] for var in non_cov_list], model = model, **NUTS_kwargs))
            else:
                steps.append(pm.NUTS([var_dict[var] for var in non_cov_vars], model = model, **NUTS_kwargs))
            
        if non_cov_vars_dense:
            if type(non_cov_vars_dense[0]) == list:
                for non_cov_vars_dense_list in non_cov_vars_dense:
                    size = 0
                    for var in non_cov_vars_dense_list:
                        size += p[var].size
                    mean = floatX(np.zeros(size))
                    var = floatX(np.eye(size))
                    potential = QuadPotentialFullAdapt(size, mean, var, 10)
                    
                    steps.append(pm.NUTS([var_dict[var] for var in non_cov_vars_dense_list], potential = potential,
                                     model = model, **NUTS_kwargs))
            else:
                size = 0
                for var in non_cov_vars_dense:
                    size += p[var].size
                mean = floatX(np.zeros(size))
                var = floatX(np.eye(size))
                potential = QuadPotentialFullAdapt(size, mean, var, 10)
                
                steps.append(pm.NUTS([var_dict[var] for var in non_cov_vars_dense], potential = potential,
                                 model = model, **NUTS_kwargs))
            
        if slice_sample:
            slice_model_vars = [var_dict[par] for par in slice_sample]
            steps.append(pm.Slice(slice_model_vars, model = model))
            
        if len(steps) == 1:
            steps = steps[0]
            
        self.trace = None
        
        try:
            for i in range(chains):
                self.start_time = datetime.now()
                idata = pm.sample(model = model, step = steps, initvals = p_fit, draws = draws,
                                  tune = tune, chains = 1, cores = cores, callback = self.save_trace,
                                  **other_sample_kwargs)
                if i == 0:
                    self.idata = idata
                else:
                    self.idata = az.concat([self.idata, idata], dim = "chain")

        except:
            datadict = {}
            coords = {}
            dims = {}

            for var in self.trace.varnames:
                vals = self.trace.get_values(var)
                vals = vals.reshape((1, vals.shape[0], vals.shape[1]))
                datadict[var] = vals
                coords[f"{var}1"] = vals.shape[2]
                dims[var] = [f"{var}1"]
            idata_raw = az.convert_to_inference_data(datadict)
            
            var_names = list(idata_raw.posterior.data_vars)
            random_var = var_names[0]
            random_var_vals = idata_raw.posterior[random_var].to_numpy()
            nonzero_ind = np.arange(random_var_vals.shape[1])[np.all(random_var_vals != 0, axis = (0, 2))]

            idata_raw = idata_raw.sel(draw=nonzero_ind)
            idata_raw.posterior = idata_raw.posterior[[var for var in self.trace.varnames if "_interval__" not in var]]
            
            self.idata = idata_raw

        if save:
            self.save(p, Y, param_bounds = self.param_bounds)
            
        return self.idata

    
    def numpyro_mcmc(self, p, Y, draws = 1000, tune = 1000, chains = 2, cores = 1, vars = None, fixed_vars = None,
             slice_sample = [], param_bounds = {}, cov_mat = None, NUTS_kwargs = {}, regularise = True, large_block_size = 50,
               regularise_const = 100., large = False, adapt_mass_matrix = False, save = True, **other_sample_kwargs):
        
        if not param_bounds:
            param_bounds = self.param_bounds

        model = self.make_numpyro_model(p, Y, vars = vars, fixed_vars = fixed_vars, param_bounds = param_bounds)

        if cov_mat is None:
            self.cov_mat, ordered_param_list = self.laplace_approx_with_bounds(p, Y, param_bounds,
                                                                               large_block_size = large_block_size,
                                                                        vars = vars, fixed_vars = fixed_vars,
                                                                        return_array = True, regularise = regularise,
                                                                        large = large, regularise_const = regularise_const)
            cov_mat = self.cov_mat
            
        elif type(cov_mat) == dict:
            cov_mat = pytree_to_array_2D(p, self.cov_mat)

        kernel_NUTS = NUTS(model,
                           init_strategy = init_to_value(values = p),
                           inverse_mass_matrix = cov_mat, 
                           adapt_mass_matrix=adapt_mass_matrix,
                           dense_mass = True,
                           regularize_mass_matrix = False)

        mcmc = MCMC(
            kernel_NUTS,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            thinning=1,
            progress_bar=True,
            **other_sample_kwargs,
        )

        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        mcmc.run(rng_key, p, Y)

        self.idata = az.from_numpyro(mcmc)
        
        if save:
            self.save(p, Y, param_bounds = self.param_bounds)

        return self.idata
    
    
    def save(self, p, Y, param_bounds = {}, save_idata = True, save_folder = None, filename_suffix = None):
        
        if save_folder is None:
            save_folder = self.save_folder
        if filename_suffix is None:
            filename_suffix = self.filename_suffix
            
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        save_folder = os.path.join(save_folder, "vars")
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        if not param_bounds:
            param_bounds = self.param_bounds
            
        save_params(p, os.path.join(save_folder, f"p_{filename_suffix}"))
        save_dict(param_bounds, os.path.join(save_folder, f"param_bounds_{filename_suffix}.pkl"))
        
        if type(self.cov_mat) == dict:
            save_dict(self.cov_mat, os.path.join(save_folder, f"cov_dict_{filename_suffix}.pkl"))
        else:
            np.save(os.path.join(save_folder, f"cov_mat_{filename_suffix}"), self.cov_mat)
            
        np.save(os.path.join(save_folder, f"x_l_{filename_suffix}.npy"), self.x_l)
        np.save(os.path.join(save_folder, f"x_t_{filename_suffix}.npy"), self.x_t)
        np.save(os.path.join(save_folder, f"Y_{filename_suffix}.npy"), Y)
        
        if save_idata and self.idata is not None:
            idata_location = f"idata_{filename_suffix}_{np.random.randint(10**9)}.json"
            self.idata.to_json(os.path.join(save_folder, idata_location));
        
        print(self.logP(p, Y))
        print("Saved at: ", os.path.join(save_folder, f"_{filename_suffix}"))
 

    def plot_corner(self, idata = None, dims = None, params = None,
                    save_folder = None, save = True,
                    filename_suffix = None, show = True, labelpad = 0.16,
                    max_par = 5, label_kwargs = {"fontsize": 23}, title_kwargs = {"fontsize": 23},
                    **kwargs,
                   ):
        
        if save_folder is None:
            save_folder = self.save_folder
        if filename_suffix is None:
            filename_suffix = self.filename_suffix

        if dims is None:
            if params is None:
                dims = [0, 1]
            else:
                dims = np.arange(self.N_l)
                
        if idata is None:
            idata = self.idata
            
        if hasattr(idata, "posterior"):
            idata = idata.posterior
                
        if params is None:
            params = list(idata.data_vars)
            

        dim_dict = {}
        for (k, v) in idata.dims.items():
            if k[:-6] in params:
                if v >= max_par:
                    dim_dict[k] = dims
        idata_corner = idata.sel(**dim_dict)

        N_chains = idata.dims["chain"]
        if N_chains > 1:
            idata_corner1 = idata_corner.sel(chain=list(range(N_chains//2)))
            idata_corner2 = idata_corner.sel(chain=list(range(N_chains//2, N_chains - N_chains % 2)))

            fig1 = corner.corner(idata_corner1, smooth = 0.4, var_names = params, data_kwargs = {"alpha":1.}, **kwargs)
            fig = corner.corner(idata_corner2, quantiles=[0.16, 0.5, 0.84], title_fmt = None,
                                 title_kwargs=title_kwargs, label_kwargs=label_kwargs, show_titles=True,
                                 smooth = 0.4, color = "r", fig = fig1, top_ticks = True, max_n_ticks = 2,
                                 labelpad = labelpad, var_names = params, data_kwargs = {"alpha":1.}, **kwargs)
        else:
            fig = corner.corner(idata_corner, quantiles=[0.16, 0.5, 0.84], title_fmt = None,
                                title_kwargs=title_kwargs, label_kwargs=label_kwargs, show_titles=True,
                                 smooth = 0.4, color = "k", top_ticks = True, max_n_ticks = 2,
                                labelpad = labelpad, var_names = params, data_kwargs = {"alpha":1.}, **kwargs)
        
        if save:
            plt.savefig(os.path.join(save_folder, f"corner_plot_{filename_suffix}.png"), facecolor='w', bbox_inches = "tight")
        
        if show:
            plt.show()
        
        return fig
        
        
    def summary(self, idata = None, corner = True, save = True, save_folder = None, filename_suffix = None, show = False, **corner_kwargs):
        
        if save_folder is None:
            save_folder = self.save_folder
        if filename_suffix is None:
            filename_suffix = self.filename_suffix
            
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        
        if idata is None:
            idata = self.idata

        df_summary = az.summary(idata, round_to = 6)
        if save:
            df_summary.to_csv(os.path.join(save_folder, f"MCMC_summary_{filename_suffix}.csv"))

        print("Max r_hat: ", df_summary["r_hat"].max())

        trace_plot = az.plot_trace(idata, divergences = None)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_folder, f"trace_plot_{filename_suffix}.png"), facecolor='w', bbox_inches = "tight")
        if show:
            plt.show()
        plt.clf()
        
        if hasattr(idata, "posterior"):
            idata = idata.posterior
        if "d" in list(idata.data_vars):
            rho_mean, rho_cov = self.get_mcmc_cov(["d"], idata = idata)
            if save:
                np.save(os.path.join(save_folder, f"rho_mean_{filename_suffix}"), rho_mean)
                np.save(os.path.join(save_folder, f"rho_cov_{filename_suffix}"), rho_cov)
            if rho_cov.size > 1:
                self.plot_spectrum(rho_mean = rho_mean, rho_cov = rho_cov, show = show, save_folder = save_folder, filename_suffix = filename_suffix)
        
        if corner:
            self.plot_corner(idata = idata, show = show, save_folder = save_folder, filename_suffix = filename_suffix, **corner_kwargs)
            
        if show:
            display(df_summary)
            
        return df_summary
            
    
    def p_from_mcmc(self, p = {}, how = "mean", idata = None):
        
        if idata is None:
            idata = self.idata#
    
        p = deepcopy(p)

        if how in ["min", "max"]:
            log_prob_mat = idata.sample_stats["lp"].to_numpy().copy()

            for chain in range(log_prob_mat.shape[0]):
                for position in range(log_prob_mat.shape[1]):
                    x = self.transf_all_from_param(self.p_from_mcmc(how = (chain, position)))
                    x_bounds = {k:x[k] for k in self.bounded_params}
                    x_flat = ravel_pytree(x_bounds)[0]
                    sigmoid_jacob = (-2*jnp.log(1+jnp.exp(-x_flat)) - x_flat).sum()

                    log_prob_mat[chain, position] -= sigmoid_jacob
            if how == "max":
                chain, position = np.unravel_index(log_prob_mat.argmax(), log_prob_mat.shape)
            else:
                chain, position = np.unravel_index(log_prob_mat.argmin(), log_prob_mat.shape)
            
        elif type(how) == tuple:
            chain, position = how
            
        elif how == "random":
            N_chains = idata.posterior.chain.size
            draws = idata.posterior.draw.size
            
            chain, position = np.random.randint(N_chains), np.random.randint(draws)
        elif how in ["mean", "median"]:
            pass
        else:
            raise Exception(f"Method {how} not permitted.")

        for par in list(idata.posterior.data_vars):
            if how == "mean":
                p[par] = idata.posterior[par].to_numpy().mean((0, 1))
            elif how == "median":
                p[par] = np.median(idata.posterior[par].to_numpy(), axis = (0, 1))
            else:
                p[par] = idata.posterior[par][chain, position, :].to_numpy()

        return p

            
    def get_mcmc_cov(self, params, zero_diag = False, idata = None):
        
        if idata is None:
            idata = self.idata#
            
        if hasattr(idata, "posterior"):
            idata = idata.posterior

        N_chains = idata.dims["chain"]
        for i in range(N_chains):
            mcmc_data = np.concatenate([idata[par][i, :, :].to_numpy() for par in params], axis = 1)

            if i == 0:
                param_mean = mcmc_data.mean(0)
                param_cov = np.cov(mcmc_data.T)
            else:
                param_mean += mcmc_data.mean(0)
                param_cov += np.cov(mcmc_data.T)

        param_mean /= N_chains
        param_cov /= N_chains
        
        if zero_diag:
            param_cov *= 1 - np.eye(param_cov.shape[0])

        return param_mean, param_cov

    
    def plot_spectrum(self, rho_mean = None, rho_cov = None, show = True, save_folder = None, filename_suffix = None):

        if rho_mean is None and rho_cov is None:
            rho_mean, rho_cov = self.get_mcmc_cov(["d"])

        fig = plt.figure(figsize = (8, 4))
        
        if type(rho_mean) == list and type(rho_cov) == list:
            min_bin_width = np.diff(self.x_l).min()
            for i in range(len(rho_mean)):
                plt.errorbar(self.x_l + i*min_bin_width/4., 100*rho_mean[i], yerr = 100*np.sqrt(np.diag(rho_cov[i])), fmt = '.')
#                 draws = np.random.multivariate_normal(rho_mean[i], rho_cov[i], 5).T
#                 plt.plot(self.x_l, 100*draws[:, :], alpha = 0.3)
        else:
            plt.errorbar(self.x_l, 100*rho_mean, yerr = 100*np.sqrt(np.diag(rho_cov)), fmt = 'k.')
            draws = np.random.multivariate_normal(rho_mean, rho_cov, 5).T
            plt.plot(self.x_l, 100*draws[:, :], alpha = 0.3, color = 'k')
            
        plt.ylabel("Transit Depth (%)")
        plt.xlabel(r"$\lambda$")
        
        if save_folder:
            plt.savefig(os.path.join(save_folder, f"trans_spec_{filename_suffix}.png"), facecolor='w', bbox_inches = "tight")
        
        if show:
            plt.show()
        plt.clf()
        
        
