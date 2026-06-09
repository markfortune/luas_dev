import numpy as np
import jax.numpy as jnp
import jax
from luas.exoplanet.transit_fns import ld_from_kipping
from .luas_types import Kernel, PyTree, JAXArray, Scalar
from typing import Optional, Callable, Tuple, Any
import jax
import jax.numpy as jnp
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import astropy.units as u
from astropy.constants import G, M_sun, R_sun, c
import jaxoplanet
from scipy.stats import pearsonr

# from harmonica.jax import harmonica_transit_quad_ld

# Ensure we are using double precision floats as JAX uses single precision by default
jax.config.update("jax_enable_x64", True)

solar_density = ((M_sun/R_sun**3)/(u.kg/u.m**3)).si
def transit_light_curve(par, t):
    # Define the orbit
    rho_s = 3*jnp.pi*par["a"]**3/(G.value*(par["P"]*86400)**2)
    central = jaxoplanet.orbits.keplerian.Central(density=rho_s/solar_density,radius=1.)

    body = jaxoplanet.orbits.keplerian.Body(
        period=par["P"],
        time_transit=par["T0"],
        radius=jnp.sqrt(par["d"]),
        impact_param=par["b"],  # All angles are in radians
        eccentricity=0.,
        omega_peri = 0.,
    )

    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body)

    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [par["u1"], par["u2"]])
    flux = lc(t)
    baseline = par["Foot"] + 24*par["Tgrad"]*(t - par["T0"])
    
    return baseline*(1 + flux)
    

def transit_light_curve_ecc(par, t):
    # Define the orbit
    rho_s = 3*jnp.pi*par["a"]**3/(G.value*(par["P"]*86400)**2*solar_density)
    central = jaxoplanet.orbits.keplerian.Central(density=rho_s,radius=1.)
    
    e = par["secosw"]**2 + par["sesinw"]**2
    sinw = par["sesinw"]/jnp.sqrt(e)
    
    incl_factor = (1 + e*sinw)/(1 - e**2)
    body = jaxoplanet.orbits.keplerian.Body(
        period=par["P"],
        time_transit=par["T0"],
        radius=jnp.sqrt(par["d"]),
        inclination = jnp.arccos(incl_factor*par["b"]/par["a"]),
        eccentricity=e,
        cos_omega_peri = par["secosw"]/jnp.sqrt(e),
        sin_omega_peri = par["sesinw"]/jnp.sqrt(e),
    )

    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body)

    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [par["u1"], par["u2"]], order = 10)
    flux = lc(t)
    baseline = par["Foot"] + 24*par["Tgrad"]*(t - par["T0"])
    
    return baseline*(1 + flux)
    

def eclipse_light_curve(par, t):
    # Define the orbit
    rho_s = 3*jnp.pi*par["a"][0]**3/(G.value*(par["P"][0]*86400)**2)
    central = jaxoplanet.orbits.keplerian.Central(density=rho_s/solar_density,radius=1.)

    body = jaxoplanet.orbits.keplerian.Body(
        period=par["P"][0],
        time_transit=par["T0"][0],
        radius=par["rho"][0],
        impact_param=par["b"][0],  # All angles are in radians
        eccentricity=0.,
        omega_peri = 0.,
    )

    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body)

    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [0., 0.])
    flux = lc(t)
    
    flux *= par["d"]/par["rho"][0]**2
    
    baseline = par["Foot"] + 24*par["Tgrad"]*(t - par["T0"][0])
    
    return baseline*(1 + flux)
    

eclipse_light_curve_vmap = jax.vmap(eclipse_light_curve,
                                    in_axes=({
                                        # Parameters to be shared between all wavelengths
                                        "T0":None, "P":None, "a":None, "b":None, 
                                        
                                        # Parameters to be separate for each wavelength
                                        "rho":None, "d":0, "Foot":0, "Tgrad":0}, 
                                        
                                        # Array of timestamps to be the same for each wavelength
                                        None, 
                                        ), 
                                    # Will output extra flux values for each light curve as additional rows
                                    out_axes = 0 
                                   ) 


def eclipse_2D(p: PyTree, X: tuple) -> JAXArray:
    r"""Uses ``jax.vmap`` on the ``transit_light_curve`` function to generate a 2D ``JAXArray`` of
    transit light curves for multiple wavelengths simultaneously.
    
    This is just meant to be a simple example for generating multiple simultaneous light curves
    in wavelength, it should be easy to modify for different limb darkening parameterisations, etc.
    See the package `jaxoplanet <https://github.com/exoplanet-dev/jaxoplanet>`_ to see the range of
    currently implemented light curve models.
    
    Note:
        Unlike ``transit_light_curve``, input limb darkening parameters are assumed to follow
        the `Kipping (2013) <https://arxiv.org/abs/1308.0009>`_ parameterisation and are converted
        to standard limb darkening coefficients. Also assumed that the transit depth d = rho^2 is
        being input which is then converted to radius ratio values for ``transit_light_curve``.
    
    .. code-block:: python

        >>> from luas.exoplanet import transit_2D
        >>> import jax.numpy as jnp
        >>> N_l = 16 # Number of wavelength channels
        >>> par = {
        >>> ... "T0":0.*jnp.ones(1),        # Central transit time (days)
        >>> ... "P":3.4*jnp.ones(1),        # Period (days)
        >>> ... "a":8.2*jnp.ones(1),        # Semi-major axis to stellar ratio (aka a/R*)
        >>> ... "d":0.01*jnp.ones(N_l),     # Transit depth (aka (Rp/R*)^2 or rho^2)
        >>> ... "b":0.5*jnp.ones(1),        # Impact parameter
        >>> ... # Kipping (2013) limb darkening parameterisation is used
        >>> ... "q1":0.36*jnp.ones(N_l),    # First quadratic limb darkening coefficient for each wv
        >>> ... "q2":0.416*jnp.ones(N_l),   # Second quadratic limb darkening coefficient for each wv
        >>> ... "Foot":1.*jnp.ones(N_l),    # Baseline flux out of transit for each wv
        >>> ... "Tgrad":0.*jnp.ones(N_l),   # Gradient in baseline flux for each wv (hrs^-1)
        >>> }
        >>> x_l = jnp.linspace(4000, 7000, N_l)
        >>> x_t = jnp.linspace(-0.1, 0.1, 100)
        >>> flux = transit_2D(par, x_l, x_t)
    
    Args:
        par (PyTree): The transit parameters stored in a PyTree/dictionary (see example above).
        x_l (JAXArray): Array of wavelengths, not used but included for compatibility with :class:`luas.GP`.
        x_t (JAXArray): Array of times to calculate the light curve at.
            
    Returns:
        JAXArray: 2D array of flux values in a wavelength by time grid of shape ``(N_l, N_t)``.
        
    """
    
    # vmap requires that we only input the parameters which have been explicitly defined how they vectorise
    transit_params = ["T0", "P", "a", "b", "rho", "d", "Foot", "Tgrad"]
    mfp = {k:p[k] for k in transit_params}
    
    # Use the vmap of transit_light_curve to calculate a 2D array of shape (M, N) of flux values
    # For M wavelengths and N time points.
    return eclipse_light_curve_vmap(mfp, X[1])



def eclipse_light_curve_ecc(par, t, light_delay = False):
    # Define the orbit
    
    # rho_s in units of kg/m^3
    # rho_s = 3*jnp.pi*par["a"][0]**3/(G*(par["P"][0]*86400)**2)
    # a_Rs = par["a"][0]

    rho_s = par["rho_s"][0]
    a_Rs = jnp.cbrt(rho_s*G.value*(par["P"][0]*86400)**2/(3*jnp.pi))
    
    central = jaxoplanet.orbits.keplerian.Central(density=rho_s/solar_density,
                                                  radius=1.)

    e = par["secosw"][0]**2 + par["sesinw"][0]**2
    cosw = par["secosw"][0]/jnp.sqrt(e)
    sinw = par["sesinw"][0]/jnp.sqrt(e)

    opsw = 1 + sinw
    E0 = 2 * jnp.arctan2(jnp.sqrt(1 - e) * cosw, jnp.sqrt(1 + e) * opsw)
    M0 = E0 - e * jnp.sin(E0)
    time_peri = par["T0"][0] - M0 * par["P"][0] / (2 * jnp.pi)
    
    incl_factor = (1 + e*sinw)/(1 - e**2)
    body_eclipse = jaxoplanet.orbits.keplerian.Body(
        period=par["P"][0],
        time_peri=time_peri,
        radius=par["rho"][0],
        inclination = jnp.arccos(incl_factor*par["b"][0]/a_Rs),
        eccentricity=e,
        cos_omega_peri=-cosw,
        sin_omega_peri=-sinw,
    )

    orbit = jaxoplanet.orbits.keplerian.OrbitalBody(central = central, body = body_eclipse)

    lc = jaxoplanet.light_curves.limb_dark.light_curve(orbit, [0., 0.])

    if light_delay:
        x1, y1, z1 = orbit.relative_position(t)
        x2, y2, z2 = orbit.relative_position(par["T0"][0])
        delay = (z1 - z2)*par["rad_s"][0]*R_sun/c
        flux = lc(t - delay.magnitude/(60*60*24))
    else:   
        flux = lc(t)
        
    flux *= par["d"][0]/par["rho"][0]**2
    baseline = par["Foot"][0] + 24*par["Tgrad"][0]*(t - par["T0"][0])
    
    return baseline*(1 + flux)


def detrend_Y(detrend_arr, Y, points_to_use = 3000, how = "all", corr = 0.07, check_corr = True, zero_mean = True, how_vec = None):
    """
    Does least-squares best-fit of array of values to Y where Y and trace are both (N_l, N_t)

    how can be "all", "corr" or a boolean vector where detrending if True
    """
    if zero_mean:
        detrend_arr = (detrend_arr.T - detrend_arr.mean(1)).T
    
    T = detrend_arr[:, -points_to_use:].copy()
    T = (T.T/(T**2).sum(1)).T
    
    detrend_vec = (T * (Y[:, -points_to_use:].T - Y[:, -points_to_use:].mean(1)).T).sum(1)

    if how == "all":
        print("Detrending all values")
    elif how == "corr":
        print(f"Detrending based on correlation threshold: {corr}")
        for i in range(Y.shape[0]):
            rho = pearsonr(detrend_arr[i, -points_to_use:], Y[i, -points_to_use:]).statistic
            if np.abs(rho) < corr:
                detrend_vec[i] = 0.
    elif how == "vec":
        print(f"Detrending based on a given vector")
        detrend_vec *= how_vec
    else:
        raise Exception("Failure to give correct detrending method!")

    no_detrended = (detrend_vec != 0).sum()
    print(f"Detrended {no_detrended} out of {Y.shape[0]} light curves")

    Y_detrend =  Y - detrend_vec[:, np.newaxis]*detrend_arr

    if check_corr:
        corr_before = np.zeros(Y.shape[0])
        corr_after = np.zeros(Y.shape[0])
        
        for i in range(Y.shape[0]):
            corr_before[i] = pearsonr(detrend_arr[i, :-points_to_use], Y[i, :-points_to_use]).statistic
            corr_after[i] = pearsonr(detrend_arr[i, :-points_to_use], Y_detrend[i, :-points_to_use]).statistic

        N_corr_decrease = ((np.abs(corr_before) - np.abs(corr_after)) > 0).sum()

        print(f"Correlation decreased on {N_corr_decrease} out of {no_detrended} ({N_corr_decrease*100/no_detrended:.1f}%) of lightcurves")
            
        return detrend_vec, Y_detrend, corr_before, corr_after
    else:
        return detrend_vec, Y_detrend
        

def make_video_timescale(data, time, flux, filename, start = 0, stop = None, fig = None, dpi = 100, figsize = (14, 6), interval = 10):
    
    if stop is None:
        stop = data.shape[0]
        
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    ax1 = fig.add_subplot(1,2,2)
    ax2 = fig.add_subplot(1,2,1)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.set_ylabel('Flux')
    ax2.get_xaxis().set_visible(True)

    short_data = data[start:stop]
    short_flux = flux[start:stop]
    short_time = time[start:stop]

    short_time_animated = short_time.copy()

    ax2.set_xlabel("Time")

    ims = []

    min, max = short_data.min(), short_data.max()

    im2, = ax2.plot(short_time_animated, short_flux, animated=True, color='black', alpha=0.5)
    for idx, i in enumerate(short_time_animated):

        im = ax1.imshow(short_data[idx], animated=True, aspect = 'auto')
        im.set_clim(vmin=min, vmax=max)

        im3 = ax2.vlines(i, np.min(short_flux), np.max(short_flux),  animated=True, color='red')
        ims.append([im, im2, im3])

    fig.suptitle('title')
    fig.colorbar(im, label='Counts', ax = ax1)
    fig.tight_layout()

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=1000)

    ani.save(filename,writer='pillow')
    

def make_video(data, filename, time_axis = 0, t = None, time_fmt = ".2f", interval = 10, fig = None, dpi = 100, figsize = (6, 6)):
    """
    data of shape (N_t, N_x, N_y)
    interval meant to be duration of each frame in milliseconds
    filename needs to end in .gif, will add it if not already there
    """

    if filename[-4:] != ".gif":
        filename += ".gif"

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
    ims = []
    ax = fig.add_subplot(111)
    min_flux, max_flux = data.min(), data.max()

    for i in range(data.shape[time_axis]):
        if time_axis == 0:
            data_i = data[i, :, :]
        elif time_axis == 1:
            data_i = data[:, i, :]
        elif time_axis == 2:
            data_i = data[:, :, i]
            
        im = plt.imshow(data_i, animated=True)
        im.set_clim(vmin=min_flux, vmax=max_flux)
        
        artists = [im]
        if t is not None:
            title = ax.text(
                0.5, 1.02,
                f"t = {t[i]:{time_fmt}}",
                transform=ax.transAxes,
                ha="center"
            )
            artists.append(title)

        ims.append(artists)
    fig.colorbar(im)
    fig.tight_layout()

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=1000)

    ani.save(filename, writer='pillow')


    


# def harmonica_light_curve(par, t):
#     r = jnp.array([par["rho"], par["rho"]*par["r1"][0], 0.])
#     f_asym = harmonica_transit_quad_ld(
#         t,
#         t0=par["T0"],
#         period=par["P"],
#         a=par["a"],
#         inc=jnp.arccos(par["b"]/par["a"]),
#         u1=par["u1"],
#         u2=par["u2"],
#         r=r
#     )
#     flux = (par["Foot"] + 24*par["Tgrad"]*(t-par["T0"]))*f_asym
    
#     return flux
# harmonica_light_curve_vmap = jax.vmap(harmonica_light_curve,
#                                     in_axes=({
#                                         "T0":None, "P":None, "a":None, "b":None, 
#                                         "rho":0, "u1":0, "u2":0, "Foot":0, "Tgrad":0, "r1":None}, 
#                                         None),  out_axes = 0) 

def harmonica_2D(p, x_l, x_t):
    
    # vmap requires that we only input the parameters which have been explicitly defined how they vectorise
    transit_params = ["T0", "P", "a", "b", "Foot", "Tgrad", "r1"]
    mfp = {k:p[k] for k in transit_params}
    
    # Calculate the radius ratio rho from the transit depth d
    mfp["rho"] = jnp.sqrt(p["d"])
    
    # Calculate limb darkening coefficients from the Kipping (2013) parameterisation.
    mfp["u1"], mfp["u2"] = ld_from_kipping(p["q1"], p["q2"])
    
    # Use the vmap of transit_light_curve to calculate a 2D array of shape (M, N) of flux values
    # For M wavelengths and N time points.
    
    M = jnp.zeros((x_l.shape[-1], x_t.shape[-1]))
    
    for i in range(x_l.size):
        p_1D = {"T0":mfp["T0"], "P":mfp["P"], "a":mfp["a"], "rho":mfp["rho"][i], "b":mfp["b"], "r1":mfp["r1"],
               "u1":mfp["u1"][i], "u2":mfp["u2"][i], "Foot":mfp["Foot"][i], "Tgrad":mfp["Tgrad"][i]}
        M = M.at[i, :].set(harmonica_light_curve(p_1D, x_t))
    return M




def bin_data(x_l, x_t, Y, l_bin, t_bin):
    N_l = Y.shape[0]
    N_t = Y.shape[1]
    
    Y_bin = Y.copy()

    if t_bin > 1:
        bin_mat_t = np.kron(np.eye(N_t//t_bin), np.ones(t_bin)/t_bin)
        clip = N_t%t_bin

        if clip:
            x_t_bin = bin_mat_t @ x_t[:-clip]
            Y_bin = (bin_mat_t @ Y_bin[:, :-clip].T).T
        else:
            x_t_bin = bin_mat_t @ x_t
            Y_bin = (bin_mat_t @ Y_bin.T).T
    else:
        x_t_bin = x_t.copy()

    if l_bin > 1:
        bin_mat_l = np.kron(np.eye(N_l//l_bin), np.ones(l_bin)/l_bin)
        clip = N_l%l_bin

        if clip:
            x_l_bin = bin_mat_l @ x_l[:-clip]
            Y_bin = (bin_mat_l @ Y_bin[:-clip, :])
        else:
            x_l_bin = bin_mat_l @ x_l
            Y_bin = (bin_mat_l @ Y_bin)
    else:
        x_l_bin = x_l.copy()
            
    return x_l_bin, x_t_bin, Y_bin


def bin_data_1D(t, y, bin_size):
    N = y.size
        
    y_bin = y.copy()

    if bin_size > 1:
        bin_mat = np.kron(np.eye(N//bin_size), np.ones(bin_size)/bin_size)
        clip = N % bin_size

        if clip:
            t_bin = bin_mat @ t[:-clip]
            y_bin = bin_mat @ y_bin[:-clip]
        else:
            t_bin = bin_mat @ t
            y_bin = bin_mat @ y_bin
    else:
        t_bin = t.copy()
            
    return t_bin, y_bin


def baseline(par, x_l, x_t):
    flux = (par["Foot"] + 24*par["Tgrad"]*(x_t-par["T0"]))
    return flux

baseline_vmap = jax.vmap(baseline, in_axes = ({"T0":None, "Foot":0, "Tgrad":0}, None, None))

def baseline_2D(p, *args, **kwargs):
    
    p_mf = {}
    for par in ["T0", "Foot", "Tgrad"]:
        p_mf[par] = p[par]
        
    return baseline_vmap(p_mf, *args, **kwargs)

def build_kronecker_K(
        self,
        Kl_fns: Optional[list[Callable]] = None,
        Kt_fns: Optional[list[Callable]] = None,
    ) -> Callable:
    
        def K_kron(
            hp: PyTree,
            x_l1: JAXArray,
            x_l2: JAXArray,
            x_t1: JAXArray,
            x_t2: JAXArray, 
            **kwargs,
        ) -> JAXArray:
        
            K = jnp.zeros((x_l1.shape[-1]*x_t1.shape[-1], x_l2.shape[-1]*x_t2.shape[-1]))
            for i in range(len(Kl_fns)):
                Kl = Kl_fns[i](hp, x_l1, x_l2, **kwargs)
                Kt = Kt_fns[i](hp, x_t1, x_t2, **kwargs)
                K += jnp.kron(Kl, Kt)

            return K
        
        return K_kron
    


    
# This code may be faster for full covariance calc for LuasKernel but need to reorder matrix more efficiently 
#     def predict(
#         self,
#         hp: PyTree,
#         x_l: JAXArray,
#         x_l_pred: JAXArray,
#         x_t: JAXArray,
#         x_t_pred: JAXArray,
#         R: JAXArray,
#         M_s: JAXArray,
#         wn = True,
#         return_std_dev = True,
#     ) -> Tuple[JAXArray, JAXArray]:
#         
#         # Calculate the decomposition of K
#         stored_values = self.decomp_fn(hp, x_l, x_t)
        
#         # Calculate the covariance between the observed and predicted points
#         Kl_s = self.Kl(hp, x_l, x_l_pred, wn = False)
#         Kt_s = self.Kt(hp, x_t, x_t_pred, wn = False)
#         Sl_s = self.Sl(hp, x_l, x_l_pred, wn = False)
#         St_s = self.St(hp, x_t, x_t_pred, wn = False)
        
#         # Calculate the covariance between predicted points with other predicted points
#         Kl_ss = self.Kl(hp, x_l_pred, x_l_pred, wn = wn)
#         Kt_ss = self.Kt(hp, x_t_pred, x_t_pred, wn = wn)
#         Sl_ss = self.Sl(hp, x_l_pred, x_l_pred, wn = wn)
#         St_ss = self.St(hp, x_t_pred, x_t_pred, wn = wn)

#         # Calculate K^-1 R
#         K_inv_R = K_inv_vec(R, stored_values)

#         # Calculates the GP mean including the deterministic mean function at the prediction locations
#         gp_mean = M_s + kron_prod(Kl_s.T, Kt_s.T, K_inv_R) + kron_prod(Sl_s.T, St_s.T, K_inv_R)

#         # Prepare matrices for calculating the predictive covariance
#         KW_l = Kl_s.T @ stored_values["W_l"]
#         KW_t = Kt_s.T @ stored_values["W_t"]
#         SW_l = Sl_s.T @ stored_values["W_l"]
#         SW_t = St_s.T @ stored_values["W_t"]

#         if return_std_dev:
#             # Efficiently solves for the diagonal of the predictive covariance
#             pred_err = jnp.outer(jnp.diag(Kl_ss), jnp.diag(Kt_ss))
#             pred_err += jnp.outer(jnp.diag(Sl_ss), jnp.diag(St_ss))

#             # K_s.T K^-1 K_s term can be broken into these three terms
#             pred_err -= kron_prod(KW_l**2, KW_t**2, stored_values["D_inv"])
#             pred_err -= kron_prod(SW_l**2, SW_t**2, stored_values["D_inv"])
#             pred_err -= 2*kron_prod(KW_l * SW_l, KW_t * SW_t, stored_values["D_inv"])
            
#             # Take the sqrt of the diagonal to get the std dev
#             pred_err = jnp.sqrt(pred_err)
            
#         else:
#             # Get the length of each dimension
#             N_l = x_l.shape[-1]
#             N_t = x_t.shape[-1]
#             N_l_pred = x_l_pred.shape[-1]
#             N_t_pred = x_t_pred.shape[-1]

#             # Useful to define to calculate elementwise products between different columns
#             def K_mult(K1, K2):
#                 return K1*K2
#             vmap_K_mult = jax.vmap(K_mult, in_axes = (0, None), out_axes = 0)

#             # First solve for the predictive covariance but in a matrix that will be
#             # of shape (N_l_pred*N_l_pred, N_t_pred*N_t_pred)
#             cov_wrong_order = jnp.zeros((N_l_pred**2, N_t_pred**2))
#             for (Kl1, Kt1) in [(KW_l, KW_t), (SW_l, SW_t)]:
#                 for (Kl2, Kt2) in [(KW_l, KW_t), (SW_l, SW_t)]:

#                     Kl_cube = vmap_K_mult(Kl1, Kl2)
#                     Kt_cube = vmap_K_mult(Kt1, Kt2)

#                     Kl_cube = Kl_cube.reshape((N_l_pred**2, N_l))
#                     Kt_cube = Kt_cube.reshape((N_t_pred**2, N_t))

#                     cov_wrong_order += Kl_cube @ stored_values["D_inv"] @ Kt_cube.T

#             # Begin reshaping to the correct shape of (N_l_pred*N_t_pred, N_l_pred*N_t_pred)
#             cov_wrong_order = cov_wrong_order.reshape((N_l_pred**2*N_t_pred, N_t_pred))
#             pred_err = jnp.zeros((N_l_pred*N_t_pred, N_l_pred*N_t_pred))
            
# #             # Loops through blocks of rows placing elements into the correct order
#             for i in range(N_l_pred):
#                 for j in range(N_l_pred):
#                     cov = cov_wrong_order[(i*N_l_pred+j)*N_t_pred:(i*N_l_pred+j+1)*N_t_pred, :].T
#                     pred_err = pred_err.at[j*N_t_pred:(j+1)*N_t_pred, i*N_t_pred:(i+1)*N_t_pred].set(-cov)
            
# #             # Add the K_ss term
#             pred_err += jnp.kron(Kl_ss, Kt_ss) + jnp.kron(Sl_ss, St_ss)
        
#         return gp_mean, cov_wrong_order

def logPrior_offset2(p):
    c1, c2 = ld_from_kipping(p["u1"], p["u2"])
    
    const_c1 = jnp.power(10, 2*p["const_c1_err"])
    const_c2 = jnp.power(10, 2*p["const_c2_err"])
    
    scale_c1 = p["scale_c1"]**-2
    scale_c2 = p["scale_c2"]**-2
    
    c1_prior_inv = scale_c1*mat1_c1 - mat2_c1*scale_c1**2*const_c1/(1+const_c1*scale_c1*c_c1)
    c2_prior_inv = scale_c2*mat1_c2 - mat2_c2*scale_c2**2*const_c2/(1+const_c2*scale_c2*c_c2)
    
    logdetC_total = jnp.log(1 + const_c1*scale_c1*c_c1) + jnp.log(1 + const_c2*scale_c2*c_c2)
    
    scale_c1 = p["scale_c1"]
    scale_c2 = p["scale_c2"]
    
    logdetC_total += logdetC + 2*N_l*jnp.log(scale_c1) + 2*N_l*jnp.log(scale_c2)
    
    res_c1 = c1 - ld_coeff[:, 0]
    res_c2 = c2 - ld_coeff[:, 1]

    # Calculates the log likelihood
    logPrior = - 0.5 * res_c1.T @ c1_prior_inv @ res_c1 - 0.5 *  res_c2.T @ c2_prior_inv @ res_c2
    logPrior += -0.5*logdetC_total
    
#     logPrior += 60*jnp.log(p["scale_c1"])
#     logPrior += 60*jnp.log(p["scale_c2"])
    
    return logPrior.sum()



"""
try:
    import luas
except ImportError:
    %mkdir downloads
    %cd downloads
    !git clone https://github.com/markfortune/luas.git
    %cd luas
    %pip install . --target=/content
    %cd /content

try:
    import jaxoplanet
except ImportError:
    %pip install -q jaxoplanet==0.0.1

try:
    import ldtk
except ImportError:
    %pip install ldtk

try:
    import numpyro
except ImportError:
    %pip install numpyro

try:
    import corner
except ImportError:
    %pip install corner
    
import jax

print(jax.devices())
"""

