"""
Inference and simulation logic for COVID-19 modeling.

This module provides functions and classes for running Bayesian inference,
simulation, and prediction using an ODE-based epidemiological model.
It leverages JAX, NumPyro, and Diffrax for efficient computation and probabilistic
modeling. Key features include:

- Calculation of time-dependent weights for hospitalizations and variant proportion data.
- Definition of the NumPyro model for hierarchical parameter inference.
- Simulation and prediction utilities for posterior and prior predictive checks.
- Integration with configuration-driven experiment management.
- Support for changepoints, anchors, and hierarchical priors.

Intended for use in COVID-19 forecasting, scenario analysis, and model validation.
"""

from copy import deepcopy
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from diffrax import Solution
from dynode.config import SimulationConfig
from dynode.infer import sample_then_resolve
from dynode.simulation import AbstractODEParams, simulate
from funcs import seasonality
from numpyro import distributions as dist
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive
from ode import get_ode_parameters, seip_ode


def calc_weights_exponential(weeks, forget_factor=0.99, max_week=None):
    """
    Calculate exponentially decaying weights for time series data.

    Parameters
    ----------
    weeks : array-like
        Array of week indices.
    forget_factor : float, optional
        Exponential decay factor (default 0.99).
    max_week : int, optional
        Maximum week index. If None, uses weeks.max().

    Returns
    -------
    weights : ndarray
        Weights array, shape (len(weeks), 1).
    """
    if max_week is None:
        max_week = weeks.max()
    weights = forget_factor ** (max_week - weeks)
    weights = weights.reshape(len(weights), 1)
    return weights


def calc_weights_step(weeks, threshold, multiplier=0.3):
    """
    Calculate stepwise weights for time series data.

    Parameters
    ----------
    weeks : array-like
        Array of week indices.
    threshold : int
        Week threshold for step change.
    multiplier : float, optional
        Weight for weeks below the threshold (default 0.3).

    Returns
    -------
    weights : ndarray
        Weights array, shape (len(weeks), 1).
    """
    weights = jnp.where(weeks < threshold, multiplier, 1.0)
    weights = weights.reshape(len(weights), 1)
    return weights


def calculate_weights(
    obs_data,
    hosp_weight_type="constant",
    var_weight_type="constant",
    hosp_weight_constant=1.0,
    var_weight_constant=1.0,
    **kwargs,
):
    """
    Calculate weights for hospitalizations and variant proportion data.

    Parameters
    ----------
    obs_data : dict
        Dictionary with observed data arrays.
    hosp_weight_type : {'constant', 'exponential'}, optional
        Type of weighting for hospitalizations.
    var_weight_type : {'constant', 'exponential'}, optional
        Type of weighting for variant data.
    hosp_weight_constant : float, optional
        Constant weight for hospitalizations.
    var_weight_constant : float, optional
        Constant weight for variant data.
    **kwargs : dict
        Additional keyword arguments (e.g., forgetting_factor).

    Returns
    -------
    hosp_weights : ndarray
        Weights for hospitalizations.
    var_weights : ndarray
        Weights for variant data.
    """
    assert hosp_weight_type in ["constant", "exponential"]
    assert var_weight_type in ["constant", "exponential"]

    obs_hosps_weeks = obs_data["obs_hosps_weeks"]
    obs_var_weeks = obs_data["obs_var_weeks"]

    if hosp_weight_type == "constant":
        hosp_weights = jnp.repeat(hosp_weight_constant, len(obs_hosps_weeks))
        hosp_weights = hosp_weights.reshape(len(hosp_weights), 1)
    elif hosp_weight_type == "exponential":
        if "forgetting_factor" in kwargs:
            hosp_weights = (
                calc_weights_exponential(
                    obs_hosps_weeks, kwargs["forgetting_factor"]
                )
                * hosp_weight_constant
            )
        else:
            raise Exception(
                "forgetting_factor must be specified in exponential mode."
            )

    if var_weight_type == "constant":
        var_weights = jnp.repeat(var_weight_constant, len(obs_var_weeks))
        var_weights = var_weights.reshape(len(var_weights), 1)
    elif var_weight_type == "exponential":
        if "forgetting_factor" in kwargs:
            var_weights = (
                calc_weights_exponential(
                    obs_var_weeks,
                    kwargs["forgetting_factor"],
                    obs_hosps_weeks.max(),
                )
                * var_weight_constant
            )
        else:
            raise Exception(
                "forgetting_factor must be specified in exponential mode."
            )

    return hosp_weights, var_weights


def likelihood(
    model_hosps,
    model_strain_incidence,
    obs_hosps_weekly,
    obs_hosps_weeks,
    obs_var_weeks,
    obs_var_prop,
    week0_day,
    duration_days,
    hosp_weights,
    var_weights,
    **kwargs,
):
    """
    Link model predictions to observed data and update the sampler.

    Parameters
    ----------
    model_hosps : ndarray
        Model-predicted hospitalizations (days, ages).
    model_strain_incidence : ndarray
        Model-predicted strain incidence (days, ...).
    obs_hosps_weekly : ndarray
        Observed weekly hospitalizations.
    obs_hosps_weeks : ndarray
        Indices of observed hospital weeks.
    obs_var_weeks : ndarray
        Indices of observed variant weeks.
    obs_var_prop : ndarray
        Observed variant proportions.
    week0_day : int
        Day index of week 0.
    duration_days : int
        Total simulation days.
    hosp_weights : ndarray
        Weights for hospitalizations.
    var_weights : ndarray
        Weights for variant data.
    **kwargs : dict
        Additional keyword arguments.
    """

    # Create binning "schema", i.e., which day belongs to week 0, 1, 2, etc.

    days_after_week0 = len(model_hosps) - (week0_day + 1)
    bins = jnp.array([0]).repeat(week0_day + 1)
    bins = jnp.append(
        bins,
        jnp.arange(1, (days_after_week0) // 7 + 1, dtype=jnp.int32).repeat(7),
    )
    bins = jnp.append(bins, (jnp.max(bins) + 1).repeat(days_after_week0 % 7))
    # definitive number of bins calculate using a constant (tf)
    nbins = int(np.ceil((duration_days - (week0_day + 1)) / 7)) + 1

    sim_hosps_weekly = jnp.array(
        [
            jnp.bincount(
                bins,
                m,
                length=nbins,
            )
            for m in model_hosps.T  # loop through age groups
        ]
    ).T
    sim_hosps_weekly_sel = sim_hosps_weekly[obs_hosps_weeks]

    base_concentration = 10.0  # Baseline concentration parameter for Negative Binomial likelihood (controls overdispersion)

    # EXTRA overdispersion (variance inflation-like approach) parameter

    overdisp_hosp = numpyro.sample(
        "overdisp_hosp",
        dist.HalfNormal(1),  # or you could use HalfCauchy(1.0)
    )

    # Effective concentration: larger overdisp_hosp (means) smaller concentration  and larger variance
    concentration = (base_concentration * sim_hosps_weekly_sel) / (
        overdisp_hosp**2
    )

    with numpyro.handlers.scale(scale=hosp_weights):  # weights
        numpyro.sample(
            "incidence",
            dist.NegativeBinomial2(sim_hosps_weekly_sel, concentration),
            obs=obs_hosps_weekly,
        )

    ## Variant proportion
    model_strain_incidence = jnp.diff(model_strain_incidence, axis=0)
    strain_incidence_weekly = jnp.array(
        [jnp.bincount(bins, m, length=nbins) for m in model_strain_incidence.T]
    ).T
    strain_incidence_weekly_sel = [
        strain_incidence_weekly[w - 1] + strain_incidence_weekly[w]
        for w in obs_var_weeks.tolist()
    ]
    sim_var_prop = jnp.array(
        [incd / jnp.sum(incd) for incd in strain_incidence_weekly_sel]
    )

    with numpyro.handlers.scale(
        scale=var_weights
    ):  # scaling_factor * var_weights
        numpyro.sample(
            "variant_proportion",
            dist.Normal(
                sim_var_prop, 0.05
            ),  # Standard deviation for variant proportion likelihood; reflects assumed measurement noise
            obs=obs_var_prop,
        )


def run_simulation(
    config: SimulationConfig,
    duration_days: int,
    vax_data: dict[str, jax.Array],
) -> Tuple[AbstractODEParams, Solution]:
    """
    Run a simulation using the ODE model and configuration.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.
    duration_days : int
        Number of days to simulate.
    vax_data : dict
        Vaccination data dictionary.

    Returns
    -------
    ode_parameters : AbstractODEParams
        ODE model parameters.
    solution : Solution
        ODE solution object.
    """
    ode_parameters = get_ode_parameters(config, vax_data)
    # print("num strain:", ode_parameters.num_strains)
    initial_state = config.initializer.get_initial_state(
        initial_infections_scale=ode_parameters.initial_infections_scale
    )
    solution = simulate(
        ode=seip_ode,
        duration_days=duration_days,
        initial_state=initial_state,
        ode_parameters=ode_parameters,
        solver_parameters=config.parameters.solver_params,
    )
    return ode_parameters, solution


def get_predictions(
    config: SimulationConfig, ode_p: AbstractODEParams, solution: Solution
):
    """
    Get model predictions of observed data given a simulation solution.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.
    ode_p : AbstractODEParams
        ODE model parameters.
    solution : Solution
        ODE solution object.

    Returns
    -------
    dict
        Dictionary of model predictions (hospitalizations, strain incidence, etc).
    """
    obs_params = deepcopy(config.parameters.observation_params)
    obs_params = sample_then_resolve(obs_params)

    ihr_ages = jnp.array(
        [
            obs_params.ihr_3 * obs_params.ihr_mult_0,
            obs_params.ihr_3 * obs_params.ihr_mult_1,
            obs_params.ihr_3 * obs_params.ihr_mult_2,
            obs_params.ihr_3,
        ]
    )

    # Immunity-induced IHR multiplier
    # Previously this was induced by previous infection or vaccinations
    # Now we are trying out waning bin-induced protection
    # Needs logic improvement
    ihr_strains_by_group = [
        1.0,
        obs_params.ihr_mult_strain_g1,
        obs_params.ihr_mult_strain_g1 * obs_params.ihr_mult_strain_g2,
        obs_params.ihr_mult_strain_g1
        * obs_params.ihr_mult_strain_g2
        * obs_params.ihr_mult_strain_g3,
    ]
    ihr_strains = [
        ihr_strains_by_group[g] for g in obs_params.ihr_strains_groups
    ]
    # Strain and history-specific IHR
    # Used to account for reinfection but no longer
    ihr_hist_strains = jnp.array(
        [ihr_strains] * (ode_p.num_strains + 1)
    )  # hist x strain

    # Strain, history, and wane-specific IHR
    # IHR multiplier comes from the config
    ihr_wane = jnp.array(obs_params.ihr_wane)
    ihr_hist_wane_str = ihr_hist_strains[:, None, :] * ihr_wane[None, :, None]

    # Age-inf_hist-strain IHR
    ihr_age_hist_wane_str = (
        ihr_ages[:, None, None, None] * ihr_hist_wane_str[None, :, :, :]
    )

    # Calculate modelled hospitalizations based on the IHRs
    # Enums have to be +1 due to the additional time component
    model_incidence = jnp.diff(
        solution.ys[ode_p.idx.c],
        axis=0,  # time dimension
    )
    model_incidence_all_vax_status = jnp.sum(
        model_incidence, axis=ode_p.idx.c.vax + 1
    )  # take vax out
    model_hosps = jnp.sum(
        model_incidence_all_vax_status * ihr_age_hist_wane_str, axis=(2, 3, 4)
    )  # day, age
    # Extract the "general" IHR, which is like the population-weighted IHR
    # model_hosps = (day, age)
    # jnp.sum(model_incidence, axis=(2, 3, 4)) = (day, age)
    general_ihr = model_hosps / jnp.sum(
        model_incidence_all_vax_status, axis=(2, 3, 4)
    )  # age-specific IHR per day

    # Strain prevalence
    strain_incidence = jnp.sum(
        solution.ys[ode_p.idx.c],
        axis=(
            ode_p.idx.c.age + 1,
            ode_p.idx.c.hist + 1,
            ode_p.idx.c.vax + 1,
            ode_p.idx.c.wane + 1,
        ),
    )
    # Extract vaccine status-specific incidence (after BOOSTER_START_DAY)
    # We are only interested in the 2024 booster
    # Needs logic improvement
    # Not boosted
    model_incidence_no_booster = jnp.sum(
        model_incidence[:, :, :, :3, :, :],
        axis=ode_p.idx.c.vax + 1,
    )
    model_hosps_no_booster = jnp.sum(
        model_incidence_no_booster * ihr_age_hist_wane_str, axis=(1, 2, 3, 4)
    )  # day
    # Boosted
    model_incidence_booster = model_incidence[:, :, :, 3, :, :]
    model_hosps_booster = jnp.sum(
        model_incidence_booster * ihr_age_hist_wane_str, axis=(1, 2, 3, 4)
    )  # day
    # Extract boosted vs not boosted population
    # Here we extracting [1:, ...] because incidence is always one time point
    # less than the state.
    pop_no_booster = (
        jnp.sum(
            solution.ys[ode_p.idx.s][1:, :, :, :3, :],
            axis=(1, 2, 3, 4),
        )
        + jnp.sum(
            solution.ys[ode_p.idx.e][1:, :, :, :3, :],
            axis=(1, 2, 3, 4),
        )
        + jnp.sum(
            solution.ys[ode_p.idx.i][1:, :, :, :3, :],
            axis=(1, 2, 3, 4),
        )
    )
    pop_booster = (
        jnp.sum(
            solution.ys[ode_p.idx.s][1:, :, :, 3, :],
            axis=(1, 2, 3),
        )
        + jnp.sum(
            solution.ys[ode_p.idx.e][1:, :, :, 3, :],
            axis=(1, 2, 3),
        )
        + jnp.sum(
            solution.ys[ode_p.idx.i][1:, :, :, 3, :],
            axis=(1, 2, 3),
        )
    )

    return {
        "model_hosps": model_hosps,
        "model_strain_incidence": strain_incidence,
        "model_general_ihr": general_ihr,
        "model_hosps_no_booster": model_hosps_no_booster,
        "model_hosps_booster": model_hosps_booster,
        "pop_no_booster": pop_no_booster,
        "pop_booster": pop_booster,
    }


def save_posteriors(
    model_hosps,
    model_strain_incidence,
    model_general_ihr,
    duration_days,
    obs_data,
    seasonality_kwargs,
    **kwargs,
):
    """
    Save deterministic posterior quantities to the numpyro trace.

    Parameters
    ----------
    model_hosps : ndarray
        Model-predicted hospitalizations.
    model_strain_incidence : ndarray
        Model-predicted strain incidence.
    model_general_ihr : ndarray
        Model-predicted general IHR.
    duration_days : int
        Number of simulation days.
    obs_data : dict
        Observed data dictionary.
    seasonality_kwargs : dict
        Arguments for seasonality function.
    **kwargs : dict
        Additional deterministic quantities to save.
    """
    obs_hosps_weeks = obs_data["obs_hosps_weeks"]
    week0_day = obs_data["week0_day"]
    # Create binning "schema", i.e., which day belongs to week 0, 1, 2, etc.
    bins = jnp.arange(len(model_hosps) // 7, dtype=jnp.int32).repeat(7)

    bins = jnp.append(bins, (jnp.max(bins) + 1).repeat(len(model_hosps) % 7))
    # definitive number of bins calculate using a constant (tf)
    nbins = int(np.ceil(duration_days / 7))
    model_strain_incidence = jnp.diff(model_strain_incidence, axis=0)
    infection_incidence = jnp.sum(model_strain_incidence, axis=1)
    infection_incidence_weekly = jnp.bincount(
        bins, infection_incidence, length=nbins
    )
    infection_incidence_weekly_sel = infection_incidence_weekly[
        obs_hosps_weeks
    ]
    numpyro.deterministic(
        "seasonality_mult",
        seasonality(obs_hosps_weeks * 7 + week0_day, **seasonality_kwargs),
    )
    numpyro.deterministic(
        "infection_incidence", infection_incidence_weekly_sel
    )
    numpyro.deterministic("general_ihr", model_general_ihr)
    # For when we need to extract values to calculate booster VE
    # Comment out if not needed to save space
    # for k, v in kwargs.items():
    #     numpyro.deterministic(k, v)


def numpyro_model(
    config: SimulationConfig,
    duration_days: int,
    data: dict[str, dict],
):
    """
    Main NumPyro model: runs simulation, computes predictions, and applies likelihood.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.
    duration_days : int
        Number of simulation days.
    data : dict
        Dictionary with observed and vaccination data.
    """
    obs_data = data["obs_data"]
    vax_data = data["vax_data"]
    ode_parameters, solution = run_simulation(config, duration_days, vax_data)
    model_predictions = get_predictions(config, ode_parameters, solution)
    likelihood(
        duration_days=duration_days,
        **model_predictions,
        **obs_data,
    )
    # observe and likelihood
    save_posteriors(
        duration_days=duration_days,
        obs_data=obs_data,
        seasonality_kwargs={
            "seasonality_amplitude": ode_parameters.seasonality_amplitude,
            "seasonality_second_wave": ode_parameters.seasonality_second_wave,
            "seasonality_dominant_wave_day": ode_parameters.seasonality_dominant_wave_day,
            "seasonality_second_wave_day": ode_parameters.seasonality_second_wave_day,
            "seasonality_wave_width": ode_parameters.seasonality_wave_width,
            "init_date": ode_parameters.init_date,
        },
        **model_predictions,
    )


def get_prior_samples(
    config,
    duration_days,
    data,
    num_samples=500,
    rng_seed=1234,
):
    """
    Generate prior samples for numpyro_model parameters.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration object.
    duration_days : int
        Number of simulation days.
    data : dict
        Data dictionary for the model.
    num_samples : int, optional
        Number of samples to draw (default 500).
    rng_seed : int, optional
        Random seed (default 1234).

    Returns
    -------
    dict
        Dictionary of prior samples for each parameter.

    Notes
    -----
    Assumes parameters are independent in the prior. If dependencies exist, modify to sample from the joint prior.
    """
    rng_key = jax.random.PRNGKey(rng_seed)
    model_trace = trace(seed(numpyro_model, rng_key)).get_trace(
        config=config, duration_days=duration_days, data=data
    )
    samp_dict = {}
    for k, v in model_trace.items():
        if (not (v["type"] == "sample")) or v["is_observed"]:
            continue
        samps = v["fn"].sample(rng_key, (num_samples,))
        samp_dict[k] = samps
    return samp_dict


def posterior_predictive(
    posterior_samples, duration_days, data, rng_seed, config
):
    """
    Produce predictive hospitalizations, variant proportions, seasonality, and infection incidence.

    Parameters
    ----------
    posterior_samples : dict
        Posterior samples from MCMC.
    duration_days : int
        Number of simulation days.
    data : dict
        Data dictionary for the model.
    rng_seed : int
        Random seed for prediction.
    config : SimulationConfig
        Simulation configuration object.

    Returns
    -------
    dict
        Dictionary of posterior predictive samples and week indices.
    """

    hosps_weeks = jnp.arange(
        1,
        int((duration_days - (data["obs_data"]["week0_day"] + 1) - 1) / 7) + 1,
    )
    var_weeks = jnp.arange(
        3,  # Start at week 3 for variant data
        int((duration_days - (data["obs_data"]["week0_day"] + 1) - 1) / 7) + 1,
        step=2,  # Use every other week for variant data (biweekly )
    )

    predictive = Predictive(
        numpyro_model,
        posterior_samples=posterior_samples,
        batch_ndims=1,
    )
    # set select keys to None to generate posterior predictions
    data["obs_data"]["obs_hosps_weekly"] = None
    data["obs_data"]["obs_var_prop"] = None
    data["obs_data"]["obs_hosps_weeks"] = hosps_weeks
    data["obs_data"]["obs_var_weeks"] = var_weeks
    posterior_pred = predictive(
        jax.random.PRNGKey(rng_seed),
        config=config,
        data=data,
        duration_days=duration_days,
    )
    posterior_pred["hosps_weeks"] = hosps_weeks
    posterior_pred["var_weeks"] = var_weeks
    return posterior_pred
