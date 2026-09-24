"""Entry point for running COVID inference and projection.

The script loads observed hospitalization and variant data, configures and
runs Bayesian inference (SVI or NUTS), generates prior and posterior
predictive samples, and writes results as JSON for downstream analysis.

Notes
-----
Intended for COVID-19 scenario analysis, forecasting, and model validation
pipelines. Configuration-driven via :class:`Covid5StrainConfig`.
"""

import argparse
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from configs import Covid5StrainConfig as cfg
from dynode import SimulationConfig, SVIProcess, log
from inference import (
    calculate_weights,
    get_prior_samples,
    numpyro_model,
    posterior_predictive,
)
from numpyro.infer import MCMC, NUTS
from numpyro.optim import Adam
from utils import (
    dict_to_json,
    get_fitting_day,
    load_vaccination_model_hill,
    load_vaccination_model_splines,
)

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

# Inference mode and sampler settings
INFERENCE_MODE = "svi"  # "svi" or "nuts"
SVI_ITERATIONS = 1000
SVI_SAMPLES = 500
SVI_STEP_SIZE = 0.15
NUTS_SAMPLES = 200
NUTS_WARMUP = 50
NUTS_CHAINS = 2
NUTS_RNG_SEED = 42

# Predictive stuff
NUM_PRIOR_SAMPLES = 500
PRI_PRED_RNG_SEED = 42
POS_PRED_RNG_SEED = 42

# Time horizon
HOSP_END_DATE = "2026-03-21"  # Last hospitalization date
PROJECTION_HORIZON_DAYS = 364
FITTING_DAY = get_fitting_day(HOSP_END_DATE)
PROJECTION_DAY = FITTING_DAY + PROJECTION_HORIZON_DAYS

# Observed hospitalizations lag infections by a week
INF_TO_HOSP_DELAY_DAYS = 7

# Data cleaning: state-specific bad reporting dates to drop
STATE_OUTLIER_DATES: dict[str, list[str]] = {
    "Maine": ["2023-09-30"],
    "New Hampshire": ["2023-09-16"],
}

# NSSP downweighting window for hospitalizations
NSSP_WINDOW_START = "2024-04-27"
NSSP_WINDOW_END = "2024-11-09"
NSSP_DOWNWEIGHT = (
    0.5  # balances summer-wave fit vs. uncertainty around its size
)

# Age-group indexing for the vaccination spline model
AGE_GROUP_INDEX = {"0-17": 0, "18-49": 1, "50-64": 2, "65+": 3}


def preprocess_observed_data(
    state_name: str, config: cfg, model_day: int
) -> dict[str, Any]:
    """Read and preprocess observed hospitalization and variant data.

    Parameters
    ----------
    state_name : str
        Full state name (e.g., ``"Maine"`` rather than ``"ME"``).
    config : cfg
        Experiment configuration.
    model_day : int
        Number of days the model will run; observed data is truncated to
        this horizon.

    Returns
    -------
    dict
        Processed observed data with keys:

        - ``week0_day`` : int, day index of the first observed week.
        - ``obs_hosps_weekly`` : jnp.ndarray, weekly hospitalizations by age.
        - ``obs_hosps_weeks`` : jnp.ndarray[int], week indices for hosp obs.
        - ``obs_var_prop`` : jnp.ndarray, renormalized variant proportions.
        - ``obs_var_weeks`` : jnp.ndarray[int], week indices for variant obs.
    """
    hosp_data_filename = f"{state_name.replace(' ', '_')}_hospitalization.csv"
    hosp_data_path = os.path.join(config.hosp_path, hosp_data_filename)
    hosp_data = pd.read_csv(hosp_data_path)
    hosp_data["date"] = pd.to_datetime(hosp_data["date"])

    # Drop known per-state outlier reporting dates
    for bad_date in STATE_OUTLIER_DATES.get(state_name, []):
        hosp_data = hosp_data[hosp_data["date"] != pd.to_datetime(bad_date)]

    # Align hosp to infections via the inf -> hosp delay
    hosp_data["day"] = (
        hosp_data["date"] - pd.to_datetime(config.initializer.initialize_date)
    ).dt.days - INF_TO_HOSP_DELAY_DAYS

    # Keep only data within the model window, sorted ascending
    model_infection_day = (
        model_day - INF_TO_HOSP_DELAY_DAYS
    )  # Supplied model_day is hosp day, account for delay
    hosp_data = hosp_data.loc[
        (hosp_data["day"] >= 0) & (hosp_data["day"] <= model_infection_day)
    ].sort_values(by=["day", "agegroup"], ascending=True, inplace=False)

    # Reshape into a day x agegroup matrix
    obs_hosps = hosp_data.groupby(["day"])["hosp"].apply(np.array)
    obs_hosps_days = jnp.array(obs_hosps.index.to_list())
    obs_hosps = jnp.array(obs_hosps.to_list())

    # Input data is weekly; record week-0 reference and compute week indices
    week0_day = obs_hosps_days[0].item()
    obs_hosps_weeks = (obs_hosps_days - week0_day) / 7

    # Enforce week >= 1 to guarantee a full week of observations
    mask = obs_hosps_weeks >= 1
    obs_hosps_weekly = obs_hosps[mask] * 7
    obs_hosps_weeks = obs_hosps_weeks[mask]

    # Variant proportions
    ## Variant data pre-formatted with 7 strains
    var_data_filename = f"{state_name.replace(' ', '_')}_strain_prop.csv"
    var_data_path = os.path.join(config.var_prop_path, var_data_filename)
    var_data = pd.read_csv(var_data_path)
    var_data["date"] = pd.to_datetime(
        var_data["date"]
    )  # No alignment shift for variants

    var_data["day"] = (
        var_data["date"] - pd.to_datetime(config.initializer.initialize_date)
    ).dt.days
    var_data = var_data.loc[
        (var_data["day"] >= 0) & (var_data["day"] <= model_day)
    ].sort_values(by=["day", "strain"], ascending=True, inplace=False)
    obs_var_prop = var_data.groupby(["day"])["share"].apply(np.array)
    obs_var_days = jnp.array(obs_var_prop.index.to_list())
    obs_var_prop = jnp.array(obs_var_prop.to_list())

    # Renormalize variant proportions, guarding against zero rows
    row_sums = jnp.sum(obs_var_prop, axis=1)[:, None]
    obs_var_prop = obs_var_prop / jnp.where(row_sums == 0, 1e-8, row_sums)

    # Convert days to weeks; require week >= 2 (a full fortnight)
    obs_var_weeks = (obs_var_days - week0_day) / 7
    mask = obs_var_weeks >= 2
    obs_var_prop = obs_var_prop[mask]
    obs_var_weeks = obs_var_weeks[mask]

    return {
        "week0_day": week0_day,
        "obs_hosps_weekly": obs_hosps_weekly,
        "obs_hosps_weeks": obs_hosps_weeks.astype(int),
        "obs_var_prop": obs_var_prop,
        "obs_var_weeks": obs_var_weeks.astype(int),
    }


def downweight_nssp_hosp_weights(
    obs_data: dict, initial_date, weight: float
) -> jnp.ndarray:
    """Downweight observed hospitalizations during the NSSP reporting window.

    Parameters
    ----------
    obs_data : dict
        Output of :func:`preprocess_observed_data`.
    initial_date : str or pandas.Timestamp
        Simulation initial date.
    weight : float
        Multiplicative weight applied to observations within the NSSP window
        (``NSSP_WINDOW_START`` to ``NSSP_WINDOW_END``); other observations
        keep a weight of ``1.0``.

    Returns
    -------
    jnp.ndarray
        Column vector of per-observation weights with shape ``(num_obs, 1)``.
    """
    # Reconstruct dates: undo the infection->hosp shift applied earlier
    obs_hosps_days = (
        obs_data["obs_hosps_weeks"] * 7
        + obs_data["week0_day"]
        + INF_TO_HOSP_DELAY_DAYS
    )
    obs_hosps_dates = pd.to_datetime(initial_date) + pd.to_timedelta(
        obs_hosps_days, unit="D"
    )
    cond = (obs_hosps_dates >= pd.to_datetime(NSSP_WINDOW_START)) & (
        obs_hosps_dates <= pd.to_datetime(NSSP_WINDOW_END)
    )
    return jnp.where(cond, weight, 1.0)[:, None]


def process_vaccination_data(
    config: SimulationConfig,
) -> dict[str, Any]:
    """Load and assemble vaccination model parameters (splines + Hill).

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``spline_model_params`` : tuple of arrays for the spline model.
        - ``hill_model_params`` : list of three Hill-model parameter tuples
          for seasons 24/25, 25/26, and 26/27 (the last reuses 25/26 with a
          ``ANNUAL_SHIFT_DAYS``-day forward shift).
    """
    spline_data_dict = load_vaccination_model_splines(
        config.vaccination_model_spline_path,
        config.region_name,
        config.num_age_groups,
        config.max_vaccination_count,
        AGE_GROUP_INDEX,
    )
    spline_model_params = (
        spline_data_dict["vaccination_model_knot_locations"],
        spline_data_dict["vaccination_model_base_equations"],
        spline_data_dict["vaccination_model_knots"],
    )
    hill_data_dict1 = load_vaccination_model_hill(
        region_name=config.region_name,
        hill_data_path=config.vaccination_model_hill_2425_path,
        num_age_groups=config.num_age_groups,
        max_vaccination_count=config.max_vaccination_count,
        init_date=config.initializer.initialize_date,
    )
    hill_data_dict2 = load_vaccination_model_hill(
        region_name=config.region_name,
        hill_data_path=config.vaccination_model_hill_2526_path,
        num_age_groups=config.num_age_groups,
        max_vaccination_count=config.max_vaccination_count,
        init_date=config.initializer.initialize_date,
    )
    hill_model_params1 = (
        hill_data_dict1["t_shifts"],
        hill_data_dict1["n_batched"],
        hill_data_dict1["t_h_batched"],
        hill_data_dict1["scales_batched"],
    )
    hill_model_params2 = (
        hill_data_dict2["t_shifts"],
        hill_data_dict2["n_batched"],
        hill_data_dict2["t_h_batched"],
        hill_data_dict2["scales_batched"],
    )
    # 26/27 vaccination curve reuses 25/26 shifted forward by one year
    hill_model_params3 = (
        hill_data_dict2["t_shifts"] + 365,
        hill_data_dict2["n_batched"],
        hill_data_dict2["t_h_batched"],
        hill_data_dict2["scales_batched"],
    )

    return {
        "spline_model_params": spline_model_params,
        "hill_model_params": [
            hill_model_params1,
            hill_model_params2,
            hill_model_params3,
        ],
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--state",
    type=str,
    help="directory for the state to run, resembles USPS code of the state",
)

parser.add_argument(
    "-j", "--jobid", type=str, help="job-id of the state being run on Azure"
)

if __name__ == "__main__":
    # Housekeeping: parse arguments, set up logging, and create the output directory
    logger = logging.getLogger("dynode")
    args = parser.parse_args()
    jobid = args.jobid
    state = args.state
    save_path = f"{Path(__file__).parent}/output/{jobid}/{state}/"
    os.makedirs(save_path, exist_ok=True)
    log.use_logging(
        level="info",
        output="console",
        log_path=os.path.join(save_path, "logs/"),
    )

    if state == "EX":
        state = "GA"

    # Load the experiment configuration
    config = cfg(state)
    logger.info("Config initialized.")

    # Load and preprocess the observed data
    obs_data = preprocess_observed_data(
        config.region_name, config, model_day=FITTING_DAY
    )

    # Calculate weights for hospitalizations and variant data
    ## We reduce likelihood weights based on state-level vs. national-level
    ## hospitalization: see `utils.calculate_scaling_factor`.
    ## For demonstration purpose, we pre-calculated the scaling factor here
    scaling_factor_lookup = json.load(
        open("./data/variant-data/scaling_factors.json")
    )
    scaling_factor = scaling_factor_lookup[config.region_name]
    hosp_weights, var_weights = calculate_weights(
        obs_data, var_weight_constant=scaling_factor
    )  # "Regular likelihood" constant for both likelihood

    # Downweight NSSP data and merge into obs_data
    hosp_weights = downweight_nssp_hosp_weights(
        obs_data,
        config.initializer.initialize_date,
        weight=NSSP_DOWNWEIGHT,
    )
    obs_data["hosp_weights"] = hosp_weights
    obs_data["var_weights"] = var_weights

    # Load the vaccination data
    vax_data = process_vaccination_data(config)

    # Combine observed and vaccination data
    data = {
        "obs_data": obs_data,
        "vax_data": vax_data,
    }
    logger.info(
        f"Data loaded: likelihood weights for variant proportion is {scaling_factor}"
    )

    # Generate and save prior predictive samples
    prior_samples = get_prior_samples(
        config,
        PROJECTION_DAY,
        data,
        num_samples=NUM_PRIOR_SAMPLES,
        rng_seed=PRI_PRED_RNG_SEED,
    )
    dict_to_json(
        prior_samples,
        save_path=os.path.join(save_path, "prior_samples.json"),
        mock_chain=True,
    )

    # Run inference (SVI or NUTS)
    t = time.time()
    posterior_samples = None

    if INFERENCE_MODE == "svi":
        inferer_svi = SVIProcess(
            numpyro_model=numpyro_model,
            num_iterations=SVI_ITERATIONS,
            num_samples=SVI_SAMPLES,
            optimizer=Adam(step_size=SVI_STEP_SIZE),
        )
        logger.info(
            f"starting SVI {SVI_ITERATIONS} iterations and {SVI_SAMPLES} samples"
        )
        inferer_svi.infer(
            config=config,
            duration_days=FITTING_DAY,
            data=data,
        )
        logger.info(f"fitting (SVI) took {time.time() - t} seconds")

        # Generate and save posterior samples.
        t = time.time()
        posterior_samples = inferer_svi.get_samples()
        dict_to_json(
            posterior_samples,
            save_path=os.path.join(save_path, "samples.json"),
            mock_chain=True,
        )
        logger.info(
            f"getting posterior samples took {time.time() - t} seconds"
        )
        t = time.time()
    else:
        # NUTS / MCMC path
        logger.info(
            f"starting NUTS {NUTS_WARMUP} warmup iterations and {NUTS_SAMPLES} samples"
        )
        nuts_kernel = NUTS(numpyro_model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=NUTS_WARMUP,
            num_samples=NUTS_SAMPLES,
            num_chains=NUTS_CHAINS,
            chain_method="parallel",
            progress_bar=True,
        )
        mcmc.run(
            jax.random.PRNGKey(NUTS_RNG_SEED),
            config=config,
            duration_days=FITTING_DAY,
            data=data,
        )
        logger.info(f"fitting (NUTS) took {time.time() - t} seconds")

        t = time.time()
        posterior_samples = mcmc.get_samples(
            group_by_chain=False
        )  # flat (total_samples, ...)
        # Exclude deterministic sites (ndim > 1) to match SVI's exclude_deterministic=True
        # behavior — they are timeseries arrays that break the medians DataFrame
        samples_to_save = {
            k: v for k, v in posterior_samples.items() if v.ndim <= 1
        }
        dict_to_json(
            samples_to_save,
            save_path=os.path.join(save_path, "samples.json"),
            mock_chain=True,  # wrap to (1, total_samples, ...) so postprocessing axis=(0,1) works
        )
        logger.info(
            f"getting posterior samples took {time.time() - t} seconds"
        )
        t = time.time()

    # Generate and save posterior predictive samples
    posterior_predictions = posterior_predictive(
        posterior_samples=posterior_samples,
        duration_days=PROJECTION_DAY,
        rng_seed=POS_PRED_RNG_SEED,
        data=deepcopy(data),
        config=config,
    )
    combined_dict = dict(posterior_predictions) | data["obs_data"]
    dict_to_json(
        combined_dict,
        save_path=os.path.join(save_path, "predictive_output.json"),
        mock_chain=False,
    )
    logger.info(
        f"generating posterior predictive took {time.time() - t} seconds"
    )
