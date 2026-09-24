"""Forward projection based on given scenarios and fitted sets of posterior
samples.
"""

import argparse
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from configs import Covid6StrainProjConfig as cfg
from dynode.utils import log
from inference import (
    calculate_weights,
    posterior_predictive,
)
from run_fitting import (
    FITTING_DAY,
    PROJECTION_DAY,
    preprocess_observed_data,
    process_vaccination_data,
)
from utils import dict_to_json


def unintroduce_x_strain(config):
    """Set introduction of the 6th strain to 0."""
    strains = config.parameters.transmission_params.strains
    ind = [i for i, s in enumerate(strains) if s.strain_name == "X"][0]
    strains[ind].introduction_percentage = 0.0
    si = config.parameters.transmission_params.strain_interactions
    si["X"] = {
        "omicron": 1.0,
        "BA2BA5": 1.0,
        "XBB": 1.0,
        "JN1": 1.0,
        "XEC": 1.0,
        "X": 1.0,
    }


def transform(posterior_samples, previous_fit_median):
    """Create new strain parameters in the posterior samples based on
    other parameters using the mappings defined below. The `target_mean` is
    the mean value we want for the new parameter. Using the US median we
    calculated via post-processing scripts in previous fit"""
    medians = pd.read_csv(previous_fit_median)
    us_medians = medians[medians["state"] == "US"]

    mappings = [
        ("strain_interactions_XEC_XBB", "strain_interactions_X_JN1"),
        ("strain_interactions_XEC_JN1", "strain_interactions_X_XEC"),
        ("strain_interactions_XEC_XEC", "strain_interactions_X_X"),
        ("strain_interactions_3_steps", "strain_interactions_X_XBB"),
        ("strain_interactions_3_steps", "strain_interactions_X_BA2BA5"),
        (
            "strains_4_introduction_percentage",
            "strains_5_introduction_percentage",
        ),
    ]
    for src, dst in mappings:
        v = posterior_samples[src]
        target_mean = us_medians[src][0]
        posterior_samples[dst] = v * target_mean / v.mean()

    return posterior_samples


def load_config_sample(
    scenario, state, previous_fit_posteriors, previous_fit_median
):
    """Load and modify config and posterior samples based on scenario."""
    config = cfg(state)
    # Casting list back to array and remove mock chain from posterior samples
    posterior_samples = json.load(open(previous_fit_posteriors, "r"))
    posterior_samples = {
        k: jnp.array(v[0]) for k, v in posterior_samples.items()
    }
    # Modify config or posterior samples based on scenario
    if scenario == "nointro":
        # KP and X not introduced
        unintroduce_x_strain(config)
    elif scenario == "intro":
        # Only KP not introduced, X is XEC-like
        # Modify posterior samples to reflect XEC-like using resampled values
        posterior_samples = transform(posterior_samples, previous_fit_median)
    else:
        raise ValueError(f"Invalid scenario: {scenario}")
    return config, posterior_samples


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
parser.add_argument(
    "-sc",
    "--scenario",
    required=True,
    type=str,
    help="Scenario ID",
    choices=[
        "nointro",
        "intro",
    ],
)
parser.add_argument("-pf", "--prev-fit", type=str, help="previous fit job id")

if __name__ == "__main__":
    logger = logging.getLogger("dynode")
    args = parser.parse_args()
    jobid = args.jobid
    state = args.state
    scenario = args.scenario
    PROJECT_ROOT = Path(__file__).parent
    save_path = f"{PROJECT_ROOT}/output/{jobid}/{state}/{scenario}"
    prev_fit = args.prev_fit
    previous_fit_posteriors = (
        f"{PROJECT_ROOT}/output/{prev_fit}/{state}/samples.json"
    )
    previous_fit_median = f"{PROJECT_ROOT}/output/{prev_fit}/medians.csv"
    os.makedirs(save_path, exist_ok=True)
    log.use_logging(
        level="info",
        output="console",
        log_path=os.path.join(save_path, "logs/"),
    )

    if state == "EX":
        state = "GA"

    # Load config and posterior samples based on scenario
    config, posterior_samples = load_config_sample(
        scenario, state, previous_fit_posteriors, previous_fit_median
    )
    logger.info(
        f"Loaded previous fit's posterior samples found in {previous_fit_posteriors}"
    )

    # Data stuff
    ## Observed data
    obs_data = preprocess_observed_data(
        config.region_name, config, model_day=FITTING_DAY
    )
    ### Padding with zeros to account for the sixth strain
    new_var_prop = np.array(obs_data["obs_var_prop"])
    new_var_prop = np.pad(new_var_prop, ((0, 0), (0, 1)))
    obs_data["obs_var_prop"] = jnp.array(new_var_prop)

    ### Calculate weights and merge into obs_data
    ### Weightage here can differ from run_task_5s but they are just a dummy
    scaling_factor_lookup = json.load(
        open("./data/variant-data/scaling_factors.json")
    )
    scaling_factor = scaling_factor_lookup[config.region_name]
    hosp_weights, var_weights = calculate_weights(
        obs_data, var_weight_constant=scaling_factor
    )  # "Regular likelihood" constant for both likelihood
    obs_data["hosp_weights"] = hosp_weights
    obs_data["var_weights"] = var_weights
    ## Vaccination data
    vax_data = process_vaccination_data(config)
    ## Combine data
    data = {
        "obs_data": obs_data,
        "vax_data": vax_data,
    }

    # Generate posterior predictive samples
    t = time.time()
    logger.info(f"Start generating projections based on {scenario} scenario")
    posterior_predictions = posterior_predictive(
        posterior_samples=posterior_samples,
        duration_days=PROJECTION_DAY,
        rng_seed=42,
        data=deepcopy(data),
        config=config,
    )
    combined_dict = {k: v for k, v in posterior_predictions.items()}
    combined_dict = combined_dict | data["obs_data"]
    dict_to_json(
        combined_dict,
        save_path=os.path.join(save_path, "predictive_output.json"),
        mock_chain=False,
    )
    logger.info(f"generating projections took {time.time() - t} seconds")
