import glob
import json
import os
from typing import Any, Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dynode import SimulationConfig
from jax import Array
from pydantic import BaseModel


def vectorize_objects(
    objs: List[Any],
    targets: str | List[str],
    filter: Callable[[Any], bool] = lambda _: True,
) -> List[Any] | tuple[List[Any]]:
    """Given a list of objects, return a list of the target attributes for each attribute, optionally applying a filter.

    Parameters
    ----------
    objs : List[obj]
        list of objects with the same attributes
    target : str | list[str]
        attribute name to pull from each object in objs.
    filter : Callable[[obj], bool], optional
        function to filter the objects in objs, by default lambda _: True

    Returns
    -------
    list[Any] | tuple[list[Any]]
        if supplied a str, returned the target attribute from each
        obj in objs. If supplied a list of targets, returns a tuple of such lists.
    """
    if isinstance(targets, str):
        return [getattr(obj, targets) for obj in objs if filter(obj)]
    return tuple(
        [
            [getattr(obj, target) for obj in objs if filter(obj)]
            for target in targets
        ]
    )


def strain_interaction_to_cross_immunity(
    num_strains: int, strain_interactions: Array
) -> Array:
    """Because we are overriding the definitions of immune history, the model must
    contain its own method for generating the cross immunity matrix, not relying on
    utils.strain_interaction_to_cross_immunity() which has complex logic
    to convert the strain interactions matrix to a cross immunity matrix

    Parameters
    ----------
    num_strains : int
        number of strains in the model
    strain_interactions : jax.Array
        the strain interactions matrix

    Returns
    -------
    jax.Array
        the cross immunity matrix which is similar to the strain
        matrix but with an added 0 layer for no previous exposure
    """
    # Construct a matching zero column for the "no previous exposure" slot.
    zero_col = jnp.zeros((num_strains, 1))
    cim = jnp.concatenate((zero_col, strain_interactions), axis=1)

    return cim


def replace_simulation_date(obj: Any):
    """Recursively searches `obj` and replaces any simulation date
    placeholders with integer values.

    Parameters
    ----------
    obj : Any
        object that may or may not contain date placeholders
    """
    if isinstance(obj, dict):
        # Create dictionary representation if not and recursively sample fields.
        obj_dict = dict(obj)
        for key, value in obj_dict.items():
            obj_dict[key] = replace_simulation_date(value)
        return dict(obj_dict)

    elif isinstance(obj, BaseModel):
        for key, value in dict(obj).items():
            # Recursively seach for date fields in BaseModel.
            obj.__setattr__(key, replace_simulation_date(value))

    elif isinstance(obj, (np.ndarray, list)):
        # Recursively sample elements in list.
        lst = [
            replace_simulation_date(
                item,
            )
            for item in obj
        ]
        return lst

    else:
        # Return value as-is if not recognized type.
        # Note that Simulation objects are no longer used in the new Dynode version
        return obj


def dict_to_json(d, save_path, mock_chain=False):
    """Convert dictionary of jax.Array to dictionary of list and dump to json."""
    listified_dict = {}
    for k, v in d.items():
        if not isinstance(v, jax.Array) or v is None:
            listified_dict[k] = v
        else:
            if mock_chain:
                listified_dict[k] = [v.tolist()]
            else:
                listified_dict[k] = v.tolist()
    json.dump(listified_dict, open(save_path, "w"), indent=None)


def get_fitting_day(last_obs_date_str, first_day_date_str="2022-02-11"):
    """Calculate number of days to fit to the data"""
    first_day = pd.to_datetime(first_day_date_str)
    last_obs_date = pd.to_datetime(last_obs_date_str)
    return (last_obs_date - first_day).days


def calculate_scaling_factor(state_name, config: SimulationConfig):
    """
    A function setup to calculate total number of hospitalizations for the US
    and a scaling factor for a specific state.
    """
    total_hosp_us = 0.0
    total_hosp_state = 0.0

    # Pattern to find all hospitalization files
    pattern = os.path.join(config.hosp_path, "*_hospitalization.csv")
    files = glob.glob(pattern)
    for file in files:
        # Extract state name from the filename
        state = (
            os.path.basename(file)
            .replace("_hospitalization.csv", "")
            .replace("_", " ")
        )
        # Load hospitalization data for the current state
        hosp_data = pd.read_csv(file)
        hosp_data["date"] = pd.to_datetime(hosp_data["date"])

        # Special handling for Maine and NH
        if state_name == "Maine":
            hosp_data = hosp_data[
                hosp_data["date"] != pd.to_datetime("2023-09-30")
            ]
        if state_name == "New Hampshire":
            hosp_data = hosp_data[
                hosp_data["date"] != pd.to_datetime("2023-09-16")
            ]

        # Sum hospitalizations for the state
        state_total = hosp_data["hosp"].sum(skipna=True)
        total_hosp_us += state_total

        # If this is the specified state, store the state total
        if state == state_name:
            total_hosp_state = state_total

    # Calculate scaling factor
    if total_hosp_us == 0:  # Avoid division by zero
        raise ValueError(
            "Total US hospitalizations are zero, cannot calculate scaling factor."
        )

    scaling_factor = (total_hosp_state * 5) / (2 * total_hosp_us)
    return scaling_factor
