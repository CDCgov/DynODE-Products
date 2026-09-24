"""A module for loading, processing, and saving datasets."""

import datetime
import glob
import json
import os
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore
from jax import Array


def get_region_name_and_pop_size(region_code, demographic_data_path):
    """Read in a population_by_region.csv file and return the region name and
    population size."""
    population_by_region = pd.read_csv(
        os.path.join(
            demographic_data_path,
            "population_by_region.csv",
        ),
    )
    region = population_by_region[
        population_by_region["stusps"] == region_code
    ]
    if len(region) == 1:
        population_size: float = region["population"].iloc[0]
        region_name: str = region["stname"].iloc[0]
        return region_name, population_size
    else:
        raise ValueError(
            f"Region code {region_code} not found in population_by_region.csv."
        )


def generate_yearly_age_bins_from_limits(age_limits: list) -> list[list[int]]:
    """Generate age bins up to 85 years old exclusive based on age limits.

    Parameters
    ----------
    age_limits : list[int]
        Boundaries of each age bin. The last bin is implicitly up to 85.

    Returns
    -------
    list[list[int]]
        List of lists containing integer years within each age bin.

    Examples
    --------
    >>> generate_yearly_age_bins_from_limits([0, 5, 10, ... 80])
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9],... [80, 81, 82, 83, 84]]
    """
    age_groups = []
    for age_idx in range(1, len(age_limits)):
        age_groups.append(
            list(range(age_limits[age_idx - 1], age_limits[age_idx]))
        )
    age_groups.append(list(range(age_limits[-1], 85)))
    return age_groups


def load_age_demographics(
    path: str,
    regions: list[str],
    age_limits: list[int],
) -> dict[str, np.ndarray]:
    """Load normalized proportions of each age bin for given regions.

    Parameters
    ----------
    path : str
        Path to the demographic-data folder.
    regions : list(str)
        List of FIPS regions
    age_limits : list(int)
        Age limits for each bin; values are exclusive upper bounds.
        Max tracked age is enforced at 84 inclusive. All
        populations older than 84 are counted as 84 years old.

    Returns
    -------
    demographic_data : dict[str, np.ndarray]
        A dictionary maping FIPS region supplied in `regions`
        to an array of length `len(age_limits)` representing
        the __relative__ population proportion of each bin, summing to 1.
    """
    assert os.path.exists(
        path
    ), "The path to population-rescaled age distributions does not exist as it should"

    demographic_data = {}
    # Create contact matrices
    for r in regions:
        try:
            # e.g., if territory is "North Carolina", pass it as "North_Carolina"
            if len(r.split()) > 1:
                region = "_".join(r.split())
            else:
                region = r
            if region != "United_States":
                region_data_file = (
                    "United_States_subnational_"
                    + region
                    + "_age_distribution_85.csv"
                )
            else:
                region_data_file = (
                    "United_States_country_level_age_distribution_85.csv"
                )

            age_distributions = np.loadtxt(
                path + region_data_file,
                delimiter=",",
                dtype=np.float64,
                skiprows=0,
            )
            binned_ages = np.array([])
            age_bin_pop = 0
            current_age = age_limits[0]
            if 84 not in age_limits:
                age_limits = age_limits + [84]
            age_limits = age_limits[1:]
            while current_age < 85:
                # get the population for the current age
                age_bin_pop += age_distributions[current_age][1]
                # add total population of that bin to the array, reset
                if current_age in age_limits:
                    binned_ages = np.append(binned_ages, age_bin_pop)
                    age_bin_pop = 0
                current_age += 1  # go to next year.
            # normalize array to proportions after all bins constructed.
            binned_ages = binned_ages / sum(binned_ages)
            demographic_data[r] = binned_ages
        except Exception as e:
            print(
                f"Something went wrong with {r} and produced the following error:\n\t{e}"
            )
    return demographic_data


def make_two_settings_matrices(
    path_to_population_data: str,
    path_to_settings_data: str,
    region: str = "United States",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and parse settings contact matricies for a given region.

    For a single region, read the two column (age, population counts) population
    csv (up to age 85) then read the 85 column interaction settings csvs by
    setting (four files) and combine them into an aggregate 85 x 85 matrix
    (twice, one for school setting and another for "other")

    Parameters
    ----------
    path_to_population_data : str
        The path to the folder that has the population size for each age
    path_to_settings_data : str
        The path to age-by-age contacts by settings
    region : str
        The FIPS region to create the 2 settings matrices

    Returns
    -------
    tuple
        A tuple containing an 85x85 school settings contact matrix, an 85x85
        other (household + community + work) contact matrix, and the population
        in each of the 85 ages groupings
    """
    # Define the settings in the interactions by settings data
    interaction_settings = ["school", "household", "community", "work"]
    # Get all the interactions by settings data
    all_settings_data_files = glob.glob(f"{path_to_settings_data}/*.csv")
    # Index where the location of the setting is after splitting by "_"
    # NB: A file name looks like United_States_country_level_F_school_setting_85.csv
    setting_index = -3
    # Collect and unpack all the settings data files, as well as the age
    # distribution, by the region inputted
    if region != "United_States":
        region_data_file = (
            "United_States_subnational_" + region + "_age_distribution_85.csv"
        )
        settings_data_dict = dict(
            [
                (
                    elt.split("_")[setting_index],
                    np.loadtxt(
                        elt, delimiter=",", dtype=np.float64, skiprows=0
                    ),
                )
                for elt in all_settings_data_files
                if region in elt
            ]
        )
    else:
        region_data_file = (
            "United_States_country_level_age_distribution_85.csv"
        )
        settings_data_dict = dict(
            [
                (
                    elt.split("_")[setting_index],
                    np.loadtxt(
                        elt, delimiter=",", dtype=np.float64, skiprows=0
                    ),
                )
                for elt in all_settings_data_files
                if ("country" in elt) and ("_F_" in elt)
            ]
        )
    # Make sure region_data_file exists
    assert os.path.exists(
        path_to_population_data + "/" + region_data_file
    ), f"The {region} file does not exist"
    # Load territory data
    region_data = pd.read_csv(
        path_to_population_data + "/" + region_data_file,
        header=None,
        names=["Age", "Population"],
    )
    # Create the empty base School and Other contact matrices
    sch_CM = np.zeros((region_data.shape[0], region_data.shape[0]))
    oth_CM = np.zeros((region_data.shape[0], region_data.shape[0]))
    # Iterate through the interaction settings, assembling the School and Other
    # contact matrices
    for setting in interaction_settings:
        if setting == "school":
            sch_CM = settings_data_dict[setting]
        # else:
        oth_CM += settings_data_dict[setting]
    return (sch_CM, oth_CM, region_data)


def create_age_grouped_CM(
    region_data: pd.DataFrame,
    setting_CM: np.ndarray,
    num_age_groups: int,
    minimum_age: int,
    age_limits,
) -> tuple[np.ndarray, list[float]]:
    """Load a contact matrix and group it into age bins.

    Parameters
    ----------
    region_data : pd.DataFrame
        A two column dataframe with the FIPS region's 85 ages and their
        population sizes
    setting_CM : np.ndarray
        An 85x85 contact matrix for a given setting (either school or other)
    num_age_groups : int
        number of age bins.
    minimum_age : int
        lowest possible tracked age in years.
    age_limits : list[int]
        Age limit for each age bin in the model, beginning with minimum age,
        values are exclusive in upper bound. so [0,18] means 0-17, 18+.

    Returns
    -------
    tuple
        A tuple containing an age-grouped
        (num_age_group x num_age_group) contact matrix and a list with
        the proportion of that FIPS region's population in each of those
        num_age_group age groups
    """
    # Check if the received setting (all ages) matrix is square
    assert (
        setting_CM.shape[0] == setting_CM.shape[1]
    ), "Baseline contact matrix is not square."
    # Check if the dc.MINIMUM_AGE is proper
    assert 0 <= minimum_age < 84, "Please correct the value of the minimum age"
    # Check if the dc.MINIMUM_AGE is an int
    assert isinstance(
        minimum_age, int
    ), "Please make sure the minimum age is an int"
    # Check to see if the age limits specified are ordered properly
    assert age_limits[-1] < 84, "The entered age limits are not compatible"
    # Check if the upper bound of the age limits is greater than the lower bound
    assert (
        age_limits[0] < age_limits[1] + 1
    ), "The bounds for the age limits are not proper"
    # Create new age groups from the age limits, e.g. if [18,66], <18,18-64,65+
    age_groups = generate_yearly_age_bins_from_limits(age_limits)
    grouped_CM = np.empty(
        (num_age_groups, num_age_groups), dtype=setting_CM.dtype
    )
    # Get the population data to be used for proportions in the
    pop_proportions = region_data["Population"].div(
        region_data["Population"].sum()
    )
    # Fill in the age-grouped contact matrix
    for i, grp_out in enumerate(age_groups):
        for j, grp_in in enumerate(age_groups):
            cm_slice = setting_CM[np.ix_(grp_out, grp_in)]
            pop_prop_slice = pop_proportions[pd.Index(grp_out)] / np.sum(
                pop_proportions[pd.Index(grp_out)]
            )
            pop_prop_slice = np.reshape(pop_prop_slice.to_numpy(), (-1, 1))
            grouped_CM[i, j] = np.sum(pop_prop_slice * cm_slice)
    # Population proportions in each age group
    N_age = [np.sum(pop_proportions[pd.Index(group)]) for group in age_groups]
    return (grouped_CM, N_age)


def load_demographic_data(
    demographics_path,
    regions,
    num_age_groups,
    minimum_age,
    age_limits,
) -> dict[str, Any]:
    """Load demography data for the specified FIPS regions.

    Contact mixing data sourced often from:
    https://github.com/mobs-lab/mixing-patterns

    Parameters
    ----------
    demographics_path : str
        path to demographic data directory, contains "contact_matrices" and
        "population_rescaled_age_distributions" directories.
    regions : list[str]
        list of FIPS regions to load.
    num_age_groups : int
        number of age bins.
    minimum_age : int
        lowest possible tracked age in years.
    age_limits : list[int]
        Age limit for each age bin in the model, beginning with minimum age,
        values are exclusive in upper bound. so [0,18] means 0-17, 18+.

    Returns
    -------
    demographic_data : dict
        A dictionary of FIPS regions, with 2 age-grouped contact matrices by
        setting (school and other, where other is comprised of work, household,
        and community settings data) and data on the population of the region
        by age group
    """
    # Get the paths to the 3 files we need
    path_to_settings_data = demographics_path + "contact_matrices"
    path_to_population_data = (
        demographics_path + "population_rescaled_age_distributions"
    )
    # Check if the paths to the files exists
    assert os.path.exists(
        demographics_path
    ), f"The base path {demographics_path} does not exist as it should"
    assert os.path.exists(
        path_to_settings_data
    ), "The path to the contact matrices does not exist as it should"
    assert os.path.exists(
        path_to_population_data
    ), "The path to population-rescaled age distributions does not exist as it should"
    # Create an empty dictionary for the demographic data
    demographic_data = dict([(r, "") for r in regions])
    # Create contact matrices
    for r in regions:
        try:
            # e.g., if territory is "North Carolina", pass it as "North_Carolina"
            if len(r.split()) > 1:
                region = "_".join(r.split())
            else:
                region = r
            # Get base school and other contact matrices (for all 85 ages) and
            # the populations of each of these ages
            sch_CM_all, avg_CM_all, region_data = make_two_settings_matrices(
                path_to_population_data,
                path_to_settings_data,
                region,
            )
            # Create the age-grouped school setting contact
            sch_CM, N_age_sch = create_age_grouped_CM(
                region_data,
                sch_CM_all,
                num_age_groups,
                minimum_age,
                age_limits,
            )
            # Create the age-grouped other an average setting contact (average being all settings combined, including school)
            avg_CM, N_age_oth = create_age_grouped_CM(
                region_data,
                avg_CM_all,
                num_age_groups,
                minimum_age,
                age_limits,
            )
            # Save one of the two N_ages (they are the same) in a new N_age var
            N_age = N_age_sch
            # Rescale contact matrices by leading eigenvalue
            avg_CM = avg_CM / np.max(np.real(np.linalg.eigvals(avg_CM)))
            sch_CM = sch_CM / np.max(np.real(np.linalg.eigvals(sch_CM)))
            # Transform Other cm with the new age limits [NB: to transpose?]
            region_demographic_data_dict = {
                "sch_CM": sch_CM.T,
                "avg_CM": avg_CM.T,
                "N_age": np.array(N_age),
                "N": np.array(np.sum(N_age)),
            }
            demographic_data[r] = region_demographic_data_dict
        except Exception as e:
            print(
                f"Something went wrong with {r} and produced the following error:\n\t{e}"
            )
            raise e
    return demographic_data


def load_vaccination_model_splines(
    spline_data_path: str,
    region_name: str,
    num_age_groups: int,
    max_vaccination_count: int,
    age_group_idxs: dict[str, int],
) -> dict[str, Array]:
    """Load parameters of a polynomial spline vaccination model.

    Returns
    -------
    the following are 3 parallel lists, each with leading dimensions
    `(num_age_groups, max_vax_count+1)` identifying the vaccination spline
    from age group I and vaccination count J to vaccination count J+1.
    (indivduals vaccinated while at `max_vax_count` generally stay in
    the same tier, but this is ODE specific).
    vaccination_model_knots: jax.Array
        array of knot coefficients for each knot located on
        `vaccination_model_knot_locations[i][j]`
    vaccination_model_knot_locations: jax.Array
        array of knot locations by model day, with 0 indicating the knot is
        placed on the initial date.
    vaccination_model_base_equations: jax.Array
        array defining the coefficients (a,b,c,d) of each
        base equation `(a + b(t) + c(t)^2 + d(t)^3)` for the spline defined
        by `vaccination_model_knot_locations[i][j]`.

    Raises
    ------
    FileNotFoundError
        if path is not a csv file or directory. Or if directory path does not
        contain region specific file matching expected naming convention.

    Notes
    -----
    Reads spline information from `spline_data_path`,
    if path given is a directory, attempts a region-specific lookup with
    `region_name`, using format
    `spline_data_path/spline_fits_{region_name}.csv`
    """
    # if the user passes a directory instead of a file path
    # check to see if the state exists in the directory and use that
    if os.path.isdir(spline_data_path):
        vax_spline_filename = "spline_fits_%s.csv" % (
            region_name.lower().replace(" ", "_")
        )
        state_path = os.path.join(spline_data_path, vax_spline_filename)
        if os.path.exists(state_path):
            parameters = pd.read_csv(state_path)
        else:
            raise FileNotFoundError(
                "Directory passed to `spline_data_path`"
                "does not contain %s which is the "
                "expected state-specific vax filename" % vax_spline_filename
            )
    # given a specific file to spline fits, use those
    elif os.path.isfile(spline_data_path):
        parameters = pd.read_csv(spline_data_path)
    else:
        raise FileNotFoundError(
            "Path given to `spline_data_path` is something other than a "
            "directory or file path, got %s. Check configuration and provide "
            "a valid directory path or filepath to vaccination splines"
            % spline_data_path
        )
    age_bins = len(parameters["age_group"].unique())
    vax_bins = len(parameters["dose"].unique())
    # change this if you start using higher degree polynomials to fit vax model
    assert age_bins == num_age_groups, (
        "the number of age bins in your model does not match the input vaccination parameters, "
        + "please provide your own vaccination parameters that match, or adjust your age bins"
    )

    assert vax_bins == max_vaccination_count + 1, (
        "the number of vaccination counts in your model does not match the input vaccination parameters, "
        + "please provide your own vaccination parameters that match, or adjust your age bins"
    )
    num_knots = len(
        parameters.iloc[0][
            [col for col in parameters.columns if "location" in col]
        ]
    )
    # store splines as a series of outward flows
    # an age_group x dose combo can identify its lost population via
    # the following 3 matricies in utils.evaluate_cubic_splines()
    vax_knots = np.zeros((age_bins, vax_bins, num_knots))
    vax_knot_locations = np.zeros((age_bins, vax_bins, num_knots))
    # always 4 base terms for cubic splines
    vax_base_equations = np.zeros((age_bins, vax_bins, 4))
    # each row in csv is one flow from vax_x -> vax_y for age group z
    for _, row in parameters.iterrows():
        age_group, vaccination = row["age_group"], row["dose"]
        intersect_and_ts = row[["intersect", "t", "t2", "t3"]].values
        # coef identifies coefficients, location identifies knot locations
        knot_coefficients = row[
            [col for col in parameters.columns if "coef" in col]
        ].values
        knot_locations = row[
            [col for col in parameters.columns if "location" in col]
        ].values
        # check that same number of knots as coefficients
        assert (
            len(knot_coefficients) == len(knot_locations)
        ), "number of knot_coefficients and number of knot locations found do not match"
        age_group_idx = age_group_idxs[age_group]
        # splines `dose` dictate the `to_dose`, but we store them as outward flows
        # thus subtract 1, we dont support skipping doses
        vax_idx = vaccination - 1
        vax_base_equations[age_group_idx, vax_idx, :] = np.array(
            intersect_and_ts
        )
        vax_knots[age_group_idx, vax_idx, :] = np.array(knot_coefficients)
        vax_knot_locations[age_group_idx, vax_idx, :] = np.array(
            knot_locations
        )
    return {
        "vaccination_model_knots": jnp.array(vax_knots),
        "vaccination_model_knot_locations": jnp.array(vax_knot_locations),
        "vaccination_model_base_equations": jnp.array(vax_base_equations),
    }


def load_vaccination_model_hill(
    region_name: str,
    hill_data_path: str,
    num_age_groups: int,
    max_vaccination_count: int,
    init_date: datetime.date,
):
    """Read and format Hill-equation based vaccination parameters from json

    Parameters
    ----------
    region_name: str
        state names, e.g., "Alabama" or "New York"
    hill_data_path: str
        path to the folder containing {region_name}_hillvax_config.json
    num_age_groups: int
        number of age groups, typically 4
    max_vaccination_count: int
        number of vaccination doses we are keeping track on, typically 3
    init_date: datetime.date
        start date of the simuation

    Returns
    -------
    dict:
        consists of 4 keys corresponding to 4 parameters required from hill
        equation. Value to each key is a jnp.array of shape (num_age_groups,
        max_vaccination_count+1)

    Notes
    -----
    Since we only use hill equation to model the seasonal booster, we only
    have parameters for vaccine status going from 2 to 3. All others are
    fixed at zeros.
    """
    vax_hill_filename = "%s_hillvax_config.json" % (
        region_name.lower().replace(" ", "_")
    )
    vax_hill_path = os.path.join(hill_data_path, vax_hill_filename)
    assert os.path.exists(vax_hill_path), (
        "Unable to locate vaccination input data file:  %s" % vax_hill_path
    )
    vax_input_file = json.load(open(vax_hill_path))
    # translate the hill equation age classes to our age indexing definitions
    age_bin_translator = {"0-17": 0, "18-49": 1, "50-64": 2, "65+": 3}
    # hill_equations: list[dict] = vax_input_file["hill_equations"]
    n_batched = jnp.zeros((num_age_groups, max_vaccination_count + 1))
    t_h_batched = jnp.zeros((num_age_groups, max_vaccination_count + 1))
    t_end_batched = jnp.zeros((num_age_groups, max_vaccination_count + 1))
    y_t_end_batched = jnp.zeros((num_age_groups, max_vaccination_count + 1))
    t_shifts = jnp.zeros((num_age_groups))
    scales_batched = jnp.zeros((num_age_groups, max_vaccination_count + 1))
    # build up the matricies of hill parameters reading in the
    # input files info
    # in this case we are only setting the 2-> 3 vax rates
    for age_specific_hill_equation in vax_input_file:
        age_idx = age_bin_translator[age_specific_hill_equation["ageclass"]]
        n_batched = n_batched.at[age_idx, 2].set(
            age_specific_hill_equation["shape"]
        )
        t_h_batched = t_h_batched.at[age_idx, 2].set(
            age_specific_hill_equation["t_halfsat"]
        )
        t_end_batched = t_end_batched.at[age_idx, 2].set(
            age_specific_hill_equation["t_anchor"]
        )
        y_t_end_batched = y_t_end_batched.at[age_idx, 2].set(
            age_specific_hill_equation["frac_vax_anchor"]
        )
        scales_batched = scales_batched.at[age_idx, 2].set(
            age_specific_hill_equation["scale"]
        )
        # get the vaccination equations start date and compare it to the
        # start date of the model to appropriately shift if needed
        t_shift = (
            datetime.datetime.strptime(
                age_specific_hill_equation["start_date"], "%Y-%m-%d"
            ).date()
            - init_date
        ).days

        t_shifts = t_shifts.at[age_idx].set(t_shift)
    return {
        "n_batched": n_batched,
        "t_h_batched": t_h_batched,
        "t_shifts": t_shifts,
        "scales_batched": scales_batched,
    }


def save_samples(samples: dict[str, Array], save_path: str, indent=None):
    """
    Save model samples to `save_path`, with JSON serializability.

    Parameters
    ----------
    samples : dict[str, Array]
        Dictionary with str keys each containing a jax array of samples or
        posteriors, often returned by MCMC.get_samples()
    save_path : str
        path, relative or absolute to save json to
    indent : Optional[int]
        optional indent spaces for pretty-printing of json,
        None is most compact and least readable, by default None
    """

    # convert np arrays to lists
    s = {param: samples[param].tolist() for param in samples.keys()}
    json.dump(s, open(save_path, "w"), indent=indent)
