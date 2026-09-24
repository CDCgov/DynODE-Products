"""run.py is a file that will create a vaccine history and infection history for the populations of any states or national. The output will be in the form of a csv and the input is from initialization_config.json."""

import argparse
import inspect
import json
import os
from shutil import copytree

import pandas as pd  # type: ignore[import-untyped]
from bin_tslie import bin_tslie
from formatting_data import format_vac_data, output_generation
from import_data import (
    import_hospitilization_data,
    import_population_data,
    import_sero_data,
    import_vaccination_data,
    import_variant_data,
)
from process_data import (
    cases_by_variant,
    individual_infection_totals,
    total_cases_using_hospital_and_sero,
    variant_tracking,
)
from tslie import get_tslie
from validate_processed_data import validate_data_processing
from validate_raw_data import validate_raw_data
from validate_tslie import validate_tslie


def generating_tslie(
    doses: int,
    start: str,
    age_groups: list[int],
    variant_types: list[str],
    data_folder_path: str,
    pop_folder_path: str,
    state: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    """Take given state and parameters from initialization_config.json to generate time since last immunization event distribution for population.

    Parameters
    ----------
    doses : int
        From 0 to 3 which accounts for how many doses of vaccination should be tracked by an individual.
    start : str
        Should be a date in format "2022-02-12" which will be the first date in which data will be tracked.
    age_groups : list[int]
        The age groups that the data and outputs will track and output results for.
        Formated like [17, 49, 64] in which are ages 0-17 year old, 18-49 year old, 50-64 year old, and 65+ year old.
    variant_types : list[str]
        The specific strains that infections that will be tracked. All strains not included will be categorized as "Other".
        Formatted ["OMICRON"].
    data_folder_path : str
        This is the relative folder location of the data, which should be defaulted as the output of generated_data.py file.
        Formatted like "covid/data/".
    pop_folder_path : str
        This is the relative folder location of the demographic data, which is currently stored in the folder "demographic-data" in the github repo.
        Formatted like "demographic-data/".
    state : pd.Series
        A single row of a dataframe containing the state name, state usps code, and state abbreviation for the given state.

    Returns
    -------
    output: pd.DataFrame
        DataFrame that contains proportion of population of a given age group that has gotten vaccinated a certain number of times
        and which variant they have been infected by, and also how recent they recieved their most recent infection/vaccination.
    list_of_errors: list[str]
        List of strings explaining all failed tests.
    """
    # Loads unprocessed vaccination data for all dates and age groups.
    vac_raw = import_vaccination_data(start, data_folder_path)
    # Loads census data of each age group.
    pop = import_population_data(state, pop_folder_path, age_groups)
    # Loads data of proportion of each infection is of different variants.
    variant_raw = import_variant_data(start, data_folder_path, state)
    # Loads weekly hospital data not age stratified.
    hospital = import_hospitilization_data(start, data_folder_path, state)
    # Loads national seroprevelence not age stratified.
    sero = import_sero_data(data_folder_path, state)
    # Validate that raw data is valid and won't cause code to crash in future.
    raw_data_errors = validate_raw_data(
        vac_raw, hospital, sero, variant_raw, pop
    )
    # If raw data fails checks for a given state or national, no analysis is done.
    if raw_data_errors == []:
        # Loads vaccination for all dates for all age groups, and vaccination rate for 'start' date.
        vac_history, vac = format_vac_data(vac_raw, start, doses)
        # Process raw variant data to include only variants specifically wished to be tracked.
        variant, variant_types_processed = variant_tracking(
            variant_raw, variant_types
        )
        # Estimates total cases using sero for total case estimates and hospitilization for distriubtion of cases.
        cases = total_cases_using_hospital_and_sero(hospital, sero, pop)
        # Unit tests and validation that formatting data did not cause errors in code.
        processed_data_errors = validate_data_processing(
            vac_history,
            vac,
            variant,
            variant_types_processed,
            cases,
            pop,
            doses,
        )
        # End code and report errors if errors exist.
        if len(processed_data_errors) == 0:
            # Estimates total cases by variant per week
            infect = cases_by_variant(
                cases, variant, pop, variant_types_processed
            )
            # Estimates number of people by age group with no infections and all variants at time 'start'.
            infections = individual_infection_totals(
                infect, pop, variant_types_processed
            )
            # Estimates proportion of population for each age group that are within categories seperated by on doses and infections
            output = output_generation(infections, vac, doses)
            # Generates desired initialization with proportion of population in certain bins seperated by total vaccinations, infection strains, and time since last immunization event.
            new_output = get_tslie(
                infect, vac_history, output, pop, variant_types_processed
            )
            # Confirm final output is logically consistent.
            list_of_errors = validate_tslie(
                new_output, pop, variant_types_processed
            )
        # If we found an error from processed data, then we save the errors and
        else:
            # List of errors after running initialization.
            list_of_errors = processed_data_errors.copy()
            # Blank dataframe for new_output.
            new_output = pd.DataFrame()
    # If the checks fail, we will send the error message as the output instead of a dataframe.
    else:
        # Renaming errors for proper output.
        list_of_errors = raw_data_errors.copy()
        # Blank dataframe for new_output.
        new_output = pd.DataFrame()

    return new_output, list_of_errors


def save_errors(errors: list[str], error_storage: str, state: str) -> None:
    """Save list of errors that caused a code to fail.

    Parameters
    ----------
    errors : pd.DataFrame
        Dataframe that stores list of errors that caused a failed run.
    error_storage : str
        Relative folder path in which list of errors for run will be stored.
    state : str
        Name of the state that the error data is generated with respect to.

    Returns
    -------
    Nothing is returned, but a csv will be saved into the error_storage file path.
    """
    # Generates the folder that the filename in the previous lines indicates.
    os.makedirs(os.path.dirname(error_storage), exist_ok=True)
    # Creates a name for the csv that will store the data from the 'output' dataframe.
    filename = error_storage + state + "_error.csv"
    filename = filename.replace(" ", "_")
    # Saves the 'output' dataframe to the folder noted by 'folder_path' and named as 'initialization.csv'
    pd.Series(errors).to_csv(filename, index=False)


def run(
    doses: int,
    start: str,
    state_sample: list[str],
    age_groups: list[int],
    variant_types: list[str],
    data_folder_path: str,
    pop_folder_path: str,
    output_folder_path: str,
    save_all_data: bool,
    tslie_bins: list[int],
    variant_bin_total: int,
    vac_bin_total: int,
    forced_population_total: int,
) -> None:
    """Take inputs from "intialization_config.json and generates distribution of infections/vaccinations by age group and variant for each state as csv.

    Parameters
    ----------
    doses : int
        From 0 to 3 which accounts for how many doses of vaccination should be tracked by an individual.
    start : str
        Should be a date in format "2022-02-12" which will be the first date in which data will be tracked.
    state_sample : list[str]
        Using USPS code for each state that is desired to run, or US for national.
        Ex: ["AL", "AK", "CA"]. Also, can use ["All"] to get all 50 states and National initialization.
    age_groups : list[int]
        The age groups that the data and outputs will track and output results for.
        Formated like [17, 49, 64] in which are ages 0-17 year old, 18-49 year old, 50-64 year old, and 65+ year old.
    variant_types : list[str]
        The specific strains that infections that will be tracked. All strains not included will be categorized as "Other".
        Formatted ["OMICRON"].
    data_folder_path : str
        This is the relative folder location of the data, which should be defaulted as the output of generated_data.py file.
        Formatted like "covid/data/".
    pop_folder_path : str
        This is the relative folder location of the demographic data, which is currently stored in the folder "demographic-data" in the github repo.
        Formatted like "demographic-data/".
    output_folder_path : str
        This is the relative folder location to the deisired folder where the output will be saved to. This is currently in a generic folder named "storage".
        Formatted like "covid/storage/".
    save_all_data: bool
        Whether user wants to save all data used to generate outputs, or only the output themself. The data saved will take up quite a bit of space, but will be allow perfect recreation.
        So, true is reasonable default and false is important if storage space is short.
    tslie_bins: list[int]
        Taking dates with people from each group and putting them into bins based on most recent immunization event.
        Formatted [12, 22, 45, 100] which represents [0-12], [13-22], [23-45], [46-100], and [100, infinity) time ranges.
        Include [] as input if you do not want the binned output.
    variant_bin_total: int
        When generating bins this is going to be the number of variants in initialization that user wants.
        If this is greater than variant_types then it will be zero for each greater than available data.
    vac_bin_total: int
        When generating bins this is going to be the number of vaccination in initialization that user wants.
        If this is greater than variant_types then it will be zero for each greater than available data.
    forced_population_total: int
        User sets all states/national population to exactly this number.
        If user inputs zero, then the output will be with their actual population.

    Returns
    -------
    This code will generate a folder named run_"current date" into the "output_folder_path". The folder will be populated with csv files.
    Each csv will have a proportion of individuals of each age group that has been infected/vaccinated by given variant within each "tslie_range" for each state.
    """
    # Creates a folder if one doesn't exist to store the ouput of run.
    os.makedirs(os.path.dirname(output_folder_path), exist_ok=True)
    # Takes folder path and adds what an addition folder distinguishing from previous runs with a timestamp.
    folder_path = output_folder_path
    # This generates the folder that the folder path indicates from the previous line of code.
    os.makedirs(os.path.dirname(folder_path), exist_ok=True)
    # Filename for another folder within 'folder_path' named 'output' which will store the 'output' dataframe.
    output_storage = folder_path
    # Filename for another folder within 'folder_path' named 'FAILURE' which will store the list of errors.
    error_storage = folder_path + "FAILURE/"
    # The two lines below are load the complete list of state names.
    filename = data_folder_path + "fips_to_name_improved.csv"
    all_states = pd.read_csv(filename)
    if "EX" in state_sample:
        all_states = all_states[all_states.st == "13"].reset_index(drop=True)
    # If "All" is inputted for the desired states, then the code runs every state and national.
    elif "All" in state_sample:
        state_sample = all_states.copy()
    # If "All" is not inputted, then the code runs the given state and national.
    else:
        # The below three lines of code removes any inputted states that are not actually states or no data is available.
        state_sample = list(set(state_sample) & set(all_states.stusps))
        all_states = all_states[all_states.stusps.isin(state_sample)]
        all_states = all_states.reset_index(drop=True)
    # For each state, the distribution of time since last immunization event will be generated.
    for i in range(len(state_sample)):
        state = all_states.loc[i, :]
        # North Dakota seroprevelance data not avaliable, so use its hhs region instead.
        if "ND" in state.stusps:
            state.stusps = "R8"
        # Take the inputs from "initialization_config.json" for given state to generate tslie.
        output, errors = generating_tslie(
            doses,
            start,
            age_groups,
            variant_types,
            data_folder_path,
            pop_folder_path,
            state,
        )
        # If no errors were recorded then save output like usual.
        if len(errors) == 0:
            # If user wants to set the population of each state to constant, then it will be done here.
            if forced_population_total >= 0:
                # Scales each date's total such that the sum totals to user requested total.
                output["total"] = (
                    output["total"]
                    / sum(output["total"])
                    * forced_population_total
                )
            # If the user wants the tslie to be binned into defined bins, the code will generate them.
            if len(tslie_bins) > 0:
                # The binning cannot be done if the user input for variant types are less than total vartiant bins requested.
                # So, we will make them match in that case.
                if variant_bin_total < len(variant_types):
                    # Inform the user with a print statement that their input was adjusted.
                    print(
                        "Variant bin total less than total variants requested. Adjusted user input to match."
                    )
                    # Adjust the variant_bin total to match number of variants.
                    variant_bin_total = len(variant_types)
                binned_output = bin_tslie(
                    output, tslie_bins, variant_bin_total, vac_bin_total
                )
                filename = (
                    output_storage
                    + state.stname
                    + "_binned_initalization.json"
                )
                if state.st == "13":
                    filename = (
                        output_storage + "Example_binned_initalization.json"
                    )
                filename = filename.replace(" ", "_")
                with open(filename, "w") as f:
                    json.dump({"S": binned_output.tolist()}, f)
        # If errors were recorded then only the list of errors will be saved since outputs are invalid.
        else:
            # Function that saves list of errors to desired folder path.
            save_errors(errors, error_storage, state.stname)
    # Two lines below saves data used in raw form and the code that pulls the raw data.
    # This data is fairly large, so its a toggle for user to include or not.
    if save_all_data:
        data_storage = folder_path + "data/"
        copytree(data_folder_path, data_storage)

    return output


def read_and_validate_config(config_path) -> tuple[dict, list[str]]:
    """Load desired inputs for 'run' function from a json file and checks that it contains all necessary arguements for the function.

    Parameters
    ----------
    config_path : str
        Relative path to a json file that contains the inputs for the 'run' function.

    Returns
    -------
    j: dict
        A dictionary containing all the information from the json file that contains the inputs for the 'run' function.
    necessary_args: list[str]
    """
    j = json.load(open(config_path))
    necessary_args = list(inspect.signature(run).parameters.keys())
    for arg in necessary_args:
        assert arg in j.keys(), (
            "Necessary parameter not found in config file %s" % arg
        )

    return j, necessary_args


# Calls the 'run' function using arguements from a file, typically "covid/initialization_config.json", and verifies that file contains all necessary arguements.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to the config file",
        default="initializers/initialization_generation/initialization_config.json",
    )
    args = parser.parse_args()
    config = args.config
    j, necessary_args = read_and_validate_config(config)
    arguements = {k: j[k] for k in necessary_args}
    run(**arguements)
