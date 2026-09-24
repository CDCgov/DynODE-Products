"""import_data.py contains multiple functions and each imports data. This code will only subdivide, add/remove columns, or change column names to make required inputs for later functions."""

import pandas as pd  # type: ignore[import-untyped]


def import_vaccination_data(start: str, data_folder_path: str) -> pd.DataFrame:
    """Load vaccination data that was retrieved and processed by 'generate_data.py' code into dataframe, and then truncates unnessesary dates.

    Parameters
    ----------
    start : str
        The final date that data will be used.
    data_folder_path : str
        Relative folder path to where the vaccination data should be found.

    Returns
    -------
    vac_raw
        The unprocessed vaccination data with only dates viable for the current run.
    """
    # Vaccination data from https://data.cdc.gov/Vaccinations/COVID-19-Vaccination-Age-and-Sex-Trends-in-the-Uni/5i5k-6cmh
    # Next two lines loads the 'vaccination_data.csv' file from the desired folder path.
    filename = data_folder_path + "vaccination_data.csv"
    vac_raw = pd.read_csv(filename)
    # Next three lines removes data from dates newer than current run desire and orders dataframe ascending by dates.
    vac_raw = vac_raw.sort_values(by=["date"], ascending=True)
    vac_raw = vac_raw.reset_index(drop=True)
    vac_raw = vac_raw[vac_raw.date <= start]
    ###### Removing error cases when data has more vaccines than people. Changing the data in this case.
    overflow_index = vac_raw.total > vac_raw.census
    vac_raw.loc[overflow_index, ["total", "percentage"]] = (
        vac_raw.loc[overflow_index, ["census"]].assign(percentage=1.0).values
    )

    return vac_raw


def import_population_data(
    state: pd.Series, pop_folder_path: str, age_groups: list[int]
) -> pd.DataFrame:
    """Convert raw, ungrouped population data and combine into desired age groups.

    Parameters
    ----------
    state : pd.Series
        Series that contains the state name, state abbreviation, and state usps code.
    pop_folder_path : str
        Relative path where the raw, ungrouped population data is stored.
    age_groups : list[str]
        List containing the integers that represent the age groups.
        For example, [17, 49, 64] would be the age groups [0,17], [18,49], [50,64], and [65, oldest_age).

    Returns
    -------
    pop: pd.DataFrame
        Dataframe that contains the number of people in each age group and what proportion of the total population that age group contains.
    """
    # Load csv containing population data into dataframe.
    # Us has different filenaming process than each state, so if-else function on process.
    if state.stusps == "US":
        # Creates the correct filename in order to call state population data.
        filename = (
            pop_folder_path
            + "United_States_country_level_age_distribution_85.csv"
        )
        # Load US csv population data into dataframe.
        age_distrbution = pd.read_csv(filename)
    else:
        # Creates the correct filename in order to call state population data.
        filename = (
            pop_folder_path
            + "United_States_subnational_"
            + state.stname
            + "_age_distribution_85.csv"
        )
        filename = filename.replace(" ", "_")
        # Load state csv population data into dataframe.
        age_distrbution = pd.read_csv(filename)
    # Labeling columns representing age and total population of given age.
    age_distrbution.columns = ["age", "census"]
    # Generates list of strings that are the name of each age group based on the age_groups integer list.
    for i in range(len(age_groups) + 1):
        # The first number in the list of integers represents the first age group, so the number 17 becomes "0-17".
        if i == 0:
            age_group_list = ["0-" + str(age_groups[0])]
        # The final number in the list of integers represents the minimum age of the oldest age group, so the number 65 becomes "65+".
        elif i == (len(age_groups)):
            age_group_list.append(str(age_groups[i - 1] + 1) + "+")
        # All other numbers will use the current number as the upper bound the previous number + 1 as the lower bound.
        else:
            age_group_list.append(
                str(age_groups[i - 1] + 1) + "-" + str(age_groups[i])
            )
    # Creating dataframe that will store total population that each age group contains.
    pop = pd.DataFrame({"age": age_group_list, "proportion": 0.0, "census": 0})
    # For-loop that takes total population of any given age and combining them into desired age groups.
    for i in range(len(pop)):
        # For final age group, we sum all remaining ages and subtract people younger than youngest person in age group.
        if i == len(pop) - 1:
            pop.loc[i, "census"] = sum(age_distrbution.census) - sum(
                pop.census
            )
        # Sums all ages younger than or equal to largest age in group and subtracts total from age groups below.
        else:
            pop.loc[i, "census"] = sum(
                age_distrbution[age_distrbution.age <= age_groups[i]].census
            ) - sum(pop.census)
    # Calculating proportion of population each age group contains.
    pop.proportion = pop.census / sum(pop.census)
    # Resets index to make all ages accounted for.
    pop = pop.reset_index(drop=True)
    return pop


def import_variant_data(
    start: str, data_folder_path: str, state: pd.Series
) -> pd.DataFrame:
    """Take variant data for all hhs regions, takes only from region the current state is in, and truncates to only include dates the current run includes.

    Parameters
    ----------
    start : str
        The final date that data will be used.
    data_folder_path : str
        Relative folder path to where the vaccination data should be found.
    state : pd.Series
        Series that contains the state name, state abbreviation, and state usps code.

    Returns
    -------
    variant: pd.DataFrame
        Dataframe including proportion of all infections that fall into desired variants for dates within desired time range.
    """
    # Variant data from https://data.cdc.gov/Laboratory-Surveillance/SARS-CoV-2-Variant-Proportions/jr58-6ysp/data_preview
    # Three lines of code below will load dataframe that contains which hhs region each state falls under and applying to the current state.
    filename = data_folder_path + "fips_to_name_improved.csv"
    region_id = pd.read_csv(filename)
    state_region = region_id[region_id.stname == state.stname].region
    # Two lines of code will load dataframce containing varient proportion of every hhs region population.
    filename = data_folder_path + "variant.csv"
    variant = pd.read_csv(filename)
    # The final lines of code removes all but data from desired hhs region and desired dates.
    variant = variant[variant.region == state_region.values[0]]
    variant = variant.sort_values(by=["date"], ascending=True)
    variant = variant.reset_index(drop=True)
    variant = variant[variant.date <= start]
    return variant


def import_hospitilization_data(
    start: str, data_folder_path: str, state: pd.Series
) -> pd.DataFrame:
    """Convert raw hospitalization dataframes by changing column names, only including desired dates, and change order of rows.

    Parameters
    ----------
    start : str
        The final date that data will be used.
    data_folder_path : str
        Relative folder path to where the vaccination data should be found.
    state : pd.Series
        Series that contains the state name, state abbreviation, and state usps code.

    Returns
    -------
    hospital: pd.DataFrame
        Dataframe that contains weekly hospitalization data for desired state.
    """
    # Hospitalization data from https://covid.cdc.gov/covid-data-tracker/#trends_weeklyhospitaladmissions_select_00
    # Three lines below load weekly hospitalization data for given state.
    filename = (
        data_folder_path + "weekly_hospital_" + state.stname.lower() + ".csv"
    )
    filename = filename.replace(" ", "_")
    hospital = pd.read_csv(filename)
    # Rename columns of weekly hospitalization data.
    hospital.columns = ["date", "state", "hospital_weekly"]
    # Removing "NA" or "Not Available" data entirely since can't make any assesment about information of those dates.
    hospital = hospital.dropna()
    # Reorder data to start at earliest date and end with most recent date.
    hospital = hospital.sort_values(by=["date"], ascending=True)
    hospital = hospital.reset_index(drop=True)

    # Removing all data from dates more recent than desired.
    hospital = hospital[hospital.date <= start]
    return hospital


def import_sero_data(data_folder_path: str, state: pd.Series) -> pd.DataFrame:
    """Take seroprevalence data for all states, removes states not used, and names the columns for easy refence.

    Parameters
    ----------
    start : str
        The final date that data will be used.
    data_folder_path : str
        Relative folder path to where the vaccination data should be found.

    Returns
    -------
    sero: pd.DataFrame
        Dataframe that stores estimated total poportion of total population that have been infected by covid.
    """
    # Sero from https://covid19serohub.nih.gov/
    # sero = pd.read_csv('seroprevolence_average.csv') # National Seroprevolence average with columns: date, sero
    # Data from https://covid.cdc.gov/covid-data-tracker/?ref=dailybrief.net#national-lab
    # Two lines of code loads seroprevelence data for every state.
    filename = data_folder_path + "seroprevalence_50states.csv"
    sero = pd.read_csv(filename)
    # Rename columns for easier reference.
    sero.columns = ["state", "age", "date", "sero"]
    # Remove data except for current state.
    sero = sero[sero.state == state.stusps]
    # Reset index so dataframe starts at 0 and increases normally
    sero = sero.reset_index(drop=True)
    sero.columns = ["state", "age", "date", "sero"]
    return sero
