"""Process_data contains multiple functions that apply some function to edit emperical data, typically by combining multiple data sources together."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def reorder_list(result: pd.DataFrame) -> list[str]:
    """Take variant data and outputs list of variants in order of first date with infections.

    Parameters
    ----------
    result : pd.DataFrame
        Dataframe with variant data over the desired date range.

    Returns
    -------
    ordered: list[str]
        List of tracked variants including "Other" in order of first date of infections.
    """
    # Filter rows where proportion > 0
    nonzero = result[result.proportion > 0]
    # Find earliest index for each variant
    first_idx = (
        nonzero.assign(idx=nonzero.index).groupby("variant")["idx"].min()
    )
    # Variants that never had >0 proportion
    all_variants = set(result.variant)
    missing_variants = all_variants - set(first_idx.index)
    # Order variants by earliest index
    ordered = list(first_idx.sort_values().index)
    # Add variants that never appear >0 at the end
    ordered += sorted(missing_variants)

    return ordered


def variant_tracking(
    variant: pd.DataFrame, variant_types: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Take raw variant data and combine untracked variant data into "Other" category.

    Parameters
    ----------
    variant : pd.DataFrame
        Dataframe with raw variant data over the desired date range.
    variant_types : list[str]
        List user input as requested variants to track.

    Returns
    -------
    result: pd.DataFrame
        Dataframe with variant data that combines all variant proportion not requested to be tracked as "Other".
    unique_variants_tracked: list[str]
        List of tracked variants including "Other" and adjusted in the write formatting.
    """
    # Copy unaltered variant data with intention of adjusting it.
    variant_tracked = variant.copy()
    # Makes user input into all caps to avoid case sensitivity issues.
    unique_variants_tracked = [s.upper() for s in variant_types]
    # If a variant input by user isn't available in data, then we make a warning and add them with zero total infections.
    if not set(unique_variants_tracked).issubset(set(variant_tracked.variant)):
        # List of variants that are missing, but user requested to track them.
        missing = list(
            set(unique_variants_tracked) - set(variant_tracked.variant)
        )
        # All region/date combinations.
        base = variant_tracked[["region", "date"]].drop_duplicates()
        # Repeat each row once per missing variant
        copies = base.loc[base.index.repeat(len(missing))].copy()
        # Making each copy with each missing variant.
        copies["variant"] = missing * len(base)
        # Filling each missing varaint dataframe with 0 for proporiton since there was no data.
        copies["proportion"] = 0.0
        # Concat dataframes of missing variants to the dataframe with data.
        variant_tracked = pd.concat(
            [variant_tracked, copies], ignore_index=True
        )
        # User warning which will be printed out for user to note.
        warning_message = f"WARNING: No data found for variants, {missing}. We assume they have zero infections and continued the code. If this is not the intention, please check spelling of variant input."
        print(warning_message)

    # Categorize every variant that user doesn't want to track as "Other".
    variant_tracked["category"] = np.where(
        variant_tracked.variant.isin(unique_variants_tracked),
        variant_tracked.variant,
        "Other",
    )
    # For every date, sums proporiton of all infections caused by "Other" category.
    result = variant_tracked.groupby(["date", "category"], as_index=False).sum(
        numeric_only=True
    )
    # Add "Other" category to list of tracked variants.
    unique_variants_tracked.append("Other")
    # List of every date with data.
    dates = list(set(result.date))
    # Generating list of all combinations of dates and tracked variants.
    full_index = pd.MultiIndex.from_product(
        [dates, unique_variants_tracked], names=["date", "category"]
    )
    # Next two lines of code adds additional rows to dataframe for when variants have no data, and make them have zero proportion of those dates.
    # Therefore, every date tracked will have row from every variant tracked.
    result = result.set_index(["date", "category"])
    result = result.reindex(full_index, fill_value=0).reset_index()
    # Reorder such dataframe is increasing from earliest date to most recent date
    result = result.sort_values("date").reset_index(drop=True)
    # Rename name of column to variant rather than category.
    result = result.rename(columns={"category": "variant"})
    # The function is going to use results to reorder unique_tracked variants by date of first infection.
    unique_variants_tracked = reorder_list(result)

    return result, unique_variants_tracked


def total_cases_using_hospital_and_sero(
    hospital: pd.DataFrame, sero: pd.DataFrame, pop: pd.DataFrame
) -> pd.DataFrame:
    """Convert cumulative seroprevalence data into weekly cases.

    Parameters
    ----------
    hospital : pd.DataFrame
        Weekly hospitalization totals.
    sero : pd.DataFrame
        Weekly cumulative seroprevalence.
    pop : pd.DataFrame
        Total population seperated by age group.

    Returns
    -------
    cases: pd.DataFrame
        Total cases that occured in a given week generated via seroprevalence data and hospitalization data for distribution of infection days.

    """
    # Generates dataframe that stores date, age group, and weekly hospitalization
    for age_group in range(len(pop)):
        # In the first age group a dataframe is created in order to store the weekly hospitalization and will eventually store estimated weekly cases.
        if age_group == 0:
            # Store weekly hospitalization data for current age group with intention to scale for weekly cases later in code.
            cases = pd.DataFrame(
                {
                    "date": hospital.date,
                    "age": pop.age[0],
                    "cases": hospital.hospital_weekly,
                }
            )
            len_cases = len(cases)
        # After the dataframe is created, this appends the dataframe to store information of the current age group.
        else:
            # Append weekly hospitalization data for current age group with intention to scale for weekly cases later in code.
            cases = pd.concat(
                [
                    cases,
                    pd.DataFrame(
                        {
                            "date": hospital.date,
                            "age": pop.age[age_group],
                            "cases": hospital.hospital_weekly,
                        }
                    ),
                ],
                ignore_index=True,
            )
    # For each age group, identifies how many infections occured during a time frame using sero data and transitions to weekly time scale by using weekly hospitalization.
    for j in range(len(pop)):
        # Making subsection of seroprevalence only including current age group.
        sero_age = sero[sero.age == pop.age[j]]
        sero_age = sero_age.reset_index(drop=True)
        # Estimated population that has been infected using sero prevelence at first time step in data
        lower_infections = sero_age.sero[0] * pop.census[j]
        # Estimated population that has been infected using sero prevelence at second time step in data
        upper_infections = sero_age.sero[1] * pop.census[j]
        # Accounting for seroprevalence data having inconsistency.
        if upper_infections < lower_infections:
            upper_infections = lower_infections
        # Index used in next for-loop to identify which sero data date lines up with hospitalization date.
        lower_index = 0
        # Closest date of sero data less than hosiptalization date
        lower = sero_age.date[lower_index]
        # Closest date of sero data less than hosiptalization date
        upper = sero_age.date[lower_index + 1]
        # Going through each day where we have hospitalization data and scale to match based on closest sero data
        for i in range(len(hospital)):
            # If the hospitalization date is more recent than current sero date, then we increase the index and continue the process.
            if cases.date[i] > upper:
                # If the next index would exceed the final index, then just use the final index.
                if cases.date[i] > max(sero_age.date):
                    # If the date is more recent than data is available then we make the cases the same as previous step, so no new cases.
                    cases.loc[i + j * len_cases, "cases"] = cases.loc[
                        i + j * len_cases - 1, "cases"
                    ]
                # Adjusts the index so that we use later hospitalization data to scale the sero data.
                else:
                    # Adjusts lower_index if current date is below upper_index
                    lower_index = lower_index + 1
                    # Identifying new closest sero date below
                    lower = sero_age.date[lower_index]
                    # Identifying new closest sero date above
                    upper = sero_age.date[lower_index + 1]
                    # Number of total infections scaling sero data for lower sero date.
                    lower_infections = (
                        sero_age.sero[lower_index] * pop.census[j]
                    )
                    # Number of total infections scaling sero data for upper sero date.
                    upper_infections = (
                        sero_age.sero[lower_index + 1] * pop.census[j]
                    )
                    # Preventing inconsistent data from allowing future dates to have less infections.
                    if upper_infections < lower_infections:
                        upper_infections = lower_infections
                    # Next two lines are finding all hospitalization data between the two sero dates
                    a = hospital[hospital.date > lower]
                    a = a[a.date <= upper]
                    # Multiplier to scale the hospitalization data such that it sums to the new infections between two sero dates
                    multiplier = (upper_infections - lower_infections) / max(
                        sum(a.hospital_weekly), 1
                    )
                    # Scaling sero cases to match the hospitalization multiplier.
                    cases.loc[i + j * len_cases, "cases"] *= multiplier
            else:
                # Next two lines are finding all hospitalization data between the two sero dates
                a = hospital[hospital.date > lower]
                a = a[a.date <= upper]
                # Multiplier to scale the hospitalization data such that it sums to the new infections between two sero dates
                multiplier = (upper_infections - lower_infections) / max(
                    sum(a.hospital_weekly), 1
                )
                # Changing cases to float in order to avoid type change error.
                cases["cases"] = cases["cases"].astype(float)
                # Scaling sero cases to match the hospitalization multiplier.
                cases.loc[i + j * len_cases, "cases"] *= multiplier
    return cases


def cases_by_variant(
    cases: pd.DataFrame,
    variant: pd.DataFrame,
    pop: pd.DataFrame,
    variant_types: list[str],
) -> pd.DataFrame:
    """Take estimated weekly cases and distributes the cases among the different variants.

    Parameters
    ----------
    cases : pd.DataFrame
        Contains estimated total cases in a given week.
    variant : pd.DataFrame
        Contains proportion of infections in a given time that belonged to each variant.
    pop : pd.DataFrame
        Contains every age groups total population.
    variant_types : list[str]
        List of variants that are intended to be tracked. 'Other' is needed since it includes all variants not specifically mentioned.

    Returns
    -------
    infect: df.DataFrame
        Dataframe containing estimated weekly infections seperated by age group and variant.
    """
    # All combinations of different strains/variants a person can have among tracked varaints.
    # Identifying location of most recent variant proportion to use in event that variant date is not identical to hospitalization date
    index = 0
    # Number of variants that we keep track of.
    num_variants = len(variant_types)
    # For each age group, identify the proportion cases for each week that are of a given variant.
    for i in range(len(pop)):
        # Two lines below subdivide the weekly case dataframe to only include data of current age group.
        df_cases = cases[cases.age == pop.age[i]]
        df_cases = df_cases.reset_index(drop=True)
        # Creating dataframe that will store cases and which variant the cases were. This will be combined later with every age group.
        df = pd.DataFrame(
            {
                "date": df_cases.date,
                "cases": df_cases.cases,
                "age": pop.age[i],
                "strain": variant_types[0],
            }
        )
        n = len(df)
        # Expands the dataframe to include enough rows to account for each variant tracked.
        if num_variants > 1:
            # Adding rows to included data for other variants
            df = pd.concat(num_variants * [df], ignore_index=True)
            # Adding an additional number of rows representing the for each additional variant.
            for q in range(1, num_variants):
                # Changing the name of variant to currnet strain.
                df.loc[q * n : (q + 1) * n, "strain"] = variant_types[q]
        # For each date with case data, scale cases to proportion of each variant.
        for j in range(n):
            # If variant data occurs before first cases, then we can assume cases for given variant is zero.
            if variant.date[0] > df.date[j]:
                df.loc[j, "cases"] = 0
            else:
                # Subdivide variant data to include only current date.
                variant_date = variant[variant.date == df.date[j]]
                # For each variant scale total cases on current date to current variants proportion of total infections in that time period.
                for cur_variant in range(num_variants):
                    # if variant date and case dates don't match exactly, uses most recent proportion
                    if len(variant_date) == 0:
                        # If new date has no data, then use most recent variant data.
                        variant_date = variant.iloc[
                            index : index + num_variants
                        ]
                    else:
                        # If new date has data, then adjust indext to match current date.
                        index = variant_date.index[0]
                    # For the given date, take only the data of the current variant.
                    single_variant_date = variant_date[
                        variant_date.variant == variant_types[cur_variant]
                    ].reset_index(drop=True)
                    # Scale cases to proportion of current variant.
                    df.loc[j + n * cur_variant, "cases"] *= (
                        single_variant_date.proportion[0]
                    )
        # Adding each age group together.
        if i == 0:
            # For the first variant, we create a new dataframe that will be used to combine all the other age groups.
            infect = df
        else:
            # Combine current variant data with other variants data.
            infect = pd.concat([infect, df], ignore_index=True)
    return infect


def individual_infection_totals(
    infect: pd.DataFrame, pop: pd.DataFrame, variant_types: list[str]
) -> pd.DataFrame:
    """Take weekly infections by age group and variant, and identifying total infections of all variants and combinations of variants.

    Parameters
    ----------
    infect : pd.DataFrame
        Weekly infections seperated by age group and variant.
    pop : pd.DataFrame
        Contains every age groups total population.
    variant_types : list[str]
        List of variants that are intended to be tracked. 'Other' is needed since it includes all variants not specifically mentioned.

    Returns
    -------
    infections: pd.DataFrame
        Total case seperated by age group and all variant combinations an individual could have.
    """
    # Converting list of variants tracked into array for later operations.
    variant_array = np.array(variant_types)
    # Number of variants tracked.
    num_variants = len(variant_array)
    # All possible combinations of variants, labeled as 0 for no and 1 as yes.
    comb_matrix = np.indices((2,) * num_variants).reshape(num_variants, -1).T
    # Converting all combination of variants into booleans instead of 0 and 1.
    comb_matrix = comb_matrix.astype(bool)
    # Which combination of variants are subsets of others. A combination of variants is a subset if all variants are contained in another combination.
    superset_bool = np.all(
        (~comb_matrix[:, None]) | comb_matrix[None, :], axis=2
    )
    # Same matrix, but with boolean values as either 0 or 1.
    # In the matrix, index ij is one if variant combination j has all variants present within variant combination i, and is zero otherwise.
    # This is going to be used to solve for the indivdual variant combination probabilities since we only have access to their joint probabilities.
    superset_matrix = superset_bool.astype(float)
    # Total infections for each strain by age group.
    infect_grouped = infect.groupby(["age", "strain"]).cases.sum()
    # Going to be a list of dataframes that will have desired outputs for each age group.
    df_age_list = []
    # For each age group, we calculate the proportion of all infections that fall exclusively into each variant combination.
    # Any people in population not assigned into a variant combination will be placed into no infection category.
    for age_val, N in zip(pop.age.values, pop.census.values):
        # Variant probabilities for curent age group.
        infect_age = infect_grouped.get(
            age_val, pd.Series(0, index=variant_array)
        )
        # Proportion of current age group's total population infected by each variant.
        proportion = infect_age.reindex(variant_array, fill_value=0).values / N
        # Product of each combination of variant, and setting 1 to the no infection category.
        comb_prob = np.prod(np.where(comb_matrix, proportion, 1), axis=1)
        # Solving for Probability Mass Function (pmf). The superset times pmf should be the joint distributions, i.e the comb_prob.
        # This function solves for the pmf of that equation.
        pmf = np.linalg.solve(superset_matrix, comb_prob)
        # Creates dataframe with total infections and percentage of population infected by each combination of variants.
        df_age = pd.DataFrame(
            comb_matrix.astype(int), columns=variant_array
        ).assign(age=age_val, total=pmf * N, percent=pmf)
        # Adding that dataframe to the list of dataframes.
        df_age_list.append(df_age)
    # Concatinating list of dataframes into a single dataframe.
    infections = pd.concat(df_age_list, ignore_index=True)

    return infections
