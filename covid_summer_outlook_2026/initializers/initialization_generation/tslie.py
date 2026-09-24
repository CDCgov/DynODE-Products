"""tslie.py takes vaccination and infection data and estimates the Time Since Last Immunization Event(tslie) for different groups."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def scaling_vaccination(
    df: pd.DataFrame,
    ratio: float,
    category_total_people: float,
    tslie: pd.DataFrame,
) -> pd.DataFrame:
    """Take vaccination data convert cumulative to weekly and scale to group total population.

    Parameters
    ----------
    df : pd.DataFrame
        Vaccination data dataframe.
    ratio : float
        Proportion of people from categroy that have specific number of vaccinations..
    category_total_people : float
        Total number of people from population that are in categroy.
    tslie : pd.DataFrame
        Dataframe with index of final output dataframe for get_tslie function.

    Returns
    -------
    df: pd.DataFrame
        Vaccination data scaled to category population total, including weekly totals, and indexed to match final output of get_tslie function.
    """
    # Creating column to calculate vaccinations within last week rather than cumlative vaccinations.
    df["new"] = df["total"].diff()
    # Accounting for .diff function not applying to index 0.
    df.loc[0, "new"] = df.total[0] * 1
    # Accounting for rounding errors or data error that can lead to negative people being vaccinated in a week.
    df.loc[df.new < 0, "new"] = 0
    # Total new vaccinations of current dosage, accounting for possible division by zero.
    tot = max(sum(df.new), 1)
    # Scaling vaccinations by total population of age group
    df.new *= 1 / tot * category_total_people * ratio
    # Creating dataframe to store tslie, but including dates only when vaccination data is available.
    tslie_vac = tslie[tslie.date.isin(df.date)]
    # Matching index for vaccination and tslie dataframes.
    df.index = tslie_vac.index
    return df


def vaccine_tslie(
    vac: pd.DataFrame,
    tslie: pd.DataFrame,
    category_total_people: float,
    age_total: int,
) -> pd.DataFrame:
    """Convert vaccination history to time since laste immunization event for current category population and output dataframe with that information.

    Parameters
    ----------
    vac : pd.DataFrame
        Vaccination history for current category.
    tslie : pd.DataFrame
        DataFrame which will store time since last immunization event due to vaccination for current category.
    category_total_people : float
        Number of people in current category.
    age_total: int
        Number of people in the age group.

    Returns
    -------
    output_vac
        DataFrame which will store time since last immunization event due to vaccination for current.
    """
    output_vac = tslie.copy()
    # If this is the last tracked dosage, but there is additional vaccination data available, we will use newer vaccination totals to account for most recent vaccination.
    # Dataframe with max available dosages that is higher than highest tracked dosage.
    # For all other cases, the max and min dose are the same, and therefore the code will run when only one dose is tracked as well.
    newest_vac = vac[vac.dose == max(vac.dose)]
    # Creates subset of dataframe for current dosage.
    df_vac = vac[vac.dose == min(vac.dose)].reset_index(drop=True)
    # Next two lines of code matches dataframe of max dosage with those of current dosage.
    newest_vac = newest_vac[newest_vac.date.isin(df_vac.date)]
    newest_vac = newest_vac.reset_index(drop=True)
    # Comparing what proportion of people who got current dosage that also would have gotten the dosage with the most doses.
    ratio_newest = max(newest_vac.total) / max(df_vac.total)
    # Ratio of total people not in newest vaccination group.
    ratio_remainder = 1 - ratio_newest
    # Function that takes vaccination data and gets weekly change in vaccination scaled by group population total for first vaccination status tracked in data available.
    df_vac = scaling_vaccination(
        df_vac, ratio_remainder, category_total_people, output_vac
    )
    # Function that takes vaccination data and gets weekly change in vaccination scaled by group population total for most recent vaccination status.
    newest_vac = scaling_vaccination(
        newest_vac, ratio_newest, category_total_people, output_vac
    )
    # Creating dataframe to store tslie, but including dates only when vaccination data is available.
    tslie_vac = output_vac[output_vac.date.isin(df_vac.date)]
    # Combinging our tslie dataframe and vaccination dataframe together.
    df_vac = (
        (tslie_vac.total + df_vac.new + newest_vac.new) + tslie.total
    ).fillna(0)
    output_vac.loc[:, "total"] = df_vac.astype(float)
    # Adding percent of total age group into tslie dataframe.
    output_vac.loc[:, "percent"] = df_vac / age_total

    return output_vac


def infection_tslie(
    infect_category: pd.DataFrame,
    tslie: pd.DataFrame,
    category_total_people: float,
    age_total: int,
) -> pd.DataFrame:
    """Take infection history of category and identifying most recent immunization event within that category.

    Parameters
    ----------
    infect_category : pd.DataFrame
        Data of all infections that occured from current category.
    tslie : pd.DataFrame
        DataFrame which will store time since last immunization event due to infections for current category.
    category_total_people : float
        Number of people in current category.
    age_total: int
        Number of people in the age group.

    Returns
    -------
    output_infect: pd.DataFrame
        DataFrame which will store time since last immunization event due to infections for current category.
    """
    # Making copy of output dataframe.
    output_infect = tslie.copy()
    # Making copy of infection data.
    df_infect = infect_category.copy()
    # If there are multiple strains in category, we choose most recent strain to account for most recent immunization event.
    # Since we only care about most recent immunization event, the more recent variant should be the more recent infection.
    # This may lead to missing some earlier infection dates from "other" category, but deemed fine.
    if len(set(df_infect.strain)) > 1:
        # Finding all days with non-zero cases.
        dates_with_infections = df_infect[df_infect.cases > 0]
        # Of cases with some cases, finding the most recent date.
        most_recent_infection_day = max(dates_with_infections.date)
        # Subsection of df_infect with only most recent date.
        infect_most_recent_day = df_infect[
            df_infect.date == most_recent_infection_day
        ]
        # Name of strain with most cases in most recent date.
        most_recent_strain = infect_most_recent_day[
            infect_most_recent_day.cases == max(infect_most_recent_day.cases)
        ].strain
        # Removing older strain from df_infect.
        df_infect = df_infect[
            df_infect.strain == most_recent_strain.iloc[0]
        ].reset_index(drop=True)
    # Total infections, and accounting for possible division by zero in the future.
    tot_infections = max(sum(df_infect.cases), 1)
    # Scaling weekly cases by current groups total population.
    df_infect.cases = df_infect["cases"].apply(
        lambda x: x * (1 / tot_infections * category_total_people)
    )
    # Creates desired tslie dataframe only including dates with data from infection data.
    tslie_infect = output_infect[output_infect.date.isin(df_infect.date)]
    # Adjust the index such that tslie_infect and df_infect have same index.
    df_infect.index = tslie_infect.index
    # Replacing na with zero to avoid errors.
    df_infect = ((tslie_infect.total + df_infect.cases) + tslie.total).fillna(
        0
    )
    # Adding infections to tslie dataframe.
    output_infect.loc[:, "total"] = df_infect.astype(float)
    # Adding percent of total age group into tslie dataframe.
    output_infect.loc[:, "percent"] = df_infect / age_total

    return output_infect


def get_tslie(
    infect: pd.DataFrame,
    vac_history: pd.DataFrame,
    output: pd.DataFrame,
    pop: pd.DataFrame,
    variant_types: list[str],
) -> pd.DataFrame:
    """Take vaccination and infection histories of total populations and estimates time since last immunization event for total population seperated by group.

    Parameters
    ----------
    infect : pd.DataFrame
        Dataframe including total infections by all combinations of each variant tracked by age group.
    vac_history : pd.DataFrame
        Dataframe including proportion of people vaccinated different number of times over time by age group.
    output : pd.DataFrame
        Dataframe that includes proportion of each age group that has been infected by every combination of variants and every tracked number of vaccinations.
    pop : pd.DataFrame
        Dataframe including all age groups total population and proportion of total population.
    variant_types: list[str]
        List containing variants tracked.

    Returns
    -------
    tslie_output: pd.DataFrame
        Dataframe containing time since last immunization event for each age group, vaccination group, and infection variant combinaiton group.
    """
    # Making list of every day that vaccination or infection data exists.
    all_days = list(set(list(set(infect.date)) + list(set(vac_history.date))))
    # Order the all_days list in dates from furthest away to most recent.
    all_days.sort()
    # List of all vaccine doses tracked
    vac_dose = sorted(list(set(output.dose)))
    # Highest dosage of vaccination tracked.
    max_vac_dose = max(vac_dose)
    # Duplicating output dataframe with intention of adding rows for every day with data.
    expanded = output.copy()
    # Replacing percent with 0 since we are going to fill up with data for every given day rather than cumulative data.
    expanded["percent"] = 0.0
    # Replacing total with 0 since we are going to fill up with data for every given day rather than cumulative data.
    expanded["total"] = 0.0
    # Adding column to identify the category this row represents.
    # Categories are combinations of vaccination dose total, different variants being infected, and age groups.
    # Each category is disjoint and the combination of all categories is the entire population.
    expanded["category"] = list(range(len(output)))
    # Each category will now have an additional row for each date with data.
    tslie_output = expanded.loc[expanded.index.repeat(len(all_days))].copy()
    # Adding dates column with each date with data.
    tslie_output["date"] = np.tile(all_days, len(expanded))
    # Reordering the index for easier referencing.
    tslie_output.reset_index(drop=True, inplace=True)
    # For every category, distribute proportion of people among each date.
    for cur_category in range(max(tslie_output.category) + 1):
        # Subsection of dataframe including only current category.
        # This for-loop will fill empty columns and add back to tslie_output dataframe.
        tslie_category = tslie_output[tslie_output.category == cur_category]
        # Variable for age group in category.
        cur_age = tslie_category.age.iloc[0]
        # Including vaccination data with only age group in current category.
        vac_category = vac_history[vac_history.age == cur_age]
        # Total population of age group associated with category.
        age_group_pop = pop[pop.age == cur_age].census.iloc[0]
        # Total people that make up the current category.
        category_total_people = (
            pop[pop.age == cur_age].census * output.percent[cur_category]
        ).iloc[0]
        # Including dosage of only current category.
        if tslie_category.dose.iloc[0] == max_vac_dose:
            # If the highest tracked dosage is lower than the dosages with data, we use higher dosage to account for most recent vaccination.
            vac_category = vac_category[
                vac_category.dose >= tslie_category.dose.iloc[0]
            ]
        else:
            vac_category = vac_category[
                vac_category.dose == tslie_category.dose.iloc[0]
            ]
        # Including infection data only of current age group.
        infect_category = infect[infect.age == cur_age]
        # Going to contain used strains.
        list_used_strains = []
        # Including only strains in current category.
        for cur_var in range(len(variant_types)):
            # Adding included strains into list.
            if tslie_category[variant_types[cur_var]].iloc[0] == 1:
                list_used_strains.append(variant_types[cur_var])
        # Removing strains with no infections in current category from dataframe.
        infect_category = infect_category[
            infect_category["strain"].isin(list_used_strains)
        ]
        # Filling tslie_category with most recent immunization event for each date for current category.
        # For categories with no infections and no vaccinations.
        if len(infect_category) == 0 and tslie_category.dose.iloc[0] == 0:
            # Sets all people in current age group into oldest date since they have no immunization event.
            tslie_category.loc[tslie_category.index[0], "total"] += (
                pop.loc[pop.age == cur_age, "census"].values[0]
                * output.percent[cur_category]
            )
            # Putting percentage into oldest date.
            tslie_category.loc[tslie_category.index[0], "percent"] = (
                output.percent[cur_category]
            )

        # For categories with no infections, but with some vaccinations.
        elif len(infect_category) == 0:
            # Function takes vaccination data
            # and adds most recent immunization event to the tslie_category dataframe.
            tslie_category = vaccine_tslie(
                vac_category,
                tslie_category,
                category_total_people,
                age_group_pop,
            )
        # For categories with no vaccinations, but with some infections.
        elif tslie_category.dose.iloc[0] == 0:
            # Function takes infection data
            # and adds most recent immunization events from those infections to tslie_category dataframe
            tslie_category = infection_tslie(
                infect_category,
                tslie_category,
                category_total_people,
                age_group_pop,
            )
        else:
            # Function takes vaccination data
            # and adds most recent immunization event to the tslie_category dataframe.
            df_vac = vaccine_tslie(
                vac_category,
                tslie_category,
                category_total_people,
                age_group_pop,
            )
            # Function takes infection data
            # and adds most recent immunization events from those infections to tslie_category dataframe
            df_infect = infection_tslie(
                infect_category,
                tslie_category,
                category_total_people,
                age_group_pop,
            )
            joint_dist = pd.DataFrame(
                {
                    "date": tslie_category.date,
                    "age": tslie_category.age,
                    "prob": 0.0,
                }
            )
            # Preventing division by zero by making zero vaccinations 1.
            tot_vac = max(sum(df_vac.total), 1)
            # Preventing division by zero by making zero infections 1.
            tot_infect = max(sum(df_infect.total), 1)
            # For each date, calculating the probabilty that a person has their most recent event on the current date.
            for j in df_infect.index:
                # First date of the current category.
                fst = min(df_infect.index)
                if j == fst:
                    # Probability that their infection and vaccination both occured on the first date with data.
                    joint_dist.loc[j, "prob"] = (
                        (df_infect.total[j])
                        / tot_infect
                        * (df_vac.total[j])
                        / tot_vac
                    )
                else:
                    ## Law of total probability.
                    # Probability infection and vaccine occured on current day
                    # plus the probability infection happened current day and vaccination happened any day before
                    # plus the probability vaccination heppend current day and infection happened any day before.
                    joint_dist.loc[j, "prob"] = (
                        (df_infect.total[j])
                        / tot_infect
                        * (df_vac.total[j])
                        / tot_vac
                        + (df_infect.total[j])
                        / tot_infect
                        * df_vac["total"].iloc[:j].sum()
                        / tot_vac
                        + df_infect["total"].iloc[:j].sum()
                        / tot_infect
                        * (df_vac.total[j])
                        / tot_vac
                    )
            # Accounting for zero people in category in joint distribution dataframe.
            joint_sum = max(sum(joint_dist.prob), 1)
            # Normalizing data to account to rounding errors and issues with discrete time steps on continious probability equation.
            joint_dist.prob = (
                joint_dist.prob / joint_sum * output.percent[cur_category]
            )
            # Filling tslie_category dataframe with TSLIE for current category.
            tslie_category.loc[:, "percent"] = joint_dist["prob"]
            tslie_category.loc[:, "total"] = joint_dist["prob"] * age_group_pop
        # Adding current category to dataframe containing all categories.
        tslie_output.loc[tslie_category.index] = tslie_category
    # Removing category column since it is not needed anymore.
    tslie_output = tslie_output.drop(columns="category")
    return tslie_output
