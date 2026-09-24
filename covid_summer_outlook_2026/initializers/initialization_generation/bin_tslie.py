"""bin_tslie.py will place most recent immunization events by date into bins by how long ago they occured."""

import bisect
from datetime import datetime

import numpy as np
import pandas as pd


def format_tslie_binned(
    tslie: pd.DataFrame,
    variant_list: list[str],
    variant_bin_total: int,
    vac_bin_total: int,
) -> list:
    """Take tslie dataframe grouped by bin and change format to match input of dynODE.

    Parameters
    ----------
    tslie : pd.DataFrame
        Dataframe with total people with most recent immunization event within different bin.
    variant_list : pd.DataFrame
        List of all tracked variants
    variant_bin_total: int
        This is the number of variant bins associated user would like, independently of how many variants actually have data.
    vac_bin_total: int
        This is the number of vaccination bins associated user would like, independently of how many variants actually have data.

    Returns
    -------
    tslie: pd.DataFrame
        Same data as the tslie input, but in a different formatting to match desired output.
    """
    # Copying dataframe in order to make edits.
    df = tslie.copy()
    # Creating new column which represents the variant which caused most recent infection.
    df["infections"] = df[variant_list].max(axis=1)
    # Column names of percentage and variants that are not in final output.
    colums_to_drop = variant_list + ["percent"]
    # Removing columns that are not going to be in final output.
    df = df.drop(columns=colums_to_drop)
    # Since we dropped the variant rows, some rows have identical categories, so we sum up their totals and combine them together.
    df = df.groupby(["age", "infections", "dose", "bin"], as_index=False).agg(
        {"total": "sum"}
    )
    # If the user requests more variant bins than data available, then we fill them up with zero infection.
    if variant_bin_total > (len(variant_list) - 1):
        # List of infection totals greater than those with actual data.
        infection_count = list(range((len(variant_list)), variant_bin_total))
        # Subset of only on infection total in order to get correct dimensions.
        single_infection_base = df[df.infections == 0]
        # Creat list of subsets of infection total, but with total infections being zero for infection counts greater than infection counts with data.
        copies = [
            single_infection_base.assign(total=0.0, infections=b)
            for b in infection_count
        ]
        # Concat all subsets into a single dataframe.
        df = pd.concat([df] + copies, ignore_index=True)
    # If the user requests more vaccination bins than data available, then we fill them up with zero infection.
    if vac_bin_total > max(tslie.dose):
        # List of infection totals greater than those with actual data.
        vac_count = list(range(max(df.dose) + 1, vac_bin_total + 1))
        # Subset of only on vaccination total in order to get correct dimensions.
        single_dose_base = df[df.dose == 0]
        # Creat list of subsets of vaccination total, but with total infections being zero for infection counts greater than infection counts with data.
        copies = [
            single_dose_base.assign(total=0.0, dose=b) for b in vac_count
        ]
        # Concat all subsets into a single dataframe.
        df = pd.concat([df] + copies, ignore_index=True)
    # Sort the non-total categorizes in the order dynODE uses.
    df = df.sort_values(["age", "infections", "dose", "bin"])
    # Grouping dataframe by each category.
    df_idx = df.sort_values(["age", "infections", "dose", "bin"]).set_index(
        ["age", "infections", "dose", "bin"]
    )
    # Unstack non-age categories.
    tensor = df_idx["total"].unstack(["infections", "dose", "bin"])
    # Reshape to the format that dynODE uses.
    arr = tensor.values.reshape(
        len(set(df.age)),
        len(set(df.infections)),
        len(set(df.dose)),
        len(set(df.bin)),
    )

    return arr


def bin_tslie(
    tslie: pd.DataFrame,
    tslie_bins: list[int],
    variant_bin_total: int,
    vac_bin_total: int,
) -> list:
    """Take data date of most recent immunization event distribution and bin by user suggested age groups.

    Parameters
    ----------
    tslie : pd.DataFrame
        Dataframe including total infections by all combinations of each variant tracked by age group.
    tslie_bins : pd.DataFrame
        Dataframe including proportion of people vaccinated different number of times over time by age group.
    variant_bin_total: int
        When generating the output, this is the number of variant bins associated user would like, independently of how many variants actually have data.
    vac_bin_total: int
        When generating the output, this is the number of vaccine bins associated user would like, independently of how many variants actually have data.

    Returns
    -------
    output: list
        List containing time since last immunization event for each age group, vaccination group, and infection variant combinaiton group.
    """
    # List with the different variants.
    variant_list = tslie.drop(
        columns=["percent", "total", "dose", "age", "date"]
    ).columns.tolist()
    # Make duplicate because we want to keep original for later, but intend to remove "Other" from this
    variant_list_minus_other = variant_list.copy()
    # Variant list but excluding Other category for purposes of tracking them.
    variant_list_minus_other.remove("Other")
    # Change values of different variants to account for which variant was most recent.
    tslie[variant_list_minus_other] = tslie[variant_list_minus_other] * (
        np.arange(len(variant_list_minus_other)) + 1
    )
    # Convert dates from string to datetime in order to apply functions to dates.
    tslie_datetime = [datetime.strptime(d, "%Y-%m-%d") for d in tslie.date]
    # The most recent date that we use.
    most_recent_date = max(tslie_datetime)
    # Finding how many days away since most recent date every date is.
    date_diff = [
        (most_recent_date - tslie_datetime[i]).days
        for i in range(len(tslie_datetime))
    ]
    # Adding the days from most recent date as a column to tslie dataframe.
    tslie["days_from_most_recent"] = date_diff
    # Sorting tslie dataframe by the newly added column.
    tslie = tslie.sort_values("days_from_most_recent").reset_index(drop=True)
    # Creating column to store bins each category is associated with.
    tslie["bin"] = len(tslie_bins)
    # For each bin, we will add the bin number to each group that had their most recent event within that bin date range.
    for i in reversed(range(len(tslie_bins))):
        # Finding the index which all values less than are outside current bin.
        count = bisect.bisect_left(
            tslie["days_from_most_recent"], tslie_bins[i]
        )
        # Fill the bin column for all rows that satisfy current bin.
        tslie.loc[: count - 1, "bin"] = i
    # Going to generate copy of dataframe to group by bin rather than date.
    tslie_binned = tslie.copy()
    # Removing date and days from most recent columns that are not needed in final output.
    tslie_binned = tslie_binned.drop(columns=["date", "days_from_most_recent"])
    # List of all columns except for percent and total with intention of grouping combination of these columns for new category.
    list_without_totals = tslie_binned.drop(
        columns=["percent", "total"]
    ).columns.tolist()
    # Summing over total and percentage for all dates within a category.
    tslie_binned = (
        tslie_binned.groupby(list_without_totals).sum().reset_index()
    )
    # Formatting output to json file which is input of dynODE code base.
    output = format_tslie_binned(
        tslie_binned, variant_list, variant_bin_total, vac_bin_total
    )

    return output
