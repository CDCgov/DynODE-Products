"""formatting_data.py takes raw data and formats them into usable dataframes for later functions."""

import pandas as pd  # type: ignore[import-untyped]


def format_vac_data(
    vac_raw: pd.DataFrame, start: str, doses: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take raw vaccination data, structures new dataframes, and accounting for overlapping individuals in different dosage totals.

    Parameters
    ----------
    vac_raw : pd.DataFrame
        Vaccination data including total people with at least n dosages for n <= "doses".
        The totals overlap so dose=1 would be people with 1 or more total dosages.
        This function will solve for all people with exactly n dosages, rather than at least n dosages, like it currently is.
    start : str
        Final date that data will be used.
    doses : int
        Number of vaccination dosage tracked.

    Returns
    -------
    vac: pd.DataFrame
        Proportion of people in each age group that have each dosage total for the final date tracked.
    vac_history: pd.DataFrame
        Proportion of people in each age group that have each dosage total for the every date with data.
    """
    # Sort vaccination data by date, age, and dose to group for vectorizing.
    vac_raw = vac_raw.sort_values(["date", "age", "dose"])
    # Group by date and age, and since we sorted by dose previously each group is in order from least to most dose.
    # We create new column where each dose has percentage of next dosage on same column. Used for calculations in future.
    vac_raw["next_percentage"] = vac_raw.groupby(["date", "age"])[
        "percentage"
    ].shift(-1)
    # We create new column where each dose has total of next dosage on same column. Used for calculations in future.
    vac_raw["next_total"] = vac_raw.groupby(["date", "age"])["total"].shift(-1)
    # Make new column which is going to be data for exactly n doses rather than n or more doses.
    vac_raw["exact_percent"] = vac_raw["percentage"]
    vac_raw["exact_total"] = vac_raw["total"]
    # Index location of rows in dataframe that are for doses more than zero, but less than highest tracked dosage.
    # Highest dosage tracked will remain the same.
    mask_exact = (vac_raw["dose"] > 0) & (vac_raw["dose"] < doses)
    # The percentage for this group will be the percentage of the current dose minus the percentage of the next highest dosage.
    vac_raw.loc[mask_exact, "exact_percent"] = vac_raw.loc[
        mask_exact, "percentage"
    ] - vac_raw.loc[mask_exact, "next_percentage"].fillna(0)
    # The total for this group will be the total of the current dose minus the total of the next highest dosage.
    vac_raw.loc[mask_exact, "exact_total"] = vac_raw.loc[
        mask_exact, "total"
    ] - vac_raw.loc[mask_exact, "next_total"].fillna(0)
    # Generate dataframe including only those with exactly 1 dose.
    dose1 = vac_raw[vac_raw["dose"] == 1].copy()
    # Create dataframe containing groups with zero infections, which are just all people minus those with vaccinations.
    dose0 = pd.DataFrame(
        {
            "date": dose1["date"],
            "age": dose1["age"],
            "dose": 0,
            "exact_percent": 1 - dose1["percentage"],
            "exact_total": dose1["census"] - dose1["total"],
        }
    )
    # Combine adjusted data for those with vaccinations and those with zero vaccinations.
    vac_history = pd.concat(
        [
            vac_raw[["date", "age", "dose", "exact_percent", "exact_total"]],
            dose0,
        ],
        ignore_index=True,
    )
    # Renaming columns to match desired output for later functions.
    vac_history.rename(
        columns={"exact_percent": "percent", "exact_total": "total"},
        inplace=True,
    )
    # Dataframe containing only final date of data.
    vac = vac_history[vac_history.date == start].reset_index(drop=True)

    return vac_history, vac


def output_generation(
    infections: pd.DataFrame,
    vac: pd.DataFrame,
    doses: int,
) -> pd.DataFrame:
    """Combine infection and vaccination proporotions of age groups and distributes them with infection and vaccination history likilihoods being idependent.

    Parameters
    ----------
    infections : pd.DataFrame
        Dataframe including proportion of people with every combination of different variants by age group.
    vac : pd.DataFrame
        Dataframe including proportion of people with different number of vaccinations by age group.
    doses : int
        Max number of different vaccines tracked among all individuals.

    Returns
    -------
    output: pd.DataFrame
        Dataframe that includes proportion of each age group that has been infected by every combination of variants and every tracked number of vaccinations.
    """
    # Copy of infection data which we will add columns to add vaccination data.
    total_data = infections.copy()
    # Generating list of which doses we track.
    list_of_doses = list(range(doses + 1))
    # Vaccination data with only data of vaccination tracked.
    vac_with_tracked_doses = vac[vac.dose.isin(list_of_doses)].reset_index(
        drop=True
    )
    # Removing columns that won't be used and changing name of column shared between columns..
    vac_with_tracked_doses = vac_with_tracked_doses.rename(
        columns={"percent": "percent_vac"}
    ).drop(columns=["date", "total"])
    # Left joining vaccination data to variant data along age groups.
    output = pd.merge(total_data, vac_with_tracked_doses, how="left", on="age")
    # Total is people infected times people in age group with specific dosage.
    output["total"] *= output["percent_vac"]
    # Percentage is calculated the same way as total.
    output["percent"] *= output["percent_vac"]
    # Remove percentage of vaccination column since we only care about combination percentage.
    output = output.drop(columns=["percent_vac"])

    return output
