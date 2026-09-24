"""tslie_validation.py verifies that the final output remains logically consistent and not likely to lead to code crashes later on."""

import pandas as pd


# Simple function that takes condition and adds to list of errors if condition is false.
def check(condition: bool, message: str, errors: list[str]):
    if not condition:
        errors.append(message)


def validate_tslie(
    tslie: pd.DataFrame, pop: pd.DataFrame, variant_types_processed: list[str]
) -> list[str]:
    """Confirm dataframe of time since last immunization event is logically consistent.

    Parameters
    ----------
    tslie : pd.DataFrame
        Time since last immunization event for the entire population.
    pop: pd.DataFrame
        Total population data by age group.
    variant_types_processed: list[str]
        List of the variants that user wants to track.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Create list to store any errors found in the tslie dataframe.
    errors = []
    # Confirm that dataframe isn't empty
    has_data = len(tslie) > 0
    check(has_data, "Tslie does not have any data", errors)
    if has_data:
        # Confirm no negative percentages.
        check(
            min(tslie.percent) >= 0,
            "Tslie contains negative percentages.",
            errors,
        )
        # Confirm no percentage over 100%.
        check(
            max(tslie.percent) <= 1,
            "Tslie contains proportions over 1.",
            errors,
        )
        # Confirm no negative people.
        check(
            min(tslie.total) >= 0,
            "Tslie contains negative totals for some dates.",
            errors,
        )
        # Confirm all age groups are present in data.
        check(
            set(tslie.age) == set(pop.age),
            "Tslie does not contain proper age groups.",
            errors,
        )
        # Confirm that total population is accounted for.
        check(
            abs(sum(tslie.total) - sum(pop.census)) < 1e-7,
            "The total in Tslie does not equal entire population.",
            errors,
        )
        # Confirm that percentages add up.
        check(
            abs(sum(tslie.percent) - len(pop)) < 1e-7,
            "Tslie percentages do not add up to equal 1 for each age group.",
            errors,
        )
        # Confirm that each age group has the same amount of data.
        check(
            tslie["age"].value_counts().all(),
            "Some age groups have more dates in Tslie output.",
            errors,
        )
        # Confirm that each date has same amount of groups.
        check(
            tslie["date"].value_counts().all(),
            "Some dates in Tslie data contains more groups than other dates.",
            errors,
        )
        # Confirm that each vaccine dose has same amount of groups.
        check(
            tslie["dose"].value_counts().all(),
            "Some vaccine doses in Tslie data contains more groups than other dates.",
            errors,
        )
        # Confirm each combination of variants have same number of dates and age groups.
        check(
            tslie.groupby(variant_types_processed).size().all(),
            "Tslie has some combination of variants with missing data.",
            errors,
        )

    return errors
