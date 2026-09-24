"""raw_data_validation.py takes raw code needed to run initialization and verifies that data is logically consistent."""

import pandas as pd


# Simple function that takes condition and adds to list of errors if condition is false.
def check(condition: bool, message: str, errors: list[str]):
    if not condition:
        errors.append(message)


def validate_vac(
    vac: pd.DataFrame, errors: list[str], pop: pd.DataFrame
) -> list[str]:
    """Run unit tests on raw vaccination data to check for issues with data.

    Parameters
    ----------
    vac: pd.DataFrame
        Raw Vaccination data.
    errors: list[str]
        List of strings explaining all previously failed tests.
    pop: pd.DataFrame
        Dataframe containing total population of each age group.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confirm vaccination data is not empty.
    has_data = len(vac) > 0
    check(has_data, "Vaccination data is blank.", errors)
    # Only run checks if vaccination data exists.
    if has_data:
        # Confirm that total isn't negative.
        check(
            min(vac.total) >= 0, "Vaccination totals less than zero.", errors
        )
        # Confirm no negative percentage.
        check(
            min(vac.percentage) >= 0,
            "Percentage of vaccination data less than zero.",
            errors,
        )
        # Confirm no percentage over 100%.
        check(
            max(vac.percentage) <= 1,
            "Percentage of vaccination data over 100%.",
            errors,
        )
        # Confirm total and percentage match.
        check(
            max(vac["percentage"] - (vac["total"] / vac["census"]).abs())
            < 1e-8,
            "Vaccination percentage and totals do not line up properly.",
            errors,
        )
        # Confirm each age group has the same data entries.
        check(
            (vac.groupby("age").size().nunique() == 1),
            "Age groups in vaccination data aren't same dimension.",
            errors,
        )
        # Comfirm each vaccination dosage have the same data entries.
        check(
            (vac.groupby("dose").size().nunique() == 1),
            "Vaccination doses don't have the same dimension.",
            errors,
        )
        # Confirm each date has the same number of data entries.
        check(
            (vac.groupby("date").size().nunique() == 1),
            "Certain dates in vaccination data are missing data.",
            errors,
        )
        # Confirm that age groups match pop age groups.
        check(
            set(vac.age) == set(pop.age),
            "Age groups in vaccination data are incorrect.",
            errors,
        )

    return errors


def validate_hosp(
    hospitalization: pd.DataFrame, errors: list[str], pop: pd.DataFrame
) -> list[str]:
    """Run unit tests on raw vaccination data to check for issues with data.

    Parameters
    ----------
    hospitalization: pd.DataFrame
        Raw hospitalization data.
    errors: list[str]
        List of strings explaining all previously failed tests.
    pop: pd.DataFrame
        Dataframe containing total population of each age group.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Check if there is data.
    has_data = len(hospitalization) > 0
    check(has_data, "Hospitalization data is blank.", errors)
    # Only run checks if there is data.
    if has_data:
        # Confirm no negative hospitalizations in a week.
        check(
            min(hospitalization.hospital_weekly) >= 0,
            "Negative hospitalizations in a week.",
            errors,
        )
        # Confirm no dates have multiple data entries.
        check(
            len(hospitalization) == len(set(hospitalization.date)),
            "Hospitalization data contains multiple data for same date.",
            errors,
        )
        # Daily hospitalization more than population.
        check(
            sum(hospitalization.hospital_weekly) <= sum(pop.census),
            "Weekly hospitalization larger than total population.",
            errors,
        )

    return errors


def validate_sero(
    sero: pd.DataFrame, errors: list[str], pop: pd.DataFrame
) -> list[str]:
    """Run unit tests on raw vaccination data to check for issues with data.

    Parameters
    ----------
    sero: pd.DataFrame
        Raw seroprevalence data.
    errors: list[str]
        List of strings explaining all previously failed tests.
    pop: pd.DataFrame
        Dataframe containing total population of each age group.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confrim sero data is not empty.
    has_data = len(sero) > 0
    check(len(sero) > 0, "Sero data is blank.", errors)
    # Only run checks if there is data.
    if has_data:
        # Confirm no negative sero.
        check(
            min(sero.sero) >= 0, "Sero data contains negative values.", errors
        )
        # Confirm no sero proporiton greater than 1.
        check(max(sero.sero) <= 1, "Sero data proportion over 1.", errors)
        # Confirm correct age groups.
        check(
            set(sero.age) == set(pop.age),
            "Sero data age groups are not proper age groups.",
            errors,
        )
        # Confirm each age group has the same data entries.
        check(
            (sero.groupby("age").size().nunique() == 1),
            "Age groups in sero data aren't same dimension.",
            errors,
        )
        # Confirm each age group has the same data entries.
        check(
            (sero.groupby("date").size().nunique() == 1),
            "Sero data missing data from some date.",
            errors,
        )

    return errors


def validate_variant(variant: pd.DataFrame, errors: list[str]) -> list[str]:
    """Run unit tests on raw vaccination data to check for issues with data.

    Parameters
    ----------
    variant: pd.DataFrame
        Raw variant data.
    errors: list[str]
        List of strings explaining all previously failed tests.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confrim sero data is not empty.
    has_data = len(variant) > 0
    check(has_data, "Variant data is blank.", errors)
    # Only run checks if there is data.
    if has_data:
        # Confirm no proportion is negative.
        check(
            min(variant.proportion) >= 0,
            "Variant data has negative proportion.",
            errors,
        )
        # Confirm no proportion is greater than 1.
        check(
            max(variant.proportion) <= 1,
            "Variant data has proportions greater than 1.",
            errors,
        )
        # Confirm that sum of all proportions sum to 1 for each date.
        check(
            (variant.groupby("date").proportion.sum() - 1).abs().max() < 1e-8,
            "In variant data, total proportion doesn't equal 1 for all dates.",
            errors,
        )
        # Confirm that no variants have data from same date more than once.
        check(
            variant.groupby("variant")["date"]
            .nunique()
            .eq(variant.groupby("variant")["date"].size())
            .all(),
            "Some variants from variant data have multiple data from same date.",
            errors,
        )

    return errors


def validate_pop(pop: pd.DataFrame, errors: list[str]) -> list[str]:
    """Run unit tests on raw vaccination data to check for issues with data.

    Parameters
    ----------
    pop: pd.DataFrame
        Raw population data.
    errors: list[str]
        List of strings explaining all previously failed tests.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confirm that there are no negative population age groups.
    check(min(pop.census) >= 0, "Negative population total", errors)
    # Confirm that the sum of the proportion of the population sums to 1.
    check(
        abs(sum(pop.proportion) - 1) < 1e-8,
        "Totoal proportion of population doesn't sum to 1.",
        errors,
    )
    # Confirm that proportion and population numbers match.
    check(
        pop["proportion"].equals(pop["census"] / sum(pop["census"])),
        "Population proportion and totals do not match.",
        errors,
    )
    # Confirm that each age group data isn't repeated.
    check(
        len(set(pop.age)) == len(pop),
        "Duplicate age group used in population data.",
        errors,
    )

    return errors


def validate_raw_data(
    vac: pd.DataFrame,
    hospitalization: pd.DataFrame,
    sero: pd.DataFrame,
    variant: pd.DataFrame,
    pop: pd.DataFrame,
) -> list[str]:
    """Run unit tests on raw data used in initialization to check for data that is either logically impossible or will lead to errors later in the code.

    Parameters
    ----------
    vac : pd.DataFrame
        Raw vaccination data.
    hospitalization : pd.DataFrame
        Raw hospitalization data.
    sero : pd.DataFrame
        Raw seroprevalence data.
    variant : pd.DataFrame
        Raw variant data.
    pop : pd.DataFrame
        Raw population data.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Going to store list of strings explaining what tests failed if any.
    errors = []
    # Function that comfirms population data is logically consistent and won't lead to errors later in the code base.
    errors = validate_pop(pop, errors)
    # Function that comfirms vaccination data is logically consistent and won't lead to errors later in the code base.
    errors = validate_vac(vac, errors, pop)
    # Function that comfirms hospitalization data is logically consistent and won't lead to errors later in the code base.
    errors = validate_hosp(hospitalization, errors, pop)
    # Function that comfirms sero data is logically consistent and won't lead to errors later in the code base.
    errors = validate_sero(sero, errors, pop)
    # Function that comfirms variant data is logically consistent and won't lead to errors later in the code base.
    errors = validate_variant(variant, errors)

    return errors
