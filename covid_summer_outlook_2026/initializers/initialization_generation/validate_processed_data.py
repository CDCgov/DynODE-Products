"""processed_data_validation.py takes the processed raw data and verfies the data remains logically consistent and not likely to lead to code crashes later on."""

import pandas as pd


# Simple function that takes condition and adds to list of errors if condition is false.
def check(condition: bool, message: str, errors: list[str]):
    if not condition:
        errors.append(message)


def vac_checks(
    vac: pd.DataFrame, vac_type: str, pop: pd.DataFrame, errors: list[str]
) -> list[str]:
    """List of checks to validate processed vaccination data.

    Parameters
    ----------
    vac : pd.DataFrame
        Processed vaccination data.
    vac_typ: pd.DataFrame
        String containing type of vaccination data tested for error message purposes.
    pop: pd.DataFrame
        Total population data by age group.
    errors: list[str]
        List of strings explaining all previously failed tests.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confirm vac history percentages don't include negative values.
    check(
        min(vac.percent) >= 0,
        f"Negative percent in {vac_type} vaccination data.",
        errors,
    )
    # Confirm vac history percentages don't include over 100%.
    check(
        min(vac.percent) <= 1,
        f"Proportion greater than 1 in {vac_type} vaccination data.",
        errors,
    )
    # Confirm vac history totals don't include negative people.
    check(
        min(vac.total) >= 0,
        f"Negative total people in {vac_type} vaccination data.",
        errors,
    )
    # Confirm that each date isn't missing data from an age or dose group.
    check(
        (vac.groupby("date").size().nunique() == 1),
        f"Certain dates in {vac_type} vaccination data are missing data.",
        errors,
    )
    # Confirm that each date isn't missing data from an age group.
    check(
        (vac.groupby("age").size().nunique() == 1),
        f"Certain ages in {vac_type} vaccination data are missing data.",
        errors,
    )
    # Confirm that each dose total isn't missing data.
    check(
        (vac.groupby("dose").size().nunique() == 1),
        f"Certain doses in {vac_type} vaccination data are missing data.",
        errors,
    )
    # Confirm only/all desired age groups are present.
    check(
        set(vac.age) == set(pop.age),
        f"Incorrect age groups in {vac_type} vaccination data.",
        errors,
    )

    return errors


def validate_vac_data(
    vac_history: pd.DataFrame,
    vac: pd.DataFrame,
    pop: pd.DataFrame,
    errors: list[str],
    doses: int,
) -> list[str]:
    """Run unit tests on processed vaccination data to check for issues with data.

    Parameters
    ----------
    vac_history: pd.DataFrame
        Vaccination data over entire time range user wants.
    vac : pd.DataFrame
        Vaccination data for most recent date user wants tracked.
    pop: pd.DataFrame
        Total population data by age group.
    errors: list[str]
        List of strings explaining all previously failed tests.
    doses: int
        Number of doses user wants to track.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confirm vac and vac_history data exists.
    vac_has_data = len(vac) > 0
    vac_history_has_data = len(vac_history) > 0
    check(
        vac_has_data, "Vaccination data of most recent date is blank.", errors
    )
    check(vac_history_has_data, "Vaccination history data is blank.", errors)
    # Only need to run additional checks if vaccination history data exists.
    if vac_history_has_data:
        # Going to run all checks for vac dataframe to check for errors.
        errors = vac_checks(vac_history, "historical", pop, errors)
    # Only need to run additional checks if vaccination of most recent data exists.
    if vac_has_data:
        # Going to run all checks for vac dataframe to check for errors.
        errors = vac_checks(vac, "recent", pop, errors)
        # Data of most recent vaccination data only including user requested doses.
    # Checks comparing the two vaccination data sets.
    if vac_history_has_data and vac_has_data:
        # Confirm that vac is the most recent date.
        check(
            max(vac_history.date) == max(vac.date),
            "Most recent vaccination date isn't actually most recent date.",
            errors,
        )

    return errors


def validate_variant_data(
    variant: pd.DataFrame, variant_types: list[str], errors: list[str]
) -> list[str]:
    """Run unit tests on processed variant data to check for issues with data.

    Parameters
    ----------
    variant: pd.DataFrame
        Variant data over entire time range user wants with only variants user requested.
    variant_types : list[str]
        List of variants that user requested.
    errors: list[str]
        List of strings explaining all previously failed tests.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confirm variant data set isn't blank.
    var_has_data = len(variant) > 0
    check(var_has_data, "Processed variant data is blank.", errors)
    # Confirm variant types list isn't blank.
    types_has_data = len(variant_types) > 0
    check(types_has_data, "List of variants tracked is blank.", errors)
    # If variant data exists then we will run checks.
    if var_has_data and types_has_data:
        # Confirm variant data contains variant types tracked.
        check(
            set(variant_types) == set(variant.variant),
            "Processed variant data missing tracked variant type.",
            errors,
        )
        # Confirm each variant has data for every date.
        check(
            (variant.groupby("date").size() == len(variant_types)).all(),
            "Some dates of processed variant data are missing data from variants.",
            errors,
        )
        # Confirm that sum of proportion for each date is 1.
        check(
            (variant.groupby("date")["proportion"].sum() - 1).abs().max()
            < 1e-8,
            "Processed variant data has proporiton for date not equal to 1.",
            errors,
        )
        # Comfirm that no date has same variant multiple times.
        check(
            (
                variant.groupby("variant")["date"].nunique()
                == variant.groupby("variant")["date"].size()
            ).all(),
            "In processed variant data, some dates have same variant multiple times.",
            errors,
        )

    return errors


def validate_case_data(
    cases: pd.DataFrame, pop: pd.DataFrame, errors: list[str]
) -> list[str]:
    """Run unit tests on processed case data to check for issues with data.

    Parameters
    ----------
    cases: pd.DataFrame
        Case data over entire time range user wants with only variants user requested.
    pop: pd.DataFrame
        Total population data by age group.
    errors: list[str]
        List of strings explaining all previously failed tests.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Confirm dataframe isn't empty.
    has_data = len(cases) > 0
    check(has_data, "Processed case data is empty.", errors)
    # Only run checks if case data isn't empty.
    if has_data:
        # Confirm that proper age groups are being used.
        check(
            set(cases.age) == set(pop.age),
            "Processed case data has incorrect age groups.",
            errors,
        )
        # Confirm no negative cases.
        check(
            min(cases.cases) >= 0,
            "Processed case data has negative cases.",
            errors,
        )
        # Confirm that each date has data for all age groups.
        check(
            (cases["date"].value_counts() == len(pop)).all(),
            "Some dates for processed case data are missing data from certain age groups.",
            errors,
        )
        # Confirm cases are not larger than population of age group.
        check(
            (
                cases.groupby("age")["cases"].max()
                < pop.groupby("age")["census"].max()
            ).all(),
            "For processed case data, some case totals surpass total population of an age group.",
            errors,
        )

    return errors


def validate_data_processing(
    vac_history: pd.DataFrame,
    vac: pd.DataFrame,
    variant: pd.DataFrame,
    variant_types: list[str],
    cases: pd.DataFrame,
    pop: pd.DataFrame,
    doses: int,
) -> list[str]:
    """Run unit tests on processed data used in initialization to check for data that is either logically impossible or will lead to errors later in the code.

    Parameters
    ----------
    vac_history : pd.DataFrame
        Vaccination data over entire time range user wants.
    vac : pd.DataFrame
        Vaccination data for most recent date user wants tracked.
    variant : pd.DataFrame
        Variant data including only variants user wants tracked.
    variant_types : list[str]
        List of all variants user wants tracked.
    cases : pd.DataFrame
        Raw variant data.
    pop: pd.DataFrame
        Total population data by age group.
    doses: int
        Number of doses user wants to track.

    Returns
    -------
    errors: list[str]
        List of strings explaining all failed tests.
    """
    # Going to store list of all errors for the data in these functions.
    errors = []
    # Function that comfirms vaccination data is logically consistent and won't lead to errors later in the code base.
    errors = validate_vac_data(vac_history, vac, pop, errors, doses)
    # Function that comfirms variant data is logically consistent and won't lead to errors later in the code base.
    errors = validate_variant_data(variant, variant_types, errors)
    # Function that comfirms case data is logically consistent and won't lead to errors later in the code base.
    errors = validate_case_data(cases, pop, errors)

    return errors
