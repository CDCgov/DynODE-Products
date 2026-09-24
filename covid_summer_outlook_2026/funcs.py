"""
Seasonality, vaccination, and external infection helper functions for ODE models.

Functions expect JAX arrays and are designed for use in epidemiological ODE models.
All time variables are in days. Array shapes and units are noted per function.
"""

from datetime import date

import jax.numpy as jnp
from dynode import evaluate_cubic_spline
from jax import Array
from jax.scipy.stats.norm import pdf
from jax.typing import ArrayLike


def seasonality(
    t: ArrayLike,
    seasonality_amplitude: Array,
    seasonality_second_wave: Array,
    seasonality_dominant_wave_day: Array,
    seasonality_second_wave_day: Array,
    seasonality_wave_width: Array,
    init_date: date,
    seasonality_second_wave_width: Array = None,
) -> ArrayLike:
    """
    Compute a seasonal multiplier for ODE seasonal forcing.

    Parameters
    ----------
    t : ArrayLike
        1D array of days since model start.
    seasonality_amplitude : Array
        Amplitude of seasonal forcing.
    seasonality_second_wave : Array
        Magnitude of the second seasonal wave.
    seasonality_dominant_wave_day : Array
        Peak day for dominant wave.
    seasonality_second_wave_day : Array
        Peak day for second wave.
    seasonality_wave_width : Array
        Width of the dominant wave.
    init_date : date
        Initial date for time alignment.
    seasonality_second_wave_width : Array, optional
        Width of the second wave.

    Returns
    -------
    ArrayLike
        Seasonal multiplier for each time point, shape matches t.

    Notes
    -----
    - Uses cosine-based seasonality.
    - Exponent controls wave sharpness.
    - If seasonality_second_wave_width is None, uses seasonality_wave_width for both waves.
    """

    def g(t: ArrayLike, phi: ArrayLike, w: int):
        return jnp.abs(jnp.cos(2 * jnp.pi * (t - phi) / 730.0)) ** w

    def f(
        t: ArrayLike,
        dominant_wave_day,
        second_wave_day,
        seasonality_second_wave: ArrayLike,
        w1: int = 10,  # default value, but actually estimated
        w2: int = 10,  # default value, but actually estimated
    ):
        # w1/w2 determine the narrowness of the dominant/second waves
        return (
            -0.5
            + 1.0 * g(t, phi=dominant_wave_day, w=w1)
            + seasonality_second_wave * g(t, phi=second_wave_day, w=w2)
        )

    shifted_dominant_wave_day = (
        seasonality_dominant_wave_day - init_date.timetuple().tm_yday
    )
    shifted_second_wave_day = (
        seasonality_second_wave_day - init_date.timetuple().tm_yday
    )
    w2 = (
        seasonality_wave_width
        if seasonality_second_wave_width is None
        else seasonality_second_wave_width
    )

    return (
        1
        + f(
            t,
            dominant_wave_day=shifted_dominant_wave_day,
            second_wave_day=shifted_second_wave_day,
            seasonality_second_wave=seasonality_second_wave,
            w1=seasonality_wave_width,
            w2=w2,
        )
        * seasonality_amplitude
        * 2
    )


def external_i(
    t: ArrayLike,
    introduction_times: Array,
    introduction_scales: Array,
    introduction_percentages: Array,
    introduction_ages: Array,
    population_by_age: Array,
    i_shape: tuple,
) -> Array:
    """
    Calculate number of external infection interacting with the population at time t.

    Parameters
    ----------
    t : ArrayLike
        1D array of days.
    introduction_times : Array
        Days of introduction events, shape (n_events,).
    introduction_scales : Array
        Stddev for each event, shape (n_events,).
    introduction_percentages : Array
        Fraction of population for each event, shape (n_events,).
    introduction_ages : Array
        Age group weights, shape (n_events, n_ages).
    population_by_age : Array
        Population per age group, shape (n_ages,).
    i_shape : tuple
        Shape of output array.

    Returns
    -------
    Array
        External infected array, shape i_shape.

    Notes
    -----
    Uses normal PDF for event timing.
    """
    ext_i = jnp.zeros(i_shape)
    densities = pdf(t, loc=introduction_times, scale=introduction_scales)
    introduction_densities = introduction_percentages * densities
    populations = introduction_ages * population_by_age
    populations_new_infections = introduction_densities[:, None] * populations
    ext_i = ext_i.at[:, 0, 0, :].set(populations_new_infections.T)
    return ext_i


def vaccination_rates(
    t: ArrayLike,
    vax_discontinuity_points: list[int],
    spline_model_params: tuple[Array, Array, Array],
    hill_model_params: list[tuple[Array, Array, Array, Array]],
) -> Array:
    """
    Compute vaccination rates for each age bin and strata.

    Parameters
    ----------
    t : ArrayLike
        1D array of days.
    vax_discontinuity_points : list of int
        Days where vaccination model changes.
    spline_model_params : tuple
        Parameters for spline model (see vaccination_rate_spline).
    hill_model_params : list
        List of Hill model parameter tuples (see vaccination_rate_hill).

    Returns
    -------
    Array
        Vaccination rates, shape depends on model.
    """
    changepoints = vax_discontinuity_points
    segments = [
        lambda t: vaccination_rate_spline(t, *spline_model_params),
    ] + [
        lambda t, p=params: vaccination_rate_hill(t, *p)
        for params in hill_model_params
    ]

    # Start from the last segment and fold backwards
    rate = segments[-1](t)
    for cp, segment in zip(reversed(changepoints), reversed(segments[:-1])):
        rate = jnp.where(t <= cp, segment(t), rate)
    return rate


def vaccination_rate_spline(
    t: ArrayLike,
    vaccination_model_knot_locations: Array,
    vaccination_model_base_equations: Array,
    vaccination_model_knots: Array,
) -> Array:
    """
    Compute vaccination rates using spline model.

    Parameters
    ----------
    t : ArrayLike
        1D array of days.
    vaccination_model_knot_locations : Array
        Knot locations for spline.
    vaccination_model_base_equations : Array
        Base equations for spline.
    vaccination_model_knots : Array
        Knot values for spline.

    Returns
    -------
    Array
        Log vaccination rates, shape matches t.

    Notes
    -----
    Zero values are masked to remain zero after exponentiation.
    """
    vaccination_rates_log = evaluate_cubic_spline(
        t,
        vaccination_model_knot_locations,
        vaccination_model_base_equations,
        vaccination_model_knots,
    )
    dummy = jnp.zeros(vaccination_rates_log.shape)
    # print(jax.dtypes(dummy))
    # one of the side effects of exp() is setting exp(0) -> 1
    # we dont want this behavior in our vaccination rates obviously
    # so we find the locations of zero and save them to remask 0 -> 0 after exp() op
    zero_mask = jnp.where(vaccination_rates_log == 0, 0, 1)
    return zero_mask * jnp.exp(vaccination_rates_log + dummy)


def vaccination_rate_hill(
    t: ArrayLike,
    t_shifts: Array,
    n_batched: Array,
    t_h_batched: Array,
    scales_batched: Array,
) -> Array:
    """
    Compute vaccination rates using a Hill equation.

    Parameters
    ----------
    t : ArrayLike
        1D array of days.
    t_shifts : Array
        Shift for each Hill curve.
    n_batched : Array
        Hill coefficient for each curve.
    t_h_batched : Array
        Half-max day for each curve.
    scales_batched : Array
        Scale for each curve.

    Returns
    -------
    Array
        Vaccination rates, shape depends on input.

    Notes
    -----
    Uses derivative of Hill equation for uptake rate.
    """

    def hill_equation_dt(
        t: ArrayLike,
        n: ArrayLike,
        t_h: ArrayLike,
        scale: ArrayLike,
    ):
        """
        Derivative of the hill equation with respect to `t`, used
        to calculate uptake at day=t with shape parameters `n`, time of
        half cumulative uptake `t_h`, target end saturation date `t_end`
        and ending cumulative vaccination uptake `y_t_end`

        """
        t1 = t[:, jnp.newaxis]
        return (
            scale
            * ((n * (t1 ** (n - 1))) * (t_h**n))
            / ((t1**n + t_h**n) ** 2)
        )

    shift_coefs = jnp.heaviside(t - t_shifts, 0)
    t_shift = jnp.where(t - t_shifts < 0, 0, t - t_shifts)
    return jnp.nan_to_num(
        shift_coefs[:, jnp.newaxis]
        * hill_equation_dt(
            t_shift,
            n_batched,
            t_h_batched,
            scales_batched,
        )
    )


def seasonality_vaccine_reset(
    t: ArrayLike, vaccine_season_change: date, init_date: date
) -> ArrayLike:
    """
    Return multiplier for resetting seasonal vaccinations in summer.

    Parameters
    ----------
    t : ArrayLike
        1D array of days.
    vaccine_season_change : date
        Date of seasonal change.
    init_date : date
        Initial date for time alignment.

    Returns
    -------
    ArrayLike
        Multiplier, shape matches t.

    Notes
    -----
    Uses sine wave with period 730 days, but the even exponent ensures non-negativity and yearly peaks.
    Exponent 1000 ensures sharp transition; adjust if needed.
    """
    # outflow function must be positive if and only if
    # it is time to move people from seasonal bin back to max ordinal bin
    # use a sine wave that occurs once a year to achieve this effect
    peak_of_function = 182.5
    # shift this value using shift_t to align with config.VACCINATION_SEASON_CHANGE
    # such that outflow_fn(config.VACCINATION_SEASON_CHANGE) == 1.0 always
    shift_t = peak_of_function - (vaccine_season_change - init_date).days
    # raise to an even exponent to remove negatives,
    # pick 1000 since too high of a value likely to be stepped over by adaptive step size
    # divide by 730 so wave only occurs 1 per every 365 days
    # multiply by 2pi since we pass days as int
    return jnp.sin((2 * jnp.pi * (t + shift_t) / 730)) ** 1000
