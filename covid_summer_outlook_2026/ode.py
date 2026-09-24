"""Class and functions related to the actual ODE structure, and the
translation of dynode config to actual ODE parameters"""

from datetime import date
from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import numpyro
from configs import Covid5StrainConfig
from dynode import AbstractODEParams, sample_then_resolve
from dynode.typing import CompartmentGradients, CompartmentState
from funcs import (
    external_i,
    seasonality,
    seasonality_vaccine_reset,
    vaccination_rates,
)
from jaxtyping import ArrayLike
from utils.helpers import (
    strain_interaction_to_cross_immunity,
    vectorize_objects,
)


@chex.dataclass
class IntroductionParameters:
    introduction_times: chex.ArrayDevice
    introduction_scales: chex.ArrayDevice
    introduction_percentages: chex.ArrayDevice
    introduction_age_groups: chex.ArrayDevice


@chex.dataclass(
    frozen=False,  # DONT TOUCH THIS BEN
    static_keynames=[
        "idx",
        "init_date",
        "num_age_groups",
        "num_strains",
        "num_waning_compartments",
        "max_vaccination_count",
        "vaccine_season_change",
        "vax_discontinuity_points",
    ],
)
class CovidODEParams(AbstractODEParams):
    """A collection of parameters that is passed to `seip_ode`.

    This is only used to define the data type for each attributes.
    Nevertheless, this is compulsory for type check purpose.
    """

    idx: SimpleNamespace
    init_date: date
    num_age_groups: int
    num_strains: int
    max_vaccination_count: int
    num_waning_compartments: int
    vaccine_season_change: date
    population_counts_by_age: chex.ArrayDevice
    # scaling initial infections
    initial_infections_scale: chex.ArrayDevice
    # for protection from vaccination and prior infection
    crossimmunity_matrix: chex.ArrayDevice
    vaccine_eff_matrix: chex.ArrayDevice
    # for waning
    waning_rates: chex.ArrayDevice
    waning_protections: chex.ArrayDevice
    min_homologous_immunity: chex.ArrayDevice
    # for force of infection calculations
    contact_matrix: chex.ArrayDevice
    beta: chex.ArrayDevice
    sigma: chex.ArrayDevice
    gamma: chex.ArrayDevice
    # for strain introduction
    introduction_times: chex.ArrayDevice
    introduction_scales: chex.ArrayDevice
    introduction_percentages: chex.ArrayDevice
    introduction_age_groups: chex.ArrayDevice
    # for seasonality
    seasonality_amplitude: chex.ArrayDevice
    seasonality_second_wave: chex.ArrayDevice
    seasonality_dominant_wave_day: chex.ArrayDevice
    seasonality_second_wave_day: chex.ArrayDevice
    seasonality_wave_width: chex.ArrayDevice
    seasonality_second_wave_width: chex.ArrayDevice
    # for vaccination with hill and spline methods
    vax_spline_model_params: tuple
    vax_hill_model_params: list[tuple]
    vax_discontinuity_points: list[int]


def get_ode_parameters(
    config: Covid5StrainConfig, vax_data: dict[str, tuple]
) -> CovidODEParams:
    """Construct ODE parameters from configuration and vaccination data.

    Samples transmission parameters from prior distributions and constructs
    a complete set of ODE parameters to be passed into `seip_ode`.

    Parameters
    ----------
    config : Covid7StrainBaseConfig
        Configuration object containing model structure, compartments, and
        parameter specifications including transmission parameters and strain
        definitions. Doesn't need to be Covid7StrainBaseConfig, any similarly
        structured SimulationConfig with similar parameters set would work.
    vax_data : dict[str, tuple]
        Dictionary containing vaccination model parameters with keys:
        - 'spline_model_params': Tuple of spline parameters for vaccination rates
        - 'hill_model_params': List of tuples for Hill model parameters

    Returns
    -------
    CovidODEParams
        Dataclass containing all parameters needed for ODE as specified in
        the class definition
    """
    transmission_params = config.parameters.transmission_params
    # This is now an object with sampled distributions within, all lists
    # should be converted to jax arrays by the ODEParams.
    transmission_params = sample_then_resolve(transmission_params)

    # Initial state and waning stuff
    initial_infections_scale = transmission_params.initial_infections_scale
    s = config.get_compartment("s")
    num_age_groups = config.num_age_groups
    waning_rates = jnp.array(
        [
            1 / wane_bin.waiting_time
            for wane_bin in s.dimensions[s.idx.wane].bins
        ]
    )
    waning_protections = jnp.array(
        [
            wane_bin.base_protection
            for wane_bin in s.dimensions[s.idx.wane].bins
        ]
    )

    # Combine all vax efficacy across strains into a single matrix
    vaccine_eff_matrix = jnp.array(
        [
            list(strain.vaccine_efficacy.values())
            for strain in transmission_params.strains
        ]
    )

    # Convert the dict strain_interactions to a matrix
    strain_interactions_matrix = jnp.array(
        [
            list(strain_interact_targets.values())
            for strain_interact_targets in transmission_params.strain_interactions.values()
        ]
    )

    # Convert to cross immunity which is based on the immune histories.
    crossimmunity_matrix = strain_interaction_to_cross_immunity(
        num_strains=config.num_strains,
        strain_interactions=strain_interactions_matrix,
    )

    # Beta, gamma, sigma
    strain_r0s = jnp.array(
        vectorize_objects(transmission_params.strains, targets="r0")
    )
    infectious_periods = jnp.array(
        vectorize_objects(
            transmission_params.strains, targets="infectious_period"
        )
    )
    exposed_to_infectious = jnp.array(
        vectorize_objects(
            transmission_params.strains, targets="exposed_to_infectious"
        )
    )
    beta = strain_r0s / infectious_periods
    sigma = 1 / exposed_to_infectious
    gamma = 1 / infectious_periods

    # Introduction
    ## need to keep this line so SimulationDate converts itself to a float.
    transmission_params.strains[0].introduction_time += 0
    introduction_times = jnp.array(
        vectorize_objects(
            transmission_params.strains, targets="introduction_time"
        )
    )
    introduction_scales = jnp.array(
        vectorize_objects(
            transmission_params.strains, targets="introduction_scale"
        )
    )
    introduction_percentages = jnp.array(
        vectorize_objects(
            transmission_params.strains, targets="introduction_percentage"
        )
    )
    introduction_age_groups = jnp.array(
        vectorize_objects(
            transmission_params.strains,
            targets="introduction_ages_mask_vector",
        )
    )

    # Seasonality
    seasonality_second_wave_day = numpyro.deterministic(
        "seasonality_second_wave_day",
        transmission_params.seasonality_dominant_wave_day
        - transmission_params.seasonality_second_wave_offset,
    )
    return CovidODEParams(
        idx=config.idx,
        init_date=config.initializer.initialize_date,
        num_age_groups=num_age_groups,
        num_strains=config.num_strains,
        num_waning_compartments=config.num_waning_compartments,
        max_vaccination_count=config.max_vaccination_count,
        vaccine_season_change=config.vaccine_season_change,
        population_counts_by_age=config.initializer._population_counts_by_age,
        initial_infections_scale=initial_infections_scale,
        crossimmunity_matrix=crossimmunity_matrix,
        vaccine_eff_matrix=vaccine_eff_matrix,
        waning_rates=waning_rates,
        waning_protections=waning_protections,
        min_homologous_immunity=transmission_params.min_homologous_immunity,
        contact_matrix=transmission_params.contact_matrix,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        introduction_times=introduction_times,
        introduction_scales=introduction_scales,
        introduction_percentages=introduction_percentages,
        introduction_age_groups=introduction_age_groups,
        seasonality_amplitude=transmission_params.seasonality_amplitude,
        seasonality_second_wave=transmission_params.seasonality_second_wave,
        seasonality_dominant_wave_day=transmission_params.seasonality_dominant_wave_day,
        seasonality_second_wave_day=seasonality_second_wave_day,
        seasonality_wave_width=transmission_params.seasonality_wave_width,
        seasonality_second_wave_width=transmission_params.seasonality_second_wave_width,
        vax_spline_model_params=vax_data["spline_model_params"],
        vax_hill_model_params=vax_data["hill_model_params"],
        vax_discontinuity_points=config.parameters.solver_params.discontinuity_points,
    )


@jax.jit
def seip_ode(
    t: ArrayLike, state: CompartmentState, p: CovidODEParams
) -> CompartmentGradients:
    """Calculate the change of each compartments at a time point

    This is the ODE specifying the flow of compartments. Specifically, it
    consists of calculations and equations to describe the in- and out-flow
    for each compartments across all strata at time `t`

    Parameters
    ----------
    t: ArrayLike
        Time point, consist of float usually
    state: CompartmentState
        Tuple of arrays reflecting the state (values) of each compartment
        and its sub-compartments.
    p: CovidODEParams
        Set of parameters that are needed for ODE, usually related to
        transmissions, immunity, seasonal forcing, strain introductions and
        vaccination.

    Returns
    -------
    CovidODEParams
        Basically a tuple of arrays that specify the changes in each
        compartment.
    """
    s, e, i, c = state
    if any([not isinstance(compartment, jax.Array) for compartment in state]):
        raise TypeError(
            "Please pass jax.numpy.array instead of np.array to ODEs"
        )
    ds, de, di, dc = (
        jnp.zeros(s.shape),
        jnp.zeros(e.shape),
        jnp.zeros(i.shape),
        jnp.zeros(c.shape),
    )
    seasonality_coef = seasonality(
        t,
        p.seasonality_amplitude,
        p.seasonality_second_wave,
        p.seasonality_dominant_wave_day,
        p.seasonality_second_wave_day,
        p.seasonality_wave_width,
        p.init_date,
        seasonality_second_wave_width=p.seasonality_second_wave_width,
    )
    # Calculating FOI (here it's beta*I/N)
    # Include external forcing ("introduction")
    force_of_infection = (
        (
            p.beta
            * seasonality_coef
            * jnp.einsum(
                "ab,bijk->ak",
                p.contact_matrix,
                i
                + external_i(
                    t,
                    p.introduction_times,
                    p.introduction_scales,
                    p.introduction_percentages,
                    p.introduction_age_groups,
                    p.population_counts_by_age,
                    i.shape,
                ),
            )
        ).transpose()
        / p.population_counts_by_age
    ).transpose()  # (age_group, strain)

    # FOI x S (factoring in immunity)
    foi_suscept = jnp.array(get_foi_suscept(p, force_of_infection))

    # Using vmap for this for loop. We select the force of infection
    # for each strain, and calculated the number of susceptibles it exposes
    # summing over wane bin since `e` has no waning bin.
    # OLD FOR LOOP FOR INTERPRETABILITY
    # for strain in range(p.NUM_STRAINS):
    #     exposed_s = s * foi_suscept[strain]
    #     de = de.at[:, :, :, strain].add(
    #         jnp.sum(exposed_s, axis=-1)
    #     )
    #     ds = jnp.add(ds, -exposed_s)
    exposed_s = jnp.moveaxis(
        jax.vmap(
            lambda s, foi_suscept: s * foi_suscept,
            in_axes=(None, 0),
        )(s, foi_suscept),
        0,
        -1,
    )  # returns shape (s.shape..., p.num_strain)

    # S has waning as last dimension, e has infected strain as last dim
    # the last two dimensions of `exposed_s` are `wane` and `strain`,
    # so lets sum over them to get the expected shape for each
    de = de + jnp.sum(exposed_s, axis=-2)  # Remove wane so matches e.shape
    ds = ds - jnp.sum(exposed_s, axis=-1)  # Remove strain so matches s.shape
    dc = exposed_s  # Add to cumulative

    # e and i shape remain same, just multiplying by a constant.
    de_to_i = e * p.sigma  # exposure -> infectious
    di_to_w0 = i * p.gamma  # infectious -> new_immune_state
    di = jnp.add(de_to_i, -di_to_w0)
    de = jnp.add(de, -de_to_i)

    # Go through all combinations of immune history and exposing strain
    # calculate new immune history after recovery, place them there.
    # immune history = most recent infected strain
    for strain in range(p.num_strains):
        ds = ds.at[:, strain + 1, :, 0].add(
            jnp.sum(di_to_w0[:, :, :, strain], axis=1)
        )

    # Waning
    # Last w group doesn't wane, waning_rates enforces a 0 at the end
    waning_array = jnp.zeros(s.shape).at[:, :, :].add(p.waning_rates)
    s_waned = waning_array * s
    ds = ds.at[:, :, :, 1:].add(s_waned[:, :, :, :-1])
    ds = ds.at[:, :, :, :-1].add(-s_waned[:, :, :, :-1])

    # Vaccination
    # Vaccination updates the vax column and also moves all to w0.
    # eg: diagonal movement from 1 shot in 4th waning compartment to 2 shots 0
    # waning compartment      s[:, 0, 1, 3] -> s[:, 0, 2, 0]
    # Input vaccination rate is per entire population, need to update to per
    # compartments first
    vax_rates = vaccination_rates(
        t,
        vax_discontinuity_points=p.vax_discontinuity_points,
        spline_model_params=p.vax_spline_model_params,
        hill_model_params=p.vax_hill_model_params,
    )
    vax_totals = vax_rates * p.population_counts_by_age[:, None]
    vax_status_counts = jnp.sum(
        s,
        axis=(p.idx.s.hist, p.idx.s.wane),  # (age, wane)
    )  # vax count per age and vax status
    updated_vax_rates = vax_totals / (
        vax_status_counts + 1e-10
    )  # Prevent div by zero
    updated_vax_rates = jnp.minimum(
        updated_vax_rates, 0.95
    )  # Prevent moving too much people out

    # Assuming that people who received 2 or more doses wouldn't get
    # additional booster too soon, i.e., when they were still within the first
    # waning compartment
    vax_counts = s * updated_vax_rates[:, jnp.newaxis, :, jnp.newaxis]
    vax_counts = vax_counts.at[:, :, p.max_vaccination_count, 0].set(0)
    vax_gained = jnp.sum(vax_counts, axis=(-1))
    ds = ds.at[:, :, p.max_vaccination_count, 0].add(
        vax_gained[:, :, p.max_vaccination_count]
    )
    ds = ds.at[:, :, 1 : (p.max_vaccination_count) + 1, 0].add(
        vax_gained[:, :, 0 : p.max_vaccination_count]
    )
    ds = ds - vax_counts

    # Seasonal vaccination outflow
    # Moving people from seasonal booster (bin 3) back t 2+ dose (bin 2),
    # usually by end of June, to easily ensure that people are not double
    # boosted for each season.
    seasonal_vaccination_outflow = seasonality_vaccine_reset(
        t, p.vaccine_season_change, p.init_date
    )
    # flow seasonal_vaccination_outflow of seasonal vaxers back to max
    # "ordinal" tier
    ds = ds.at[:, :, p.max_vaccination_count - 1, :].add(
        seasonal_vaccination_outflow * s[:, :, p.max_vaccination_count, :]
    )
    # remove these people from the seasonal vaccination tier
    ds = ds.at[:, :, p.max_vaccination_count, :].add(
        -seasonal_vaccination_outflow * s[:, :, p.max_vaccination_count, :]
    )
    # do the same process for e and i compartments
    de = de.at[:, :, p.max_vaccination_count - 1, :].add(
        seasonal_vaccination_outflow * e[:, :, p.max_vaccination_count, :]
    )
    de = de.at[:, :, p.max_vaccination_count, :].add(
        -seasonal_vaccination_outflow * e[:, :, p.max_vaccination_count, :]
    )
    di = di.at[:, :, p.max_vaccination_count - 1, :].add(
        seasonal_vaccination_outflow * i[:, :, p.max_vaccination_count, :]
    )
    di = di.at[:, :, p.max_vaccination_count, :].add(
        -seasonal_vaccination_outflow * i[:, :, p.max_vaccination_count, :]
    )

    return (ds, de, di, dc)


def get_foi_suscept(p: CovidODEParams, force_of_infection):
    """
    Calculate the force of infections experienced by the susceptibles, _after_
    factoring their immunity (infection history, vaccination and waning status).

    Parameters
    ----------
    `p` : CovidODEParams
        a CovidODEParams object.
    `force_of_infection`: jnp.array
        an array of (NUM_AGE_GROUPS, NUM_STRAINS) that quantifies the force of
        infection experienced by age group by strain.

    Returns
    -------
    jnp.array:
        an array of FOI after factoring in immunity, by the shape of
        (NUM_STRAINS, s.shape)

    Notes
    -----
    The FOI input implemented in `seip_ode` is beta*I/N. Although using beta*I
    would still work here.
    """
    foi_suscept = []
    for strain in range(p.num_strains):
        force_of_infection_strain = force_of_infection[
            :, strain
        ]  # (num_age_groups,)

        crossimmunity_matrix = p.crossimmunity_matrix[strain, :]
        vax_efficacy_strain = p.vaccine_eff_matrix[strain, :]
        initial_immunity = 1 - jnp.einsum(
            "j, k",
            1 - crossimmunity_matrix,
            1 - vax_efficacy_strain,
        )
        # renormalize the waning curve to have minimum of `final_immunity` after full waning
        # and maximum of `initial_immunity` right after rec
        final_immunity = (
            jnp.ones(shape=initial_immunity.shape)
            * crossimmunity_matrix[:, jnp.newaxis]
            * p.min_homologous_immunity
        )

        waned_immunity_baseline = jnp.einsum(
            "jk,l",
            initial_immunity,
            p.waning_protections,
        )
        # find the lower bound of immunity for a homologous exposure against this challenging strain
        waned_immunity_min = (1 - waned_immunity_baseline) * final_immunity[
            :, :, jnp.newaxis
        ]
        waned_immunity = waned_immunity_baseline + waned_immunity_min
        foi_suscept_strain = jnp.einsum(
            "i, jkl", force_of_infection_strain, 1 - waned_immunity
        )
        foi_suscept.append(foi_suscept_strain)

    return foi_suscept
