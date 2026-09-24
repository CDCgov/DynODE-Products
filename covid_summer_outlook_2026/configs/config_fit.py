from datetime import date
from pathlib import Path
from typing import List

import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.distributions.constraints as const
import numpyro.distributions.transforms as tran
from dynode import (
    AgeBin,
    Bin,
    Compartment,
    DeterministicParameter,
    Dimension,
    LastStrainImmuneHistoryDimension,
    SimulationConfig,
    SolverParams,
    Strain,
    TransmissionParams,
    VaccinationDimension,
    WaneBin,
)
from initializers import CovidInitializer
from jax import Array
from utils import (
    get_region_name_and_pop_size,
    strain_interaction_to_cross_immunity,
)
from utils.data_utils import load_demographic_data

from .params import ExpandedParams, ObservationParams

# path to data folder
DATA_DIR = str(Path(__file__).parent.parent / "data")
OUTPUT_DIR = str(Path(__file__).parent.parent / "output")


class Covid5StrainConfig(SimulationConfig):
    save_path: str
    hosp_path: str
    var_prop_path: str
    vaccination_model_spline_path: str
    vaccination_model_hill_2425_path: str
    vaccination_model_hill_2526_path: str
    region_name: str
    region_population_size: int
    scenario_name: str
    demographic_data_path: str
    initialization_data_path: str
    num_age_groups: int
    num_strains: int
    num_waning_compartments: int
    max_vaccination_count: int
    vaccine_season_change: date

    def __init__(self, region_code: str, scenario_name: str = "base"):
        demographic_data_path = DATA_DIR + "/demographic-newcensus-data/"
        initialization_data_path = DATA_DIR + "/initialization-data/"
        save_path = OUTPUT_DIR
        hosp_path = DATA_DIR + "/hospitalization-data/20220219_to_20260404/"
        var_prop_path = DATA_DIR + "/variant-data/fitting-2602"
        vaccination_model_spline_path = (
            DATA_DIR + "/vaccination-data/2022_02_11_to_2024_04_27_v2"
        )

        vaccination_model_hill_2425_path = (
            DATA_DIR + "/vaccination-data/2602_2425_hill_eqx/"
        )

        vaccination_model_hill_2526_path = (
            DATA_DIR + "/vaccination-data/2602_2526_hill_eqx/"
        )

        vaccine_season_change = date(2023, 4, 25)

        region_name, region_population_size = get_region_name_and_pop_size(
            region_code, demographic_data_path
        )
        strains = self._get_strains()
        compartments = self._get_compartments(strains)
        contact_matrix = self._get_contact_matrix(
            demographic_data_path,
            region_name,
            compartments[0].dimensions[0].bins,
        )
        solver_params = self._get_solver_params()
        transmission_params = self._get_transmission_params(
            strains, contact_matrix=contact_matrix
        )
        observation_params = self._get_observation_params()

        # Construct a mock crossimmunity matrix for initializer
        cross_immunity_matrix = self._get_cross_immunity_matrix(
            transmission_params.strain_interactions
        )

        parameters = ExpandedParams(
            solver_params=solver_params,
            transmission_params=transmission_params,
            observation_params=observation_params,
        )
        # load the initializer with the current state.
        initializer = CovidInitializer(
            cross_immunity_matrix,
            compartments,
            len(parameters.transmission_params.strains),
            strains,
            demographic_data_path,
            initialization_data_path,
            region_name=region_name,
            population_size=region_population_size,
            initial_infections_base_rate=parameters.transmission_params.initial_infections_base_rate,
        )
        super().__init__(
            initializer=initializer,
            compartments=compartments,
            num_age_groups=len(compartments[0].dimensions[0]),
            max_vaccination_count=len(compartments[0].dimensions[2]) - 1,
            num_strains=len(strains),
            num_waning_compartments=len(compartments[0].dimensions[-1]),
            parameters=parameters,
            region_name=region_name,
            region_population_size=region_population_size,
            scenario_name=scenario_name,
            save_path=save_path,
            hosp_path=hosp_path,
            var_prop_path=var_prop_path,
            vaccination_model_spline_path=vaccination_model_spline_path,
            vaccination_model_hill_2425_path=vaccination_model_hill_2425_path,
            vaccination_model_hill_2526_path=vaccination_model_hill_2526_path,
            demographic_data_path=demographic_data_path,
            initialization_data_path=initialization_data_path,
            vaccine_season_change=vaccine_season_change,
        )

    def _get_strains(self) -> List[Strain]:
        """Define the strains and their parameter distributions."""
        omicron = Strain(
            strain_name="omicron",
            r0=dist.TransformedDistribution(
                dist.Beta(concentration1=50, concentration0=50),
                tran.AffineTransform(
                    loc=1.0, scale=3.0, domain=const.unit_interval
                ),
            ),
            infectious_period=4.0,
            exposed_to_infectious=2.0,
            vaccine_efficacy={0: 0, 1: 0.175, 2: 0.35, 3: 0.35},
            is_introduced=True,
            introduction_time=-88,
            introduction_percentage=0.040,
            introduction_scale=5,
            introduction_ages=[
                AgeBin(18, 49),
            ],
        )
        ba2ba5 = Strain(
            strain_name="BA2BA5",
            r0=dist.TransformedDistribution(
                dist.Beta(concentration1=50, concentration0=50),
                tran.AffineTransform(
                    loc=1.0, scale=3.0, domain=const.unit_interval
                ),
            ),
            infectious_period=4.0,
            exposed_to_infectious=2.0,
            vaccine_efficacy={0: 0, 1: 0.15, 2: 0.30, 3: 0.30},
            is_introduced=True,
            introduction_time=dist.TruncatedNormal(
                loc=20,
                scale=5,
                low=10,
                high=180,
            ),
            introduction_percentage=0.040,
            introduction_scale=15,
            introduction_ages=[
                AgeBin(18, 49),
            ],
        )
        xbb = Strain(
            strain_name="XBB",
            r0=dist.TransformedDistribution(
                dist.Beta(concentration1=50, concentration0=50),
                tran.AffineTransform(
                    loc=1.0, scale=3.0, domain=const.unit_interval
                ),
            ),
            infectious_period=4.0,
            exposed_to_infectious=2.0,
            vaccine_efficacy={0: 0, 1: 0.125, 2: 0.25, 3: 0.25},
            is_introduced=True,
            introduction_time=dist.TruncatedNormal(
                loc=230,
                scale=5,
                low=190,
                high=350,
            ),
            introduction_percentage=0.040,
            introduction_scale=15,
            introduction_ages=[
                AgeBin(18, 49),
            ],
        )
        jn1 = Strain(
            strain_name="JN1",
            r0=DeterministicParameter("jn1_r0"),
            infectious_period=4.0,
            exposed_to_infectious=2.0,
            vaccine_efficacy={0: 0, 1: 0, 2: 0, 3: 0},
            is_introduced=True,
            introduction_time=dist.TruncatedNormal(
                loc=640,
                scale=5,
                low=600,
                high=720,
            ),
            introduction_percentage=dist.TransformedDistribution(
                base_distribution=dist.Beta(25, 25),
                transforms=tran.AffineTransform(
                    loc=0.0, scale=0.05, domain=const.unit_interval
                ),
            ),
            introduction_scale=15,
            introduction_ages=[
                AgeBin(18, 49),
            ],
        )
        xec = Strain(
            strain_name="XEC",
            r0=DeterministicParameter("jn1_r0"),
            infectious_period=4.0,
            exposed_to_infectious=2.0,
            vaccine_efficacy={0: 0, 1: 0.0, 2: 0.0, 3: 0.0},
            is_introduced=True,
            introduction_time=dist.TruncatedNormal(
                loc=920,
                scale=10,
                low=880,
                high=1000,
            ),
            introduction_percentage=dist.TransformedDistribution(
                base_distribution=dist.Beta(25, 25),
                transforms=tran.AffineTransform(
                    loc=0.0, scale=0.05, domain=const.unit_interval
                ),
            ),
            introduction_scale=15,
            introduction_ages=[
                AgeBin(18, 49),
            ],
        )
        return [omicron, ba2ba5, xbb, jn1, xec]

    def _get_compartments(self, strains: list[Strain]) -> List[Compartment]:
        """Define the compartment structure given the strains expected in the model."""
        age_dimension = Dimension(
            name="age",
            bins=[
                AgeBin(0, 17),
                AgeBin(18, 49),
                AgeBin(50, 64),
                AgeBin(65, 99),
            ],
        )
        immune_history_dimension = LastStrainImmuneHistoryDimension(strains)
        vaccination_dimension = VaccinationDimension(
            max_ordinal_vaccinations=2, seasonal_vaccination=True
        )
        waning_dimension = Dimension(
            name="wane",
            bins=[
                WaneBin(name="W0", waiting_time=70, base_protection=1.0),
                WaneBin(name="W1", waiting_time=70, base_protection=1.0),
                WaneBin(name="W2", waiting_time=70, base_protection=1.0),
                # W3 can be used to model slower waning of IHR protection
                WaneBin(name="W3", waiting_time=100, base_protection=0.0),
                WaneBin(name="W4", waiting_time=jnp.inf, base_protection=0.0),
            ],
        )
        infecting_strain_dimension = Dimension(
            name="strain",
            bins=[Bin(name=strain.strain_name) for strain in strains],
        )
        s_compartment = Compartment(
            name="s",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                waning_dimension,
            ],
        )
        e_compartment = Compartment(
            name="e",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                infecting_strain_dimension,
            ],
        )
        i_compartment = Compartment(
            name="i",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                infecting_strain_dimension,
            ],
        )
        c_compartment = Compartment(
            name="c",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                waning_dimension,
                infecting_strain_dimension,
            ],
        )
        return [
            s_compartment,
            e_compartment,
            i_compartment,
            c_compartment,
        ]

    def _get_solver_params(self) -> SolverParams:
        return SolverParams(
            ode_solver_rel_tolerance=1e-7,
            ode_solver_abs_tolerance=1e-8,
            max_steps=5e6,
            # Days when our vaccination data switches from one curve to another.
            # Corresponds to "2024-04-21", "2025-05-26" and "2026-05-26".
            # Could potentially adjust these dates to be "nice", but not done
            # here yet, need to assess impact first.
            discontinuity_points=[800.0, 1200.0, 1565.0],
        )

    def _get_transmission_params(
        self, strains: list[Strain], contact_matrix: Array
    ) -> TransmissionParams:
        transmission_params = TransmissionParams(
            contact_matrix=contact_matrix,
            initial_infections_base_rate=0.04,  # multiplied by initial_infections_scale
            initial_infections_scale=dist.TruncatedNormal(
                loc=1.0, scale=0.1, low=0.5, high=2.5
            ),
            min_homologous_immunity=0.4,
            jn1_r0=dist.TransformedDistribution(
                dist.Beta(concentration1=50, concentration0=50),
                tran.AffineTransform(
                    loc=1.0, scale=3.0, domain=const.unit_interval
                ),
            ),
            strain_interactions_2_steps=dist.TransformedDistribution(
                base_distribution=dist.Beta(10, 40),
                transforms=tran.AffineTransform(
                    loc=0.5, scale=0.5, domain=const.unit_interval
                ),
            ),
            strain_interactions_3_steps=dist.Beta(12.5, 37.5),
            seasonality_amplitude=dist.TransformedDistribution(
                base_distribution=dist.Beta(120, 180),
                transforms=tran.AffineTransform(
                    loc=0.02, scale=0.2, domain=const.unit_interval
                ),
            ),
            seasonality_second_wave=dist.Beta(10, 10),
            seasonality_dominant_wave_day=dist.TruncatedNormal(
                loc=345,
                scale=5,
                low=325,
                high=365,
            ),
            seasonality_second_wave_offset=dist.TruncatedNormal(
                loc=140,
                scale=5,
                low=90,
                high=180,
            ),
            seasonality_wave_width=dist.Uniform(low=5, high=80),
            seasonality_second_wave_width=dist.Uniform(low=5, high=80),
            strains=strains,
            strain_interactions={
                "omicron": {
                    "omicron": 0.75,
                    "BA2BA5": 1.0,
                    "XBB": 1.0,
                    "JN1": 1.0,
                    "XEC": 1.0,
                },
                "BA2BA5": {
                    "omicron": dist.TransformedDistribution(
                        base_distribution=dist.Beta(20, 80),
                        transforms=tran.AffineTransform(
                            loc=0.5, scale=0.5, domain=const.unit_interval
                        ),
                    ),
                    "BA2BA5": 1.0,
                    "XBB": 1.0,
                    "JN1": 1.0,
                    "XEC": 1.0,
                },
                "XBB": {
                    "omicron": DeterministicParameter(
                        "strain_interactions_2_steps"
                    ),
                    "BA2BA5": dist.TransformedDistribution(
                        base_distribution=dist.Beta(50, 50),
                        transforms=tran.AffineTransform(
                            loc=0.5, scale=0.5, domain=const.unit_interval
                        ),
                    ),
                    "XBB": 1.0,
                    "JN1": 1.0,
                    "XEC": 1.0,
                },
                "JN1": {
                    "omicron": DeterministicParameter(
                        "strain_interactions_3_steps"
                    ),
                    "BA2BA5": DeterministicParameter(
                        "strain_interactions_2_steps"
                    ),
                    "XBB": dist.TransformedDistribution(
                        base_distribution=dist.Beta(50, 50),
                        transforms=tran.AffineTransform(
                            loc=0.5, scale=0.5, domain=const.unit_interval
                        ),
                    ),
                    "JN1": 1.0,
                    "XEC": 1.0,
                },
                "XEC": {
                    "omicron": DeterministicParameter(
                        "strain_interactions_3_steps"
                    ),
                    "BA2BA5": DeterministicParameter(
                        "strain_interactions_3_steps"
                    ),
                    "XBB": dist.Beta(50, 50),
                    "JN1": dist.TransformedDistribution(
                        base_distribution=dist.Beta(50, 50),
                        transforms=tran.AffineTransform(
                            loc=0.5, scale=0.5, domain=const.unit_interval
                        ),
                    ),
                    "XEC": dist.TransformedDistribution(
                        base_distribution=dist.Beta(80, 20),
                        transforms=tran.AffineTransform(
                            loc=0.5, scale=0.5, domain=const.unit_interval
                        ),
                    ),
                },
            },
        )
        return transmission_params

    def _get_observation_params(self):
        return ObservationParams(
            ihr_3=dist.Beta(concentration1=50, concentration0=950),
            ihr_mult_0=dist.Beta(concentration1=4.4, concentration0=208.6),
            ihr_mult_1=dist.Beta(concentration1=32.4, concentration0=633.3),
            ihr_mult_2=dist.Beta(concentration1=57.7, concentration0=353.8),
            ihr_immune_mult=dist.Beta(concentration1=600, concentration0=1800),
            # IHR multipliers for strains in group > 0
            # Group 2 = g1 * g2, Group 3 = g1 * g2 * g3
            ihr_mult_strain_g1=dist.Beta(concentration1=95, concentration0=5),
            ihr_mult_strain_g2=dist.Beta(concentration1=80, concentration0=20),
            ihr_mult_strain_g3=dist.Uniform(low=0.7, high=1.0),
            # IHR protection defaults to "5 bins" behaviour
            ihr_wane=[DeterministicParameter("ihr_immune_mult")] * 4 + [1.0],
            ihr_strains_groups=[0, 0, 0, 1, 2],
        )

    def _get_contact_matrix(
        self, demographic_data_path, region_name, age_bins: List[AgeBin]
    ) -> Array:
        min_age = age_bins[0].min_value
        # A0_17, A18_49, A50_64, A65_99
        # [0, 18, 50, 65]
        age_limits = [b.min_value for b in age_bins]
        return jnp.array(
            load_demographic_data(
                demographic_data_path,
                [region_name],
                len(age_bins),
                min_age,
                age_limits,
            )[region_name]["avg_CM"]
        )

    def _get_cross_immunity_matrix(self, si):
        """Create a 'mock' cross immunity matrix for initializer

        In this config, the only values that matter are the zeros for people
        with no immune history, and the Omicron x Omicron immunity. This
        circumvents the issue that many of the strain interactions are priors
        and not real values.
        """
        strain_interaction_matrix = jnp.array(
            [
                [v if isinstance(v, float) else 0.75 for v in si[i].values()]
                for i in si.keys()
            ]
        )
        cross_immunity_matrix = strain_interaction_to_cross_immunity(
            len(si.keys()), strain_interaction_matrix
        )

        return cross_immunity_matrix
