import json
import os
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
from dynode import Initializer
from dynode.config import Compartment, Strain
from dynode.utils import logger
from pydantic import ConfigDict
from utils import (
    load_age_demographics,
    load_demographic_data,
)


class CovidInitializer(Initializer):
    """A specialized initializer for a 6 strain covid model for Feb 11th 2022."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    initial_infections: float

    def __init__(
        self,
        cross_immunity_matrix: jax.Array,
        compartments: list[Compartment],
        num_strains: int,
        strain_data: list[Strain],
        demographic_data_path: str,
        initialization_data_path: str,
        region_name: str,
        population_size: float,
        initial_infections_base_rate: float,
    ):
        # round to 3 significant figures
        initial_infections = int(
            float("%.3g" % (population_size * initial_infections_base_rate))
        )
        # round to 3 significant figures
        population_size = int(float("%.3g" % population_size))
        description = (
            str(num_strains) + " strain initializer for Feb 11th 2022."
        )
        super().__init__(
            description=description,
            initialize_date=date(2022, 2, 11),
            population_size=population_size,
            initial_infections=initial_infections,
        )
        self._sero_initializer = CovidSeroInitializer(
            strain_data,
            demographic_data_path,
            initialization_data_path,
            region_name,
            compartments,
            population_size,
            cross_immunity_matrix,
            initial_infections,
        )
        initial_state = self._sero_initializer.INITIAL_STATE
        s, e, i, _ = initial_state
        population_counts_by_age = (
            jnp.sum(s, axis=tuple(range(1, s.ndim)))  # sum over all but age
            + jnp.sum(e, axis=tuple(range(1, e.ndim)))
            + jnp.sum(i, axis=tuple(range(1, i.ndim)))
        )
        self._population_counts_by_age = population_counts_by_age
        self._initial_state = add_c_compartment(initial_state)

    def get_initial_state(self, **kwargs):
        """Load the initial state via serology data"""
        initial_infection_scale = kwargs.get("initial_infections_scale", None)
        if initial_infection_scale is None:
            raise ValueError(
                "initial_infections_scale must be provided to get_initial_state"
            )
        initial_state = self._initial_state
        # negative if scale_factor < 1.0
        e_new = initial_state[1] * initial_infection_scale
        i_new = initial_state[2] * initial_infection_scale
        age_stratified_infections_delta = jnp.sum(
            e_new + i_new,
            axis=(1, 2, 3),
        ) - jnp.sum(
            initial_state[1] + initial_state[2],
            axis=(1, 2, 3),
        )
        age_stratified_infections_scaling = 1 - (
            age_stratified_infections_delta
            / jnp.sum(
                initial_state[0],
                axis=(1, 2, 3),
            )
        )
        # multiplying E and I by the same scale_factor preserves their relative ratio
        s_new = (
            age_stratified_infections_scaling[:, None, None, None]
            * initial_state[0]
        )
        return (
            s_new,
            e_new,
            i_new,
            initial_state[-1],
        )


def add_c_compartment(initial_state):
    """Adding C compartment to the initial state

    C has 5 dimensions: (age, hist, vax, wane, strain). So we use S and
    I's dimensions to make C.
    """
    c_new_shape = list(initial_state[0].shape)
    c_new_shape.append(initial_state[2].shape[3])
    c_new = jnp.zeros(tuple(c_new_shape))

    initial_state = (
        initial_state[0],
        initial_state[1],
        initial_state[2],
        c_new,
    )
    return initial_state


class CovidSeroInitializer:
    """A Covid specific initializer class using serology input data to stratify immunity."""

    def __init__(
        self,
        strain_data: list[Strain],
        demographic_data_path: str,
        initialization_data_path: str,
        region_name,
        compartments,
        population_size,
        cross_immunity_matrix,
        initial_infections,
    ):
        """Create an initializer using generated input (from serological
        data).

        Parameters
        ----------
        strain_data : list
            List of Strain used in the config
        demographic_data_path : str
            Path to demographic data
        initialization_data_path : str
            Path to initialization data
        region_name : str
            Full name of state, or hhs regions, e.g., Georgia or HHS4
        compartments : list[Compartment]
            List of Compartment used in config
        population_size : float
            Population size for the corresponding region
        cross_immunity_matrix : np.ndarray
            2-Dimensional array specifying the pairwise crossimmunity values
        initial_infections : int
            Number of infected (E or I) initially
        """
        if not hasattr(self, "INITIAL_POPULATION_FRACTIONS"):
            age_limits = [
                b.min_value for b in compartments[0].dimensions[0].bins
            ]
            self.INITIAL_POPULATION_FRACTIONS = (
                self.load_initial_population_fractions(
                    demographic_data_path, region_name, age_limits
                )
            )

        self.POPULATION = population_size * self.INITIAL_POPULATION_FRACTIONS
        # self.POPULATION.shape = (NUM_AGE_GROUPS,)
        if not hasattr(self, "INIT_IMMUNE_HISTORY"):
            self.INIT_IMMUNE_HISTORY = (
                self.load_immune_history_via_serological_data(
                    len(strain_data), region_name, initialization_data_path
                )
            )
        # self.INIT_IMMUNE_HISTORY.shape = (age, hist, num_vax, waning)
        if not hasattr(self, "CONTACT_MATRIX"):
            self.CONTACT_MATRIX = self.load_contact_matrix(
                demographic_data_path, region_name, compartments
            )
        # NEED TO WORK ON
        if not hasattr(self, "CROSSIMMUNITY_MATRIX"):
            self.CROSSIMMUNITY_MATRIX = cross_immunity_matrix
        # stratify initial infections appropriately across age, hist, vax counts
        if not (
            hasattr(self, "INIT_INFECTIOUS_DIST")
            and hasattr(self, "INIT_EXPOSED_DIST")
        ):
            wane_protection = np.array(
                [
                    b.base_protection
                    for b in compartments[0].dimensions[-1].bins
                ]
            )

            self.INIT_INFECTION_DIST = (
                self.load_initial_infection_dist_via_contact_matrix(
                    strain_data,
                    wane_protection,
                )
            )
            self.INIT_INFECTIOUS_DIST = (
                self.get_initial_infectious_distribution(strain_data)
            )
            self.INIT_EXPOSED_DIST = self.get_initial_exposed_distribution(
                strain_data
            )

        # load initial state using
        # INIT_IMMUNE_HISTORY, INIT_INFECTIOUS_DIST, and INIT_EXPOSED_DIST
        self.INITIAL_STATE = self.load_initial_state(initial_infections)

    def load_initial_population_fractions(
        self, demographic_data_path, region_name, age_limits
    ) -> np.ndarray:
        """Load age demographics for the specified region.

        Parameters
        ----------
        demographic_data_path : str
            Path to demographic data
        region_name : str
            Full name of state, or hhs regions, e.g., Georgia or HHS4
        age_limits : list
            A list of lower bound of each age group

        Returns
        -------
        numpy.ndarray
            Proportion of the total population that falls into each age group.
        """
        logger.debug(
            "Creating populations_path based on demographic_data_path."
        )

        populations_path = (
            demographic_data_path + "population_rescaled_age_distributions/"
        )

        logger.debug(f"Set populations path as {populations_path}.")
        logger.debug("Returning values from utils.load_age_demographics()")
        return load_age_demographics(
            populations_path, [region_name], age_limits
        )[region_name]

    def load_initial_state(
        self,
        initial_infections: float,
    ) -> tuple:
        """Disperse initial infections across infectious and exposed compartments.

        Parameters
        ----------
        initial_infections : float
            the number of infections to disperse between infectious and exposed
            compartments.

        Returns
        -------
        tuple
            a tuple of len 4 representing the S, E, I, and C compartment
        """
        # create population distribution with INIT_INFECTIOUS_DIST then sum by age
        initial_infectious_count = (
            initial_infections * self.INIT_INFECTIOUS_DIST
        )
        initial_infectious_count_ages = jnp.sum(
            initial_infectious_count,
            axis=(
                1,
                2,
                3,
            ),
        )
        # create population distribution with INIT_EXPOSED_DIST then sum by age
        initial_exposed_count = initial_infections * self.INIT_EXPOSED_DIST
        initial_exposed_count_ages = jnp.sum(
            initial_exposed_count,
            axis=(
                1,
                2,
                3,
            ),
        )
        # susceptible / partial susceptible =
        # Total population - infected_count - exposed_count
        initial_susceptible_count = (
            self.POPULATION
            - initial_infectious_count_ages
            - initial_exposed_count_ages
        )[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] * self.INIT_IMMUNE_HISTORY
        # cumulative count always starts at zero
        return (
            initial_susceptible_count,  # s
            initial_exposed_count,  # e
            initial_infectious_count,  # i
            jnp.zeros(initial_exposed_count.shape),  # c
        )

    def load_immune_history_via_serological_data(
        self, num_strains, region_name, initialization_data_path
    ) -> np.ndarray:
        """Load the serology init file and adjusts based on desired strains.

        Parameters
        ----------
        num_strains : int
            Number of strains in the model
        region_name : str
            Full name of state, or hhs regions, e.g., Georgia or HHS4
        initialization_data_path : str
            Path to initialization data

        Returns
        -------
        np.ndarray
            The initial immune history of susceptibles accounting for the
            various dimensions
        """
        # Loading initialization bins generated by other code.
        filename = os.path.join(
            initialization_data_path,
            str(region_name).replace(" ", "_") + "_binned_initalization.json",
        )
        js = json.load(open(filename, "r"))
        sero_matrix = np.array(js["S"])
        # Removing strains if less strains desired.
        if sero_matrix.shape[1] > (num_strains + 1):
            diff = (num_strains + 1) - sero_matrix.shape[1]
            sero_matrix = sero_matrix[:, :diff, :, :]
        # Adding zeros and new column if more strain storage needed.
        if sero_matrix.shape[1] < (num_strains + 1):
            diff = (num_strains + 1) - sero_matrix.shape[1]
            sero_matrix = np.pad(
                sero_matrix, ((0, 0), (0, diff), (0, 0), (0, 0))
            )

        # Normalize sero_matrix by age (so each age group sum to 1)
        age_sum = sero_matrix.sum(axis=(1, 2, 3))
        sero_matrix = sero_matrix / age_sum[:, None, None, None]

        return sero_matrix

    def load_initial_infection_dist_via_contact_matrix(
        self, strain_data, wane_protection
    ) -> np.ndarray:
        """Estimates the demographics and immune histories of initial infections.

        Based on current susceptible population level of protection as well as
        the contact matrix, estimates the distribution of initial infections
        across all stratification dimension.

        Parameters
        ----------
        strain_data : list[Strain]
            List of Strain as specified in config
        wane_protection : np.ndarray[float]
            Numpy array of immunity protection for each waning bin

        Returns
        -------
        np.ndarray
            Array describing the proportion of new infections falling under
            each stratification of the compartment.
        """
        # use contact matrix to get the infection age distributions
        eig_data = np.linalg.eig(self.CONTACT_MATRIX)
        max_index = np.argmax(eig_data[0])
        infection_age_dists = abs(eig_data[1][:, max_index])
        # normalize so they sum to 1
        infection_age_dists = infection_age_dists / np.sum(infection_age_dists)
        omicron_index = next(
            i for i, s in enumerate(strain_data) if s.strain_name == "omicron"
        )
        # we will now weight the population in each strata of the S compartment by their
        # relative susceptibility to the incoming omicron strain
        crossimmunity_matrix = self.CROSSIMMUNITY_MATRIX[omicron_index, :]
        # p.vax_susceptibility_strain.shape = (MAX_VAX_COUNT,)
        vax_efficacy_strain = np.array(
            list(strain_data[0].vaccine_efficacy.values())
        )
        # susceptibility by hist and vax status
        initial_immunity = 1 - np.einsum(
            "j, k",
            1 - crossimmunity_matrix,
            1 - vax_efficacy_strain,
        )
        # susceptibility by hist, vax, and wane status
        waned_immunity = np.einsum(
            "jk,l", initial_immunity, wane_protection
        )  # shape=(hist, vax, wane)
        # for each age bin, multiply the pop counts of each
        # hist, vax, wane strata by the protect afforded by that strata
        pop_sizes_with_relative_immunity = (
            1 - waned_immunity[np.newaxis, ...]
        ) * (
            self.POPULATION[:, np.newaxis, np.newaxis, np.newaxis]
            * self.INIT_IMMUNE_HISTORY
        )  # shape(age, hist, vax, wane)

        # normalize so each age bin in pop_sizes_with_relative_immunity sums to 1
        pop_sizes_with_relative_immunity = (
            pop_sizes_with_relative_immunity
            / np.sum(
                pop_sizes_with_relative_immunity, axis=(1, 2, 3), keepdims=True
            )
        )
        # bring back the infection age distributions from the contact matrix eig value
        # since sum(infection_age_dists) = 1, it is now the case that sum(pop_sizes_with_relative_immunity) = 1
        pop_sizes_with_relative_immunity = (
            infection_age_dists[:, np.newaxis, np.newaxis, np.newaxis]
            * pop_sizes_with_relative_immunity
        )
        # sum across wane axis since we dont track that for E or I compartments
        infection_dist = np.sum(pop_sizes_with_relative_immunity, axis=-1)
        infection_dist = np.repeat(
            infection_dist[:, :, :, np.newaxis],
            repeats=len(strain_data),
            axis=-1,
        )
        # set all strains that arent omicron to zero, since we repeated them 3 times above
        infection_dist[
            :,
            :,
            :,
            list(
                set(list(range(infection_dist.shape[-1])))
                - set([omicron_index])
            ),
        ] = 0
        # disperse infections across E and I compartments by exposed_to_infectous_ratio
        return infection_dist

    def get_initial_infectious_distribution(self, strain_data) -> np.ndarray:
        """Get actively infectious proportion of initial infections.

        Returns
        -------
        np.ndarray
            actively infectious compartment as a proportion of
            `INIT_INFECTION_DIST`.

        Raises
        ------
        RuntimeError
            if self.INIT_INFECTION_DIST does not exist. Usually created
            via self.load_initial_infection_dist_via_contact_matrix()
        """
        if not hasattr(self, "INIT_INFECTION_DIST"):
            raise RuntimeError(
                "this function requires `self.INIT_INFECTION_DIST`"
                "set if via load_initial_infection_dist_via_contact_matrix()"
                "before calling this function"
            )
        # use relative wait times in each compartment to get distribution
        #  of infections across infected vs exposed compartments
        exposed_to_infectous_ratio = strain_data[0].exposed_to_infectious / (
            strain_data[0].exposed_to_infectious
            + strain_data[0].infectious_period
        )
        return (1 - exposed_to_infectous_ratio) * self.INIT_INFECTION_DIST

    def get_initial_exposed_distribution(self, strain_data) -> np.ndarray:
        """Get exposed proportion of initial infections.

        Returns
        -------
        np.ndarray
            actively exposed compartment as a proportion of `INIT_INFECTION_DIST`.

        Raises
        ------
        RuntimeError
            if self.INIT_INFECTION_DIST does not exist. Usually created
            via self.load_initial_infection_dist_via_contact_matrix()

        Notes
        -----
        Ratio of initial infections across the E and I compartments
        dictated by the ratio of their waiting times.
        ```
        self.EXPOSED_TO_INFECTIOUS
        / (self.EXPOSED_TO_INFECTIOUS + self.INFECTIOUS_PERIOD)
        ```
        """
        if not hasattr(self, "INIT_INFECTION_DIST"):
            raise RuntimeError(
                "this function requires `self.INIT_INFECTION_DIST`"
                "set if via load_initial_infection_dist_via_contact_matrix()"
                "before calling this function"
            )
        # use relative wait times in each compartment to get distribution
        # of infections across infected vs exposed compartments
        exposed_to_infectous_ratio = strain_data[0].exposed_to_infectious / (
            strain_data[0].exposed_to_infectious
            + strain_data[0].infectious_period
        )
        return exposed_to_infectous_ratio * self.INIT_INFECTION_DIST

    def load_contact_matrix(
        self, demographic_data_path, region_name, compartments
    ) -> np.ndarray:
        """Load the region specific contact matrix.

        Usually sourced from https://github.com/mobs-lab/mixing-patterns

        Returns
        -------
        numpy.ndarray
            a matrix of shape (num_age_groups, num_age_group) refers to the
            pairwise interaction rates between age groups
        """
        age_limits = [b.min_value for b in compartments[0].dimensions[0].bins]
        return load_demographic_data(
            demographic_data_path,
            [region_name],
            len(age_limits),
            age_limits[0],
            age_limits,
        )[region_name]["avg_CM"]
