from dynode.config import (
    AgeBin,
    DeterministicParameter,
    PlaceholderSample,
    Strain,
)
from initializers import CovidInitializer
from numpyro import distributions as dist

from .config_fit import Covid5StrainConfig


class Covid6StrainProjConfig(Covid5StrainConfig):
    # This is a 6-strain projection model, basing on 5-strain base fit
    def __init__(self, region_code: str, scenario_name: str = "base"):
        super().__init__(region_code, scenario_name)

        # Make a XEC-like X
        ## Recreate definition of X, then append to strains list
        x = Strain(
            strain_name="X",
            r0=DeterministicParameter("jn1_r0"),
            infectious_period=4.0,
            exposed_to_infectious=2.0,
            vaccine_efficacy={0: 0, 1: 0.0, 2: 0.0, 3: 0.0},
            is_introduced=True,
            introduction_time=dist.Uniform(1585, 1585 + 90),
            introduction_percentage=PlaceholderSample(),
            introduction_scale=15,
            introduction_ages=[
                AgeBin(18, 49),
            ],
        )

        self.parameters.transmission_params.strains.append(x)
        self._create_introduction_ages_mask_encoding()

        ## Add X to strain interactions
        si = self.parameters.transmission_params.strain_interactions
        si["X"] = {
            "omicron": 0.0,  # No matching step in XEC
            "BA2BA5": PlaceholderSample(),
            "XBB": PlaceholderSample(),
            "JN1": PlaceholderSample(),
            "XEC": PlaceholderSample(),
            "X": PlaceholderSample(),
        }

        for strain_name in si:
            if strain_name != "X":
                si[strain_name]["X"] = 1.0

        # Update num_strains to reflect addition
        # Modify for both transmission_params and top-level
        self.parameters.transmission_params.num_strains = len(
            self.parameters.transmission_params.strains
        )
        self.num_strains = self.parameters.transmission_params.num_strains

        # Rebuild compartments with the added strain list
        self.compartments = self._get_compartments(
            self.parameters.transmission_params.strains
        )
        # Clear cached idx so it rebuilds from new compartments
        self.__dict__.pop("idx", None)

        # Change initializer
        ini = self.parameters.transmission_params.initial_infections_base_rate
        cross_immunity_matrix = self._get_cross_immunity_matrix(
            self.parameters.transmission_params.strain_interactions
        )
        initializer = CovidInitializer(
            cross_immunity_matrix,
            self.compartments,
            self.parameters.transmission_params.num_strains,
            self.parameters.transmission_params.strains,
            self.demographic_data_path,
            self.initialization_data_path,
            self.region_name,
            self.region_population_size,
            ini,
        )
        self.initializer = initializer

        # IHR by strains groups need to be updated too
        self.parameters.observation_params.ihr_strains_groups = [
            0,
            0,
            0,
            1,
            2,
            3,
        ]
