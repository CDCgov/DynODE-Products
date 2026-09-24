from dynode.config import Params
from pydantic import BaseModel, ConfigDict


class ObservationParams(BaseModel):
    """Parameters for the observation model e.g., IHR"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class ExpandedParams(Params):
    """Miscellaneous parameters of an ODE model."""

    observation_params: ObservationParams
