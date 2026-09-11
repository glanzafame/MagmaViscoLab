from abc import ABC, abstractmethod

from core.sample import Sample


class BaseViscosityModel(ABC):

    # --- Model metadata --------------------------------------------

    name = "Base melt-viscosity model"
    parameters = {}
    required_oxides = ()
    required_melt_physical_parameters = ()

    # --- Viscosity calculation -------------------------------------

    @abstractmethod
    def calculate(
        self,
        sample: Sample,
        **kwargs,
    ) -> dict:
        """
        Calculate melt viscosity for one sample.

        Returns a dictionary containing the model name,
        log₁₀ melt viscosity in Pa·s and calculation details.
        """
        raise NotImplementedError