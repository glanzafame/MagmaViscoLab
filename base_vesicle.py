from abc import ABC, abstractmethod


class BaseVesicleCorrection(ABC):

    name = "Base vesicle model"
    parameters = {}

    @abstractmethod
    def calculate(
        self,
        vesicle_fraction,
        **parameters
    ):
        """
        Calculate the vesicle relative viscosity correction.

        Returns:
            dict: {
                "model": str,
                "relative_viscosity": float,
                "details": dict
            }
        """
        pass