from abc import ABC, abstractmethod


class BaseCrystalCorrection(ABC):

    name = "Base crystal model"
    parameters = {}

    @abstractmethod
    def calculate(
        self,
        crystal_fraction,
        **parameters
    ):
        """
        Calculate the crystal relative viscosity correction.

        Returns:
            dict: {
                "model": str,
                "relative_viscosity": float,
                "details": dict
            }
        """
        pass