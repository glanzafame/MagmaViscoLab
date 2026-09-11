import math

from models.corrections.vesicles.base_vesicle import (
    BaseVesicleCorrection
)


class Mackenzie1950(BaseVesicleCorrection):

    name = "Mackenzie (1950)"

    parameters = {}

    model_physical_limits = {
        "Vesicles": (0.0, 60.0)
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        vesicle_fraction
    ):

        vesicles = float(
            vesicle_fraction
        )

        minimum, maximum = (
            self.model_physical_limits[
                "Vesicles"
            ]
        )

        if not minimum <= vesicles < maximum:

            raise ValueError(
                "Mackenzie (1950) requires "
                f"vesicle fraction ≥ {minimum:g} "
                f"and < {maximum:g} vol%."
            )

        phi = (
            vesicles / 100.0
        )

        # =================================================
        # MACKENZIE (1950)
        # =================================================

        eta_relative = (
            1.0
            - (5.0 / 3.0) * phi
        )

        # =================================================
        # RESULT VALIDATION
        # =================================================

        if (
            not math.isfinite(
                eta_relative
            )
            or eta_relative <= 0
        ):

            raise ValueError(
                "Calculated vesicle relative viscosity "
                "is not valid."
            )

        # =================================================
        # RETURN
        # =================================================

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,

            "details": {
                "vesicle_fraction": vesicles,
                "phi": phi
            }
        }