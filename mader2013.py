import math

from models.corrections.vesicles.base_vesicle import (
    BaseVesicleCorrection
)


class Mader2013(BaseVesicleCorrection):

    name = "Mader et al. (2013)"

    LOW_CAPILLARY_MODE = (
        "low Ca (spherical vesicles)"
    )

    HIGH_CAPILLARY_MODE = (
        "high Ca (deformed vesicles)"
    )

    parameters = {
        "mode": {
            "label": "Vesicle regime",
            "type": "choice",
            "options": [
                LOW_CAPILLARY_MODE,
                HIGH_CAPILLARY_MODE
            ],
            "default": LOW_CAPILLARY_MODE
        }
    }

    model_physical_limits = {
        "Vesicles": (0.0, 100.0)
    }

    MODE_ALIASES = {
        LOW_CAPILLARY_MODE:
            LOW_CAPILLARY_MODE,

        HIGH_CAPILLARY_MODE:
            HIGH_CAPILLARY_MODE,

        # Compatibility with previous MVL values.
        "spherical":
            LOW_CAPILLARY_MODE,

        "deformed":
            HIGH_CAPILLARY_MODE,

        "low Ca (spherical shape)":
            LOW_CAPILLARY_MODE,

        "high Ca (deformed shape)":
            HIGH_CAPILLARY_MODE
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        vesicle_fraction,
        mode=LOW_CAPILLARY_MODE
    ):

        try:
            vesicles = float(vesicle_fraction)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Vesicle fraction must be numeric."
            ) from error

        # =================================================
        # VALIDATION
        # =================================================

        if not math.isfinite(vesicles):
            raise ValueError(
                "Vesicle fraction must be finite."
            )

        minimum, maximum = (
            self.model_physical_limits[
                "Vesicles"
            ]
        )

        if not minimum <= vesicles < maximum:
            raise ValueError(
                "Mader et al. (2013) requires "
                f"vesicle fraction ≥ {minimum:g} "
                f"and < {maximum:g} vol%."
            )

        normalized_mode = self.MODE_ALIASES.get(mode)

        if normalized_mode is None:
            raise ValueError(
                f"Unknown vesicle regime: {mode}."
            )

        phi = vesicles / 100.0

        # =================================================
        # RELATIVE VISCOSITY
        # =================================================

        if normalized_mode == self.LOW_CAPILLARY_MODE:
            eta_relative = (
                1.0 - phi
            ) ** -1.0

        else:
            eta_relative = (
                1.0 - phi
            ) ** (5.0 / 3.0)

        # =================================================
        # RESULT VALIDATION
        # =================================================

        if (
            not math.isfinite(eta_relative)
            or eta_relative <= 0.0
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
                "mode": normalized_mode,
                "vesicle_fraction": vesicles,
                "phi": phi
            }
        }