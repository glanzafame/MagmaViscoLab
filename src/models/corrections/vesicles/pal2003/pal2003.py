import math

from models.corrections.vesicles.base_vesicle import (
    BaseVesicleCorrection
)


class Pal2003(BaseVesicleCorrection):

    name = "Pal (2003)"

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
    
    model_parameter_limits = {
        "phi_m": (0.0, 1.0)
    }

    MODE_ALIASES = {
        LOW_CAPILLARY_MODE: LOW_CAPILLARY_MODE,
        HIGH_CAPILLARY_MODE: HIGH_CAPILLARY_MODE,
        "spherical": LOW_CAPILLARY_MODE,
        "deformed": HIGH_CAPILLARY_MODE
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        vesicle_fraction,
        mode=LOW_CAPILLARY_MODE,
        phi_m=0.70
    ):

        try:
            vesicles = float(vesicle_fraction)
            phi_m = float(phi_m)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Vesicle fraction and φm must be numeric."
            ) from error

        if not math.isfinite(vesicles):
            raise ValueError(
                "Vesicle fraction must be finite."
            )

        if vesicles < 0.0:
            raise ValueError(
                "Vesicle fraction cannot be negative."
            )

        if not math.isfinite(phi_m):
            raise ValueError(
                "φm must be finite."
            )

        if not 0.0 < phi_m <= 1.0:
            raise ValueError(
                "φm must be greater than 0 and ≤ 1."
            )

        normalized_mode = self.MODE_ALIASES.get(mode)

        if normalized_mode is None:
            raise ValueError(
                f"Unknown bubble regime: {mode}."
            )

        phi = vesicles / 100.0

        if phi >= phi_m:
            raise ValueError(
                f"Vesicle fraction ({phi:.4f}) must be "
                f"lower than φm ({phi_m:.4f})."
            )

        base = 1.0 - phi / phi_m

        if not math.isfinite(base) or base <= 0.0:
            raise ValueError(
                "The Pal model base is not valid."
            )

        if normalized_mode == self.LOW_CAPILLARY_MODE:
            exponent = -3.0 * phi_m
        else:
            exponent = 5.0 * phi_m / 3.0

        try:
            eta_relative = base ** exponent

        except OverflowError as error:
            raise ValueError(
                "Calculated vesicle relative viscosity "
                "is too large."
            ) from error

        if (
            not math.isfinite(eta_relative)
            or eta_relative <= 0.0
        ):
            raise ValueError(
                "Calculated vesicle relative viscosity "
                "is not valid."
            )

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,
            "details": {
                "mode": normalized_mode,
                "vesicle_fraction": vesicles,
                "phi": phi,
                "phi_m": phi_m,
                "exponent": exponent,
                "base": base
            }
        }

    # =====================================================
    # DYNAMIC PHYSICAL LIMITS
    # =====================================================

    def get_dynamic_physical_limits(
        self,
        parameters
    ):

        try:
            phi_m = float(
                parameters.get(
                    "phi_m",
                    0.70
                )
            )

        except (TypeError, ValueError, AttributeError):
            return {}

        if (
            not math.isfinite(phi_m)
            or not 0.0 < phi_m <= 1.0
        ):
            return {}

        return {
            "Vesicles": (
                0.0,
                phi_m * 100.0
            )
        }