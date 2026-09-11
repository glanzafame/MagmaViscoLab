import math

from models.corrections.vesicles.base_vesicle import BaseVesicleCorrection


class Vona2016(BaseVesicleCorrection):

    name = "Vona et al. (2016)"

    required_physical_parameters = [
        "vesicles"
    ]

    parameters = {
        "alpha": {
            "label": "α",
            "default": 1.47
        },
        "beta": {
            "label": "β",
            "default": 0.48
        }
    }

    model_parameter_limits = {
        "alpha": (0.0, None),
        "beta": (0.0, None)
    }

    model_physical_limits = {
        "Vesicles": (0.0, 66.0)
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        vesicle_fraction,
        alpha,
        beta
    ):

        vesicles_percent = float(
            vesicle_fraction
        )

        alpha = float(
            alpha
        )

        beta = float(
            beta
        )

        # =================================================
        # VALIDATION
        # =================================================

        if not math.isfinite(
            vesicles_percent
        ):
            raise ValueError(
                "Vesicle content must be finite."
            )

        if not 0.0 <= vesicles_percent <= 66.0:
            raise ValueError(
                "Vona et al. (2016) is calibrated for "
                "vesicle contents between 0 and 66 vol%."
            )

        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(
                "α must be finite and greater than zero."
            )

        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(
                "β must be finite and greater than zero."
            )

        # ViscosityEngine supplies vesicles in vol%.
        phi = vesicles_percent / 100.0

        # =================================================
        # RELATIVE VISCOSITY
        # =================================================

        void_ratio = phi / (1.0 - phi)

        log10_eta_relative = -alpha * (
            void_ratio ** beta
        )

        eta_relative = 10.0 ** (
            log10_eta_relative
        )

        # =================================================
        # RESULT VALIDATION
        # =================================================

        if (
            not math.isfinite(eta_relative)
            or eta_relative <= 0.0
        ):
            raise ValueError(
                "Calculated vesicle relative viscosity is not valid."
            )

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,

            "details": {
                "vesicles_percent": vesicles_percent,
                "phi": phi,
                "alpha": alpha,
                "beta": beta,
                "void_ratio": void_ratio,
                "log10_eta_relative": log10_eta_relative
            }
        }