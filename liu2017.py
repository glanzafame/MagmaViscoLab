import math

from models.corrections.base_correction import BaseCrystalCorrection


class Liu2017(BaseCrystalCorrection):

    name = "Liu et al. (2017)"

    required_physical_parameters = [
        "crystals"
    ]

    parameters = {
        "eta_r_max": {
            "label": "ηᵣ,ₘₐₓ",
            "default": 10000.0
        },
        "k": {
            "label": "k",
            "default": 4.0
        },
        "n": {
            "label": "n",
            "default": 3.5
        }
    }

    model_parameter_limits = {
        "eta_r_max": (1.0, None),
        "k": (0.0, None),
        "n": (0.0, None)
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        crystal_fraction,
        eta_r_max,
        k,
        n
    ):

        phi = float(crystal_fraction)
        eta_r_max = float(eta_r_max)
        k = float(k)
        n = float(n)

        # =================================================
        # VALIDATION
        # =================================================

        if not math.isfinite(phi):
            raise ValueError(
                "Crystal fraction must be finite."
            )

        if not 0.0 <= phi <= 1.0:
            raise ValueError(
                "Crystal fraction must be between 0 and 1."
            )

        if (
            not math.isfinite(eta_r_max)
            or eta_r_max < 1.0
        ):
            raise ValueError(
                "ηr(max) must be finite and greater than or equal to 1."
            )

        if not math.isfinite(k) or k <= 0.0:
            raise ValueError(
                "k must be finite and greater than zero."
            )

        if not math.isfinite(n) or n <= 0.0:
            raise ValueError(
                "n must be finite and greater than zero."
            )

        # =================================================
        # RELATIVE VISCOSITY
        # =================================================

        phi_power = phi ** n

        numerator = -math.expm1(
            -k * phi_power
        )

        denominator = -math.expm1(
            -k
        )

        if (
            not math.isfinite(denominator)
            or denominator <= 0.0
        ):
            raise ValueError(
                "The Liu et al. denominator is not valid."
            )

        exponent = numerator / denominator

        eta_relative = eta_r_max ** exponent

        # =================================================
        # RESULT VALIDATION
        # =================================================

        if (
            not math.isfinite(eta_relative)
            or eta_relative <= 0.0
        ):
            raise ValueError(
                "Calculated crystal relative viscosity is not valid."
            )

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,

            "details": {
                "phi": phi,
                "eta_r_max": eta_r_max,
                "k": k,
                "n": n,
                "phi_power": phi_power,
                "numerator": numerator,
                "denominator": denominator,
                "exponent": exponent
            }
        }

    # =====================================================
    # DYNAMIC PHYSICAL LIMITS
    # =====================================================

    def get_dynamic_physical_limits(
        self,
        parameters
    ):

        return {
            "Crystals": (
                0.0,
                100.0
            )
        }