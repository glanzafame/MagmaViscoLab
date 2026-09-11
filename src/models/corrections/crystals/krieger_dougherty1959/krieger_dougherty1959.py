import math

from models.corrections.base_correction import BaseCrystalCorrection


class KriegerDougherty1959(BaseCrystalCorrection):

    name = "Krieger & Dougherty (1959)"

    required_physical_parameters = [
        "crystals"
    ]

    DEFAULT_PHI_M = 0.60
    DEFAULT_INTRINSIC_VISCOSITY = 2.5
    MIN_POSITIVE = 1e-12

    parameters = {
        "phi_m": {
            "label": "φₘ",
            "default": DEFAULT_PHI_M
        },
        "intrinsic_viscosity": {
            "label": "η",
            "default": DEFAULT_INTRINSIC_VISCOSITY
        }
    }

    model_parameter_limits = {
        "phi_m": (
            MIN_POSITIVE,
            1.0
        ),
        "intrinsic_viscosity": (
            MIN_POSITIVE,
            None
        )
    }

    # =====================================================
    # DYNAMIC PHYSICAL LIMITS
    # =====================================================

    def get_dynamic_physical_limits(self, parameters):

        parameters = parameters or {}

        try:
            phi_m = float(
                parameters.get(
                    "phi_m",
                    self.DEFAULT_PHI_M
                )
            )

        except (TypeError, ValueError):
            return {}

        if (
            not math.isfinite(phi_m)
            or not 0.0 < phi_m <= 1.0
        ):
            return {}

        # The upper limit is strict because relative viscosity
        # diverges when the crystal fraction approaches phi_m.
        return {
            "Crystals": (
                0.0,
                100.0 * phi_m
            )
        }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        crystal_fraction,
        phi_m,
        intrinsic_viscosity
    ):

        values = {
            "Crystal fraction": crystal_fraction,
            "φₘ": phi_m,
            "[η]": intrinsic_viscosity
        }

        converted = {}

        for name, value in values.items():

            try:
                converted[name] = float(value)

            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} must be numeric."
                ) from error

            if not math.isfinite(converted[name]):
                raise ValueError(
                    f"{name} must be finite."
                )

        phi = converted["Crystal fraction"]
        phi_m = converted["φₘ"]
        intrinsic_viscosity = converted["[η]"]

        # =================================================
        # INPUT VALIDATION
        # =================================================

        if phi < 0.0:
            raise ValueError(
                "Crystal fraction cannot be negative."
            )

        if not 0.0 < phi_m <= 1.0:
            raise ValueError(
                "φₘ must satisfy 0 < φₘ ≤ 1."
            )

        if intrinsic_viscosity <= 0.0:
            raise ValueError(
                "[η] must be greater than zero."
            )

        if phi >= phi_m:
            raise ValueError(
                f"Crystal fraction ({phi:.6g}) must be lower "
                f"than maximum packing fraction φₘ "
                f"({phi_m:.6g})."
            )

        # =================================================
        # KRIEGER & DOUGHERTY (1959)
        # eta_r = (1 - phi / phi_m)^(-[eta] * phi_m)
        # =================================================

        base = (
            1.0
            - phi / phi_m
        )

        exponent = (
            -intrinsic_viscosity
            * phi_m
        )

        if (
            not math.isfinite(base)
            or base <= 0.0
        ):
            raise ValueError(
                "Krieger-Dougherty equation base "
                "must be finite and greater than zero."
            )

        try:
            eta_relative = (
                base**exponent
            )

        except OverflowError as error:
            raise ValueError(
                "Calculated crystal relative viscosity "
                "is too large."
            ) from error

        if (
            not math.isfinite(eta_relative)
            or eta_relative <= 0.0
        ):
            raise ValueError(
                "Calculated crystal relative viscosity "
                "is not valid."
            )

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,
            "details": {
                "phi": phi,
                "crystals_vol_percent": 100.0 * phi,
                "phi_m": phi_m,
                "maximum_packing_vol_percent": 100.0 * phi_m,
                "intrinsic_viscosity": intrinsic_viscosity,
                "base": base,
                "exponent": exponent,
                "log10_relative_viscosity": math.log10(
                    eta_relative
                )
            }
        }