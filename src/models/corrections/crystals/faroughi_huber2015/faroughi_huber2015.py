import math

from models.corrections.base_correction import BaseCrystalCorrection


class Faroughi_Huber2015(BaseCrystalCorrection):

    name = "Faroughi & Huber (2015)"

    required_physical_parameters = [
        "crystals"
    ]

    # =====================================================
    # FIXED MODEL CONSTANTS
    # Faroughi & Huber (2015), Eq. 49
    # =====================================================

    PHI_MAX = 0.637

    # No user-adjustable model parameters
    parameters = {}

    model_parameter_limits = {}

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        crystal_fraction
    ):

        try:
            phi = float(crystal_fraction)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Crystal fraction must be numeric."
            ) from error

        phi_max = self.PHI_MAX

        # =================================================
        # INPUT VALIDATION
        # =================================================

        if not math.isfinite(phi):
            raise ValueError(
                "Crystal fraction must be finite."
            )

        if phi < 0.0:
            raise ValueError(
                "Crystal fraction cannot be negative."
            )

        if phi >= phi_max:
            raise ValueError(
                f"Faroughi & Huber (2015) requires the "
                f"crystal fraction to be lower than φₘ. "
                f"Current crystal fraction: "
                f"{phi * 100.0:.4g} vol%; "
                f"φₘ: {phi_max * 100.0:.4g} vol%."
            )

        # =================================================
        # FAROUGHI & HUBER (2015), EQ. 49
        # RIGID SOLID PARTICLES
        #
        # eta_r =
        # [
        #   (phi_m - phi)
        #   /
        #   (phi_m * (1 - phi))
        # ] ^
        # [
        #   -2.5 * phi_m / (1 - phi_m)
        # ]
        #
        # This is the rigid-particle limit:
        # lambda -> infinity
        # =================================================

        numerator = (
            phi_max
            - phi
        )

        denominator = (
            phi_max
            * (1.0 - phi)
        )

        if (
            not math.isfinite(denominator)
            or denominator <= 0.0
        ):
            raise ValueError(
                "The Faroughi & Huber denominator "
                "is not valid."
            )

        base = (
            numerator
            / denominator
        )

        exponent = (
            -2.5
            * phi_max
            / (1.0 - phi_max)
        )

        if (
            not math.isfinite(base)
            or base <= 0.0
        ):
            raise ValueError(
                "The Faroughi & Huber equation base "
                "is not valid."
            )

        # =================================================
        # NUMERICAL EVALUATION
        # =================================================

        try:
            log_eta_relative = (
                exponent
                * math.log(base)
            )

            eta_relative = math.exp(
                log_eta_relative
            )

        except OverflowError as error:
            raise ValueError(
                "Calculated crystal relative viscosity "
                "is too large."
            ) from error

        # =================================================
        # RESULT VALIDATION
        # =================================================

        if (
            not math.isfinite(eta_relative)
            or eta_relative <= 0.0
        ):
            raise ValueError(
                "Calculated crystal relative viscosity "
                "is not valid."
            )

        # =================================================
        # AUXILIARY QUANTITIES
        # =================================================

        # Self-crowding factor from Faroughi & Huber
        # Eq. 47:
        #
        # omega = (1 - phi_m) / phi_m
        #
        # It is not an independent parameter in Eq. 49.
        omega = (
            (1.0 - phi_max)
            / phi_max
        )

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,

            "details": {
                "phi": phi,
                "crystals_vol_percent":
                    phi * 100.0,

                "phi_max": phi_max,
                "maximum_packing_vol_percent":
                    phi_max * 100.0,

                "omega": omega,

                "base": base,
                "exponent": exponent,

                "ln_eta_relative":
                    log_eta_relative,

                "log10_eta_relative":
                    log_eta_relative
                    / math.log(10.0)
            }
        }

    # =====================================================
    # DYNAMIC PHYSICAL LIMITS
    # =====================================================

    def get_dynamic_physical_limits(
        self,
        parameters=None
    ):

        # The mathematical domain is:
        #
        # 0 <= phi < phi_m
        #
        # phi_m = 0.637 is fixed.

        upper_limit = math.nextafter(
            self.PHI_MAX * 100.0,
            0.0
        )

        return {
            "Crystals": (
                0.0,
                upper_limit
            )
        }