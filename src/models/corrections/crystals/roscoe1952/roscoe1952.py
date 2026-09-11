import math

from models.corrections.base_correction import BaseCrystalCorrection


class Roscoe1952(BaseCrystalCorrection):

    name = "Roscoe (1952)"

    required_physical_parameters = [
        "crystals"
    ]

    # Roscoe's original parameterization does not contain
    # user-adjustable parameters.
    parameters = {}
    model_parameter_limits = {}

    # Einstein-Roscoe equation:
    # eta_r = (1 - K * phi) ** EXPONENT
    K = 1.35
    EXPONENT = -2.5
    PHI_M = 1.0 / K

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(self, crystal_fraction):

        try:
            phi = float(crystal_fraction)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Crystal fraction must be numeric."
            ) from error

        # =================================================
        # VALIDATION
        # =================================================

        if not math.isfinite(phi):
            raise ValueError(
                "Crystal fraction must be finite."
            )

        if phi < 0.0:
            raise ValueError(
                "Crystal fraction cannot be negative."
            )

        if phi >= self.PHI_M:
            raise ValueError(
                f"Crystal fraction ({phi:.6f}) must be lower "
                "than the Roscoe (1952) limiting fraction "
                f"1/1.35 = {self.PHI_M:.6f} "
                f"({self.PHI_M * 100.0:.3f} vol%)."
            )

        # =================================================
        # ROSCOE (1952)
        # eta_r = (1 - 1.35 * phi) ** -2.5
        # =================================================

        base = 1.0 - self.K * phi

        if not math.isfinite(base) or base <= 0.0:
            raise ValueError(
                "The Roscoe (1952) equation base "
                "must be greater than zero."
            )

        try:
            eta_relative = base ** self.EXPONENT

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

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,

            "details": {
                "phi": phi,
                "crystals_vol_percent": phi * 100.0,
                "K": self.K,
                "phi_m": self.PHI_M,
                "maximum_crystals_vol_percent":
                    self.PHI_M * 100.0,
                "base": base,
                "exponent": self.EXPONENT
            }
        }

    # =====================================================
    # DYNAMIC PHYSICAL LIMITS
    # =====================================================

    def get_dynamic_physical_limits(self, parameters=None):

        # The upper limit is strict because the equation
        # diverges when phi = 1 / 1.35.
        return {
            "Crystals": (
                0.0,
                self.PHI_M * 100.0
            )
        }