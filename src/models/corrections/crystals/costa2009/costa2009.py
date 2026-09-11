import math

from models.corrections.base_correction import (
    BaseCrystalCorrection,
)


class Costa2009(BaseCrystalCorrection):

    name = "Costa et al. (2009)"

    required_physical_parameters = [
        "crystals"
    ]

    # Costa et al. (2009):
    # B is fixed to the nominal Einstein coefficient.
    # Delta is derived from delta = A - gamma.
    A_RELATION = 13.0
    FIXED_B = 2.5

    # Representative values discussed by Costa et al. (2009).
    # They do not constitute a universal calibration.
    DEFAULT_PHI_STAR = 0.591
    DEFAULT_GAMMA_EXPONENT = 5.76
    DEFAULT_XI = 4.63e-4

    MIN_POSITIVE = 1e-12

    # Only these three parameters are user-adjustable.
    parameters = {
        "phi_star": {
            "label": "φ⁎",
            "default": DEFAULT_PHI_STAR,
        },
        "gamma_exponent": {
            "label": "γ",
            "default": DEFAULT_GAMMA_EXPONENT,
        },
        "xi": {
            "label": "ξ",
            "default": DEFAULT_XI,
        },
    }

    model_parameter_limits = {
        "phi_star": (
            MIN_POSITIVE,
            1.0,
        ),
        "gamma_exponent": (
            MIN_POSITIVE,
            A_RELATION - MIN_POSITIVE,
        ),
        "xi": (
            MIN_POSITIVE,
            1.0 - MIN_POSITIVE,
        ),
    }

    # The model is used from the dilute limit to 80 vol%.
    # The principal experimental comparison presented by
    # Costa et al. (2009) concerns approximately 30–80 vol%.
    model_physical_limits = {
        "Crystals": (0.0, 80.0)
    }

    # =====================================================
    # NUMERIC CONVERSION
    # =====================================================

    @staticmethod
    def _finite_float(value, name):
        """Return a finite floating-point value."""
        try:
            converted = float(value)

        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be numeric."
            )

        if not math.isfinite(converted):
            raise ValueError(
                f"{name} must be finite."
            )

        return converted

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        crystal_fraction,
        phi_star,
        gamma_exponent,
        xi,
    ):
        """
        Calculate the crystal relative-viscosity factor.

        crystal_fraction is expressed as a fraction between
        0 and 1. MVL converts the GUI value from vol% before
        calling this method.

        B is fixed to 2.5 and delta is calculated from:

            delta = 13 - gamma
        """

        phi = self._finite_float(
            crystal_fraction,
            "Crystal fraction",
        )
        phi_star = self._finite_float(
            phi_star,
            "φ*",
        )
        gamma_exponent = self._finite_float(
            gamma_exponent,
            "γ",
        )
        xi = self._finite_float(
            xi,
            "ξ",
        )

        # Fixed and derived parameters
        B = self.FIXED_B
        delta = (
            self.A_RELATION
            - gamma_exponent
        )

        # =================================================
        # CRYSTAL FRACTION
        # =================================================

        phi_min_percent, phi_max_percent = (
            self.model_physical_limits[
                "Crystals"
            ]
        )

        phi_percent = 100.0 * phi

        if not (
            phi_min_percent
            <= phi_percent
            <= phi_max_percent
        ):
            raise ValueError(
                "Costa et al. (2009) is implemented "
                f"for {phi_min_percent:g}–"
                f"{phi_max_percent:g} vol% crystals. "
                "Higher crystal fractions represent an "
                "unsupported solid-framework extrapolation."
            )

        # =================================================
        # MODEL PARAMETERS
        # =================================================

        if not 0.0 < phi_star <= 1.0:
            raise ValueError(
                "φ* must satisfy 0 < φ* ≤ 1."
            )

        if not (
            0.0
            < gamma_exponent
            < self.A_RELATION
        ):
            raise ValueError(
                "γ must satisfy 0 < γ < 13 so that "
                "the derived exponent δ = 13 − γ "
                "remains positive."
            )

        if not 0.0 < xi < 1.0:
            raise ValueError(
                "ξ must satisfy 0 < ξ < 1."
            )

        # =================================================
        # COSTA ET AL. (2009), EQUATIONS 1–2
        # =================================================

        varphi = phi / phi_star

        try:
            varphi_gamma = (
                varphi**gamma_exponent
            )

            erf_argument = (
                math.sqrt(math.pi)
                / (
                    2.0
                    * (1.0 - xi)
                )
                * varphi
                * (
                    1.0
                    + varphi_gamma
                )
            )

            F = (
                (1.0 - xi)
                * math.erf(erf_argument)
            )

            denominator = 1.0 - F

            if (
                not math.isfinite(denominator)
                or denominator <= 0.0
            ):
                raise ValueError(
                    "Costa et al. (2009) produced an "
                    "invalid denominator for the selected "
                    "parameters."
                )

            numerator = (
                1.0
                + varphi**delta
            )

            denominator_power = (
                denominator
                ** (B * phi_star)
            )

            if (
                not math.isfinite(
                    denominator_power
                )
                or denominator_power <= 0.0
            ):
                raise ValueError(
                    "Costa et al. (2009) produced an "
                    "invalid denominator power."
                )

            eta_relative = (
                numerator
                / denominator_power
            )

        except (OverflowError, ZeroDivisionError):
            raise ValueError(
                "Costa et al. (2009) overflowed for "
                "the selected parameters."
            )

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
                "phi_percent": phi_percent,
                "phi_star": phi_star,
                "varphi": varphi,
                "B": B,
                "B_is_fixed": True,
                "gamma_exponent": gamma_exponent,
                "delta": delta,
                "delta_is_derived": True,
                "delta_relation": "δ = 13 − γ",
                "xi": xi,
                "varphi_gamma": varphi_gamma,
                "erf_argument": erf_argument,
                "F": F,
                "denominator": denominator,
                "numerator": numerator,
            },
        }