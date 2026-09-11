import math

from models.corrections.base_correction import BaseCrystalCorrection


class Caricchi2007(BaseCrystalCorrection):

    name = "Caricchi et al. (2007)"

    required_physical_parameters = [
        "crystals"
    ]

    parameters = {
        "strain_rate": {
            "label": "γ̇ (s⁻¹)",
            "default": 1e-5
        }
    }

    model_physical_limits = {
        "Crystals": (0.0, 80.0)
    }

    model_parameter_limits = {
        "strain_rate": (
            1e-7,
            10**-0.5
        )
    }

    B = 2.5

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        crystal_fraction,
        strain_rate
    ):

        try:
            phi = float(crystal_fraction)
            strain_rate = float(strain_rate)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Crystal fraction and strain rate "
                "must be numeric."
            ) from error

        # =================================================
        # INPUT VALIDATION
        # =================================================

        if not math.isfinite(phi):
            raise ValueError(
                "Crystal fraction must be finite."
            )

        phi_min_vol, phi_max_vol = (
            self.model_physical_limits[
                "Crystals"
            ]
        )

        phi_min = phi_min_vol / 100.0
        phi_max_limit = phi_max_vol / 100.0

        if not phi_min <= phi <= phi_max_limit:
            raise ValueError(
                "Caricchi et al. (2007) is calibrated "
                f"for {phi_min_vol:g}–{phi_max_vol:g} vol% "
                "crystals."
            )

        if not math.isfinite(strain_rate):
            raise ValueError(
                "Strain rate must be finite."
            )

        rate_min, rate_max = (
            self.model_parameter_limits[
                "strain_rate"
            ]
        )

        if not rate_min <= strain_rate <= rate_max:
            raise ValueError(
                "Caricchi et al. (2007) is parameterized "
                f"for strain rates between "
                f"{rate_min:.1e} and {rate_max:.3g} s⁻¹."
            )

        log_rate = math.log10(strain_rate)

        # =================================================
        # EQS. 6–9
        # =================================================

        phi_max = self.calculate_phi_max(log_rate)
        delta = self.calculate_delta(log_rate)
        alpha = self.calculate_alpha(log_rate)
        gamma = self.calculate_gamma(log_rate)

        if (
            not math.isfinite(phi_max)
            or phi_max <= 0.0
        ):
            raise ValueError(
                "Calculated maximum packing fraction "
                "is not valid."
            )

        if (
            not math.isfinite(alpha)
            or alpha <= 0.0
        ):
            raise ValueError(
                "Calculated alpha parameter is not valid."
            )

        # =================================================
        # NORMALIZED CRYSTAL FRACTION
        # =================================================

        varphi = phi / phi_max

        # =================================================
        # EQ. 3
        # =================================================

        erf_argument = (
            math.sqrt(math.pi)
            / (2.0 * alpha)
            * varphi
            * (
                1.0
                + varphi**gamma
            )
        )

        denominator = (
            1.0
            - alpha
            * math.erf(erf_argument)
        )

        if (
            not math.isfinite(denominator)
            or denominator <= 0.0
        ):
            raise ValueError(
                "Caricchi et al. (2007) denominator "
                "is not valid for the selected parameters."
            )

        eta_relative = (
            (
                1.0
                + varphi**delta
            )
            /
            denominator**(
                self.B * phi_max
            )
        )

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
                "strain_rate": strain_rate,
                "log10_strain_rate": log_rate,
                "phi_max": phi_max,
                "delta": delta,
                "alpha": alpha,
                "gamma": gamma,
                "B": self.B,
                "varphi": varphi,
                "erf_argument": erf_argument,
                "denominator": denominator
            }
        }

    # =====================================================
    # EQ. 6
    # =====================================================

    def calculate_phi_max(self, log_rate):

        return (
            0.066499
            * math.tanh(
                0.913424
                * log_rate
                + 3.850623
            )
            + 0.591806
        )

    # =====================================================
    # EQ. 7
    # =====================================================

    def calculate_delta(self, log_rate):

        return (
            -6.301095
            * math.tanh(
                
                0.818496
                * log_rate
                + 2.86
            )
            + 7.462405
        )

    # =====================================================
    # EQ. 8
    # =====================================================

    def calculate_alpha(self, log_rate):

        return (
            -0.000378
            * math.tanh(
                1.148101
                * log_rate
                + 3.92
            )
            + 0.999572
        )

    # =====================================================
    # EQ. 9
    # =====================================================

    def calculate_gamma(self, log_rate):

        return (
            3.987815
            * math.tanh(
                0.8908
                * log_rate
                + 3.24
            )
            + 5.099645
        )