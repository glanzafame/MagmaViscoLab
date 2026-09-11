import math

from models.base_models import BaseViscosityModel


class Dominijanni2026(BaseViscosityModel):

    name = "Dominijanni et al. (2026)"

    required_oxides = []

    required_melt_physical_parameters = [
        "temperature",
        "water"
    ]

    parameters = {}

    model_physical_limits = {
        "H₂O": (0.0, 5.18)
    }

    A = -2.93

    TG_DRY = 917.05
    TG_WATER = 136.0

    M0 = 35.92
    M1 = 0.18857

    B = 0.1322
    C = 1.54232
    D = -1.40674

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(self, sample):

        T = float(sample.temperature) + 273.15
        water = float(sample.H2O)

        if T <= 0:
            raise ValueError(
                "Temperature must be greater than 0 K."
            )

        minimum, maximum = self.model_physical_limits[
            "H₂O"
        ]

        if not minimum <= water <= maximum:
            raise ValueError(
                "Dominijanni et al. (2026) is calibrated "
                f"for {minimum:g}–{maximum:g} wt% H₂O."
            )

        x_h2o = self.water_wt_to_mol_percent(
            water
        )

        if not 0 <= x_h2o < 100:
            raise ValueError(
                "Calculated H₂O mol% is not valid."
            )

        fragility = self.calculate_fragility(
            x_h2o
        )

        Tg = self.calculate_tg(
            x_h2o
        )

        log_eta = self.calculate_myega(
            T,
            Tg,
            fragility
        )

        if not math.isfinite(log_eta):
            raise ValueError(
                "Calculated Dominijanni viscosity "
                "is not valid."
            )

        return {
            "model": self.name,
            "log10_eta": log_eta,
            "details": {
                "temperature_K": T,
                "H2O_wt": water,
                "H2O_mol_percent": x_h2o,
                "Tg_K": Tg,
                "fragility": fragility
            }
        }

    # =====================================================
    # H2O WT% -> MOL%
    # =====================================================

    def water_wt_to_mol_percent(self, water):

        return (
            -0.0801 * water**2
            + 3.6258 * water
        )

    # =====================================================
    # FRAGILITY
    # =====================================================

    def calculate_fragility(self, x_h2o):

        return (
            self.M0
            + self.M1 * x_h2o
        )

    # =====================================================
    # GLASS TRANSITION TEMPERATURE
    # =====================================================

    def calculate_tg(self, x_h2o):

        denominator = (
            self.B * (100.0 - x_h2o)
            + x_h2o
        )

        if denominator <= 0:
            raise ValueError(
                "Invalid H₂O content for Tg calculation."
            )

        w1 = (
            x_h2o
            / denominator
        )

        w2 = (
            self.B
            * (100.0 - x_h2o)
            / denominator
        )

        delta_tg = (
            self.TG_DRY
            - self.TG_WATER
        )

        return (
            w1 * self.TG_WATER
            + w2 * self.TG_DRY
            + self.C * w1 * w2 * delta_tg
            + self.D * w1 * w2**2 * delta_tg
        )

    # =====================================================
    # MYEGA
    # =====================================================

    def calculate_myega(
        self,
        T,
        Tg,
        fragility
    ):

        delta = (
            12.0
            - self.A
        )

        exponent = (
            (
                fragility / delta
                - 1.0
            )
            * (
                Tg / T
                - 1.0
            )
        )

        return (
            self.A
            + delta
            * (Tg / T)
            * math.exp(exponent)
        )