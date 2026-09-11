import math

from models.base_models import BaseViscosityModel


class Valdivia2023(BaseViscosityModel):

    name = "Valdivia et al. (2023)"

    required_oxides = []

    required_melt_physical_parameters = [
        "temperature",
        "water"
    ]

    parameters = {}

    model_physical_limits = {
        "H₂O": (0.0, 10)
    }

    # =====================================================
    # FITTING PARAMETERS
    # =====================================================

    A = -2.93
    TG_DRY = 940.1
    TG_WATER = 136.0

    FRAGILITY = 40.7

    B = 0.2351
    C = 1.234
    D = -1.47

    H2O_COEFFICIENTS = {
        "a5": 7.635e-7,
        "a4": -5.209e-5,
        "a3": 0.002282,
        "a2": -0.09059,
        "a1": 3.552
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(self, sample):

        temperature = float(sample.temperature)
        water = float(sample.H2O)

        T = (
            temperature
            + 273.15
        )

        if T <= 0:
            raise ValueError(
                "Temperature must be greater than 0 K."
            )

        h_min, h_max = self.model_physical_limits[
            "H₂O"
        ]

        if not h_min <= water <= h_max:
            raise ValueError(
                "Valdivia et al. (2023) is calibrated "
                f"for {h_min:g}–{h_max:g} wt% H₂O."
            )

        x_h2o = self.water_wt_to_mol_percent(
            water
        )

        Tg = self.calculate_tg(
            x_h2o
        )

        log_eta = self.calculate_myega(
            T,
            Tg
        )

        if not math.isfinite(
            log_eta
        ):
            raise ValueError(
                "Calculated Valdivia viscosity "
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
                "fragility": self.FRAGILITY
            }
        }

    # =====================================================
    # H2O WT% -> MOL%
    # =====================================================

    def water_wt_to_mol_percent(self, water):

        a5 = self.H2O_COEFFICIENTS["a5"]
        a4 = self.H2O_COEFFICIENTS["a4"]
        a3 = self.H2O_COEFFICIENTS["a3"]
        a2 = self.H2O_COEFFICIENTS["a2"]
        a1 = self.H2O_COEFFICIENTS["a1"]

        return (
            a5 * water**5
            + a4 * water**4
            + a3 * water**3
            + a2 * water**2
            + a1 * water
        )

    # =====================================================
    # GLASS TRANSITION TEMPERATURE
    # =====================================================

    def calculate_tg(self, x_h2o):

        denominator = (
            self.B
            * (100.0 - x_h2o)
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
            + self.C
            * w1
            * w2
            * delta_tg
            + self.D
            * w1
            * w2**2
            * delta_tg
        )

    # =====================================================
    # MYEGA
    # =====================================================

    def calculate_myega(
        self,
        T,
        Tg
    ):

        delta = (
            12.0
            - self.A
        )

        exponent = (
            (
                self.FRAGILITY
                / delta
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
            * math.exp(
                exponent
            )
        )