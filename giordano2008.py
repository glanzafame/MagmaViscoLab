import math

from models.base_models import BaseViscosityModel


class Giordano2008(BaseViscosityModel):

    name = "Giordano et al. (2008)"
    parameters = {}

    required_oxides = [
        "SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3",
        "MnO", "MgO", "CaO", "Na2O", "K2O",
        "P2O5", "H2O", "F2O_1"
    ]

    required_melt_physical_parameters = [
        "temperature",
        "water"
    ]

    model_physical_limits = {
        "Temperature": (245.0, 1705.0),
        "H₂O": (0.0, 8.0)
    }

    model_composition_limits = {
        "SiO2": (41.0, 79.0),
        "TiO2": (0.0, 3.0),
        "Al2O3": (0.0, 23.0),
        "FeO_total": (0.0, 12.0),
        "MnO": (0.0, 0.3),
        "MgO": (0.0, 32.0),
        "CaO": (0.0, 26.0),
        "Na2O": (0.0, 11.0),
        "K2O": (0.3, 9.0),
        "P2O5": (0.0, 1.2),
        "H2O": (0.0, 8.0),
        "F2O_1": (0.0, 4.0)
    }

    MOLAR_MASSES = {
        "SiO2": 60.0843,
        "TiO2": 79.8658,
        "Al2O3": 101.9613,
        "FeO": 71.8444,
        "Fe2O3": 159.6882,
        "MnO": 70.9374,
        "MgO": 40.3044,
        "CaO": 56.0774,
        "Na2O": 61.9789,
        "K2O": 94.1960,
        "P2O5": 141.9446,
        "Cr2O3": 151.9904,
        "H2O": 18.0153,
        "F2O_1": 37.9960
    }

    A = -4.55

    B_COEFFICIENTS = {
        "b1": 159.56,
        "b2": -173.34,
        "b3": 72.13,
        "b4": 75.69,
        "b5": -38.98,
        "b6": -84.08,
        "b7": 141.54,
        "b11": -2.43,
        "b12": -0.91,
        "b13": 17.62
    }

    C_COEFFICIENTS = {
        "C1": 2.75,
        "C2": 15.72,
        "C3": 8.32,
        "C4": 10.20,
        "C5": -12.29,
        "C6": -99.54,
        "C11": 0.30
    }

    def calculate(self, sample):

        temperature = float(sample.temperature)
        water = float(sample.H2O)

        self.validate_physical_parameters(
            temperature,
            water
        )

        feo = float(
            getattr(sample, "FeO", 0.0)
        )
        fe2o3 = float(
            getattr(sample, "Fe2O3", 0.0)
        )

        feo_total = (
            feo
            + fe2o3
            * (
                2.0
                * self.MOLAR_MASSES["FeO"]
                / self.MOLAR_MASSES["Fe2O3"]
            )
        )

        composition = self.oxide_mole_percent(
            sample,
            feo_total
        )

        terms = self.calculate_terms(
            composition
        )
        B = self.calculate_B(terms)
        C = self.calculate_C(terms)

        temperature_K = (
            temperature + 273.15
        )
        denominator = (
            temperature_K - C
        )

        if denominator <= 0:
            raise ValueError(
                "Invalid VFT denominator."
            )

        log_eta = (
            self.A
            + B / denominator
        )

        if not math.isfinite(log_eta):
            raise ValueError(
                "Calculated Giordano viscosity "
                "is not valid."
            )

        return {
            "model": self.name,
            "log10_eta": log_eta,
            "details": {
                "A": self.A,
                "B": B,
                "C": C,
                "temperature_K": temperature_K,
                "FeO_total_wt": feo_total,
                "terms": terms
            }
        }

    def validate_physical_parameters(
        self,
        temperature,
        water
    ):

        t_min, t_max = (
            self.model_physical_limits[
                "Temperature"
            ]
        )

        if not t_min <= temperature <= t_max:
            raise ValueError(
                "Giordano et al. (2008) calibration "
                f"temperature range is "
                f"{t_min:g}–{t_max:g} °C."
            )

        h_min, h_max = (
            self.model_physical_limits[
                "H₂O"
            ]
        )

        if not h_min <= water <= h_max:
            raise ValueError(
                "Giordano et al. (2008) calibration "
                f"H₂O range is "
                f"{h_min:g}–{h_max:g} wt%."
            )

    @classmethod
    def oxide_mole_percent(
        cls,
        sample,
        feo_total
    ):
        """
        Normalize anhydrous oxides to 100 - H2O wt%
        while keeping H2O fixed, then convert to mol%.
        """

        composition = {
            oxide: float(
                getattr(
                    sample,
                    oxide,
                    0.0
                )
            )
            for oxide in cls.MOLAR_MASSES
        }

        # Use all iron as equivalent FeO.
        composition["FeO"] = feo_total
        composition["Fe2O3"] = 0.0

        # Cr2O3 is not included in the GRD 2008 model.
        composition["Cr2O3"] = 0.0

        h2o = composition["H2O"]

        if not 0.0 <= h2o < 100.0:
            raise ValueError(
                "H2O must be between "
                "0 and 100 wt%."
            )

        anhydrous_total = sum(
            value
            for oxide, value
            in composition.items()
            if oxide != "H2O"
        )

        if anhydrous_total <= 0:
            raise ValueError(
                "Total anhydrous composition "
                "must be greater than zero."
            )

        normalization_factor = (
            (100.0 - h2o)
            / anhydrous_total
        )

        moles = {
            oxide: (
                (
                    h2o
                    if oxide == "H2O"
                    else value
                    * normalization_factor
                )
                / cls.MOLAR_MASSES[oxide]
            )
            for oxide, value
            in composition.items()
        }

        total_moles = sum(
            moles.values()
        )

        if total_moles <= 0:
            raise ValueError(
                "Total number of moles "
                "must be greater than zero."
            )

        return {
            oxide: (
                value
                / total_moles
                * 100.0
            )
            for oxide, value
            in moles.items()
        }

    @staticmethod
    def calculate_terms(X):

        SiO2 = X.get("SiO2", 0.0)
        TiO2 = X.get("TiO2", 0.0)
        Al2O3 = X.get("Al2O3", 0.0)
        FeO = X.get("FeO", 0.0)
        MnO = X.get("MnO", 0.0)
        MgO = X.get("MgO", 0.0)
        CaO = X.get("CaO", 0.0)
        Na2O = X.get("Na2O", 0.0)
        K2O = X.get("K2O", 0.0)
        P2O5 = X.get("P2O5", 0.0)
        H2O = X.get("H2O", 0.0)
        F2O_1 = X.get("F2O_1", 0.0)

        volatile_content = (
            H2O + F2O_1
        )

        if (
            H2O < -1.0
            or volatile_content < -1.0
        ):
            raise ValueError(
                "Invalid volatile content "
                "for logarithmic term."
            )

        return {
            "b1": SiO2 + TiO2,
            "b2": Al2O3,
            "b3": FeO + MnO + P2O5,
            "b4": MgO,
            "b5": CaO,
            "b6": Na2O + volatile_content,
            "b7": (
                volatile_content
                + math.log(1.0 + H2O)
            ),
            "b11": (
                (SiO2 + TiO2)
                * (FeO + MnO + MgO)
            ),
            "b12": (
                (
                    SiO2
                    + TiO2
                    + Al2O3
                    + P2O5
                )
                * (
                    Na2O
                    + K2O
                    + H2O
                )
            ),
            "b13": (
                Al2O3
                * (Na2O + K2O)
            ),
            "C1": SiO2,
            "C2": Al2O3 + TiO2,
            "C3": FeO + MnO + MgO,
            "C4": CaO,
            "C5": Na2O + K2O,
            "C6": math.log(
                1.0 + volatile_content
            ),
            "C11": (
                (
                    Al2O3
                    + FeO
                    + MnO
                    + MgO
                    + CaO
                    - P2O5
                )
                * (
                    Na2O
                    + K2O
                    + volatile_content
                )
            )
        }

    def calculate_B(self, terms):

        return sum(
            coefficient
            * terms.get(key, 0.0)
            for key, coefficient
            in self.B_COEFFICIENTS.items()
        )

    def calculate_C(self, terms):

        return sum(
            coefficient
            * terms.get(key, 0.0)
            for key, coefficient
            in self.C_COEFFICIENTS.items()
        )