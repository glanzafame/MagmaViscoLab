import math

from models.base_models import BaseViscosityModel


MOLAR_MASS = {
    "SiO2": 60.0843,
    "TiO2": 79.8658,
    "Al2O3": 101.9613,
    "FeO": 71.8444,
    "MnO": 70.9374,
    "MgO": 40.3044,
    "CaO": 56.0774,
    "Na2O": 61.9789,
    "K2O": 94.1960,
    "P2O5": 141.9446,
    "H2O": 18.0153
}


class HuiZhang2007(BaseViscosityModel):

    name = "Hui & Zhang (2007)"

    parameters = {}

    required_oxides = [
        "SiO2",
        "TiO2",
        "Al2O3",
        "FeO",
        "Fe2O3",
        "MnO",
        "MgO",
        "CaO",
        "Na2O",
        "K2O",
        "P2O5",
        "H2O"
    ]

    required_melt_physical_parameters = [
        "temperature",
        "water"
    ]

    model_physical_limits = {
        "Temperature": (299.85, None),
        "H₂O": (0.0, 12.3)
    }

    model_output_limits = {
        "log10_eta": (None, 15.0)
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(self, sample):

        temperature = float(sample.temperature)
        water = float(sample.H2O)
        T = temperature + 273.15

        t_min, _ = self.model_physical_limits[
            "Temperature"
        ]

        if temperature < t_min:
            raise ValueError(
                "Hui & Zhang (2007) should not be "
                f"extrapolated below {t_min:g} °C "
                "(573 K)."
            )

        h_min, h_max = self.model_physical_limits[
            "H₂O"
        ]

        if not h_min <= water <= h_max:
            raise ValueError(
                "Hui & Zhang (2007) is calibrated "
                f"for {h_min:g}–{h_max:g} wt% H₂O."
            )

        if float(
            getattr(
                sample,
                "F2O_1",
                0.0
            )
        ) > 0:
            raise ValueError(
                "Hui & Zhang (2007) does not "
                "explicitly include F."
            )

        X = self.calculate_mole_fractions(
            sample
        )

        components = self.calculate_components(
            X,
            T
        )

        log_eta = self.calculate_log_viscosity(
            components,
            T
        )

        if not math.isfinite(log_eta):
            raise ValueError(
                "Calculated Hui & Zhang viscosity "
                "is not valid."
            )

        _, eta_max = self.model_output_limits[
            "log10_eta"
        ]

        if log_eta > eta_max:
            raise ValueError(
                "Hui & Zhang (2007) should not be "
                f"extrapolated above log₁₀(η) = {eta_max:g}."
            )

        return {
            "model": self.name,
            "log10_eta": log_eta,
            "details": {
                "temperature_K": T,
                "H2O_wt": water,
                "mole_fractions": X,
                "components": components
            }
        }

    # =====================================================
    # WT% -> MOLE FRACTIONS
    # =====================================================

    def calculate_mole_fractions(self, sample):

        feo_total = (
            float(
                getattr(
                    sample,
                    "FeO",
                    0.0
                )
            )
            + float(
                getattr(
                    sample,
                    "Fe2O3",
                    0.0
                )
            )
            * (
                2.0 * 71.8444
                / 159.6882
            )
        )

        wt = {
            "SiO2": float(sample.SiO2),
            "TiO2": float(sample.TiO2),
            "Al2O3": float(sample.Al2O3),
            "FeO": feo_total,
            "MnO": float(sample.MnO),
            "MgO": float(sample.MgO),
            "CaO": float(sample.CaO),
            "Na2O": float(sample.Na2O),
            "K2O": float(sample.K2O),
            "P2O5": float(sample.P2O5),
            "H2O": float(sample.H2O)
        }

        for oxide, value in wt.items():

            if value < 0:
                raise ValueError(
                    f"{oxide} cannot be negative."
                )

        moles = {
            oxide:
                value / MOLAR_MASS[oxide]

            for oxide, value
            in wt.items()
        }

        total = sum(
            moles.values()
        )

        if total <= 0:
            raise ValueError(
                "Total number of moles must "
                "be greater than zero."
            )

        return {
            oxide:
                value / total

            for oxide, value
            in moles.items()
        }

    # =====================================================
    # COMPONENTS
    # =====================================================

    def calculate_components(self, X, T):

        alkalis = (
            X["Na2O"]
            + X["K2O"]
        )

        aluminum = X["Al2O3"]

        paired = min(
            aluminum,
            alkalis
        )

        al_excess = max(
            aluminum - alkalis,
            0.0
        )

        alkali_excess = max(
            alkalis - aluminum,
            0.0
        )

        nak_alo2 = (
            2.0 * paired
        )

        fe_mn = (
            X["FeO"]
            + X["MnO"]
        )

        water = X["H2O"]

        Z = (
            water ** (
                1.0
                / (
                    1.0
                    + 185.797 / T
                )
            )
            if water > 0
            else 0.0
        )

        return {
            "SiO2": X["SiO2"],
            "TiO2": X["TiO2"],
            "Al2O3_ex": al_excess,
            "FeMnO": fe_mn,
            "MgO": X["MgO"],
            "CaO": X["CaO"],
            "NaK2O_ex": alkali_excess,
            "P2O5": X["P2O5"],
            "H2O": water,
            "NaKAlO2": nak_alo2,
            "Z": Z
        }

    # =====================================================
    # EQ. 12
    # =====================================================

    def calculate_log_viscosity(self, X, T):

        A = (
            -6.83 * X["SiO2"]
            -170.79 * X["TiO2"]
            -14.71 * X["Al2O3_ex"]
            -18.01 * X["MgO"]
            -19.76 * X["CaO"]
            +34.31 * X["NaK2O_ex"]
            -140.38 * X["Z"]
            +159.26 * X["H2O"]
            -8.43 * X["NaKAlO2"]
        )

        B = (
            18.14 * X["SiO2"]
            +248.93 * X["TiO2"]
            +32.61 * X["Al2O3_ex"]
            +25.96 * X["MgO"]
            +22.64 * X["CaO"]
            -68.29 * X["NaK2O_ex"]
            +38.84 * X["Z"]
            -48.55 * X["H2O"]
            +16.12 * X["NaKAlO2"]
        )

        C = (
            21.73 * X["Al2O3_ex"]
            -61.98 * X["FeMnO"]
            -105.53 * X["MgO"]
            -69.92 * X["CaO"]
            -85.67 * X["NaK2O_ex"]
            +332.01 * X["Z"]
            -432.22 * X["H2O"]
            -3.16 * X["NaKAlO2"]
        )

        D = (
            2.16 * X["SiO2"]
            -143.05 * X["TiO2"]
            -22.10 * X["Al2O3_ex"]
            +38.56 * X["FeMnO"]
            +110.83 * X["MgO"]
            +67.12 * X["CaO"]
            +58.01 * X["NaK2O_ex"]
            +384.77 * X["P2O5"]
            -404.97 * X["Z"]
            +513.75 * X["H2O"]
        )

        exponent = (
            C
            + D * 1000.0 / T
        )

        try:
            exponential_term = math.exp(
                exponent
            )

        except OverflowError:
            raise ValueError(
                "Exponential term overflow in "
                "Hui & Zhang (2007)."
            )

        return (
            A
            + B * 1000.0 / T
            + exponential_term
        )