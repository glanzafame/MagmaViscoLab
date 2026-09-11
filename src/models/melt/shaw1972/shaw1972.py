import math

from models.base_models import BaseViscosityModel


MOLAR_MASS = {
    "SiO2": 60.0843,
    "TiO2": 79.8658,
    "Al2O3": 101.9613,
    "FeO": 71.8444,
    "Fe2O3": 159.6882,
    "MgO": 40.3044,
    "CaO": 56.0774,
    "Na2O": 61.9789,
    "K2O": 94.1960,
    "H2O": 18.0153,
}


class Shaw1972(BaseViscosityModel):

    name = "Shaw (1972)"

    parameters = {}

    # Fe2O3 is optional. If present in the Sample object, it is converted
    # internally to equivalent FeO and added to the entered FeO content.
    required_oxides = [
        "SiO2",
        "TiO2",
        "Al2O3",
        "FeO",
        "MgO",
        "CaO",
        "Na2O",
        "K2O",
        "H2O",
    ]

    optional_oxides = [
        "Fe2O3",
    ]

    required_melt_physical_parameters = [
        "temperature",
        "water",
    ]

    # This is an approximate guard rather than a formal calibration range.
    model_physical_limits = {
        "H₂O": (0.0, 12.0),
    }

    model_composition_limits = {
        "X_SiO2": (0.4, 0.8),
    }

    C_ETA = -6.40
    C_T = 1.50

    S_VALUES = {
        "H2O": 2.0,
        "alkalis": 2.8,
        "FeMg": 3.4,
        "CaTi": 4.5,
        "AlO2": 6.7,
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(self, sample):

        temperature_c = self._read_number(
            sample,
            "temperature",
            "Temperature",
        )

        temperature_k = temperature_c + 273.15

        if temperature_k <= 0.0:
            raise ValueError(
                "Temperature must be greater than 0 K."
            )

        water_wt = self._read_oxide(
            sample,
            "H2O",
        )

        water_min, water_max = self.model_physical_limits[
            "H₂O"
        ]

        if not water_min <= water_wt <= water_max:
            raise ValueError(
                "Shaw (1972) should not be extrapolated "
                f"outside approximately {water_min:g}–"
                f"{water_max:g} wt% H₂O."
            )

        mole_fractions, iron_details = (
            self.calculate_mole_fractions(sample)
        )

        x_sio2 = mole_fractions["SiO2"]

        x_min, x_max = self.model_composition_limits[
            "X_SiO2"
        ]

        if not x_min <= x_sio2 <= x_max:
            raise ValueError(
                "Shaw (1972) should not be extrapolated "
                "outside approximately "
                f"{x_min:g} ≤ XSiO₂ ≤ {x_max:g}. "
                f"The calculated value is {x_sio2:.4f}."
            )

        mean_slope = self.calculate_mean_slope(
            mole_fractions
        )

        ln_eta_poise = (
            mean_slope
            * (10000.0 / temperature_k)
            - self.C_T * mean_slope
            + self.C_ETA
        )

        log10_eta_pa_s = (
            ln_eta_poise / math.log(10.0)
            - 1.0
        )

        if not math.isfinite(log10_eta_pa_s):
            raise ValueError(
                "Calculated Shaw viscosity is not valid."
            )

        return {
            "model": self.name,
            "log10_eta": log10_eta_pa_s,
            "details": {
                "temperature_C": temperature_c,
                "temperature_K": temperature_k,
                "H2O_wt_percent": water_wt,
                "X_SiO2": x_sio2,
                "mean_slope_s": mean_slope,
                "ln_eta_poise": ln_eta_poise,
                "log10_eta_Pa_s": log10_eta_pa_s,
                **iron_details,
                "mole_fractions": mole_fractions,
            },
        }

    # =====================================================
    # WT% -> MOLE FRACTIONS
    # =====================================================

    def calculate_mole_fractions(self, sample):

        oxide_wt = {
            oxide: self._read_oxide(sample, oxide)
            for oxide in self.required_oxides
        }

        fe2o3_wt = self._read_oxide(
            sample,
            "Fe2O3",
            optional=True,
        )

        feo_from_fe2o3_wt = (
            fe2o3_wt
            * (
                2.0 * MOLAR_MASS["FeO"]
                / MOLAR_MASS["Fe2O3"]
            )
        )

        feo_total_wt = (
            oxide_wt["FeO"]
            + feo_from_fe2o3_wt
        )

        # Shaw's component basis:
        # - two AlO2 units for each mole of Al2O3;
        # - Fe2O3 expressed as two FeO units;
        # - minor components such as MnO and P2O5 are ignored.
        component_moles = {
            "SiO2": (
                oxide_wt["SiO2"]
                / MOLAR_MASS["SiO2"]
            ),
            "TiO2": (
                oxide_wt["TiO2"]
                / MOLAR_MASS["TiO2"]
            ),
            "AlO2": (
                2.0
                * oxide_wt["Al2O3"]
                / MOLAR_MASS["Al2O3"]
            ),
            "FeO": (
                feo_total_wt
                / MOLAR_MASS["FeO"]
            ),
            "MgO": (
                oxide_wt["MgO"]
                / MOLAR_MASS["MgO"]
            ),
            "CaO": (
                oxide_wt["CaO"]
                / MOLAR_MASS["CaO"]
            ),
            "Na2O": (
                oxide_wt["Na2O"]
                / MOLAR_MASS["Na2O"]
            ),
            "K2O": (
                oxide_wt["K2O"]
                / MOLAR_MASS["K2O"]
            ),
            "H2O": (
                oxide_wt["H2O"]
                / MOLAR_MASS["H2O"]
            ),
        }

        total_moles = sum(component_moles.values())

        if (
            not math.isfinite(total_moles)
            or total_moles <= 0.0
        ):
            raise ValueError(
                "Total number of moles must be greater "
                "than zero."
            )

        mole_fractions = {
            component: moles / total_moles
            for component, moles in component_moles.items()
        }

        iron_details = {
            "FeO_input_wt_percent": oxide_wt["FeO"],
            "Fe2O3_input_wt_percent": fe2o3_wt,
            "FeO_from_Fe2O3_wt_percent": feo_from_fe2o3_wt,
            "FeO_total_equivalent_wt_percent": feo_total_wt,
        }

        return mole_fractions, iron_details

    # =====================================================
    # MEAN SLOPE
    # =====================================================

    def calculate_mean_slope(self, mole_fractions):

        x_sio2 = mole_fractions["SiO2"]
        denominator = 1.0 - x_sio2

        if denominator <= 0.0:
            raise ValueError(
                "Invalid SiO₂ mole fraction."
            )

        groups = {
            "H2O": mole_fractions.get("H2O", 0.0),
            "alkalis": (
                mole_fractions.get("Na2O", 0.0)
                + mole_fractions.get("K2O", 0.0)
            ),
            "FeMg": (
                mole_fractions.get("FeO", 0.0)
                + mole_fractions.get("MgO", 0.0)
            ),
            "CaTi": (
                mole_fractions.get("CaO", 0.0)
                + mole_fractions.get("TiO2", 0.0)
            ),
            "AlO2": mole_fractions.get("AlO2", 0.0),
        }

        contributions = {
            group: (
                fraction
                * self.S_VALUES[group]
                * x_sio2
            )
            for group, fraction in groups.items()
        }

        mean_slope = (
            sum(contributions.values())
            / denominator
        )

        if not math.isfinite(mean_slope):
            raise ValueError(
                "Calculated Shaw mean slope is not valid."
            )

        return mean_slope

    # =====================================================
    # INPUT HELPERS
    # =====================================================

    @staticmethod
    def _read_number(sample, attribute, label):

        if not hasattr(sample, attribute):
            raise ValueError(
                f"Missing value for {label}."
            )

        try:
            value = float(getattr(sample, attribute))
        except (TypeError, ValueError):
            raise ValueError(
                f"{label} must be numeric."
            )

        if not math.isfinite(value):
            raise ValueError(
                f"{label} must be finite."
            )

        return value

    @classmethod
    def _read_oxide(cls, sample, oxide, optional=False):

        raw_value = getattr(sample, oxide, None)

        if optional and (
            raw_value is None
            or raw_value == ""
        ):
            return 0.0

        if raw_value is None:
            raise ValueError(
                f"Missing oxide in sample: {oxide}."
            )

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{oxide} must be numeric."
            )

        if optional and math.isnan(value):
            return 0.0

        if not math.isfinite(value):
            raise ValueError(
                f"{oxide} must be finite."
            )

        if value < 0.0:
            raise ValueError(
                f"{oxide} cannot be negative."
            )

        return value