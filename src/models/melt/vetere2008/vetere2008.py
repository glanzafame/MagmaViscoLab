import math

from models.base_models import BaseViscosityModel


MOLAR_MASS = {
    "FeO": 71.8444,
    "Fe2O3": 159.6882,
}


class Vetere2008(BaseViscosityModel):

    name = "Vetere et al. (2008)"

    # Keep the complete oxide composition editable. The Vetere equation
    # uses H2O and iron redox, while the remaining oxides define whether
    # the sample is compositionally comparable to the studied andesites.
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
        "H2O",
        "F2O_1",
    ]

    required_melt_physical_parameters = [
        "temperature",
        "water",
    ]

    # Fe2+/Fetot is calculated from FeO and Fe2O3 and is therefore
    # no longer an editable model parameter.
    parameters = {}
    model_parameter_limits = {}

    computed_fields = {
        "fe_ratio": {
            "label": "Fe²⁺/Feₜₒₜ",
            "method": "calculate_fe_ratio",
            "decimals": 4,
            "tooltip": (
                "Calculated automatically from FeO (ferrous iron) "
                "and Fe₂O₃ (ferric iron), using moles of Fe atoms."
            ),
        },
    }

    model_composition_limits = {
        "Fe2+/Fetot": (0.3, 1.0),
    }

    model_physical_limits = {
        "H₂O": (0.0, 6.2),
    }

    model_output_limits = {
        "log10_eta": (0.5, 13.0),
    }

    LOG10_ETA_AT_TG = 12.0

    # Constants of the published empirical equation.
    INTERCEPT = -4.00
    TERM_1_NUMERATOR = 5515.26
    TERM_1_TEMPERATURE = 275.43
    TERM_2_NUMERATOR = 1907.23
    TERM_2_TEMPERATURE = 626.24
    REDOX_COEFFICIENT = 218.66
    WATER_COEFFICIENT = 469.1

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

        if temperature_k <= self.TERM_2_TEMPERATURE:
            raise ValueError(
                "Vetere et al. (2008) requires a temperature "
                f"greater than {self.TERM_2_TEMPERATURE:g} K."
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
                "Vetere et al. (2008) is calibrated for "
                f"{water_min:g}–{water_max:g} wt% H₂O. "
                f"The sample contains {water_wt:.4g} wt% H₂O."
            )

        iron_data = self.calculate_iron_redox_data(
            sample
        )

        fe_ratio = iron_data["Fe2+/Fetot"]

        ratio_min, ratio_max = self.model_composition_limits[
            "Fe2+/Fetot"
        ]

        if not ratio_min < fe_ratio <= ratio_max:
            raise ValueError(
                "Vetere et al. (2008) requires "
                f"Fe²⁺/Feₜₒₜ > {ratio_min:g} and "
                f"≤ {ratio_max:g}. The calculated value is "
                f"{fe_ratio:.4f}."
            )

        glass_transition_k = (
            self.calculate_glass_transition_temperature(
                water_wt,
                fe_ratio,
            )
        )

        if temperature_k <= glass_transition_k:
            raise ValueError(
                "Vetere et al. (2008) should be applied only "
                "above the glass-transition temperature. "
                f"For this sample, Tg = {glass_transition_k:.2f} K "
                f"({glass_transition_k - 273.15:.2f} °C)."
            )

        equation_data = self.calculate_equation_terms(
            temperature_k,
            water_wt,
            fe_ratio,
        )

        log_eta = equation_data["log10_eta"]

        if not math.isfinite(log_eta):
            raise ValueError(
                "Calculated Vetere viscosity is not valid."
            )

        eta_min, eta_max = self.model_output_limits[
            "log10_eta"
        ]

        if not eta_min <= log_eta <= eta_max:
            raise ValueError(
                "Vetere et al. (2008) prediction is outside "
                "the experimental viscosity range "
                f"log₁₀(η / Pa s) = {eta_min:g}–{eta_max:g}. "
                f"The calculated value is {log_eta:.4g}."
            )

        return {
            "model": self.name,
            "log10_eta": log_eta,
            "details": {
                "temperature_C": temperature_c,
                "temperature_K": temperature_k,
                "H2O_wt_percent": water_wt,
                "FeO_wt_percent": iron_data[
                    "FeO_wt_percent"
                ],
                "Fe2O3_wt_percent": iron_data[
                    "Fe2O3_wt_percent"
                ],
                "Fe2_moles": iron_data["Fe2_moles"],
                "Fe3_moles": iron_data["Fe3_moles"],
                "Fe_total_moles": iron_data[
                    "Fe_total_moles"
                ],
                "Fe2+/Fetot": fe_ratio,
                "Tg_C": glass_transition_k - 273.15,
                "Tg_K": glass_transition_k,
                "term_1": equation_data["term_1"],
                "term_2": equation_data["term_2"],
                "redox_term": equation_data["redox_term"],
                "water_term": equation_data["water_term"],
            },
        }

    # =====================================================
    # FeO + Fe2O3 -> Fe2+/Fetot
    # =====================================================

    def calculate_fe_ratio(self, sample):

        return self.calculate_iron_redox_data(
            sample
        )["Fe2+/Fetot"]

    def calculate_iron_redox_data(self, sample):

        feo_wt = self._read_oxide(
            sample,
            "FeO",
        )

        fe2o3_wt = self._read_oxide(
            sample,
            "Fe2O3",
        )

        # One mole of FeO contains one mole of Fe2+.
        fe2_moles = (
            feo_wt / MOLAR_MASS["FeO"]
        )

        # One mole of Fe2O3 contains two moles of Fe3+.
        fe3_moles = (
            2.0 * fe2o3_wt
            / MOLAR_MASS["Fe2O3"]
        )

        fe_total_moles = fe2_moles + fe3_moles

        if (
            not math.isfinite(fe_total_moles)
            or fe_total_moles <= 0.0
        ):
            raise ValueError(
                "FeO + Fe₂O₃ must contain a positive "
                "amount of total iron."
            )

        fe_ratio = fe2_moles / fe_total_moles

        if not math.isfinite(fe_ratio):
            raise ValueError(
                "Calculated Fe²⁺/Feₜₒₜ is not valid."
            )

        return {
            "FeO_wt_percent": feo_wt,
            "Fe2O3_wt_percent": fe2o3_wt,
            "Fe2_moles": fe2_moles,
            "Fe3_moles": fe3_moles,
            "Fe_total_moles": fe_total_moles,
            "Fe2+/Fetot": fe_ratio,
        }

    # =====================================================
    # VETERE EQUATION
    # =====================================================

    def calculate_equation_terms(
        self,
        temperature_k,
        water_wt,
        fe_ratio,
    ):

        if temperature_k <= self.TERM_2_TEMPERATURE:
            raise ValueError(
                "Temperature is outside the mathematical domain "
                "of the Vetere equation."
            )

        term_1 = (
            self.TERM_1_NUMERATOR
            / (
                temperature_k
                - self.TERM_1_TEMPERATURE
            )
        )

        term_2 = (
            self.TERM_2_NUMERATOR
            / (
                temperature_k
                - self.TERM_2_TEMPERATURE
            )
        )

        redox_term = math.exp(
            self.REDOX_COEFFICIENT
            / (fe_ratio * temperature_k)
        )

        water_term = math.exp(
            -self.WATER_COEFFICIENT
            * water_wt
            / temperature_k
        )

        log_eta = (
            self.INTERCEPT
            + term_1
            + term_2
            * redox_term
            * water_term
        )

        return {
            "log10_eta": log_eta,
            "term_1": term_1,
            "term_2": term_2,
            "redox_term": redox_term,
            "water_term": water_term,
        }

    # =====================================================
    # GLASS-TRANSITION TEMPERATURE
    # =====================================================

    def calculate_glass_transition_temperature(
        self,
        water_wt,
        fe_ratio,
    ):

        # Tg is determined from log10(eta / Pa s) = 12.
        lower = self.TERM_2_TEMPERATURE + 1.0e-6
        upper = 3000.0

        lower_value = (
            self.calculate_equation_terms(
                lower,
                water_wt,
                fe_ratio,
            )["log10_eta"]
            - self.LOG10_ETA_AT_TG
        )

        upper_value = (
            self.calculate_equation_terms(
                upper,
                water_wt,
                fe_ratio,
            )["log10_eta"]
            - self.LOG10_ETA_AT_TG
        )

        if lower_value * upper_value > 0.0:
            raise ValueError(
                "Unable to determine Tg for the selected "
                "H₂O content and Fe²⁺/Feₜₒₜ ratio."
            )

        for _ in range(100):

            midpoint = 0.5 * (lower + upper)

            midpoint_value = (
                self.calculate_equation_terms(
                    midpoint,
                    water_wt,
                    fe_ratio,
                )["log10_eta"]
                - self.LOG10_ETA_AT_TG
            )

            if midpoint_value > 0.0:
                lower = midpoint
            else:
                upper = midpoint

        return 0.5 * (lower + upper)

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
    def _read_oxide(cls, sample, oxide):

        if not hasattr(sample, oxide):
            raise ValueError(
                f"Missing oxide in sample: {oxide}."
            )

        try:
            value = float(getattr(sample, oxide))
        except (TypeError, ValueError):
            raise ValueError(
                f"{oxide} must be numeric."
            )

        if (
            not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                f"{oxide} cannot be negative or non-finite."
            )

        return value