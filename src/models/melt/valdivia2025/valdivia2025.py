import math

from models.base_models import BaseViscosityModel


MOLAR_MASS = {
    "FeO": 71.8444,
    "Fe2O3": 159.6882,
}


class Valdivia2025(BaseViscosityModel):

    name = "Valdivia et al. (2025)"

    # Components used to place the sample on the anhydrous
    # AND0-AND100 compositional scale.
    DRY_OXIDES = (
        "SiO2",
        "TiO2",
        "Al2O3",
        "FeO",
        "MnO",
        "MgO",
        "CaO",
        "Na2O",
        "K2O",
        "P2O5",
        "F2O_1",
    )

    required_oxides = list(DRY_OXIDES)

    # If available, Fe2O3 is converted to equivalent FeO and
    # added to the FeO entered for the sample.
    optional_oxides = [
        "Fe2O3",
    ]

    required_melt_physical_parameters = [
        "temperature",
        "water",
    ]

    # Valdivia et al. (2025) parameterized anhydrous melts.
    # ModelsParametersPanel and the 2D/3D plot windows use this
    # declaration to set H2O to zero and disable its widgets.
    fixed_melt_physical_parameters = {
        "water": 0.0,
    }

    parameters = {}
    model_parameter_limits = {}

    model_physical_limits = {
        "H₂O": (0.0, 0.0),
    }

    model_composition_limits = {
        "relative_transition_metals": (0.0, 100.0),
    }

    # Read-only value displayed by ModelsParametersPanel.
    computed_fields = {
        "relative_transition_metals": {
            "label": "Relative transition metals (%)",
            "method": "calculate_relative_transition_metals",
            "decimals": 3,
            "tooltip": (
                "Calculated automatically on an anhydrous basis "
                "from FeOtot + TiO₂ + MnO, relative to AND100."
            ),
        },
    }

    A = -2.93

    # AND100 composition used in the official spreadsheet (wt%).
    AND100_FEO_TOTAL = 6.76731707317073
    AND100_TIO2 = 0.793663414634147
    AND100_MNO = 0.17289512195122
    AND100_TOTAL = 99.69246829268295
    AND0_TOTAL = 99.96227272727272

    AND100_TRANSITION_METALS = (
        AND100_FEO_TOTAL
        + AND100_TIO2
        + AND100_MNO
    )

    NUMERICAL_TOLERANCE = 1.0e-9

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

        if water_wt > 0.0:
            raise ValueError(
                "Valdivia et al. (2025) is parameterized "
                "for anhydrous melts (H₂O = 0 wt%). "
                f"The sample contains {water_wt:.4g} wt% H₂O."
            )

        transition_data = self.calculate_transition_metal_data(
            sample
        )

        relative_transition_metals = transition_data[
            "relative_transition_metals"
        ]

        minimum, maximum = self.model_composition_limits[
            "relative_transition_metals"
        ]

        if (
            relative_transition_metals
            < minimum - self.NUMERICAL_TOLERANCE
            or relative_transition_metals
            > maximum + self.NUMERICAL_TOLERANCE
        ):
            raise ValueError(
                "Valdivia et al. (2025) is parameterized "
                "for compositions between AND0 and AND100 "
                f"({minimum:g}–{maximum:g}% relative transition "
                "metals). The calculated value is "
                f"{relative_transition_metals:.4g}%."
            )

        # Remove only negligible floating-point drift at the endmembers.
        relative_transition_metals = min(
            maximum,
            max(minimum, relative_transition_metals),
        )

        glass_transition_c = self.calculate_tg(
            relative_transition_metals
        )

        glass_transition_k = (
            glass_transition_c + 273.15
        )

        fragility = self.calculate_fragility(
            relative_transition_metals
        )

        log_eta = self.calculate_myega(
            temperature_k,
            glass_transition_k,
            fragility,
        )

        if not math.isfinite(log_eta):
            raise ValueError(
                "Calculated Valdivia viscosity is not valid."
            )

        return {
            "model": self.name,
            "log10_eta": log_eta,
            "details": {
                "temperature_C": temperature_c,
                "temperature_K": temperature_k,
                "H2O_wt_percent": water_wt,
                "relative_transition_metals": (
                    relative_transition_metals
                ),
                "FeO_input_wt_percent": transition_data[
                    "FeO_input_wt_percent"
                ],
                "Fe2O3_input_wt_percent": transition_data[
                    "Fe2O3_input_wt_percent"
                ],
                "FeO_from_Fe2O3_wt_percent": transition_data[
                    "FeO_from_Fe2O3_wt_percent"
                ],
                "FeO_total_equivalent_wt_percent": transition_data[
                    "FeO_total_equivalent_wt_percent"
                ],
                "transition_metals_wt_percent": transition_data[
                    "transition_metals_wt_percent"
                ],
                "dry_total_wt_percent": transition_data[
                    "dry_total_wt_percent"
                ],
                "normalized_transition_metals_wt_percent": (
                    transition_data[
                        "normalized_transition_metals_wt_percent"
                    ]
                ),
                "Tg_C": glass_transition_c,
                "Tg_K": glass_transition_k,
                "fragility": fragility,
            },
        }

    # =====================================================
    # TRANSITION-METAL CONTENT
    # =====================================================

    def calculate_relative_transition_metals(self, sample):

        return self.calculate_transition_metal_data(
            sample
        )["relative_transition_metals"]

    def calculate_transition_metal_data(self, sample):

        oxide_wt = {
            oxide: self._read_oxide(sample, oxide)
            for oxide in self.DRY_OXIDES
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

        transition_metals_wt = (
            feo_total_wt
            + oxide_wt["TiO2"]
            + oxide_wt["MnO"]
        )

        # The total is expressed on the same FeOtot basis used by
        # the published AND100 reference composition.
        dry_total_wt = (
            sum(oxide_wt.values())
            + feo_from_fe2o3_wt
        )

        if (
            not math.isfinite(dry_total_wt)
            or dry_total_wt <= 0.0
        ):
            raise ValueError(
                "The total anhydrous composition must be "
                "greater than zero."
            )

        transition_metal_fraction = (
            transition_metals_wt
            / dry_total_wt
        )

        # The official spreadsheet generates intermediate compositions
        # by mixing the unnormalised AND0 and AND100 endmembers. The
        # equation below analytically inverts that mixing relation, so
        # the same TM value is recovered before and after MVL normalises
        # the sample composition to 100 wt%.
        mixing_denominator = (
            self.AND100_TRANSITION_METALS
            - transition_metal_fraction
            * (self.AND100_TOTAL - self.AND0_TOTAL)
        )

        if mixing_denominator <= 0.0:
            raise ValueError(
                "Invalid transition-metal composition for the "
                "AND0-AND100 mixing relation."
            )

        relative_transition_metals = (
            100.0
            * transition_metal_fraction
            * self.AND0_TOTAL
            / mixing_denominator
        )

        normalized_transition_metals_wt = (
            100.0 * transition_metal_fraction
        )

        if not math.isfinite(relative_transition_metals):
            raise ValueError(
                "Calculated relative transition-metal content "
                "is not valid."
            )

        return {
            "relative_transition_metals": (
                relative_transition_metals
            ),
            "FeO_input_wt_percent": oxide_wt["FeO"],
            "Fe2O3_input_wt_percent": fe2o3_wt,
            "FeO_from_Fe2O3_wt_percent": (
                feo_from_fe2o3_wt
            ),
            "FeO_total_equivalent_wt_percent": feo_total_wt,
            "transition_metals_wt_percent": (
                transition_metals_wt
            ),
            "dry_total_wt_percent": dry_total_wt,
            "normalized_transition_metals_wt_percent": (
                normalized_transition_metals_wt
            ),
        }

    # =====================================================
    # GLASS TRANSITION AND FRAGILITY
    # =====================================================

    @staticmethod
    def calculate_tg(relative_transition_metals):

        return (
            737.0
            - 0.26077 * relative_transition_metals
            - 0.00569 * relative_transition_metals**2
        )

    @staticmethod
    def calculate_fragility(relative_transition_metals):

        return (
            31.8
            - 0.013 * relative_transition_metals
        )

    # =====================================================
    # MYEGA
    # =====================================================

    def calculate_myega(
        self,
        temperature_k,
        glass_transition_k,
        fragility,
    ):

        delta = 12.0 - self.A

        exponent = (
            (
                fragility / delta
                - 1.0
            )
            * (
                glass_transition_k / temperature_k
                - 1.0
            )
        )

        return (
            self.A
            + delta
            * (glass_transition_k / temperature_k)
            * math.exp(exponent)
        )

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

        if (
            not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                f"{oxide} cannot be negative or non-finite."
            )

        return value