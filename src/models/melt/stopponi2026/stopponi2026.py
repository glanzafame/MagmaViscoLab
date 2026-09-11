import math

from models.base_models import BaseViscosityModel


MOLAR_MASS = {
    "SiO2": 60.0843,
    "Al2O3": 101.9613,
    "Na2O": 61.9789,
    "K2O": 94.1960,
    "H2O": 18.0153,
}


class Stopponi2026(BaseViscosityModel):

    name = "Stopponi et al. (2026)"

    DRY_OXIDES = (
        "SiO2",
        "Al2O3",
        "Na2O",
        "K2O",
    )

    required_oxides = list(DRY_OXIDES)

    required_melt_physical_parameters = [
        "temperature",
        "water",
    ]

    # Physical values imposed by the model in the MainWindow and plots.
    fixed_melt_physical_parameters = {
        "water": 0.0,
    }

    parameters = {}
    model_parameter_limits = {}

    # Read-only values displayed automatically by ModelsParametersPanel.
    computed_fields = {
        "al2o3_mol": {
            "label": "Al₂O₃ (mol%)",
            "method": "calculate_al2o3_mol_percent",
            "decimals": 3,
            "tooltip": (
                "Calculated automatically from the sample composition."
            ),
        },
        "na2o_mol": {
            "label": "Na₂O (mol%)",
            "method": "calculate_na2o_mol_percent",
            "decimals": 3,
            "tooltip": (
                "Calculated automatically from the sample composition."
            ),
        },
        "k2o_mol": {
            "label": "K₂O (mol%)",
            "method": "calculate_k2o_mol_percent",
            "decimals": 3,
            "tooltip": (
                "Calculated automatically from the sample composition."
            ),
        },
        "excess_na2o": {
            "label": "Excess Na₂O (mol%)",
            "method": "calculate_excess_alkalis",
            "decimals": 3,
            "tooltip": (
                "Calculated on an anhydrous basis as "
                "Na₂O + K₂O - Al₂O₃, in mol%."
            ),
        },
    }

    model_physical_limits = {
        "H₂O": (0.0, 0.0),
    }

    # No hard composition interval is enforced. Values outside the
    # experimental range of Stopponi et al. (2026) are extrapolations.
    model_composition_limits = {}

    ETA_INF_A = -1.55
    ETA_INF_B = 6.35
    ETA_INF_BASE = 0.55

    TG_A = 1034.0
    TG_EXPONENT = -0.10

    FRAGILITY_A = 22.95
    FRAGILITY_B = 0.70

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

        mole_data = self.calculate_mole_data(sample)
        water_wt = mole_data["wt_percent"]["H2O"]

        if water_wt > 0.0:
            raise ValueError(
                "Stopponi et al. (2026) is calibrated for "
                "anhydrous melts (H₂O = 0 wt%). "
                f"The sample contains {water_wt:.4g} wt% H₂O, "
                "which is outside the model calibration."
            )

        excess_alkalis = mole_data[
            "excess_alkalis_mol_percent"
        ]

        log_eta_inf = self.calculate_eta_inf(
            excess_alkalis
        )

        glass_transition = self.calculate_tg(
            excess_alkalis
        )

        fragility = self.calculate_fragility(
            excess_alkalis
        )

        log_eta = self.calculate_myega(
            temperature_k,
            glass_transition,
            fragility,
            log_eta_inf,
        )

        if not math.isfinite(log_eta):
            raise ValueError(
                "Calculated Stopponi viscosity is not valid."
            )

        return {
            "model": self.name,
            "log10_eta": log_eta,
            "details": {
                "temperature_C": temperature_c,
                "temperature_K": temperature_k,
                "H2O_wt_percent": water_wt,
                "H2O_mol_percent": (
                    mole_data["mole_percent"]["H2O"]
                ),
                "mole_percent": mole_data["mole_percent"],
                "dry_mole_percent": (
                    mole_data["dry_mole_percent"]
                ),
                "excess_alkalis_mol_percent": excess_alkalis,
                "log10_eta_inf": log_eta_inf,
                "Tg_K": glass_transition,
                "fragility": fragility,
            },
        }

    # =====================================================
    # WT% -> MOL%
    # =====================================================

    def calculate_mole_data(self, sample):

        wt_percent = {
            oxide: self._read_oxide(sample, oxide)
            for oxide in self.DRY_OXIDES
        }

        wt_percent["H2O"] = self._read_oxide(
            sample,
            "H2O",
        )

        moles = {
            oxide: value / MOLAR_MASS[oxide]
            for oxide, value in wt_percent.items()
        }

        dry_total_moles = sum(
            moles[oxide]
            for oxide in self.DRY_OXIDES
        )

        if (
            not math.isfinite(dry_total_moles)
            or dry_total_moles <= 0.0
        ):
            raise ValueError(
                "The total amount of anhydrous "
                "SiO₂-Al₂O₃-Na₂O-K₂O components must "
                "be greater than zero."
            )

        total_moles = dry_total_moles + moles["H2O"]

        if (
            not math.isfinite(total_moles)
            or total_moles <= 0.0
        ):
            raise ValueError(
                "The total number of moles must be greater "
                "than zero."
            )

        mole_percent = {
            oxide: 100.0 * value / total_moles
            for oxide, value in moles.items()
        }

        dry_mole_percent = {
            oxide: 100.0 * moles[oxide] / dry_total_moles
            for oxide in self.DRY_OXIDES
        }

        excess_alkalis = (
            dry_mole_percent["Na2O"]
            + dry_mole_percent["K2O"]
            - dry_mole_percent["Al2O3"]
        )

        if not math.isfinite(excess_alkalis):
            raise ValueError(
                "Calculated excess Na₂O is not valid."
            )

        return {
            "wt_percent": wt_percent,
            "moles": moles,
            "total_moles": total_moles,
            "dry_total_moles": dry_total_moles,
            "mole_percent": mole_percent,
            "dry_mole_percent": dry_mole_percent,
            "excess_alkalis_mol_percent": excess_alkalis,
        }

    def calculate_sio2_mol_percent(self, sample):

        return self.calculate_mole_data(
            sample
        )["mole_percent"]["SiO2"]

    def calculate_al2o3_mol_percent(self, sample):

        return self.calculate_mole_data(
            sample
        )["mole_percent"]["Al2O3"]

    def calculate_na2o_mol_percent(self, sample):

        return self.calculate_mole_data(
            sample
        )["mole_percent"]["Na2O"]

    def calculate_k2o_mol_percent(self, sample):

        return self.calculate_mole_data(
            sample
        )["mole_percent"]["K2O"]

    def calculate_h2o_mol_percent(self, sample):

        return self.calculate_mole_data(
            sample
        )["mole_percent"]["H2O"]

    # =====================================================
    # EXCESS ALKALIS
    # =====================================================

    def calculate_excess_alkalis(self, sample):

        return self.calculate_mole_data(
            sample
        )["excess_alkalis_mol_percent"]

    # =====================================================
    # MODEL PARAMETERS
    # =====================================================

    def calculate_eta_inf(self, excess_alkalis):

        return (
            self.ETA_INF_A
            - self.ETA_INF_B
            * self.ETA_INF_BASE**excess_alkalis
        )

    def calculate_tg(self, excess_alkalis):

        base = 1.0 + excess_alkalis

        if base <= 0.0:
            raise ValueError(
                "Excess Na₂O must be greater than -1 mol% "
                "for the Stopponi Tg equation."
            )

        return (
            self.TG_A
            * base**self.TG_EXPONENT
        )

    def calculate_fragility(self, excess_alkalis):

        return (
            self.FRAGILITY_A
            + self.FRAGILITY_B * excess_alkalis
        )

    # =====================================================
    # MYEGA
    # =====================================================

    def calculate_myega(
        self,
        temperature_k,
        glass_transition,
        fragility,
        log_eta_inf,
    ):

        delta = 12.0 - log_eta_inf

        if delta <= 0.0:
            raise ValueError(
                "Invalid viscosity at infinite temperature."
            )

        exponent = (
            (
                fragility / delta
                - 1.0
            )
            * (
                glass_transition / temperature_k
                - 1.0
            )
        )

        return (
            log_eta_inf
            + delta
            * (glass_transition / temperature_k)
            * math.exp(exponent)
        )

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    @staticmethod
    def _read_number(sample, attribute, label):

        if not hasattr(sample, attribute):
            raise ValueError(
                f"Missing sample value: {label}."
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

    def _read_oxide(self, sample, oxide):

        value = self._read_number(
            sample,
            oxide,
            oxide,
        )

        if value < 0.0:
            raise ValueError(
                f"{oxide} cannot be negative."
            )

        return value