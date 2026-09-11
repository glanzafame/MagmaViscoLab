import math

from models.base_models import BaseViscosityModel


class DiGenova2023(BaseViscosityModel):

    name = "Di Genova et al. (2023)"

    DRY_OXIDE_MOLAR_MASSES = {
        "SiO2": 60.0843,
        "TiO2": 79.8660,
        "Al2O3": 101.9613,
        "FeO": 71.8440,
        "MnO": 70.9374,
        "MgO": 40.3044,
        "CaO": 56.0774,
        "Na2O": 61.9789,
        "K2O": 94.1960,
        "P2O5": 141.9445,
        "F2O_1": 21.9974
    }

    H2O_MOLAR_MASS = 18.01528

    required_oxides = list(
        DRY_OXIDE_MOLAR_MASSES.keys()
    )

    required_melt_physical_parameters = [
        "temperature",
        "water"
    ]

    # No editable model-specific parameters.
    parameters = {}
    model_parameter_limits = {}

    # Read-only values displayed by ModelsParametersPanel.
    computed_fields = {
        "water_mol": {
            "label": "H₂O (mol%)",
            "method": "calculate_water_mol_percent",
            "decimals": 3,
            "tooltip": (
                "Calculated automatically from the sample H₂O "
                "content and oxide composition."
            )
        }
    }

    WATER_MOL_LIMITS = (0.0, 12.0)

    A = -2.93

    TG_DRY = 993.3
    TG_WATER = 136.0

    B = 0.2737
    C = 1.686
    D = -1.779

    M0 = 59.069
    M1 = -0.6281

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(self, sample):

        temperature_k = (
            float(sample.temperature)
            + 273.15
        )

        if (
            not math.isfinite(temperature_k)
            or temperature_k <= 0
        ):
            raise ValueError(
                "Temperature must be greater than 0 K."
            )

        water_data = self.calculate_water_data(sample)
        water_mol_percent = water_data["mol_percent"]

        minimum, maximum = self.WATER_MOL_LIMITS

        if not minimum <= water_mol_percent <= maximum:
            raise ValueError(
                "Di Genova et al. (2023) is calibrated "
                f"for {minimum:g}–{maximum:g} mol% H₂O. "
                f"The sample contains "
                f"{float(sample.H2O):.4g} wt% H₂O, "
                f"corresponding to "
                f"{water_mol_percent:.4g} mol% H₂O."
            )

        fragility = self.calculate_fragility(
            water_mol_percent
        )

        glass_transition = self.calculate_tg(
            water_mol_percent
        )

        log_eta = self.calculate_myega(
            temperature_k,
            glass_transition,
            fragility
        )

        if not math.isfinite(log_eta):
            raise ValueError(
                "Calculated Di Genova viscosity "
                "is not valid."
            )

        return {
            "model": self.name,
            "log10_eta": log_eta,
            "details": {
                "temperature_K": temperature_k,
                "H2O_wt_percent": float(sample.H2O),
                "H2O_mol_percent": water_mol_percent,
                "H2O_moles": water_data["water_moles"],
                "dry_oxide_moles": water_data["dry_moles"],
                "Tg_K": glass_transition,
                "fragility": fragility
            }
        }

    # =====================================================
    # WATER: WT% TO MOL%
    # =====================================================

    def calculate_water_mol_percent(self, sample):

        return self.calculate_water_data(
            sample
        )["mol_percent"]

    def calculate_water_data(self, sample):

        try:
            water_wt = float(sample.H2O)

        except (TypeError, ValueError, AttributeError):
            raise ValueError(
                "The sample H₂O content must be numeric."
            )

        if (
            not math.isfinite(water_wt)
            or water_wt < 0
        ):
            raise ValueError(
                "The sample H₂O content cannot "
                "be negative or non-finite."
            )

        dry_moles = 0.0

        for oxide, molar_mass in (
            self.DRY_OXIDE_MOLAR_MASSES.items()
        ):

            if not hasattr(sample, oxide):
                raise ValueError(
                    f"Missing oxide in sample: {oxide}."
                )

            try:
                oxide_wt = float(
                    getattr(sample, oxide)
                )

            except (TypeError, ValueError):
                raise ValueError(
                    f"{oxide} must be numeric."
                )

            if (
                not math.isfinite(oxide_wt)
                or oxide_wt < 0
            ):
                raise ValueError(
                    f"{oxide} cannot be negative "
                    "or non-finite."
                )

            dry_moles += oxide_wt / molar_mass

        if dry_moles <= 0:
            raise ValueError(
                "The total amount of dry oxides "
                "must be greater than zero."
            )

        water_moles = (
            water_wt / self.H2O_MOLAR_MASS
        )

        total_moles = (
            dry_moles + water_moles
        )

        water_mol_percent = (
            100.0
            * water_moles
            / total_moles
        )

        if not math.isfinite(water_mol_percent):
            raise ValueError(
                "Calculated H₂O mol% is not valid."
            )

        return {
            "mol_percent": water_mol_percent,
            "water_moles": water_moles,
            "dry_moles": dry_moles
        }

    # =====================================================
    # FRAGILITY
    # =====================================================

    def calculate_fragility(
        self,
        water_mol_percent
    ):

        return (
            self.M0
            + self.M1 * water_mol_percent
        )

    # =====================================================
    # GLASS TRANSITION TEMPERATURE
    # =====================================================

    def calculate_tg(
        self,
        water_mol_percent
    ):

        denominator = (
            self.B
            * (100.0 - water_mol_percent)
            + water_mol_percent
        )

        if denominator <= 0:
            raise ValueError(
                "Invalid H₂O content for "
                "Tg calculation."
            )

        w1 = (
            water_mol_percent
            / denominator
        )

        w2 = (
            self.B
            * (100.0 - water_mol_percent)
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
    # MYEGA EQUATION
    # =====================================================

    def calculate_myega(
        self,
        temperature_k,
        glass_transition,
        fragility
    ):

        delta = (
            12.0 - self.A
        )

        exponent = (
            (
                fragility / delta
                - 1.0
            )
            * (
                glass_transition
                / temperature_k
                - 1.0
            )
        )

        return (
            self.A
            + delta
            * (
                glass_transition
                / temperature_k
            )
            * math.exp(exponent)
        )