from collections.abc import Mapping
from copy import deepcopy
import math

from models.model_manager import ModelManager
from models.corrections.crystal_manager import CrystalManager
from models.corrections.vesicle_manager import VesicleManager


class ViscosityEngine:
    """Calculate viscosities and compare models for one sample at one point.

    Logarithmic results use log10 and viscosity in Pa·s. Relative-viscosity
    factors are dimensionless. Model comparisons return results and errors
    separately, so the caller can leave gaps in curves at invalid points.
    """

    MODEL_MANAGERS = {
        "melt": "melt_manager",
        "crystal": "crystal_manager",
        "vesicle": "vesicle_manager",
    }

    OUTPUT_PHASES = {
        "log10_eta_m": ("melt",),
        "log10_eta_mc": ("melt", "crystal"),
        "log10_eta_mb": ("melt", "vesicle"),
        "log10_eta_mcb": ("melt", "crystal", "vesicle"),
        "eta_r_c": ("crystal",),
        "eta_r_b": ("vesicle",),
    }

    DEFAULT_COMPARISON_OUTPUTS = {
        "melt": ("log10_eta_m",),
        "crystal": ("eta_r_c",),
        "vesicle": ("eta_r_b",),
    }

    SAMPLE_PHYSICAL_ATTRIBUTES = {
        "temperature": ("temperature", "Temperature", "°C"),
        "water": ("H2O", "H₂O", "wt.%"),
        "crystals": ("crystals", "Crystals", "vol.%"),
        "vesicles": ("Vesicles", "Vesicles", "vol.%"),
    }

    # --- Model managers ---------------------------------------------

    def __init__(self):
        """Initialize the liquid, crystal and vesicle model managers."""
        self.melt_manager = ModelManager()
        self.crystal_manager = CrystalManager()
        self.vesicle_manager = VesicleManager()

    def get_model_names(self, group):
        """Return available model names in their manager's display order."""
        if group not in self.MODEL_MANAGERS:
            raise ValueError(f"Unknown model group: {group}")
        manager = getattr(self, self.MODEL_MANAGERS[group])
        return tuple(manager.models.keys())

    # --- Output and correction-factor validation --------------------

    @classmethod
    def _normalize_outputs(cls, outputs):
        """Validate output names while retaining the requested order."""
        if outputs is None:
            return tuple(cls.OUTPUT_PHASES)
        if isinstance(outputs, str):
            outputs = (outputs,)
        else:
            outputs = tuple(outputs)
        if not outputs:
            raise ValueError("Select at least one viscosity output.")
        for name in outputs:
            if not isinstance(name, str) or name not in cls.OUTPUT_PHASES:
                raise ValueError(f"Unknown viscosity output: {name}")
        return tuple(dict.fromkeys(outputs))

    @staticmethod
    def _get_correction_factor(result, phase):
        """Extract and validate a positive relative-viscosity factor."""
        value = float(result["relative_viscosity"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {phase} relative-viscosity factor.")
        return value

    # --- Complete viscosity calculation -----------------------------

    def calculate(
        self,
        sample,
        melt_model,
        melt_parameters,
        crystal_model,
        vesicle_model,
        crystal_parameters,
        vesicle_parameters,
    ):
        """Calculate all four viscosities and both relative-viscosity factors.

        This is the entry point used by MainWindow and the existing plots.
        Its arguments and six result keys follow the existing MVL interface.
        """
        settings = {
            "melt_model": melt_model,
            "melt_parameters": melt_parameters,
            "crystal_model": crystal_model,
            "crystal_parameters": crystal_parameters,
            "vesicle_model": vesicle_model,
            "vesicle_parameters": vesicle_parameters,
        }
        return self.calculate_outputs(sample, settings)

    def calculate_outputs(self, sample, settings, outputs=None):
        """Calculate only the phases needed by the requested output keys.

        settings uses the six keys returned by ModelsParametersPanel.
        Settings for unused phases may be omitted. With outputs=None, all
        six canonical results are calculated. Numerical/model errors are
        raised to the caller. This low-level method uses the supplied sample;
        compare_models() creates independent snapshots before calling it.
        """
        if not isinstance(settings, Mapping):
            raise TypeError("Model settings must be a mapping.")
        outputs = self._normalize_outputs(outputs)
        required = {
            phase for output in outputs for phase in self.OUTPUT_PHASES[output]
        }

        log10_eta_m = None
        eta_r_c = None
        eta_r_b = None

        if "melt" in required:
            melt_result = self.melt_manager.get_model(
                settings["melt_model"]
            ).calculate(sample, **(settings.get("melt_parameters") or {}))
            log10_eta_m = float(melt_result["log10_eta"])
            if not math.isfinite(log10_eta_m):
                raise ValueError("Invalid melt viscosity result.")

        if "crystal" in required:
            crystal_result = self.crystal_manager.get_model(
                settings["crystal_model"]
            ).calculate(
                crystal_fraction=sample.crystals / 100.0,
                **(settings.get("crystal_parameters") or {}),
            )
            eta_r_c = self._get_correction_factor(crystal_result, "crystal")

        if "vesicle" in required:
            # Existing MVL contract: these models receive vesicles in vol.%.
            vesicle_result = self.vesicle_manager.get_model(
                settings["vesicle_model"]
            ).calculate(
                vesicle_fraction=sample.Vesicles,
                **(settings.get("vesicle_parameters") or {}),
            )
            eta_r_b = self._get_correction_factor(vesicle_result, "bubble")

        result = {}
        for output in outputs:
            if output == "eta_r_c":
                result[output] = eta_r_c
            elif output == "eta_r_b":
                result[output] = eta_r_b
            else:
                # Add logarithms of the multiplicative correction factors.
                value = log10_eta_m
                phases = self.OUTPUT_PHASES[output]
                if "crystal" in phases:
                    value += math.log10(eta_r_c)
                if "vesicle" in phases:
                    value += math.log10(eta_r_b)
                result[output] = value
        return result

    # --- Conditions fixed by a melt model ---------------------------

    def _validate_comparison_melt_inputs(self, sample, model_name):
        """Reject a comparison that conflicts with declared fixed inputs.

        For example, a dry model declaring fixed water=0 cannot be included
        for a shared sample containing H2O>0. The sample is never adjusted
        here. Model-specific calibration/domain checks remain in calculate().
        """
        model = self.melt_manager.get_model(model_name)
        fixed = getattr(model, "fixed_melt_physical_parameters", {}) or {}
        for parameter, expected in fixed.items():
            attribute, label, unit = self.SAMPLE_PHYSICAL_ATTRIBUTES.get(
                parameter, (parameter, parameter, "")
            )
            if not hasattr(sample, attribute):
                raise ValueError(
                    f"Cannot check {model_name}: the shared sample has no {label} value."
                )
            expected = float(expected)
            actual = float(getattr(sample, attribute))
            if not math.isfinite(expected) or not math.isfinite(actual):
                raise ValueError(f"{label} must have a finite value for {model_name}.")
            if actual != expected:
                suffix = f" {unit}" if unit else ""
                raise ValueError(
                    f"{model_name} requires {label} = {expected:g}{suffix}; "
                    f"the shared sample has {actual:g}{suffix}. "
                    "Use matching shared inputs or omit this model."
                )

    # --- Same-sample model comparison -------------------------------

    def compare_models(
        self,
        sample,
        group,
        models,
        base_settings=None,
        outputs=None,
    ):
        """Compare selected models from one group at one physical point.

        Arguments
        ---------
        sample:
            The common sample, with composition and physical properties.
        group:
            "melt", "crystal" or "vesicle".
        models:
            An ordered mapping {model_name: parameter_dict}. Each dictionary
            belongs only to that model; None means use its calculate defaults.
            The GUI must supply equal values for shared physical parameters
            stored in these dictionaries, such as strain rate, using each
            model's own parameter name. Parameter units are not transformed.
        base_settings:
            The usual ModelsParametersPanel settings. Models/parameters of
            the other groups stay fixed. Only required phases need settings.
        outputs:
            One output key or an iterable of keys. Defaults to log10_eta_m,
            eta_r_c or eta_r_b for melt, crystal or vesicle comparisons.
            Every output must depend on the group being compared.

        Returns
        -------
        A dictionary with group, sample_name, outputs, results and errors.
        results maps successful model names to dictionaries of requested
        values. errors maps failed names to messages. Failed calculations
        never produce a numerical placeholder. Configuration errors raise;
        failures of individual models are collected and do not stop others.

        A parameter sweep calls this method for each point. For each model,
        both the sample and settings are deep-copied from the same snapshot.
        The returned factors eta_r_c and eta_r_b are linear; the GUI may take
        their log10 when plotting a logarithmic relative correction.
        """
        if group not in self.MODEL_MANAGERS:
            raise ValueError(f"Unknown model group: {group}")
        if not isinstance(models, Mapping):
            raise TypeError("models must map model names to parameter dictionaries.")
        if not models:
            raise ValueError("Select at least one model to compare.")
        for name in models:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Every compared model must have a non-empty name.")
        if base_settings is None:
            base_settings = {}
        if not isinstance(base_settings, Mapping):
            raise TypeError("base_settings must be a mapping.")

        outputs = self._normalize_outputs(
            self.DEFAULT_COMPARISON_OUTPUTS[group] if outputs is None else outputs
        )
        for output in outputs:
            if group not in self.OUTPUT_PHASES[output]:
                raise ValueError(f"{output} does not depend on the {group} model.")
        needs_melt = any("melt" in self.OUTPUT_PHASES[key] for key in outputs)

        sample_snapshot = deepcopy(sample)
        settings_snapshot = deepcopy(dict(base_settings))
        results, errors = {}, {}

        for model_name, parameters in models.items():
            try:
                if parameters is None:
                    parameters = {}
                if not isinstance(parameters, Mapping):
                    raise TypeError("Model parameters must be a mapping or None.")

                current_sample = deepcopy(sample_snapshot)
                settings = deepcopy(settings_snapshot)
                settings[f"{group}_model"] = model_name
                settings[f"{group}_parameters"] = deepcopy(dict(parameters))

                if needs_melt:
                    self._validate_comparison_melt_inputs(
                        current_sample, settings["melt_model"]
                    )
                results[model_name] = self.calculate_outputs(
                    current_sample, settings, outputs
                )
            except Exception as error:
                errors[model_name] = str(error) or type(error).__name__

        return {
            "group": group,
            "sample_name": str(getattr(sample_snapshot, "name", "")),
            "outputs": outputs,
            "results": results,
            "errors": errors,
        }