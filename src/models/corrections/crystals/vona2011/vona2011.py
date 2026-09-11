import math

from models.corrections.base_correction import BaseCrystalCorrection


class Vona2011(BaseCrystalCorrection):

    name = "Vona et al. (2011)"

    required_physical_parameters = [
        "crystals"
    ]

    
    parameters = {
        "rp": {
            "label": "R",
            "default": 2.0
        },
        "alpha": {
            "label": "α",
            "default": 0.06
        },
        "gamma": {
            "label": "γ̇ (s⁻¹)",
            "default": 0.1
        }
    }

    model_parameter_limits = {
        "rp": (1.0, None),
        "alpha": (0.0, None),
        "gamma": (0.0, None)
    }

    # Experimental ranges reported by Vona et al. (2011). They are
    # retained as metadata; the calculator can operate up to phi < phim.
    experimental_ranges = {
        "Crystals": (6.0, 27.0),
        "strain_rate": (0.02, 4.26),
        "rp": (5.57, 7.25),
        "Temperature": (1131.0, 1187.0)
    }

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        crystal_fraction,
        rp,
        alpha,
        gamma
    ):

        try:
            phi = float(crystal_fraction)
            rp = float(rp)
            alpha = float(alpha)
            gamma = float(gamma)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Crystal fraction and Vona model parameters "
                "must be numeric."
            ) from error

        values = {
            "Crystal fraction": phi,
            "Aspect ratio": rp,
            "α": alpha,
            "Strain rate": gamma
        }

        for label, value in values.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"{label} must be finite."
                )

        # =================================================
        # INPUT VALIDATION
        # =================================================

        if phi < 0.0:
            raise ValueError(
                "Crystal fraction cannot be negative."
            )

        if rp < 1.0:
            raise ValueError(
                "The Vona et al. (2011) packing relation "
                "requires an aspect ratio greater than or "
                "equal to 1."
            )

        if alpha < 0.0:
            raise ValueError(
                "α cannot be negative."
            )

        if gamma <= 0.0:
            raise ValueError(
                "Strain rate γ̇ must be greater than zero."
            )

        # =================================================
        # MAXIMUM PACKING FRACTION
        # =================================================

        phim = self.calculate_phim(rp)

        if (
            not math.isfinite(phim)
            or not 0.0 < phim <= 1.0
        ):
            raise ValueError(
                "Calculated maximum packing fraction "
                "is not valid."
            )

        if phi >= phim:
            raise ValueError(
                f"Crystal fraction ({phi:.4f}) must be "
                f"lower than maximum packing fraction "
                f"φm ({phim:.4f})."
            )

        # =================================================
        # VONA ET AL. (2011)
        # =================================================

        relative_packing = phi / phim
        base = 1.0 - relative_packing
        log10_strain_rate = math.log10(gamma)

        relative_consistency = base ** -2.0

        flow_index = (
            1.0
            + 2.0
            * alpha
            * math.log10(base)
        )

        exponent = (
            -2.0
            * (
                1.0
                - alpha * log10_strain_rate
            )
        )

        try:
            eta_relative = base ** exponent

        except OverflowError as error:
            raise ValueError(
                "Calculated crystal relative viscosity "
                "is too large."
            ) from error

        # =================================================
        # RESULT VALIDATION
        # =================================================

        results = {
            "relative viscosity": eta_relative,
            "relative consistency": relative_consistency,
            "flow index": flow_index,
            "exponent": exponent
        }

        for label, value in results.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Calculated {label} is not valid."
                )

        if eta_relative <= 0.0:
            raise ValueError(
                "Calculated crystal relative viscosity "
                "must be greater than zero."
            )

        return {
            "model": self.name,
            "relative_viscosity": eta_relative,

            "details": {
                "phi": phi,
                "crystals_vol_percent": phi * 100.0,
                "rp": rp,
                "alpha": alpha,
                "gamma": gamma,
                "strain_rate": gamma,
                "log10_strain_rate": log10_strain_rate,
                "phim": phim,
                "relative_packing": relative_packing,
                "base": base,
                "exponent": exponent,
                "relative_consistency": relative_consistency,
                "flow_index": flow_index
            }
        }

    # =====================================================
    # MAXIMUM PACKING FRACTION
    # =====================================================

    @staticmethod
    def calculate_phim(rp):

        return (
            2.0
            / (
                0.321 * rp
                + 3.02
            )
        )

    # =====================================================
    # DYNAMIC PHYSICAL LIMITS
    # =====================================================

    def get_dynamic_physical_limits(
        self,
        parameters
    ):

        parameters = parameters or {}

        try:
            rp = float(
                parameters.get(
                    "rp",
                    2.0
                )
            )

        except (TypeError, ValueError):
            return {}

        if (
            not math.isfinite(rp)
            or rp < 1.0
        ):
            return {}

        phim = self.calculate_phim(rp)

        if (
            not math.isfinite(phim)
            or not 0.0 < phim <= 1.0
        ):
            return {}

        return {
            "Crystals": (
                0.0,
                phim * 100.0
            )
        }