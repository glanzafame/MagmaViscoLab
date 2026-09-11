import math

from models.corrections.base_correction import BaseCrystalCorrection


class Mueller2011(BaseCrystalCorrection):

    name = "Mueller et al. (2011)"

    required_physical_parameters = [
        "crystals"
    ]

    parameters = {
        "rp": {
            "label": "rp",
            "default": 1.0
        }
    }

    model_parameter_limits = {
        "rp": (0.04, 22.0)
    }

    # Mueller et al. (2011), Eq. 4.
    PHI_M1 = 0.656
    B = 1.08

    # Conservative domain in which relative consistency K_r
    # can be used as a proxy for relative apparent viscosity.
    MAX_RELATIVE_PACKING = 0.70

    # =====================================================
    # CALCULATE
    # =====================================================

    def calculate(
        self,
        crystal_fraction,
        rp
    ):

        try:
            phi = float(crystal_fraction)
            rp = float(rp)

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Crystal fraction and aspect ratio "
                "must be numeric."
            ) from error

        # =================================================
        # INPUT VALIDATION
        # =================================================

        if not math.isfinite(phi):
            raise ValueError(
                "Crystal fraction must be finite."
            )

        if phi < 0.0:
            raise ValueError(
                "Crystal fraction cannot be negative."
            )

        if not math.isfinite(rp):
            raise ValueError(
                "Aspect ratio rₚ must be finite."
            )

        rp_min, rp_max = (
            self.model_parameter_limits["rp"]
        )

        if not rp_min <= rp <= rp_max:
            raise ValueError(
                "Mueller et al. (2011) is calibrated "
                "for aspect ratios rₚ between "
                f"{rp_min:g} and {rp_max:g}."
            )

        # =================================================
        # MAXIMUM PACKING FRACTION - EQ. 4
        # =================================================

        phi_m = self.calculate_phi_m(rp)

        # =================================================
        # RELATIVE PACKING
        # =================================================

        relative_packing = phi / phi_m

        if not math.isfinite(relative_packing):
            raise ValueError(
                "Calculated relative packing is not valid."
            )

        if relative_packing >= 1.0:
            raise ValueError(
                f"Crystal fraction ({phi:.4f}) must be "
                "lower than maximum packing fraction "
                f"Φₘ ({phi_m:.4f})."
            )

        above_proxy_limit = (
            relative_packing
            > self.MAX_RELATIVE_PACKING
            and not math.isclose(
                relative_packing,
                self.MAX_RELATIVE_PACKING,
                rel_tol=1e-12,
                abs_tol=1e-12
            )
        )

        if above_proxy_limit:
            max_crystals = (
                self.MAX_RELATIVE_PACKING
                * phi_m
                * 100.0
            )

            raise ValueError(
                "Mueller et al. (2011) uses relative "
                "consistency Kᵣ as a proxy for relative "
                "apparent viscosity only for "
                f"Φ/Φₘ ≤ {self.MAX_RELATIVE_PACKING:g}. "
                "For the selected aspect ratio this "
                f"corresponds to crystals ≤ "
                f"{max_crystals:.4g} vol%."
            )

        # =================================================
        # RELATIVE CONSISTENCY - EQ. 3
        # =================================================

        denominator = 1.0 - relative_packing

        relative_consistency = denominator ** -2.0

        if (
            not math.isfinite(relative_consistency)
            or relative_consistency <= 0.0
        ):
            raise ValueError(
                "Calculated relative consistency "
                "is not valid."
            )

        # K_r is retained under relative_viscosity because
        # this is the correction-factor key expected by MVL.
        return {
            "model": self.name,
            "relative_viscosity": relative_consistency,
            "relative_consistency": relative_consistency,

            "details": {
                "phi": phi,
                "crystals_vol_percent": phi * 100.0,
                "rp": rp,
                "aspect_ratio_definition": "r_p = l_a / l_b",
                "particle_shape": self.classify_particle_shape(rp),
                "phi_m": phi_m,
                "phi_m1": self.PHI_M1,
                "b": self.B,
                "relative_packing": relative_packing,
                "max_relative_packing":
                    self.MAX_RELATIVE_PACKING,
                "max_crystals_vol_percent": (
                    self.MAX_RELATIVE_PACKING
                    * phi_m
                    * 100.0
                ),
                "denominator": denominator,
                "Kr": relative_consistency,
                "proxy_for_eta_r": True
            }
        }

    # =====================================================
    # MAXIMUM PACKING FRACTION - EQ. 4
    # =====================================================

    def calculate_phi_m(self, rp):

        phi_m = (
            self.PHI_M1
            * math.exp(
                -(
                    math.log10(rp) ** 2
                )
                / (
                    2.0 * self.B**2
                )
            )
        )

        if (
            not math.isfinite(phi_m)
            or phi_m <= 0.0
        ):
            raise ValueError(
                "Calculated maximum packing fraction "
                "is not valid."
            )

        return phi_m

    # =====================================================
    # PARTICLE SHAPE
    # =====================================================

    @staticmethod
    def classify_particle_shape(rp):

        if math.isclose(
            rp,
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12
        ):
            return "equant"

        if rp < 1.0:
            return "oblate"

        return "prolate"

    # =====================================================
    # DYNAMIC PHYSICAL LIMITS
    # =====================================================

    def get_dynamic_physical_limits(
        self,
        parameters
    ):

        try:
            rp = float(
                parameters.get(
                    "rp",
                    self.parameters["rp"]["default"]
                )
            )

        except (TypeError, ValueError, AttributeError):
            return {}

        rp_min, rp_max = (
            self.model_parameter_limits["rp"]
        )

        if (
            not math.isfinite(rp)
            or not rp_min <= rp <= rp_max
        ):
            return {}

        try:
            phi_m = self.calculate_phi_m(rp)

        except ValueError:
            return {}

        max_crystals = (
            self.MAX_RELATIVE_PACKING
            * phi_m
            * 100.0
        )

        if (
            not math.isfinite(max_crystals)
            or max_crystals <= 0.0
        ):
            return {}

        return {
            "Crystals": (
                0.0,
                max_crystals
            )
        }