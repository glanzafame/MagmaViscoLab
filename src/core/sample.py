from copy import deepcopy
from dataclasses import asdict, dataclass


from copy import deepcopy
from dataclasses import asdict, dataclass


OXIDES = (
    "SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3", "MnO", "MgO",
    "CaO", "Na2O", "K2O", "P2O5", "Cr2O3", "H2O", "F2O_1"
)


@dataclass
class Sample:

    # --- Sample identification and composition ---------------------

    name: str = ""

    SiO2: float = 0.0
    TiO2: float = 0.0
    Al2O3: float = 0.0
    FeO: float = 0.0
    Fe2O3: float = 0.0
    MnO: float = 0.0
    MgO: float = 0.0
    CaO: float = 0.0
    Na2O: float = 0.0
    K2O: float = 0.0
    P2O5: float = 0.0
    Cr2O3: float = 0.0
    H2O: float = 0.0
    F2O_1: float = 0.0

    # --- Physical properties ---------------------------------------

    temperature: float = 1200.0
    crystals: float = 0.0
    Vesicles: float = 0.0

    # --- Calculated viscosities and correction factors --------------

    log10_eta_m: float | None = None
    log10_eta_mc: float | None = None
    log10_eta_mb: float | None = None
    log10_eta_mcb: float | None = None
    eta_r_c: float | None = None
    eta_r_b: float | None = None

    # --- Object utilities -------------------------------------------

    def copy(self):
        """Return an independent copy of the sample."""
        return deepcopy(self)

    def to_dict(self):
        """Convert all sample fields to a dictionary."""
        return asdict(self)

    # --- Construction from Excel data -------------------------------

    @classmethod
    def from_dict(cls, data):
        """Create a sample from one imported spreadsheet row."""
        return cls(
            name=data.get("Sample", ""),
            **{
                oxide: data.get(oxide, 0.0)
                for oxide in OXIDES
            },
            temperature=data.get(
                "Temperature",
                1200.0
            ),
            crystals=data.get(
                "Crystals",
                0.0
            ),
            Vesicles=data.get(
                "Vesicles",
                0.0
            )
        )