import importlib
import inspect
from pathlib import Path

from models.base_models import BaseViscosityModel


class ModelManager:

    # --- Melt-model location ---------------------------------------

    MODEL_PACKAGE = "models.melt"
    MODEL_FOLDER = (
        Path(__file__).resolve().parent
        / "melt"
    )

    # --- Manager initialization ------------------------------------

    def __init__(self):
        """Discover and initialize all available melt models."""
        self.models = {}
        self.load_errors = {}
        self.load_models()

    # --- Model discovery -------------------------------------------

    def load_models(self):
        """Load concrete melt models from their package folders."""
        self.models.clear()
        self.load_errors.clear()

        if not self.MODEL_FOLDER.is_dir():
            raise FileNotFoundError(
                "Melt-model folder not found: "
                f"{self.MODEL_FOLDER}"
            )

        packages = sorted(
            path
            for path in self.MODEL_FOLDER.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith("_")
                and (
                    path / "__init__.py"
                ).is_file()
            )
        )

        for package in packages:
            try:
                self._load_package(
                    package.name
                )

            except Exception as error:
                self.load_errors[
                    package.name
                ] = str(error)

    def _load_package(self, package_name):
        """Import and register the models exposed by one package."""
        module_name = (
            f"{self.MODEL_PACKAGE}."
            f"{package_name}"
        )
        module = importlib.import_module(
            module_name
        )

        model_classes = []
        seen_classes = set()

        for obj in vars(module).values():
            if (
                not inspect.isclass(obj)
                or obj in seen_classes
                or obj is BaseViscosityModel
                or inspect.isabstract(obj)
                or not issubclass(
                    obj,
                    BaseViscosityModel,
                )
                or not obj.__module__.startswith(
                    module_name
                )
            ):
                continue

            seen_classes.add(obj)
            model_classes.append(obj)

        if not model_classes:
            raise ValueError(
                "No concrete melt-viscosity "
                "model was found."
            )

        for model_class in model_classes:
            instance = model_class()
            model_name = getattr(
                instance,
                "name",
                "",
            )

            if (
                not isinstance(model_name, str)
                or not model_name.strip()
            ):
                raise ValueError(
                    f"{model_class.__name__} has "
                    "no valid model name."
                )

            if model_name in self.models:
                raise ValueError(
                    "Duplicated melt-model name: "
                    f"{model_name}"
                )

            self.models[model_name] = instance

    # --- Model access ----------------------------------------------

    def get_model(self, name):
        """Return a loaded melt model by its display name."""
        try:
            return self.models[name]

        except KeyError as error:
            available = (
                ", ".join(self.models)
                or "none"
            )

            raise ValueError(
                f"Melt model not available: {name}. "
                f"Available models: {available}."
            ) from error