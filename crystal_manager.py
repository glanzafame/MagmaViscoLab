import os
import importlib


class CrystalManager:

    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):

        folder = os.path.join(
            os.path.dirname(__file__),
            "crystals"
        )

        for name in os.listdir(folder):

            path = os.path.join(folder, name)

            if not os.path.isdir(path) or name == "__pycache__":
                continue

            try:
                module = importlib.import_module(
                    f"models.corrections.crystals.{name}"
                )

                for attribute in dir(module):

                    obj = getattr(module, attribute)

                    if isinstance(obj, type) and hasattr(obj, "name"):
                        instance = obj()
                        self.models[instance.name] = instance

            except Exception as e:
                print(
                    f"Cannot load crystal model '{name}': {e}"
                )

    def get_model(self, name):

        if name not in self.models:
            raise ValueError(
                f"Crystal model not available: {name}"
            )

        return self.models[name]