import os
import importlib


class VesicleManager:

    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):

        folder = os.path.join(
            os.path.dirname(__file__),
            "vesicles"
        )

        for name in os.listdir(folder):

            path = os.path.join(folder, name)

            if not os.path.isdir(path) or name == "__pycache__":
                continue

            try:
                module = importlib.import_module(
                    f"models.corrections.vesicles.{name}"
                )

                for attribute in dir(module):

                    obj = getattr(module, attribute)

                    if isinstance(obj, type) and hasattr(obj, "name"):
                        instance = obj()
                        self.models[instance.name] = instance

            except Exception as e:
                print(
                    f"Cannot load vesicle model '{name}': {e}"
                )

    def get_model(self, name):

        if name not in self.models:
            raise ValueError(
                f"Vesicle model not available: {name}"
            )

        return self.models[name]