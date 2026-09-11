import sys
from PySide6.QtWidgets import QApplication
import core.viscosity_engine

print("LOADED:", core.viscosity_engine.__file__)
print("CONTENT:", dir(core.viscosity_engine))

from gui.main_window import MainWindow


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec())