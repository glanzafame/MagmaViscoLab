from PySide6.QtWidgets import *


class SamplePanel(QGroupBox):

    def __init__(self):
        super().__init__("Samples")

        layout = QVBoxLayout()

        self.samples_list = QListWidget()

        layout.addWidget(
            self.samples_list
        )

        self.setLayout(layout)