from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout
)


class ResultsPanel(QGroupBox):

    # --- Result-panel definitions ----------------------------------

    RESULT_BOXES = (
        (
            "Melt viscosity",
            "No melt viscosity calculated",
            "melt_label"
        ),
        (
            "Crystal-bearing magma viscosity",
            "No crystal-bearing magma viscosity calculated",
            "melt_crystal_label"
        ),
        (
            "Vesicle-bearing magma viscosity",
            "No vesicle-bearing magma viscosity calculated",
            "melt_vesicle_label"
        ),
        (
            "Three-phase magma viscosity",
            "No three-phase magma viscosity calculated",
            "three_phase_label"
        )
    )

    # --- Panel construction ----------------------------------------

    def __init__(self):
        """Create the four magma-viscosity result panels."""
        super().__init__("Results")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for title, text, attribute in self.RESULT_BOXES:
            layout.addWidget(
                self.create_result_box(
                    title,
                    text,
                    attribute
                ),
                1
            )

    # --- Individual result box -------------------------------------

    def create_result_box(self, title, text, attribute):
        """Create one expandable box and expose its result label."""
        box = QGroupBox(title)
        box.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        layout = QVBoxLayout(box)

        label = QLabel(text)
        label.setStyleSheet(
            "color: black;"
            "font-size: 11pt;"
            "font-weight: normal;"
        )
        label.setWordWrap(True)

        setattr(self, attribute, label)

        layout.addStretch()
        layout.addWidget(label)

        return box