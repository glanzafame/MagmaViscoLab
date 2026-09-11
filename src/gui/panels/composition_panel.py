from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class CompositionPanel(QGroupBox):

    # --- Oxide definitions -----------------------------------------

    h2o_changed = Signal(str)

    OXIDES = (
        "SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3", "MnO", "MgO",
        "CaO", "Na2O", "K2O", "P2O5", "Cr2O3", "H2O", "F2O_1",
    )

    OXIDE_LABELS = {
        "SiO2": "SiO₂",
        "TiO2": "TiO₂",
        "Al2O3": "Al₂O₃",
        "FeO": "FeO",
        "Fe2O3": "Fe₂O₃",
        "MnO": "MnO",
        "MgO": "MgO",
        "CaO": "CaO",
        "Na2O": "Na₂O",
        "K2O": "K₂O",
        "P2O5": "P₂O₅",
        "Cr2O3": "Cr₂O₃",
        "H2O": "H₂O",
        "F2O_1": "F₂O₋₁",
    }

    H2O_ROW = OXIDES.index("H2O")

    # --- Panel initialization --------------------------------------

    def __init__(self):
        """Create the editable oxide-composition table."""
        super().__init__("Composition")

        self.enabled_oxides = set(self.OXIDES)

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(
            ["Oxide", "wt%"]
        )
        self.table.cellChanged.connect(
            self.update_water_signal
        )

        layout.addWidget(self.table)

    # --- Sample display --------------------------------------------

    def update_table(self, sample):
        """Display the composition of the selected sample."""
        self.table.blockSignals(True)

        try:
            self.table.setRowCount(
                len(self.OXIDES)
            )

            for row, oxide in enumerate(
                self.OXIDES
            ):
                oxide_item = QTableWidgetItem(
                    self.OXIDE_LABELS[oxide]
                )
                oxide_item.setFlags(
                    oxide_item.flags()
                    & ~Qt.ItemIsEditable
                )

                value_item = QTableWidgetItem(
                    f"{getattr(sample, oxide, 0.0):.3f}"
                )

                self.table.setItem(
                    row,
                    0,
                    oxide_item,
                )
                self.table.setItem(
                    row,
                    1,
                    value_item,
                )

        finally:
            self.table.blockSignals(False)

        self.apply_enabled_state()

    # --- Model-dependent oxide state -------------------------------

    def set_enabled_oxides(self, oxides):
        """Enable only oxides used by the selected melt model."""
        self.enabled_oxides = set(oxides)
        self.apply_enabled_state()

    def apply_enabled_state(self):
        """Update editability and colours of all oxide rows."""
        for row, oxide in enumerate(
            self.OXIDES
        ):
            oxide_item = self.table.item(
                row,
                0,
            )
            value_item = self.table.item(
                row,
                1,
            )

            if value_item is None:
                continue

            enabled = (
                oxide in self.enabled_oxides
            )

            value_item.setFlags(
                value_item.flags() | Qt.ItemIsEditable
                if enabled
                else value_item.flags()
                & ~Qt.ItemIsEditable
            )

            background, foreground = (
                ("white", "black")
                if enabled
                else ("#eeeeee", "#777777")
            )

            for item in (
                oxide_item,
                value_item,
            ):
                if item is not None:
                    item.setBackground(
                        QColor(background)
                    )
                    item.setForeground(
                        QColor(foreground)
                    )

    # --- H₂O synchronization ---------------------------------------

    def update_water_signal(self, row, column):
        """Send table H₂O changes to the physical-properties panel."""
        if (
            column != 1
            or row != self.H2O_ROW
        ):
            return

        item = self.table.item(
            self.H2O_ROW,
            1,
        )

        if item is not None:
            self.h2o_changed.emit(
                item.text()
            )

    def set_h2o(self, value):
        """Update table H₂O from the physical-properties panel."""
        self.table.blockSignals(True)

        try:
            item = self.table.item(
                self.H2O_ROW,
                1,
            )

            if item is not None:
                item.setText(str(value))

        finally:
            self.table.blockSignals(False)