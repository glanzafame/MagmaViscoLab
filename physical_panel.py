from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QDoubleValidator,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class PhysicalPanel(QGroupBox):

    # --- Physical-parameter definitions ----------------------------

    water_changed = Signal(str)

    PROPERTIES = (
        (
            "temperature",
            "resources/icons/temperature.png",
            "Temperature",
            "°C",
        ),
        (
            "water",
            "resources/icons/water.png",
            "H₂O",
            "wt%",
        ),
        (
            "crystals",
            "resources/icons/crystals.png",
            "Crystals",
            "vol%",
        ),
        (
            "vesicles",
            "resources/icons/vesicles.png",
            "Vesicles",
            "vol%",
        ),
    )

    ROW_STYLE = """
        QFrame#propertyRow {
            background-color: #f8fafb;
            border: 1px solid #dfe7eb;
            border-radius: 8px;
        }

        QLabel#propertyName {
            color: #253843;
            font-size: 11pt;
            font-weight: 600;
            border: none;
            background: transparent;
        }

        QLabel#propertyUnit {
            color: #7d8b93;
            font-size: 9pt;
            border: none;
            background: transparent;
        }

        QLineEdit {
            background-color: white;
            color: #263740;
            border: 1px solid #cbd7dd;
            border-radius: 5px;
            padding: 4px 7px;
            font-size: 10pt;
        }

        QLineEdit:focus {
            border: 1px solid #1380a0;
        }
    """

    DISABLED_FIELD_STYLE = """
        background-color: #eeeeee;
        color: #888888;
        border: 1px solid #d8d8d8;
        border-radius: 5px;
        padding: 4px 7px;
    """

    # --- Panel initialization --------------------------------------

    def __init__(self):
        """Create the physical-property input controls."""
        super().__init__("Physical Properties")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            14,
            18,
            14,
            14,
        )
        layout.setSpacing(10)

        self.parameter_widgets = {}

        for name, icon, title, unit in (
            self.PROPERTIES
        ):
            edit = QLineEdit()
            edit.setFixedSize(110, 34)
            edit.setAlignment(Qt.AlignRight)

            self.set_numeric_validator(edit)

            setattr(
                self,
                f"{name}_edit",
                edit,
            )
            self.parameter_widgets[name] = edit

            row = self.create_property_row(
                icon,
                title,
                unit,
                edit,
            )

            setattr(
                self,
                f"{name}_row",
                row,
            )
            layout.addWidget(row)

        self.enabled_parameters = set(
            self.parameter_widgets
        )

        layout.addStretch()

        self.water_edit.editingFinished.connect(
            self.emit_water_changed
        )

    # --- Property-row construction --------------------------------

    def create_property_row(
        self,
        icon_path,
        title,
        unit,
        edit,
    ):
        """Create one icon, label, unit and value row."""
        row = QFrame()
        row.setObjectName("propertyRow")
        row.setFixedHeight(74)
        row.setStyleSheet(self.ROW_STYLE)

        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            12,
            8,
            12,
            8,
        )
        layout.setSpacing(14)

        icon_label = QLabel()
        icon_label.setFixedSize(54, 54)
        icon_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(icon_path)

        if not pixmap.isNull():
            icon_label.setPixmap(
                pixmap.scaled(
                    50,
                    50,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        text_widget = QWidget()
        text_layout = QVBoxLayout(
            text_widget
        )
        text_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        text_layout.setSpacing(0)
        text_layout.setAlignment(
            Qt.AlignVCenter
        )

        name_label = QLabel(title)
        name_label.setObjectName(
            "propertyName"
        )
        name_label.setFixedHeight(24)

        unit_label = QLabel(unit)
        unit_label.setObjectName(
            "propertyUnit"
        )
        unit_label.setFixedHeight(20)

        text_layout.addWidget(name_label)
        text_layout.addWidget(unit_label)

        layout.addWidget(
            icon_label,
            0,
            Qt.AlignVCenter,
        )
        layout.addWidget(
            text_widget,
            0,
            Qt.AlignVCenter,
        )
        layout.addStretch()
        layout.addWidget(
            edit,
            0,
            Qt.AlignVCenter,
        )

        return row

    # --- Numeric validation ----------------------------------------

    def set_numeric_validator(self, widget):
        """Restrict a line edit to finite decimal input."""
        validator = QDoubleValidator(
            -999999,
            999999,
            6,
            self,
        )
        validator.setNotation(
            QDoubleValidator.StandardNotation
        )
        widget.setValidator(validator)

    # --- Model-dependent state -------------------------------------

    def set_enabled_parameters(
        self,
        parameters,
    ):
        """Enable physical inputs used by the selected melt model."""
        self.enabled_parameters = (
            set(parameters)
            | {"crystals", "vesicles"}
        )
        self.apply_enabled_state()

    def apply_enabled_state(self):
        """Update the editability and appearance of every input."""
        for name, widget in (
            self.parameter_widgets.items()
        ):
            enabled = (
                name in self.enabled_parameters
            )

            widget.setEnabled(True)
            widget.setReadOnly(not enabled)

            if enabled:
                widget.setStyleSheet("")
                widget.setToolTip("")

            else:
                widget.setStyleSheet(
                    self.DISABLED_FIELD_STYLE
                )
                widget.setToolTip(
                    "This parameter is not used by "
                    "the selected melt-viscosity model."
                )

    # --- H₂O synchronization ---------------------------------------

    def emit_water_changed(self):
        """Send an editable H₂O value to the composition table."""
        if not self.water_edit.isReadOnly():
            self.water_changed.emit(
                self.water_edit.text()
            )

    # --- Displayed values ------------------------------------------

    def set_values(
        self,
        temperature=1200,
        water=0,
        crystals=0,
        vesicles=0,
    ):
        """Display the physical properties of a selected sample."""
        values = {
            "temperature": temperature,
            "water": water,
            "crystals": crystals,
            "vesicles": vesicles,
        }

        for name, value in values.items():
            self.parameter_widgets[
                name
            ].setText(str(value))

        self.apply_enabled_state()

    # --- Numeric value access --------------------------------------

    @staticmethod
    def _value(widget):
        """Convert one line-edit value into a float."""
        return float(
            widget.text().replace(",", ".")
        )

    def get_temperature(self):
        """Return temperature in degrees Celsius."""
        return self._value(
            self.temperature_edit
        )

    def get_water(self):
        """Return dissolved H₂O in wt%."""
        return self._value(
            self.water_edit
        )

    def get_crystals(self):
        """Return crystal content in vol%."""
        return self._value(
            self.crystals_edit
        )

    def get_vesicles(self):
        """Return vesicle content in vol%."""
        return self._value(
            self.vesicles_edit
        )