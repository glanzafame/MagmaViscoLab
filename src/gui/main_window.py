from copy import deepcopy
import math

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.parameter_limits import PARAMETER_LIMITS
from core.sample_manager import SampleManager
from core.viscosity_engine import ViscosityEngine
from gui.panels.composition_panel import CompositionPanel
from gui.panels.models_parameters_panel import ModelsParametersPanel
from gui.panels.physical_panel import PhysicalPanel
from gui.panels.results_panel import ResultsPanel
from gui.panels.sample_panel import SamplePanel


class MainWindow(QMainWindow):

    # --- Canonical result names -------------------------------------

    RESULT_ATTRIBUTES = (
        "log10_eta_m",
        "log10_eta_mc",
        "log10_eta_mb",
        "log10_eta_mcb",
        "eta_r_c",
        "eta_r_b",
    )

    # --- Application initialization --------------------------------

    def __init__(self):
        """Initialize data, calculation engine, panels and interface."""
        super().__init__()

        self.data = None
        self.current_sample = None
        self.samples = []
        self.last_calculation_records = []
        self.last_calculation_scope = None
        self.model_comparison_window = None

        self.sample_manager = SampleManager()
        self.viscosity_engine = ViscosityEngine()

        self.sample_panel = SamplePanel()
        self.composition_panel = CompositionPanel()
        self.physical_panel = PhysicalPanel()
        self.models_parameters_panel = ModelsParametersPanel(self)
        self.results_panel = ResultsPanel()

        self.models_parameters_panel.setTitle("")
        self.results_panel.setTitle("")

        self.composition_panel.h2o_changed.connect(self.water_changed)
        self.physical_panel.water_changed.connect(self.update_composition_h2o)

        self.setWindowTitle("MagmaViscoLab 1.0")
        self.setWindowIcon(QIcon("resources/logo.png"))
        self.resize(1500, 900)
        self.setMinimumSize(1150, 720)
        self.menuBar().hide()

        self.create_interface()
        self.apply_interface_style()

    # --- Interface helpers ------------------------------------------

    @staticmethod
    def create_section_title(text):
        """Create a title for a main interface section."""
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    @staticmethod
    def create_sidebar_button(text, function):
        """Create a sidebar button connected to a function."""
        button = QPushButton(text)
        button.setObjectName("sidebarButton")
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(46)
        button.clicked.connect(function)
        return button

    # --- Main interface ---------------------------------------------

    def create_interface(self):
        """Build the sidebar, input panels, results and action buttons."""
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(205)

        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 24, 18, 20)
        side.setSpacing(8)

        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        logo.setMinimumHeight(120)
        pixmap = QPixmap("resources/logo.png")
        if not pixmap.isNull():
            logo.setPixmap(
                pixmap.scaled(
                    155, 105, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        side.addWidget(logo)

        software_name = QLabel("MagmaViscoLab 1.0")
        software_name.setObjectName("sidebarTitle")
        software_name.setAlignment(Qt.AlignCenter)
        software_name.setWordWrap(True)
        side.addWidget(software_name)
        side.addSpacing(24)

        for text, function in (
            ("Open Excel", self.open_excel),
            ("Export Excel", self.export_excel),
        ):
            side.addWidget(self.create_sidebar_button(text, function))

        side.addSpacing(20)

        for text, function in (
            ("2D Plot", self.open_plot_window),
            ("3D Plot", self.open_3d_plot_window),
            ("Compare models", self.open_model_comparison_window),
        ):
            side.addWidget(self.create_sidebar_button(text, function))

        side.addStretch()
        side.addWidget(self.create_sidebar_button("About", self.show_about))
        root.addWidget(sidebar)

        # Main area
        main = QWidget()
        main.setObjectName("mainArea")
        layout = QVBoxLayout(main)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(9)
        layout.addWidget(self.create_section_title("Input"))

        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self.sample_panel.samples_list.itemClicked.connect(self.show_sample)
        self.sample_panel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.composition_panel.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding
        )
        self.physical_panel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.configure_composition_panel()

        input_row.addWidget(self.sample_panel, 1)
        input_row.addWidget(self.composition_panel, 0)
        input_row.addWidget(self.physical_panel, 2)
        layout.addLayout(input_row, 3)

        layout.addWidget(self.create_section_title("Models and Parameters"))
        self.models_parameters_panel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self.models_parameters_panel, 2)

        layout.addWidget(self.create_section_title("Results"))
        self.results_panel.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        layout.addWidget(self.results_panel, 1)

        # Actions
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 5, 0, 0)
        actions.setSpacing(10)
        self.status_label = QLabel("●  Ready")
        self.status_label.setObjectName("statusLabel")
        actions.addWidget(self.status_label)
        actions.addStretch()

        for text, object_name, size, function in (
            (
                "Normalize", "secondaryActionButton", (140, 42),
                self.normalize_sample,
            ),
            (
                "Calculate", "calculateButton", (165, 42),
                self.calculate_viscosity,
            ),
            (
                "Calculate all samples", "calculateAllButton", (190, 42),
                self.calculate_all_samples,
            ),
        ):
            button = QPushButton(text)
            button.setObjectName(object_name)
            button.setMinimumSize(*size)
            button.clicked.connect(function)
            actions.addWidget(button)

        layout.addLayout(actions)
        root.addWidget(main, 1)

    # --- Interface style --------------------------------------------

    def apply_interface_style(self):
        """Apply the common visual style to the application."""
        self.setStyleSheet("""
            QMainWindow,
            #mainArea {
                background-color: #f4f7f9;
            }

            #sidebar {
                background-color: #073c56;
            }

            #sidebarTitle {
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 4px 0;
            }

            #sidebarButton {
                background-color: transparent;
                color: #e3eef2;
                border: 1px solid #2e6076;
                border-radius: 8px;
                text-align: left;
                padding: 10px 14px;
                font-size: 10pt;
            }

            #sidebarButton:hover {
                background-color: #0c6d91;
                color: white;
                border-color: #2185a5;
            }

            #sidebarButton:pressed {
                background-color: #095a78;
            }

            #sectionTitle {
                color: #18384a;
                font-size: 12.5pt;
                font-weight: bold;
                padding: 3px 2px;
            }

            #statusLabel {
                color: #168c62;
                font-size: 10pt;
                font-weight: bold;
            }

            QGroupBox {
                background-color: white;
                border: 1px solid #d2dde3;
                border-radius: 8px;
                margin-top: 9px;
                padding-top: 9px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 11px;
                padding: 0 5px;
                color: #273e4a;
            }

            QLineEdit,
            QComboBox {
                min-height: 27px;
                background-color: white;
                border: 1px solid #c8d3d9;
                border-radius: 5px;
                padding: 2px 7px;
            }

            QLineEdit:focus,
            QComboBox:focus {
                border: 1px solid #1380a0;
            }

            QListWidget,
            QTableWidget {
                background-color: white;
                border: 1px solid #d3dde2;
                border-radius: 5px;
            }

            QListWidget::item {
                padding: 6px;
            }

            QListWidget::item:selected {
                background-color: #dceff6;
                color: #123b4d;
            }

            QPushButton {
                background-color: white;
                color: #30434d;
                border: 1px solid #c5d0d6;
                border-radius: 7px;
                padding: 6px 13px;
            }

            QPushButton:hover {
                background-color: #eef4f6;
                border-color: #9fb2bc;
            }

            #secondaryActionButton {
                background-color: white;
                color: #163f52;
                border: 1px solid #6f9db2;
                font-size: 10pt;
                font-weight: bold;
            }

            #secondaryActionButton:hover {
                background-color: #eaf3f7;
            }

            #calculateButton {
                background-color: #0785a8;
                color: white;
                border: none;
                font-size: 11pt;
                font-weight: bold;
                padding: 8px 24px;
            }

            #calculateButton:hover {
                background-color: #076f8c;
            }

            #calculateAllButton {
                background-color: #075f79;
                color: white;
                border: none;
                font-size: 10.5pt;
                font-weight: bold;
                padding: 8px 20px;
            }

            #calculateAllButton:hover {
                background-color: #064d63;
            }
        """)

    # --- Composition table ------------------------------------------

    def configure_composition_panel(self):
        """Resize the composition panel and stretch both table columns."""
        table = self.composition_panel.table
        header = table.horizontalHeader()
        for column in range(table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        table.resizeColumnsToContents()

        compact_width = (
            table.verticalHeader().sizeHint().width()
            + sum(table.columnWidth(column) for column in range(table.columnCount()))
            + table.verticalScrollBar().sizeHint().width()
            + table.frameWidth() * 2
            + 28
        )
        self.composition_panel.setFixedWidth(compact_width * 2)
        for column in range(table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
        header.setStretchLastSection(False)
        self.update_fluorine_display_label()

    def update_fluorine_display_label(self):
        """Display the fluorine-equivalent component as F₂O₋₁."""
        oxides = list(getattr(self.composition_panel, "OXIDES", []))
        try:
            row = oxides.index("F2O_1")
        except ValueError:
            return

        table = self.composition_panel.table
        for item in (table.item(row, 0), table.verticalHeaderItem(row)):
            if item is not None:
                item.setText("F₂O₋₁")

    # --- Water synchronization --------------------------------------

    def water_changed(self, value):
        """Copy H₂O changes from composition to physical properties."""
        self.physical_panel.water_edit.setText(value)

    def update_composition_h2o(self):
        """Copy H₂O changes from physical properties to composition."""
        self.composition_panel.set_h2o(self.physical_panel.water_edit.text())

    # --- Excel input ------------------------------------------------

    @staticmethod
    def format_sample_name(value, excel_row=None):
        """Convert a spreadsheet sample identifier into a valid name."""
        row_text = f" in Excel row {excel_row}" if excel_row is not None else ""
        if pd.isna(value):
            raise ValueError(f"Sample name{row_text} cannot be empty.")

        if pd.api.types.is_number(value) and not pd.api.types.is_bool(value):
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Sample name{row_text} must be finite.")
            name = (
                str(int(numeric_value))
                if numeric_value.is_integer()
                else format(numeric_value, ".15g")
            )
        else:
            name = str(value).strip()

        if not name:
            raise ValueError(f"Sample name{row_text} cannot be empty.")
        return name

    def normalize_loaded_sample_names(self):
        """Normalize sample names and reject duplicated identifiers."""
        normalized_names = []
        for index, sample in enumerate(self.samples, start=2):
            sample.name = self.format_sample_name(sample.name, excel_row=index)
            normalized_names.append(sample.name)

        duplicates = sorted({
            name for name in normalized_names if normalized_names.count(name) > 1
        })
        if duplicates:
            raise ValueError(
                "Sample names must be unique. Duplicated name(s): "
                + ", ".join(duplicates) + "."
            )

        if self.data is not None and "Sample" in self.data.columns:
            self.data.loc[:, "Sample"] = [
                self.format_sample_name(value, excel_row=index)
                for index, value in enumerate(self.data["Sample"], start=2)
            ]

    def open_excel(self):
        """Load samples from an Excel workbook and reset old results."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Excel file", "", "Excel Files (*.xlsx)"
        )
        if not filename:
            return

        try:
            self.samples = self.sample_manager.load_excel(filename)
            self.data = self.sample_manager.data
            self.normalize_loaded_sample_names()
            self.last_calculation_records = []
            self.last_calculation_scope = None
            self.sample_panel.samples_list.clear()

            for sample in self.samples:
                self.sample_panel.samples_list.addItem(sample.name)

            if self.samples:
                self.current_sample = self.samples[0]
                self.show_current_sample()
            else:
                self.current_sample = None

            self.results_panel.melt_label.setText(f"Loaded {len(self.samples)} samples")
            self.results_panel.melt_crystal_label.setText(
                "No crystal-bearing magma viscosity calculated"
            )
            self.results_panel.melt_vesicle_label.setText(
                "No vesicle-bearing magma viscosity calculated"
            )
            self.results_panel.three_phase_label.setText(
                "No three-phase magma viscosity calculated"
            )
            self.configure_composition_panel()

        except Exception as error:
            QMessageBox.critical(
                self, "Excel loading error", f"Unable to load Excel file.\n\n{error}"
            )

    # --- Sample selection -------------------------------------------

    def show_sample(self, item):
        """Select the sample corresponding to the clicked list item."""
        selected_name = item.text()
        self.current_sample = next(
            (sample for sample in self.samples if sample.name == selected_name), None
        )
        if self.current_sample is not None:
            self.show_current_sample()

    def show_current_sample(self):
        """Display composition and physical properties of the sample."""
        if self.current_sample is None:
            return

        self.composition_panel.update_table(self.current_sample)
        self.update_fluorine_display_label()
        self.physical_panel.set_values(
            temperature=self.current_sample.temperature,
            water=self.current_sample.H2O,
            crystals=self.current_sample.crystals,
            vesicles=self.current_sample.Vesicles,
        )
        self.models_parameters_panel.refresh_computed_fields()

    # --- Input changes and normalization ----------------------------

    def apply_changes(self):
        """Store the values currently displayed in the selected sample."""
        if self.current_sample is None:
            return False

        for row, oxide in enumerate(self.composition_panel.OXIDES):
            item = self.composition_panel.table.item(row, 1)
            if item is None:
                continue
            value = self.get_table_numeric_value(item, oxide)
            if value is None:
                return False
            setattr(self.current_sample, oxide, value)

        fields = {
            "temperature": (self.physical_panel.temperature_edit, "Temperature"),
            "crystals": (self.physical_panel.crystals_edit, "Crystals"),
            "Vesicles": (self.physical_panel.vesicles_edit, "Vesicles"),
        }
        for attribute, (widget, name) in fields.items():
            value = self.get_numeric_value(widget, name)
            if value is None:
                return False
            setattr(self.current_sample, attribute, value)
        return True

    def normalize_sample(self):
        """Normalize all displayed composition values to 100 wt%."""
        table = self.composition_panel.table
        try:
            values = [
                float(table.item(row, 1).text().replace(",", "."))
                for row in range(table.rowCount())
            ]
        except (ValueError, AttributeError):
            QMessageBox.warning(
                self, "Invalid composition", "All composition values must be numeric."
            )
            return

        total = sum(values)
        if total <= 0:
            QMessageBox.warning(
                self, "Invalid composition", "Composition total must be greater than zero."
            )
            return

        factor = 100.0 / total
        table.blockSignals(True)
        try:
            for row, value in enumerate(values):
                table.setItem(row, 1, QTableWidgetItem(f"{value * factor:.3f}"))
        finally:
            table.blockSignals(False)

        self.composition_panel.apply_enabled_state()
        self.models_parameters_panel.refresh_computed_fields()

    # --- Numeric input ----------------------------------------------

    def get_numeric_value(self, widget, name):
        """Read a numeric line-edit value or display an input error."""
        try:
            value = float(widget.text().replace(",", "."))
            widget.setStyleSheet("")
            return value
        except ValueError:
            widget.setStyleSheet("""
                border: 2px solid red;
                background-color: #ffe6e6;
            """)
            widget.setFocus()
            widget.selectAll()
            QMessageBox.warning(
                self, "Invalid input", f"{name} must be a numeric value."
            )
            return None

    def get_table_numeric_value(self, item, name):
        """Read a numeric composition-table value."""
        try:
            return float(item.text().replace(",", "."))
        except ValueError:
            QMessageBox.warning(
                self, "Invalid composition value", f"{name} must be numeric."
            )
            return None

    # --- Physical-parameter validation ------------------------------

    def validate_sample_parameters(self, sample):
        """Validate the selected sample and highlight invalid GUI fields."""
        checks = {
            "Temperature": (sample.temperature, self.physical_panel.temperature_edit),
            "H₂O": (sample.H2O, self.physical_panel.water_edit),
            "Crystals": (sample.crystals, self.physical_panel.crystals_edit),
            "Vesicles": (sample.Vesicles, self.physical_panel.vesicles_edit),
        }
        for name, (value, widget) in checks.items():
            minimum, maximum = PARAMETER_LIMITS[name]
            if not minimum <= value <= maximum:
                widget.setStyleSheet("""
                    border: 2px solid red;
                    background-color: #ffe6e6;
                """)
                widget.setFocus()
                widget.selectAll()
                QMessageBox.warning(
                    self, "Invalid parameter",
                    f"{name} must be between {minimum} and {maximum}.",
                )
                return False
            widget.setStyleSheet("")
        return True

    @staticmethod
    def get_sample_parameter_error(sample):
        """Return a batch-safe parameter error or None."""
        checks = {
            "Temperature": sample.temperature,
            "H₂O": sample.H2O,
            "Crystals": sample.crystals,
            "Vesicles": sample.Vesicles,
        }
        for name, raw_value in checks.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return f"{name} must be numeric."
            if not math.isfinite(value):
                return f"{name} must be finite."

            minimum, maximum = PARAMETER_LIMITS[name]
            if minimum is not None and value < minimum:
                return (
                    f"{name} must be greater than or equal to {minimum:g}; "
                    f"current value: {value:g}."
                )
            if maximum is not None and value > maximum:
                return (
                    f"{name} must be lower than or equal to {maximum:g}; "
                    f"current value: {value:g}."
                )
        return None

    # --- Viscosity calculations -------------------------------------

    def calculate_sample(self, sample, settings):
        """Calculate all four viscosities for one sample."""
        return self.viscosity_engine.calculate(
            sample=sample,
            melt_model=settings["melt_model"],
            melt_parameters=settings["melt_parameters"],
            crystal_model=settings["crystal_model"],
            crystal_parameters=settings["crystal_parameters"],
            vesicle_model=settings["vesicle_model"],
            vesicle_parameters=settings["vesicle_parameters"],
        )

    def calculate_viscosity(self):
        """Calculate and store results for the selected sample."""
        if self.current_sample is None:
            return
        if not self.apply_changes():
            return
        if not self.validate_sample_parameters(self.current_sample):
            return

        settings = self.models_parameters_panel.get_parameters()
        if settings is None:
            return

        try:
            result = self.calculate_sample(self.current_sample, settings)
        except Exception as error:
            QMessageBox.critical(
                self, "Calculation error", f"Unable to calculate viscosity.\n\n{error}"
            )
            return

        self.store_calculation_result(self.current_sample, result)
        self.last_calculation_records = [
            self.create_calculation_record(self.current_sample, result, settings)
        ]
        self.last_calculation_scope = "single"
        self.update_results_panel(result)
        self.status_label.setText(f"●  Calculated: {self.current_sample.name}")

    def calculate_all_samples(self):
        """Calculate every loaded sample using the current models."""
        if not self.samples:
            QMessageBox.warning(
                self, "No samples", "Load at least one sample before calculating."
            )
            return
        if self.current_sample is not None and not self.apply_changes():
            return

        settings = self.models_parameters_panel.get_parameters()
        if settings is None:
            return

        progress = QProgressDialog(
            "Calculating viscosities...", "Cancel", 0, len(self.samples), self
        )
        progress.setWindowTitle("Calculate all samples")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        successful_results = {}
        batch_records = []
        errors = []
        cancelled = False

        for index, sample in enumerate(self.samples, start=1):
            progress.setLabelText(
                f"Calculating {sample.name} ({index}/{len(self.samples)})..."
            )
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled = True
                break

            validation_error = self.get_sample_parameter_error(sample)
            if validation_error is not None:
                errors.append((sample.name, validation_error))
                progress.setValue(index)
                continue

            try:
                result = self.calculate_sample(sample, settings)
                self.store_calculation_result(sample, result)
                successful_results[sample.name] = result
                batch_records.append(
                    self.create_calculation_record(sample, result, settings)
                )
            except Exception as error:
                errors.append((sample.name, str(error)))
            progress.setValue(index)

        progress.close()
        self.last_calculation_records = batch_records
        self.last_calculation_scope = "all"
        current_name = getattr(self.current_sample, "name", None)
        if current_name in successful_results:
            self.update_results_panel(successful_results[current_name])

        completed = len(successful_results)
        attempted = completed + len(errors)
        if cancelled:
            self.status_label.setText(f"●  Cancelled: {completed} sample(s) calculated")
        elif errors:
            self.status_label.setText(f"●  Calculated {completed}/{len(self.samples)} samples")
        else:
            self.status_label.setText(f"●  Calculated all {completed} samples")

        summary = f"Successfully calculated: {completed}\nErrors: {len(errors)}"
        if cancelled:
            summary += f"\nNot processed: {len(self.samples) - attempted}"

        if errors:
            displayed_errors = errors[:10]
            details = "\n\n" + "\n".join(
                f"• {name}: {message}" for name, message in displayed_errors
            )
            if len(errors) > len(displayed_errors):
                details += f"\n• ...and {len(errors) - len(displayed_errors)} more."
            QMessageBox.warning(
                self, "Batch calculation completed", summary + details
            )
        elif cancelled:
            QMessageBox.information(self, "Batch calculation cancelled", summary)
        else:
            QMessageBox.information(self, "Batch calculation completed", summary)

    def store_calculation_result(self, sample, result):
        """Store calculated values in the sample and source DataFrame."""
        for attribute in self.RESULT_ATTRIBUTES:
            setattr(sample, attribute, result[attribute])
        self.update_dataframe_results(sample)

    @classmethod
    def create_calculation_record(cls, sample, result, settings):
        """Create an immutable snapshot for the next Excel export."""
        physical_parameters = {
            "Temperature (°C)": sample.temperature,
            "H₂O (wt%)": sample.H2O,
            "Crystals (vol%)": sample.crystals,
            "Vesicles (vol%)": sample.Vesicles,
        }

        sample_inputs = {
            "Sample": str(sample.name),
            **physical_parameters,
        }

        attributes = (
            vars(sample)
            if hasattr(sample, "__dict__")
            else {}
        )

        excluded = {
            "name",
            "temperature",
            "H2O",
            "crystals",
            "Vesicles",
            *cls.RESULT_ATTRIBUTES,
        }

        for attribute, value in attributes.items():
            if attribute in excluded:
                continue

            if value is None or pd.api.types.is_scalar(value):
                sample_inputs[attribute] = value

        return {
            "sample": str(sample.name),
            "result": {
                name: result[name]
                for name in cls.RESULT_ATTRIBUTES
            },
            "physical_parameters": physical_parameters,
            "sample_inputs": sample_inputs,
            "settings": deepcopy(settings),
        }

    # --- Result display ---------------------------------------------

    def update_results_panel(self, result):
        """Display viscosities and relative-viscosity factors."""
        green = "font-size:12pt;color:#228B22;font-weight:bold;"
        black = "font-size:11pt;color:black;"

        self.results_panel.melt_label.setText(
            f"""
            <span style='{green}'>
            log₁₀ η<sub>m</sub> (Pa·s) =
            {result["log10_eta_m"]:.3f}
            </span>
            """
        )
        self.results_panel.melt_crystal_label.setText(
            f"""
            <span style='{black}'>
            η<sub>r,c</sub> =
            {result["eta_r_c"]:.3f}
            </span>
            <br><br>
            <span style='{green}'>
            log₁₀ η<sub>mc</sub> (Pa·s) =
            {result["log10_eta_mc"]:.3f}
            </span>
            """
        )
        self.results_panel.melt_vesicle_label.setText(
            f"""
            <span style='{black}'>
            η<sub>r,b</sub> =
            {result["eta_r_b"]:.3f}
            </span>
            <br><br>
            <span style='{green}'>
            log₁₀ η<sub>mb</sub> (Pa·s) =
            {result["log10_eta_mb"]:.3f}
            </span>
            """
        )
        self.results_panel.three_phase_label.setText(
            f"""
            <span style='{green}'>
            log₁₀ η<sub>mcb</sub> (Pa·s) =
            {result["log10_eta_mcb"]:.3f}
            </span>
            """
        )

    # --- Excel export -----------------------------------------------

    def export_excel(self):
        """Export the latest calculation using the standard MVL workbook layout."""
        if not self.last_calculation_records:
            QMessageBox.warning(
                self,
                "No calculation results",
                "Run Calculate or Calculate all samples before exporting.",
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Excel",
            "MVL_calculation.xlsx",
            "Excel Files (*.xlsx)",
        )

        if not filename:
            return

        if not filename.lower().endswith(".xlsx"):
            filename += ".xlsx"

        try:
            self._write_calculation_excel(filename)

            source = (
                "Calculate all samples"
                if self.last_calculation_scope == "all"
                else "Calculate"
            )

            QMessageBox.information(
                self,
                "Export completed",
                (
                    f"Saved {len(self.last_calculation_records)} "
                    "sample sheet(s), all four viscosities, "
                    "ηr,c, ηr,b and calculation settings "
                    f"from the last {source} operation."
                    f"\n\n{filename}"
                ),
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Export error",
                f"Unable to export Excel file.\n\n{error}",
            )

    def _write_calculation_excel(self, filename):
        """Write MainWindow results with the same structure as 2D and 3D exports."""
        records = self.last_calculation_records
        used_names = {"parameters", "samples"}

        settings = records[0]["settings"]

        source = (
            "Calculate all samples"
            if self.last_calculation_scope == "all"
            else "Calculate"
        )

        metadata = [
            ("Calculation source", source),
            ("Calculated samples", len(records)),
        ]

        for group in ("melt", "crystal", "vesicle"):
            metadata.append(
                (
                    f"{group.title()} model",
                    settings[f"{group}_model"],
                )
            )

            metadata.extend(
                (
                    f"{group.title()} parameter — {name}",
                    value,
                )
                for name, value in
                settings[f"{group}_parameters"].items()
            )

        metadata.append(
            (
                "Sample inputs",
                "Samples sheet contains the input snapshot "
                "used for the exported calculation.",
            )
        )

        oxide_names = set(
            getattr(
                self.composition_panel,
                "OXIDES",
                (),
            )
        )

        snapshots = []

        for record in records:
            snapshot = {}

            for name, value in record.get(
                "sample_inputs",
                {},
            ).items():
                if name in oxide_names:
                    label = (
                        "F₂O₋₁ (wt%)"
                        if name == "F2O_1"
                        else f"{name} (wt%)"
                    )
                else:
                    label = name

                snapshot[label] = value

            if not snapshot:
                snapshot = {
                    "Sample": record["sample"],
                    **record["physical_parameters"],
                }

            snapshots.append(snapshot)

        with pd.ExcelWriter(
            filename,
            engine="openpyxl",
        ) as writer:

            for record in records:
                row = (
                    self.calculation_record_to_export_row(
                        record
                    )
                )

                sheet_name = self.make_export_sheet_name(
                    record["sample"],
                    used_names,
                )

                pd.DataFrame([row]).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )

            pd.DataFrame(
                metadata,
                columns=["Parameter", "Value"],
            ).to_excel(
                writer,
                sheet_name="Parameters",
                index=False,
            )

            pd.DataFrame(
                snapshots,
            ).to_excel(
                writer,
                sheet_name="Samples",
                index=False,
            )

            for sheet in writer.book.worksheets:
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions

                for cells in sheet.iter_cols(
                    min_row=1,
                    max_row=1,
                ):
                    cell = cells[0]
                    sheet.column_dimensions[
                        cell.column_letter
                    ].width = min(
                        40,
                        max(
                            17,
                            len(str(cell.value)) + 3,
                        ),
                    )

        # Reuse the same η-subscript formatting used by the 2D/3D exports.
        from gui.plot_window import PlotWindow

        PlotWindow._format_excel_subscripts(
            filename
        )

    @staticmethod
    def calculation_record_to_export_row(record):
        """Convert one calculation snapshot into a standard MVL data row."""
        result = record["result"]

        return {
            "Sample":
                record["sample"],

            "Melt viscosity "
            "log₁₀ ηm (Pa·s)":
                result["log10_eta_m"],

            "Crystal correction "
            "factor ηr,c":
                result["eta_r_c"],

            "Crystal-bearing magma viscosity "
            "log₁₀ ηmc (Pa·s)":
                result["log10_eta_mc"],

            "Vesicle correction "
            "factor ηr,b":
                result["eta_r_b"],

            "Vesicle-bearing magma viscosity "
            "log₁₀ ηmb (Pa·s)":
                result["log10_eta_mb"],

            "Three-phase magma viscosity "
            "log₁₀ ηmcb (Pa·s)":
                result["log10_eta_mcb"],

            "Status":
                "OK",

            "Error":
                "",
        }

    @staticmethod
    def make_export_sheet_name(name, used_names):
        """Create a valid, unique Excel worksheet name."""
        sheet_name = str(name)

        for character in "\\/?*[]:":
            sheet_name = sheet_name.replace(
                character,
                "_",
            )

        sheet_name = (
            sheet_name[:31]
            or "Sample"
        )

        original = sheet_name
        counter = 1

        while sheet_name.casefold() in used_names:
            suffix = f"_{counter}"
            sheet_name = (
                original[
                    :31 - len(suffix)
                ]
                + suffix
            )
            counter += 1

        used_names.add(
            sheet_name.casefold()
        )

        return sheet_name

    # --- Plot windows ------------------------------------------------

    def open_plot_window(self):
        """Open the multi-sample two-dimensional plot window."""
        if not self.samples:
            self.results_panel.melt_label.setText("No samples loaded")
            return

        from gui.plot_window import PlotWindow

        self.plot_window = PlotWindow(samples=self.samples, main_window=self)
        self.plot_window.show()

    def open_3d_plot_window(self):
        """Open the three-dimensional plot window."""
        if self.current_sample is None:
            self.results_panel.melt_label.setText("No sample selected")
            return

        from gui.plot3d_window import Plot3DWindow

        self.plot3d_window = Plot3DWindow(self.current_sample, self)
        self.plot3d_window.show()

    # --- Model comparison window ------------------------------------

    def open_model_comparison_window(self):
        """Open a comparison using a snapshot of the current inputs.

        ModelComparisonWindow is provided by gui/model_comparison_window.py.
        Its constructor accepts samples, main_window, initial_sample_name
        and initial_settings. Each sample and the selected model settings
        are copied so that parameter sweeps operate on independent inputs.
        """
        if not self.samples:
            QMessageBox.warning(
                self, "No samples",
                "Load at least one sample before comparing models.",
            )
            return
        if self.current_sample is None:
            QMessageBox.warning(
                self, "No sample selected", "Select a sample to compare models."
            )
            return

        # Lazy import allows MainWindow to run during the staged update.
        try:
            from gui.model_comparison_window import ModelComparisonWindow
        except ModuleNotFoundError as error:
            if error.name == "gui.model_comparison_window":
                QMessageBox.information(
                    self, "Model comparison not installed",
                    "The model-comparison window is not installed yet.\n\n"
                    "Add gui/model_comparison_window.py to enable this feature.",
                )
            else:
                QMessageBox.critical(
                    self, "Model comparison import error",
                    f"Unable to load the model-comparison window.\n\n{error}",
                )
            return
        except Exception as error:
            QMessageBox.critical(
                self, "Model comparison import error",
                f"Unable to load the model-comparison window.\n\n{error}",
            )
            return

        if not self.apply_changes():
            return
        if not self.validate_sample_parameters(self.current_sample):
            return

        try:
            settings = self.models_parameters_panel.get_parameters()
            if settings is None:
                return

            window = ModelComparisonWindow(
                samples=deepcopy(self.samples),
                main_window=self,
                initial_sample_name=self.current_sample.name,
                initial_settings=deepcopy(settings),
            )
        except Exception as error:
            QMessageBox.critical(
                self, "Model comparison error",
                f"Unable to open the model-comparison window.\n\n{error}",
            )
            return

        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(
            lambda _=None, closed_window=window:
                self._clear_model_comparison_window(closed_window)
        )

        # Replace the previous comparison only after the new one is ready.
        previous_window = self.model_comparison_window
        self.model_comparison_window = window
        if previous_window is not None:
            previous_window.close()

        window.show()
        window.raise_()
        window.activateWindow()
        self.status_label.setText(f"●  Model comparison: {self.current_sample.name}")

    def _clear_model_comparison_window(self, closed_window):
        """Release a closed comparison without clearing a newer window."""
        if self.model_comparison_window is closed_window:
            self.model_comparison_window = None

    # --- Source DataFrame -------------------------------------------

    def update_dataframe_results(self, sample=None):
        """Write one sample's latest results into the source DataFrame."""
        sample = sample or self.current_sample
        if self.data is None or sample is None:
            return

        indexes = self.data[self.data["Sample"] == sample.name].index
        if len(indexes) == 0:
            return

        index = indexes[0]
        for column in self.RESULT_ATTRIBUTES:
            self.data.loc[index, column] = getattr(sample, column)

    # --- About dialog ------------------------------------------------

    def show_about(self):
        """Display software, project and institutional information."""
        dialog = QDialog(self)
        dialog.setWindowTitle("About MagmaViscoLab 1.0")
        dialog.setModal(True)
        dialog.resize(620, 700)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(18)

        mvl_logo = QLabel()
        mvl_logo.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("resources/logo.png")
        if not pixmap.isNull():
            mvl_logo.setPixmap(
                pixmap.scaled(
                    260, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        layout.addWidget(mvl_logo)

        description = QLabel("""
            <div align="center">
            <span style="font-size:18pt;font-weight:bold;">
            MagmaViscoLab 1.0
            </span>
            <br><br>
            Scientific software for the calculation
            and visualization of magma and lava
            viscosity.
            <br>
            Melt viscosity and the rheological
            contributions of crystals and vesicles
            can be combined through selectable models.
            </div>
        """)
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)

        klara_logo = QLabel()
        klara_logo.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("resources/klara_logo.png")
        if not pixmap.isNull():
            klara_logo.setPixmap(
                pixmap.scaled(
                    220, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        layout.addWidget(klara_logo)

        project_info = QLabel("""
            <div align="center">
            MagmaViscoLab was developed within
            the research project
            <br>
            <b>
            KLARA – Kinetics of Lava Flows
            Crystallization
            </b>
            <br><br>
            funded under the
            FIS2 – Fondo Italiano per la Scienza
            programme
            <br>
            Project No. FIS-03129
            <br><br>
            University of Catania<br>
            Department of Biological, Geological
            and Environmental Sciences
            </div>
        """)
        project_info.setWordWrap(True)
        project_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(project_info)

        close_button = QPushButton("Close")
        close_button.setFixedWidth(120)
        close_button.clicked.connect(dialog.accept)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(close_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        dialog.exec()
