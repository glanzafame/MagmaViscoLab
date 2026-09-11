"""Multi-sample MVL 2D plots with editable axes and persistent curve styles."""
from copy import deepcopy
from pathlib import Path
import math
import os
import re
from html import escape
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from PySide6.QtCore import QLocale, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QTextCharFormat, QTextDocument, QValidator
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QStyle, QStyledItemDelegate,
    QStyleOptionComboBox, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)
from core.parameter_limits import PARAMETER_LIMITS
from gui.panels.models_parameters_panel import ModelsParametersPanel


class RichTextDelegate(QStyledItemDelegate):
    """Render viscosity symbols and subscripts in a combo-box menu."""

    @staticmethod
    def document(text, font, color=None):
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(font)
        if color is not None:
            document.setDefaultStyleSheet(f"body {{ color: {color.name()}; }}")
        document.setHtml(text)
        return document

    def paint(self, painter, option, index):
        painter.save()
        painter.setClipRect(option.rect)
        selected = bool(option.state & QStyle.State_Selected)
        if selected:
            painter.fillRect(option.rect, option.palette.highlight())
        color = option.palette.highlightedText().color() if selected else option.palette.text().color()
        document = self.document(str(index.data(Qt.DisplayRole)), option.font, color)
        painter.translate(option.rect.left() + 6,
                          option.rect.top() + (option.rect.height() - document.size().height()) / 2)
        document.drawContents(painter)
        painter.restore()

    def sizeHint(self, option, index):
        document = self.document(str(index.data(Qt.DisplayRole)), option.font)
        return QSize(int(document.idealWidth()) + 16, max(30, int(document.size().height()) + 6))

class RichTextComboBox(QComboBox):
    """Display the selected HTML label with the same notation as the menu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(RichTextDelegate(self))
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(18)

    def paintEvent(self, event):
        painter = QPainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        self.style().drawComplexControl(QStyle.CC_ComboBox, option, painter, self)
        rect = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
        document = RichTextDelegate.document(self.currentText(), self.font(), self.palette().text().color())
        painter.setClipRect(rect)
        painter.translate(rect.left() + 3, rect.top() + (rect.height() - document.size().height()) / 2)
        document.drawContents(painter)


class Plot2DToolbar(NavigationToolbar2QT):
    """Keep the standard navigation icons and use MVL editing dialogs."""

    toolitems = tuple(
        ("Edit axes", "Edit axes and sample curves", icon, callback)
        if callback == "edit_parameters" else
        ("Configure plot", "Configure plot margins", icon, callback)
        if callback == "configure_subplots" else
        ("Home", "Restore the original plot configuration", icon, callback)
        if callback == "home" else (text, tooltip, icon, callback)
        for index, (text, tooltip, icon, callback) in enumerate(NavigationToolbar2QT.toolitems)
        if callback is not None or index + 1 == len(NavigationToolbar2QT.toolitems)
        or NavigationToolbar2QT.toolitems[index + 1][-1] != "save_figure"
    )

    def __init__(self, canvas, parent, plot_window):
        self.plot_window = plot_window
        super().__init__(canvas, parent)

    def edit_parameters(self, *_):
        dialog = self.plot_window._make_axes_dialog()
        dialog.exec()
        dialog.deleteLater()

    def configure_subplots(self, *_):
        dialog = self.plot_window._make_layout_dialog()
        dialog.exec()
        dialog.deleteLater()

    def home(self, *_):
        self.plot_window.restore_original_plot()


class XValueSpinBox(QDoubleSpinBox):
    """Editable physical X value, including decimal commas and exponents."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setDecimals(12)
        self.setRange(-1e100, 1e100)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setMinimumWidth(145)

    def textFromValue(self, value):
        return format(value, ".12g")

    def valueFromText(self, text):
        try:
            value = float(text.strip().replace(",", "."))
        except ValueError:
            return self.value()
        return value if math.isfinite(value) else self.value()

    def validate(self, text, position):
        value_text = text.strip().replace(",", ".")
        if value_text in {"", "+", "-", ".", "+.", "-."}:
            return QValidator.Intermediate, text, position
        try:
            value = float(value_text)
        except ValueError:
            partial = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?", value_text)
            return (QValidator.Intermediate if partial else QValidator.Invalid), text, position
        state = (QValidator.Acceptable if math.isfinite(value) and self.minimum() <= value <= self.maximum()
                 else QValidator.Intermediate if math.isfinite(value) else QValidator.Invalid)
        return state, text, position


class PlotWindow(QMainWindow):

    # --- Supported results -----------------------------------------

    CARICCHI_MODEL = "Caricchi et al. (2007)"

    RESULT_KEYS = (
        "log10_eta_m",
        "log10_eta_mc",
        "log10_eta_mb",
        "log10_eta_mcb",
        "eta_r_c",
        "eta_r_b",
    )

    RESULTS = {'m': {'combo': 'Melt viscosity · log<sub>10</sub> η<sub>m</sub> (Pa·s)',
           'plot': '$\\log_{10}\\eta_{\\mathrm{m}}\\;(\\mathrm{Pa\\,s})$',
           'key': 'log10_eta_m',
           'title': 'Melt viscosity'},
     'mc': {'combo': 'Melt + crystals · log<sub>10</sub> η<sub>mc</sub> (Pa·s)',
            'plot': '$\\log_{10}\\eta_{\\mathrm{mc}}\\;(\\mathrm{Pa\\,s})$',
            'key': 'log10_eta_mc',
            'title': 'Crystal-bearing magma viscosity'},
     'mb': {'combo': 'Melt + vesicles · log<sub>10</sub> η<sub>mb</sub> (Pa·s)',
            'plot': '$\\log_{10}\\eta_{\\mathrm{mb}}\\;(\\mathrm{Pa\\,s})$',
            'key': 'log10_eta_mb',
            'title': 'Vesicle-bearing magma viscosity'},
     'mcb': {'combo': 'Three phases · log<sub>10</sub> η<sub>mcb</sub> (Pa·s)',
             'plot': '$\\log_{10}\\eta_{\\mathrm{mcb}}\\;(\\mathrm{Pa\\,s})$',
             'key': 'log10_eta_mcb',
             'title': 'Three-phase magma viscosity'}}

    # --- Physical parameters and axes ------------------------------

    DEFAULT_RANGES = {
        "Temperature": ("900", "1200", "50"),
        "H₂O": ("0", "4", "0.5"),
        "Crystals": ("0", "50", "5"),
        "Vesicles": ("0", "50", "5"),
        "γ̇ (strain rate)": ("0.01", "1", "0.05"),
    }

    SAMPLE_ATTRIBUTES = {
        "Temperature": "temperature",
        "H₂O": "H2O",
        "Crystals": "crystals",
        "Vesicles": "Vesicles",
    }

    MODEL_FIXED_TO_PLOT_PARAMETER = {
        "temperature": "Temperature",
        "water": "H₂O",
        "crystals": "Crystals",
        "vesicles": "Vesicles",
    }

    X_LABELS = {
        "Temperature": "Temperature (°C)",
        "H₂O": "H₂O (wt%)",
        "Crystals": "Crystals (vol%)",
        "Vesicles": "Vesicles (vol%)",
        "γ̇ (strain rate)":
            r"$\dot{\gamma}$ (s$^{-1}$)",
    }

    EXPORT_AXIS_LABELS = {
        "Temperature": "Temperature (°C)",
        "H₂O": "H₂O (wt%)",
        "Crystals": "Crystals (vol%)",
        "Vesicles": "Vesicles (vol%)",
        "γ̇ (strain rate)": "γ̇ (s⁻¹)",
    }

    PARAMETER_UNITS = {
        "Temperature": "°C",
        "H₂O": "wt%",
        "Crystals": "vol%",
        "Vesicles": "vol%",
        "γ̇ (strain rate)": "s⁻¹",
    }

    FIXED_FIELD_STYLE = """
        QLineEdit,
        QLineEdit:disabled {
            background-color: #e1e4e7;
            color: #66717b;
            border: 1px solid #b9c0c6;
        }
    """

    STYLE = """
                QMainWindow { background: #f3f5f7; color: #26313a; }
                #controlPanel { background: #eef1f4; }
                #panelTitle { color: #a63b2a; font-size: 18pt; font-weight: 700; }
                #plotTitle { color: #26313a; font-size: 14pt; font-weight: 600; }
                #plotSubtitle { color: #64717b; font-size: 9pt; }
                QGroupBox { background: white; border: 1px solid #d8dde2;
                    border-radius: 8px; margin-top: 12px; padding: 11px 8px 8px;
                    font-weight: 600; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px;
                    padding: 0 4px; color: #414b54; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: white;
                    border: 1px solid #cbd1d7; border-radius: 5px;
                    padding: 3px 6px; min-height: 23px; }
                QLineEdit:focus, QComboBox:focus { border-color: #1380a0; }
                QLineEdit:disabled { background: #e1e4e7; color: #66717b; }
                QComboBox:disabled, QSpinBox:disabled { background: #eef0f2; color: #78818a; }
                QCheckBox { spacing: 6px; min-height: 25px; }
                QPushButton { min-height: 28px; padding: 4px 10px; background: white;
                    color: #30434d; border: 1px solid #c6ccd2; border-radius: 6px; }
                QPushButton:hover { background: #eaf3f7; border-color: #6f9db2; }
                QPushButton:disabled { color: #9ca4ac; background: #f1f3f5; }
                #smallButton { min-height: 23px; padding: 2px 7px; font-size: 9pt; }
                #calculateButton { background: #0785a8; color: white; border: none;
                    font-size: 10.5pt; font-weight: bold; padding: 7px 18px; }
                #calculateButton:hover { background: #076f8c; }
                #calculateButton:disabled { background: #8fb6c1; }
                QTabBar::tab { padding: 9px 20px; background: #e7ecef; color: #52616d;
                    border: 1px solid #d3dce1; margin-right: 3px; }
                QTabBar::tab:selected { background: #073c56; color: white; }
                QScrollArea { border: none; background: #eef1f4; }
                QTableWidget { background: white; border: 1px solid #d8dde2;
                    gridline-color: #e7ebee; selection-background-color: #dceff6;
                    selection-color: #123b4d; }
                QHeaderView::section { background: #eef2f5; color: #40515e;
                    padding: 5px; border: none; border-bottom: 1px solid #d8dde2; }
                QSplitter::handle { background: #d8dde2; }

    """

    # --- Window initialization -------------------------------------

    def __init__(self, samples=None, main_window=None):
        """Create local plot inputs, calculation snapshots and editable curves."""
        super().__init__()
        self.samples = deepcopy(list(samples or []))
        self.main_window = main_window
        self.viscosity_engine = getattr(main_window, "viscosity_engine", None)
        self.last_plot_values = None
        self.last_plot_results = None
        self.last_plot_parameter = None
        self.last_plot_settings = None
        self.last_plot_samples = None
        self.last_plot_errors = []
        self._error_messages = {}
        self.fixed_parameters_dirty = False
        self._initializing = True
        self._inputs_dirty = False
        self._curve_artists = {}
        self._axis_options = {"x": {}, "y": {}}
        self._plot_options = {"title": "", "grid": True, "legend": True}
        self._layout_margins = None
        self._original_plot_configuration = None
        self._point_cache_x = None
        self._point_cache = {}
        self.last_plot_step = 1.0
        colors = ("#007d9d", "#d45a3c", "#7956a3", "#279165", "#c18b16",
                  "#4577bd", "#c4578b", "#686b70", "#728835", "#905d41")
        self._sample_styles = {
            str(sample.name): {"color": colors[index % len(colors)], "linestyle": "-",
                               "linewidth": 1.8, "marker": "o", "markersize": 4.5,
                               "label": str(sample.name)}
            for index, sample in enumerate(self.samples)
        }
        self.setWindowTitle("MagmaViscoLab – 2D Plot")
        self.setWindowIcon(self._mvl_icon())
        self.resize(1500, 920)
        self.setMinimumSize(1120, 740)
        self.setStyleSheet(self.STYLE)
        self.create_interface()
        self._initializing = False
        self._remember_original_plot()

    # --- Main interface --------------------------------------------

    def create_interface(self):
        """Use the same control/figure layout as Compare models."""
        root = QWidget(objectName="plotRoot")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.addWidget(QLabel("2D Plot", objectName="panelTitle"))
        outer.addWidget(self._note("Plot selected samples using the same models and a common X range."))
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        controls = QWidget(objectName="controlPanel")
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(10, 8, 10, 10)
        control_layout.setSpacing(10)
        control_layout.addWidget(self._create_samples_box())
        control_layout.addWidget(self._create_axes_box())
        control_layout.addWidget(self._create_fixed_box())

        self.models_parameters_panel = ModelsParametersPanel(
            self.main_window, manage_main_inputs=False,
            viscosity_engine=self.viscosity_engine, sample_provider=self._sample_for_model_preview,
        )
        self.models_parameters_panel.setParent(controls)
        self.models_parameters_panel.hide()
        self.copy_initial_model_settings()
        model_box = QGroupBox("Models and parameters")
        model_layout = QVBoxLayout(model_box)
        self.model_parameter_buttons = {}
        for group, (title, selector_name, _manager) in ModelsParametersPanel.MODEL_GROUPS.items():
            model_layout.addWidget(QLabel(title))
            row = QHBoxLayout()
            selector = getattr(self.models_parameters_panel, selector_name)
            selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(selector, 1)
            button = self._button("Parameters…", lambda _=False, group=group: self.edit_model_parameters(group), "smallButton")
            self.model_parameter_buttons[group] = button
            row.addWidget(button)
            model_layout.addLayout(row)
        control_layout.addWidget(model_box)
        control_layout.addStretch()
        self.models_parameters_panel.model_selector.currentTextChanged.connect(self.sync_fixed_melt_physical_parameters)
        self.models_parameters_panel.crystal_model.currentTextChanged.connect(self.sync_crystal_model_controls)
        self.models_parameters_panel.parameters_changed.connect(self._inputs_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(520)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 0, 0, 0)
        self.plot_title = QLabel("Viscosity curves", objectName="plotTitle")
        self.plot_subtitle = self._note("Each selected sample is shown as a separate curve.")
        plot_layout.addWidget(self.plot_title)
        plot_layout.addWidget(self.plot_subtitle)
        self.figure = Figure(figsize=(8, 6), facecolor="white", layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(350)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = Plot2DToolbar(self.canvas, plot_panel, self)
        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(self.toolbar, 1)
        plot_layout.addLayout(toolbar_row)
        plot_layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("draw_event", self._capture_curve_styles)
        self.canvas.mpl_connect("button_press_event", self._plot_clicked)
        point_row = QHBoxLayout()
        point_row.addWidget(QLabel("Values at point"))
        point_row.addWidget(QLabel("X ="))
        self.point_selector = XValueSpinBox()
        self.point_selector.setEnabled(False)
        self.point_selector.valueChanged.connect(self._update_point_table)
        self.point_selector.editingFinished.connect(self._update_point_table)
        self.point_decrease_button = self._button("−", self.point_selector.stepDown, "smallButton")
        self.point_increase_button = self._button("+", self.point_selector.stepUp, "smallButton")
        for button in (self.point_decrease_button, self.point_increase_button):
            button.setFixedWidth(32)
            button.setEnabled(False)
        point_row.addWidget(self.point_decrease_button)
        point_row.addWidget(self.point_selector)
        point_row.addWidget(self.point_increase_button)
        self.point_label = QLabel("—")
        point_row.addWidget(self.point_label)
        point_row.addStretch()
        self.errors_button = self._button("Point errors…", self.show_plot_errors)
        self.errors_button.setEnabled(False)
        point_row.addWidget(self.errors_button)
        plot_layout.addLayout(point_row)
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(("Sample", "log₁₀ viscosity", "Valid points"))
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().hide()
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.setMinimumHeight(105)
        self.results_table.setMaximumHeight(175)
        plot_layout.addWidget(self.results_table)
        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([445, 1020])

        footer = QHBoxLayout()
        self.status_label = self._note("Ready — select samples and calculate.")
        footer.addWidget(self.status_label, 1)
        self.export_excel_button = self._button("Export Excel", self.export_excel)
        self.export_plot_button = self._button("Export figure", self.export_plot)
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        self.calculate_button = self._button("Calculate plot", self.calculate_plot, "calculateButton")
        self.calculate_button.setMinimumWidth(210)
        footer.addWidget(self.export_excel_button)
        footer.addWidget(self.export_plot_button)
        footer.addWidget(self.calculate_button)
        outer.addLayout(footer)
        self.sync_fixed_melt_physical_parameters()
        self._draw_empty_plot()

    def _create_samples_box(self):
        box = QGroupBox("Samples")
        layout = QVBoxLayout(box)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button("Select all", lambda: self._set_all_samples(Qt.Checked), "smallButton"))
        buttons.addWidget(self._button("Clear", lambda: self._set_all_samples(Qt.Unchecked), "smallButton"))
        buttons.addStretch()
        layout.addLayout(buttons)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(125)
        scroll.setMaximumHeight(210)
        container = QWidget()
        rows = QVBoxLayout(container)
        rows.setContentsMargins(0, 0, 5, 0)
        rows.setSpacing(3)
        current = getattr(getattr(self.main_window, "current_sample", None), "name", None)
        available = any(sample.name == current for sample in self.samples)
        self.sample_checks = []
        for index, sample in enumerate(self.samples):
            row = QHBoxLayout()
            checkbox = QCheckBox(str(sample.name))
            checkbox.setChecked(sample.name == current if available else index == 0)
            checkbox.stateChanged.connect(self.update_fixed_parameters)
            checkbox.stateChanged.connect(self._inputs_changed)
            self.sample_checks.append((checkbox, sample))
            row.addWidget(checkbox, 1)
            row.addWidget(self._button("Style…", lambda _=False, name=str(sample.name): self.edit_sample_style(name), "smallButton"))
            rows.addLayout(row)
        rows.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
        return box

    def _create_axes_box(self):
        box = QGroupBox("Plot axes and range")
        layout = QVBoxLayout(box)
        form = QFormLayout()
        self.x_selector = QComboBox()
        self.x_selector.addItems(list(self.DEFAULT_RANGES))
        self.y_selector = RichTextComboBox()
        for result_id, result in self.RESULTS.items():
            self.y_selector.addItem(result["combo"], result_id)
        form.addRow("X axis", self.x_selector)
        form.addRow("Y axis", self.y_selector)
        layout.addLayout(form)
        self.from_edit, self.to_edit, self.step_edit = (QLineEdit() for _ in range(3))
        row = QHBoxLayout()
        for title, edit in zip(("From", "To", "Step"), (self.from_edit, self.to_edit, self.step_edit)):
            column = QVBoxLayout()
            column.setSpacing(3)
            column.addWidget(QLabel(title))
            edit.setMinimumWidth(55)
            edit.textEdited.connect(self._inputs_changed)
            column.addWidget(edit)
            row.addLayout(column, 1)
        layout.addLayout(row)
        self.range_note = self._note("")
        layout.addWidget(self.range_note)
        self.x_selector.currentTextChanged.connect(self.update_defaults)
        self.x_selector.currentTextChanged.connect(self.update_fixed_parameters)
        self.x_selector.currentTextChanged.connect(self._inputs_changed)
        self.y_selector.currentIndexChanged.connect(self._refresh_result_view)
        return box

    def _create_fixed_box(self):
        box = QGroupBox("Fixed physical parameters")
        layout = QVBoxLayout(box)
        self.fixed_layout = QFormLayout()
        self.fixed_widgets = {}
        for parameter, text in (("Temperature", "Temperature (°C)"), ("H₂O", "H₂O (wt%)"),
                                ("Crystals", "Crystals (vol%)"), ("Vesicles", "Vesicles (vol%)")):
            label, edit = QLabel(text), QLineEdit()
            edit.textEdited.connect(self.mark_fixed_parameters_dirty)
            self.fixed_widgets[parameter] = label, edit
            self.fixed_layout.addRow(label, edit)
        layout.addLayout(self.fixed_layout)
        layout.addWidget(self._note("Values shown for the first selected sample. Apply to use these fixed values for all selected samples in this window."))
        self.apply_button = self._button("Apply fixed values to selected samples", self.apply_fixed_parameters)
        layout.addWidget(self.apply_button)
        return box

    # --- Sample and result controls --------------------------------

    def _draw_empty_plot(self):
        self.figure.clear()
        self._curve_artists.clear()
        self.ax = self.figure.add_subplot(111)
        self._style_axes(self.ax)
        info = self.get_selected_result()
        self.ax.set_label(info["title"])
        self.ax.set_xlabel(self.X_LABELS[self.x_selector.currentText()], fontsize=11)
        self.ax.set_ylabel(info["plot"], fontsize=11)
        self.ax.text(0.5, 0.5, "Select one or more samples and calculate the plot",
                     transform=self.ax.transAxes, ha="center", va="center", color="#83929c")
        self._apply_axis_options()
        self.toolbar.update()
        self.canvas.draw_idle()

    def _set_all_samples(self, state):
        for checkbox, _ in self.sample_checks:
            checkbox.blockSignals(True)
            checkbox.setChecked(state == Qt.Checked)
            checkbox.blockSignals(False)
        self.update_fixed_parameters()
        self._inputs_changed()

    def get_selected_samples(self):
        """Return all samples selected in the checkbox list."""
        return [
            sample
            for checkbox, sample in self.sample_checks
            if checkbox.isChecked()
        ]

    def get_selected_result(self):
        """Return the definition of the selected Y result."""
        return self.RESULTS[
            self.y_selector.currentData()
        ]

    @staticmethod
    def _result_value(result, info):
        """Read the selected viscosity from a result dictionary."""
        value = result.get(info["key"])
        return (
            value
            if value is not None
            else np.nan
        )

    @classmethod
    def _empty_result(cls):
        """Return an invalid result used when one point fails."""
        return {
            key: np.nan
            for key in cls.RESULT_KEYS
        }

    # --- Axis defaults ----------------------------------------------

    def update_defaults(self, parameter):
        """Load the default range for the selected X parameter."""
        if parameter in self.DEFAULT_RANGES:
            start, end, step = (
                self.DEFAULT_RANGES[parameter]
            )

            self.from_edit.setText(start)
            self.to_edit.setText(end)
            self.step_edit.setText(step)

        self.update_axis_units(parameter)

    def update_axis_units(self, parameter=None):
        unit = self.PARAMETER_UNITS.get(parameter or self.x_selector.currentText(), "")
        self.range_note.setText(f"Units: {unit}. Use a positive Step for ascending or descending ranges.")

    # --- Model settings ---------------------------------------------

    def copy_initial_model_settings(self):
        """Copy model selections and parameters from the MainWindow."""
        if self.main_window is None:
            return

        source = getattr(
            self.main_window,
            "models_parameters_panel",
            None,
        )

        if source is None:
            return

        target = self.models_parameters_panel

        for source_selector, target_selector in (
            (
                source.model_selector,
                target.model_selector,
            ),
            (
                source.crystal_model,
                target.crystal_model,
            ),
            (
                source.vesicle_model,
                target.vesicle_model,
            ),
        ):
            target_selector.setCurrentText(
                source_selector.currentText()
            )

        for source_widgets, target_widgets in (
            (
                source.melt_parameters,
                target.melt_parameters,
            ),
            (
                source.crystal_parameters,
                target.crystal_parameters,
            ),
            (
                source.vesicle_parameters,
                target.vesicle_parameters,
            ),
        ):
            self._copy_parameter_values(
                source_widgets,
                target_widgets,
            )

    @staticmethod
    def _copy_parameter_values(source, target):
        """Copy matching line-edit and combo-box parameter values."""
        for name, target_widget in target.items():
            source_widget = source.get(name)

            if (
                isinstance(source_widget, QLineEdit)
                and isinstance(
                    target_widget,
                    QLineEdit,
                )
            ):
                target_widget.setText(
                    source_widget.text()
                )

            elif (
                isinstance(source_widget, QComboBox)
                and isinstance(
                    target_widget,
                    QComboBox,
                )
            ):
                target_widget.setCurrentText(
                    source_widget.currentText()
                )

    # --- Fixed physical parameters ---------------------------------

    def mark_fixed_parameters_dirty(self, *args):
        """Mark edited fixed parameters as not yet applied."""
        self._inputs_changed()
        self.fixed_parameters_dirty = True

        self.apply_button.setText(
            "Apply fixed values to selected samples  ●"
        )
        self.apply_button.setToolTip(
            "Fixed values changed: apply "
            "them before calculating."
        )
        self.apply_button.setStyleSheet(
            "background:#f4c86b;"
            "border:1px solid #d79a25;"
            "color:#392b14;"
            "font-weight:700;"
        )

    def clear_fixed_parameters_dirty(self):
        """Clear the unapplied fixed-parameter warning."""
        self.fixed_parameters_dirty = False
        self.apply_button.setText(
            "Apply fixed values to selected samples"
        )
        self.apply_button.setToolTip("")
        self.apply_button.setStyleSheet("")

    def update_fixed_parameters(self, *_):
        variable = self.x_selector.currentText()
        fixed = self.get_fixed_melt_plot_parameters()
        model = self.get_selected_model("melt")
        required = getattr(model, "required_melt_physical_parameters", ("temperature", "water")) or ()
        physical_names = {"Temperature": "temperature", "H₂O": "water"}
        selected = self.get_selected_samples()
        for parameter, (label, edit) in self.fixed_widgets.items():
            unused = parameter in physical_names and physical_names[parameter] not in required
            locked = parameter in fixed
            disabled = locked or unused or parameter == variable
            edit.setEnabled(not disabled)
            edit.setReadOnly(disabled)
            edit.setStyleSheet(self.FIXED_FIELD_STYLE if disabled else "")
            if locked:
                tooltip = "Fixed by the selected melt model."
            elif parameter == variable:
                tooltip = "Values come from From, To and Step."
            elif unused:
                tooltip = "Not used by the selected melt model."
            else:
                tooltip = "Apply to assign this value to all selected samples in this plot."
            edit.setToolTip(tooltip)
            if locked:
                value = fixed[parameter]
            elif selected:
                value = getattr(selected[0], self.SAMPLE_ATTRIBUTES[parameter])
            else:
                edit.clear()
                continue
            edit.setText(f"{value:.12g}")
        self.sync_variable_parameter_visibility()
        self.clear_fixed_parameters_dirty()
        if hasattr(self, "model_parameter_buttons"):
            caricchi_sweep = variable == "γ̇ (strain rate)" and self.models_parameters_panel.crystal_model.currentText() == self.CARICCHI_MODEL
            button = self.model_parameter_buttons["crystal"]
            button.setEnabled(not caricchi_sweep)
            button.setToolTip("Strain rate is supplied by the X range." if caricchi_sweep else "Edit crystal-model parameters.")

    def apply_fixed_parameters(self):
        """Apply fixed physical values to every selected sample."""
        selected = self.get_selected_samples()

        if not selected:
            QMessageBox.warning(
                self,
                "No samples",
                "Select at least one sample.",
            )
            return

        if (
            not self.validate_model_parameters()
            or not self.validate_fixed_physical_parameters()
        ):
            return

        variable = self.x_selector.currentText()

        try:
            values = {
                self.SAMPLE_ATTRIBUTES[parameter]:
                    self._float(edit)

                for parameter, (_, edit)
                in self.fixed_widgets.items()

                if parameter != variable
            }

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid fixed parameters",
                "Fixed parameters must be numerical values.",
            )
            return

        for sample in selected:
            for attribute, value in values.items():
                setattr(
                    sample,
                    attribute,
                    value,
                )

        self.clear_fixed_parameters_dirty()
        self._inputs_changed()

        QMessageBox.information(
            self,
            "Parameters updated",
            "Fixed parameters updated for "
            f"{len(selected)} selected sample(s).",
        )

    def get_fixed_melt_plot_parameters(self):
        """Return physical values fixed by the selected melt model."""
        model = self.get_selected_model("melt")

        if model is None:
            return {}

        fixed = getattr(
            model,
            "fixed_melt_physical_parameters",
            {},
        ) or {}

        return {
            self.MODEL_FIXED_TO_PLOT_PARAMETER[name]:
                float(value)

            for name, value in fixed.items()

            if name
            in self.MODEL_FIXED_TO_PLOT_PARAMETER
        }

    def sync_fixed_melt_physical_parameters(self, *_):
        fixed = self.get_fixed_melt_plot_parameters()
        model = self.get_selected_model("melt")
        required = getattr(model, "required_melt_physical_parameters", ("temperature", "water")) or ()
        unavailable = set(fixed)
        if "temperature" not in required:
            unavailable.add("Temperature")
        if "water" not in required:
            unavailable.add("H₂O")
        if self.get_strain_rate_parameter_name() is None:
            unavailable.add("γ̇ (strain rate)")
        available = [parameter for parameter in self.DEFAULT_RANGES if parameter not in unavailable]
        current = self.x_selector.currentText()
        self.x_selector.blockSignals(True)
        self.x_selector.clear()
        self.x_selector.addItems(available)
        if current in available:
            self.x_selector.setCurrentText(current)
        self.x_selector.blockSignals(False)
        if current != self.x_selector.currentText() or not self.from_edit.text():
            self.update_defaults(self.x_selector.currentText())
        self.update_fixed_parameters()

    def apply_fixed_melt_parameters_to_sample(
        self,
        sample,
    ):
        """Apply physical values imposed by the selected melt model."""
        for parameter, value in (
            self.get_fixed_melt_plot_parameters()
            .items()
        ):
            attribute = (
                self.SAMPLE_ATTRIBUTES.get(
                    parameter
                )
            )

            if attribute:
                setattr(
                    sample,
                    attribute,
                    value,
                )

        return sample

    # --- Model access and validation -------------------------------

    def get_selected_model(self, model_type):
        """Return the currently selected model of a given type."""
        if self.viscosity_engine is None:
            return None

        selector, manager = {
            "melt": (
                self.models_parameters_panel
                .model_selector,

                self.viscosity_engine
                .melt_manager,
            ),

            "crystal": (
                self.models_parameters_panel
                .crystal_model,

                self.viscosity_engine
                .crystal_manager,
            ),

            "vesicle": (
                self.models_parameters_panel
                .vesicle_model,

                self.viscosity_engine
                .vesicle_manager,
            ),
        }[model_type]

        try:
            return manager.get_model(
                selector.currentText()
            )
        except Exception:
            return None

    @staticmethod
    def read_parameter_widgets(widgets):
        """Read model parameters from line edits and combo boxes."""
        values = {}

        for name, widget in widgets.items():
            if isinstance(widget, QComboBox):
                values[name] = (
                    widget.currentText()
                )
                continue

            try:
                values[name] = float(
                    widget.text().replace(
                        ",",
                        ".",
                    )
                )
            except ValueError:
                return None

        return values

    def get_strain_rate_parameter_name(
        self,
        parameters=None,
    ):
        """Return the strain-rate key used by the crystal model."""
        parameters = (
            self.models_parameters_panel
            .crystal_parameters

            if parameters is None
            else parameters
        )

        if "strain_rate" in parameters:
            return "strain_rate"

        if "gamma" not in parameters:
            return None

        model = self.get_selected_model(
            "crystal"
        )

        definition = getattr(
            model,
            "parameters",
            {},
        ).get(
            "gamma",
            {},
        )

        label = str(
            definition.get("label", "")
        ).lower()

        if any(
            text in label
            for text in (
                "strain",
                "γ̇",
                "s⁻¹",
                "s^-1",
            )
        ):
            return "gamma"

        return None

    def validate_model_parameters(self):
        """Validate all currently displayed model parameters."""
        groups = (
            (
                "melt",
                self.models_parameters_panel
                .melt_parameters,
            ),
            (
                "crystal",
                self.models_parameters_panel
                .crystal_parameters,
            ),
            (
                "vesicle",
                self.models_parameters_panel
                .vesicle_parameters,
            ),
        )

        variable = self.x_selector.currentText()

        for model_type, widgets in groups:
            model = self.get_selected_model(
                model_type
            )

            if model is None:
                continue

            values = self.read_parameter_widgets(
                widgets
            )

            if values is None:
                QMessageBox.warning(
                    self,
                    "Invalid model parameter",
                    f"{model.name} parameters "
                    "must be valid values.",
                )
                return False

            strain_name = (
                self.get_strain_rate_parameter_name(
                    values
                )
            )

            for name, (
                minimum,
                maximum,
            ) in getattr(
                model,
                "model_parameter_limits",
                {},
            ).items():
                if name not in values:
                    continue

                if (
                    model_type == "crystal"
                    and variable
                    == "γ̇ (strain rate)"
                    and name == strain_name
                ):
                    continue

                value = values[name]

                if not isinstance(
                    value,
                    (int, float),
                ):
                    continue

                if (
                    minimum is not None
                    and value < minimum
                ):
                    QMessageBox.warning(
                        self,
                        "Outside model limits",
                        f"{model.name}\n\n"
                        f"Parameter: {name}\n"
                        f"Current value: {value:g}\n"
                        f"Minimum: {minimum:g}",
                    )
                    return False

                if (
                    maximum is not None
                    and value > maximum
                ):
                    QMessageBox.warning(
                        self,
                        "Outside model limits",
                        f"{model.name}\n\n"
                        f"Parameter: {name}\n"
                        f"Current value: {value:g}\n"
                        f"Maximum: {maximum:g}",
                    )
                    return False

        return True

    def get_combined_physical_limits(
        self,
        model_type,
        parameters=None,
    ):
        """Combine fixed and parameter-dependent physical limits."""
        model = self.get_selected_model(
            model_type
        )

        if model is None:
            return {}, set()

        limits = deepcopy(
            getattr(
                model,
                "model_physical_limits",
                {},
            )
        )
        dynamic_names = set()

        dynamic_method = getattr(
            model,
            "get_dynamic_physical_limits",
            None,
        )

        if callable(dynamic_method):
            try:
                for parameter, value in (
                    dynamic_method(
                        parameters or {}
                    ).items()
                ):
                    limits[parameter] = value
                    dynamic_names.add(parameter)

            except Exception:
                pass

        return limits, dynamic_names

    @staticmethod
    def value_within_limit(
        value,
        minimum,
        maximum,
        parameter,
        dynamic=False,
    ):
        """Check one physical value against a model limit."""
        if (
            minimum is not None
            and value < minimum
        ):
            return False

        if maximum is None:
            return True

        if dynamic or parameter == "Vesicles":
            return value < maximum

        return value <= maximum

    @staticmethod
    def _limit_text(
        parameter,
        minimum,
        maximum,
        strict_maximum=False,
    ):
        """Format a physical-limit warning."""
        if (
            strict_maximum
            and maximum is not None
        ):
            return (
                f"{parameter} must be lower "
                f"than {maximum:g}."
            )

        parts = []

        if minimum is not None:
            parts.append(
                f"{parameter} ≥ {minimum:g}"
            )

        if maximum is not None:
            parts.append(
                f"{parameter} ≤ {maximum:g}"
            )

        return " and ".join(parts)

    def validate_fixed_physical_parameters(self):
        """Validate all physical parameters not used as the X axis."""
        variable = self.x_selector.currentText()

        settings = (
            self.models_parameters_panel
            .get_parameters()
        )

        if settings is None:
            return False

        groups = (
            (
                "melt",
                settings["melt_parameters"],
            ),
            (
                "crystal",
                settings["crystal_parameters"],
            ),
            (
                "vesicle",
                settings["vesicle_parameters"],
            ),
        )

        for model_type, parameters in groups:
            model = self.get_selected_model(
                model_type
            )

            if model is None:
                continue

            limits, dynamic_names = (
                self.get_combined_physical_limits(
                    model_type,
                    parameters,
                )
            )

            for parameter, (
                minimum,
                maximum,
            ) in limits.items():
                if (
                    parameter == variable
                    or parameter
                    not in self.fixed_widgets
                ):
                    continue

                try:
                    value = self._float(
                        self.fixed_widgets[
                            parameter
                        ][1]
                    )
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid fixed parameter",
                        f"{parameter} must be numerical.",
                    )
                    return False

                dynamic = (
                    parameter in dynamic_names
                )

                if self.value_within_limit(
                    value,
                    minimum,
                    maximum,
                    parameter,
                    dynamic,
                ):
                    continue

                text = self._limit_text(
                    parameter,
                    minimum,
                    maximum,
                    dynamic
                    or parameter == "Vesicles",
                )

                QMessageBox.warning(
                    self,
                    "Outside model limits",
                    f"{model.name}\n\n{text}\n"
                    f"Current fixed value: {value:g}",
                )
                return False

        return True

    # --- Model-dependent control visibility ------------------------

    def schedule_variable_parameter_visibility(
        self,
        *args,
    ):
        """Refresh controls after the crystal model changes."""
        QTimer.singleShot(
            0,
            self.sync_crystal_model_controls,
        )

    def sync_crystal_model_controls(self):
        """Synchronize axes and crystal-model parameters."""
        self.sync_fixed_melt_physical_parameters()
        self.sync_variable_parameter_visibility()

    def sync_variable_parameter_visibility(
        self,
        *args,
    ):
        """Hide a strain-rate field when strain rate is the X axis."""
        panel = self.models_parameters_panel
        parameters = panel.crystal_parameters
        model_name = (
            panel.crystal_model.currentText()
        )

        hide_parameters_box = (
            model_name == self.CARICCHI_MODEL
            and self.x_selector.currentText()
            == "γ̇ (strain rate)"
        )

        if hide_parameters_box:
            panel.crystal_parameters_box.setTitle(
                ""
            )
            panel.crystal_parameters_box.hide()

        elif parameters:
            panel.crystal_parameters_box.setTitle(
                f"{model_name} parameters"
            )
            panel.crystal_parameters_box.show()

        else:
            panel.crystal_parameters_box.hide()

        name = (
            self.get_strain_rate_parameter_name(
                parameters
            )
        )

        if name is None:
            return

        widget = parameters[name]
        label = (
            panel.crystal_parameters_layout
            .labelForField(widget)
        )

        widget.setVisible(
            not hide_parameters_box
        )

        if label is not None:
            label.setVisible(
                not hide_parameters_box
            )

    # --- Plot range -------------------------------------------------

    @staticmethod
    def _float(widget):
        value = float(widget.text().replace(",", "."))
        if not math.isfinite(value):
            raise ValueError("Values must be finite numbers.")
        return value

    @staticmethod
    def _parameter_limit(parameter):
        """Return the global limit of a plotted parameter."""
        aliases = {
            "H₂O": ("H₂O", "H2O"),
            "Vesicles": (
                "Vesicles",
                "Bubbles",
            ),
        }

        for name in aliases.get(
            parameter,
            (parameter,),
        ):
            if name in PARAMETER_LIMITS:
                return PARAMETER_LIMITS[name]

        return None

    def validate_variable_model_limits(
        self,
        parameter,
        start,
        end,
        settings,
    ):
        """Validate the requested X range against every selected model."""
        if parameter == "γ̇ (strain rate)":
            model = self.get_selected_model(
                "crystal"
            )
            strain_name = (
                self.get_strain_rate_parameter_name(
                    settings[
                        "crystal_parameters"
                    ]
                )
            )

            if (
                model is None
                or strain_name is None
            ):
                QMessageBox.warning(
                    self,
                    "Parameter unavailable",
                    "The selected crystal model "
                    "does not use γ̇ (strain rate).",
                )
                return False

            limits = getattr(
                model,
                "model_parameter_limits",
                {},
            )

            if strain_name not in limits:
                return True

            minimum, maximum = limits[
                strain_name
            ]

            if (
                minimum is not None
                and min(start, end) < minimum
            ):
                QMessageBox.warning(
                    self,
                    "Outside model limits",
                    f"{model.name} requires "
                    f"γ̇ ≥ {minimum:g} s⁻¹.",
                )
                return False

            if (
                maximum is not None
                and max(start, end) > maximum
            ):
                QMessageBox.warning(
                    self,
                    "Outside model limits",
                    f"{model.name} requires "
                    f"γ̇ ≤ {maximum:g} s⁻¹.",
                )
                return False

            return True

        groups = (
            (
                "melt",
                settings["melt_parameters"],
            ),
            (
                "crystal",
                settings["crystal_parameters"],
            ),
            (
                "vesicle",
                settings["vesicle_parameters"],
            ),
        )

        for model_type, parameters in groups:
            model = self.get_selected_model(
                model_type
            )

            if model is None:
                continue

            limits, dynamic_names = (
                self.get_combined_physical_limits(
                    model_type,
                    parameters,
                )
            )

            if parameter not in limits:
                continue

            minimum, maximum = limits[
                parameter
            ]
            dynamic = (
                parameter in dynamic_names
            )

            if all(
                self.value_within_limit(
                    value,
                    minimum,
                    maximum,
                    parameter,
                    dynamic,
                )
                for value in (start, end)
            ):
                continue

            text = self._limit_text(
                parameter,
                minimum,
                maximum,
                dynamic
                or parameter == "Vesicles",
            )

            QMessageBox.warning(
                self,
                "Outside model limits",
                f"{model.name}\n\n{text}",
            )
            return False

        return True

    def get_range_values(self):
        """Validate domain and allocation size before building the X values."""
        try:
            start, end, step = map(self._float, (self.from_edit, self.to_edit, self.step_edit))
            if step <= 0:
                raise ValueError("Step must be greater than zero.")
            count_float = abs(end - start) / step
            if not math.isfinite(count_float) or count_float >= 20000:
                raise ValueError("Choose a larger Step: at most 20,000 points are allowed.")
        except ValueError as error:
            QMessageBox.warning(self, "Invalid range", str(error))
            return None
        parameter = self.x_selector.currentText()
        limits = self._parameter_limit(parameter)
        if limits is not None:
            minimum, maximum = limits
            if any((minimum is not None and value < minimum) or (maximum is not None and value > maximum) for value in (start, end)):
                QMessageBox.warning(self, "Invalid range", self._limit_text(parameter, minimum, maximum))
                return None
        settings = self.models_parameters_panel.get_parameters()
        if settings is None or not self.validate_variable_model_limits(parameter, start, end, settings):
            return None
        direction = 1 if end >= start else -1
        count = math.floor(count_float + 1e-12)
        values = start + direction * step * np.arange(count + 1, dtype=float)
        if np.isclose(values[-1], end, rtol=1e-12, atol=1e-12):
            values[-1] = end
        else:
            values = np.append(values, end)
        if len(values) > 20000:
            QMessageBox.warning(self, "Range too large", "The selected range exceeds 20,000 points.")
            return None
        return values.tolist()

    # --- Plot calculation ------------------------------------------

    def apply_variable_parameter(
        self,
        sample,
        settings,
        parameter,
        value,
    ):
        """Apply one X value to a sample or model parameter."""
        if parameter in self.SAMPLE_ATTRIBUTES:
            setattr(
                sample,
                self.SAMPLE_ATTRIBUTES[parameter],
                value,
            )

        elif parameter == "γ̇ (strain rate)":
            name = (
                self.get_strain_rate_parameter_name(
                    settings[
                        "crystal_parameters"
                    ]
                )
            )

            if name is not None:
                settings[
                    "crystal_parameters"
                ][name] = value

    def calculate_plot(self):
        """Calculate complete viscosity curves for selected samples."""
        selected = self.get_selected_samples()

        if not selected:
            QMessageBox.warning(
                self,
                "No samples",
                "Select at least one sample.",
            )
            return

        if self.viscosity_engine is None:
            QMessageBox.critical(
                self,
                "Calculation error",
                "ViscosityEngine is not available.",
            )
            return

        if self.fixed_parameters_dirty:
            QMessageBox.warning(
                self,
                "Fixed parameters not applied",
                "Fixed parameters have been modified "
                "but not applied.\n\n"
                "Click 'Apply fixed values to selected "
                "samples' before calculating.",
            )
            return

        if (
            not self.validate_model_parameters()
            or not self.validate_fixed_physical_parameters()
        ):
            return

        values = self.get_range_values()

        if values is None:
            return

        settings = (
            self.models_parameters_panel
            .get_parameters()
        )

        if settings is None:
            return

        parameter = self.x_selector.currentText()
        results = {}
        resolved_samples = {}
        errors = []

        for base_sample in selected:
            model_sample = (
                self.apply_fixed_melt_parameters_to_sample(
                    deepcopy(base_sample)
                )
            )

            resolved_samples[
                base_sample.name
            ] = deepcopy(model_sample)

            sample_results = []

            for value in values:
                sample = deepcopy(model_sample)
                point_settings = deepcopy(settings)

                self.apply_variable_parameter(
                    sample,
                    point_settings,
                    parameter,
                    value,
                )

                try:
                    result = (
                        self.viscosity_engine
                        .calculate(
                            sample=sample,

                            melt_model=point_settings[
                                "melt_model"
                            ],

                            melt_parameters=point_settings[
                                "melt_parameters"
                            ],

                            crystal_model=point_settings[
                                "crystal_model"
                            ],

                            crystal_parameters=point_settings[
                                "crystal_parameters"
                            ],

                            vesicle_model=point_settings[
                                "vesicle_model"
                            ],

                            vesicle_parameters=point_settings[
                                "vesicle_parameters"
                            ],
                        )
                    )

                except Exception as error:
                    errors.append(
                        (
                            base_sample.name,
                            value,
                            str(error),
                        )
                    )
                    result = self._empty_result()

                sample_results.append(result)

            results[
                base_sample.name
            ] = sample_results

        self.last_plot_errors = errors
        self._error_messages = {(name, value): message for name, value, message in errors}
        self._inputs_dirty = False
        self.status_label.setText(f"Completed · {len(results)} samples · {len(values)} X points · {len(errors)} errors.")
        self.last_plot_values = values
        self.last_plot_results = results
        self.last_plot_parameter = parameter
        self.last_plot_settings = deepcopy(
            settings
        )
        self.last_plot_samples = resolved_samples
        self.last_plot_step = self._float(self.step_edit)
        self._point_cache_x, self._point_cache = None, {}
        self._configure_point_selector(reset=True)

        self.plot_results(
            values,
            results,
            parameter,
        )
        self._remember_original_plot()

        if errors:
            first = errors[0]

            QMessageBox.warning(
                self,
                "Some points could not be calculated",
                f"{len(errors)} point(s) returned "
                "no result and are not shown."
                f"\n\nFirst error: {first[0]}, "
                f"{parameter} = {first[1]:g}"
                f"\n{first[2]}",
            )

    # --- Plot rendering --------------------------------------------

    def _refresh_result_view(self, *_):
        if self._initializing:
            return
        if self.last_plot_results is not None:
            self.plot_results(self.last_plot_values, self.last_plot_results, self.last_plot_parameter)
        else:
            self._draw_empty_plot()

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor("white")
        ax.grid(True, color="#e6ebef", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9, colors="#44545f")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#adb9c1")

    def plot_results(self, values, results, parameter):
        """Draw complete arrays so invalid points remain gaps in each curve."""
        self._capture_curve_styles()
        self.figure.clear()
        self._curve_artists.clear()
        self.ax = self.figure.add_subplot(111)
        self._style_axes(self.ax)
        info = self.get_selected_result()
        self.ax.set_label(info["title"])
        x = np.asarray(values, dtype=float)
        all_y = []
        for name, sample_results in results.items():
            y = np.asarray([self._result_value(result, info) for result in sample_results], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not valid.any():
                continue
            all_y.extend(y[valid].tolist())
            y[~valid] = np.nan
            line, = self.ax.plot(x, y, **self._sample_styles[str(name)])
            line.set_gid(str(name))
            self._curve_artists[str(name)] = line
        self.ax.set_xlabel(self.X_LABELS.get(parameter, parameter), fontsize=11)
        self.ax.set_ylabel(info["plot"], fontsize=11)
        self._set_axis_limits(self.ax, list(values), all_y)
        if self._curve_artists:
            self.ax.legend(frameon=True, facecolor="white", edgecolor="#d5dfe5", fontsize=9)
        else:
            self.ax.text(0.5, 0.5, "No valid viscosity results", transform=self.ax.transAxes,
                         ha="center", va="center", color="#a63b2a")
        self._apply_axis_options()
        self.plot_title.setText(info["title"])
        subtitle = f"{len(results)} samples · {len(values)} X points · use Style… or Edit axes in the toolbar to customize."
        if self._inputs_dirty:
            subtitle = "Previous calculation · settings have changed. " + subtitle
        self.plot_subtitle.setText(subtitle)
        self._configure_point_selector()
        self.errors_button.setEnabled(bool(self.last_plot_errors))
        self.errors_button.setText(f"Point errors ({len(self.last_plot_errors)})…")
        self.export_plot_button.setEnabled(bool(self._curve_artists) and not self._inputs_dirty)
        self.export_excel_button.setEnabled(bool(results) and not self._inputs_dirty)
        self.toolbar.update()
        self._update_point_table()
        self.canvas.draw_idle()

    @staticmethod
    def _set_axis_limits(ax, x, y):
        """Add a small margin around calculated X and Y values."""
        for values, setter in (
            (x, ax.set_xlim),
            (y, ax.set_ylim),
        ):
            if not values:
                continue

            minimum = min(values)
            maximum = max(values)

            margin = (
                abs(minimum) * 0.05 or 1
                if minimum == maximum
                else (maximum - minimum) * 0.05
            )

            setter(
                minimum - margin,
                maximum + margin,
            )

    # --- Excel export ----------------------------------------------

    def export_excel(self):
        """Export the latest plot calculation, one sheet per sample."""
        if self._inputs_dirty:
            QMessageBox.warning(self, "Settings changed", "Calculate the current settings before exporting.")
            return

        if (
            self.last_plot_results is None
            or self.last_plot_values is None
        ):
            QMessageBox.warning(
                self,
                "No data",
                "Calculate a plot before exporting.",
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot data",
            "",
            "Excel Files (*.xlsx)",
        )

        if not filename:
            return

        if not filename.lower().endswith(
            ".xlsx"
        ):
            filename += ".xlsx"

        try:
            with pd.ExcelWriter(
                filename,
                engine="openpyxl",
            ) as writer:
                used_names = set()

                for sample_name, sample_results in (
                    self.last_plot_results.items()
                ):
                    sample = (
                        self.last_plot_samples
                        or {}
                    ).get(sample_name)

                    if sample is None:
                        continue

                    rows = [
                        self._export_row(
                            sample,
                            value,
                            result,
                        )
                        for value, result in zip(
                            self.last_plot_values,
                            sample_results,
                        )
                    ]

                    if rows:
                        pd.DataFrame(rows).to_excel(
                            writer,
                            sheet_name=self.make_sheet_name(
                                sample_name,
                                used_names,
                            ),
                            index=False,
                        )

            self._format_excel_subscripts(filename)

            QMessageBox.information(
                self,
                "Export completed",
                "Excel file successfully saved:"
                f"\n\n{filename}",
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Export error",
                "Unable to export Excel file."
                f"\n\n{error}",
            )

    def _export_row(
        self,
        sample,
        x_value,
        result,
    ):
        """Convert one calculated plot point into an Excel row."""
        parameter = self.last_plot_parameter
        settings = self.last_plot_settings

        x_column = self.EXPORT_AXIS_LABELS.get(
            parameter,
            parameter,
        )

        fixed_physical_parameters = {
            "Temperature (°C)":
                sample.temperature,

            "H₂O (wt%)":
                sample.H2O,

            "Crystals (vol%)":
                sample.crystals,

            "Vesicles (vol%)":
                sample.Vesicles,
        }

        variable_column = (
            self.EXPORT_AXIS_LABELS.get(
                parameter
            )
            if parameter
            in self.SAMPLE_ATTRIBUTES
            else None
        )

        if variable_column is not None:
            fixed_physical_parameters.pop(
                variable_column,
                None,
            )

        row = {
            "Sample":
                sample.name,

            x_column:
                x_value,

            "Melt viscosity log₁₀ ηm (Pa·s)":
                result.get(
                    "log10_eta_m",
                    np.nan,
                ),

            "Crystal correction factor ηr,c":
                result.get(
                    "eta_r_c",
                    np.nan,
                ),

            "Crystal-bearing magma viscosity "
            "log₁₀ ηmc (Pa·s)":
                result.get(
                    "log10_eta_mc",
                    np.nan,
                ),

            "Vesicle correction factor ηr,b":
                result.get(
                    "eta_r_b",
                    np.nan,
                ),

            "Vesicle-bearing magma viscosity "
            "log₁₀ ηmb (Pa·s)":
                result.get(
                    "log10_eta_mb",
                    np.nan,
                ),

            "Three-phase magma viscosity "
            "log₁₀ ηmcb (Pa·s)":
                result.get(
                    "log10_eta_mcb",
                    np.nan,
                ),

            "Liquid viscosity model":
                settings["melt_model"],

            "Crystal correction model":
                settings["crystal_model"],

            "Vesicle correction model":
                settings["vesicle_model"],

            **fixed_physical_parameters,
        }

        for key, value in (
            settings["melt_parameters"].items()
        ):
            row[
                f"Liquid parameter - {key}"
            ] = value

        crystal_parameters = deepcopy(
            settings["crystal_parameters"]
        )

        if parameter == "γ̇ (strain rate)":
            strain_name = (
                self.get_strain_rate_parameter_name(
                    crystal_parameters
                )
            )

            if strain_name is not None:
                crystal_parameters.pop(
                    strain_name,
                    None,
                )

        for key, value in (
            crystal_parameters.items()
        ):
            row[
                f"Crystal parameter - {key}"
            ] = value

        for key, value in (
            settings["vesicle_parameters"].items()
        ):
            row[
                f"Vesicle parameter - {key}"
            ] = value

        if self.last_plot_errors:
            row["Error"] = self._error_messages.get((sample.name, x_value), "")
        return row

    @staticmethod
    def make_sheet_name(name, used_names):
        """Create a valid and unique Excel worksheet name."""
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

        while sheet_name in used_names:
            suffix = f"_{counter}"
            sheet_name = (
                original[
                    :31 - len(suffix)
                ]
                + suffix
            )
            counter += 1

        used_names.add(sheet_name)
        return sheet_name

    # --- Figure export ---------------------------------------------

    def export_plot(self):
        """Export the displayed figure as PNG, PDF or SVG."""
        if self._inputs_dirty or self.last_plot_results is None:
            return

        filename, selected_filter = (
            QFileDialog.getSaveFileName(
                self,
                "Save plot",
                "",
                "PNG (*.png);;"
                "PDF (*.pdf);;"
                "SVG (*.svg)",
            )
        )

        if not filename:
            return

        suffixes = {
            "PNG": ".png",
            "PDF": ".pdf",
            "SVG": ".svg",
        }

        if not filename.lower().endswith(
            (".png", ".pdf", ".svg")
        ):
            filename += next(
                (
                    suffix
                    for name, suffix
                    in suffixes.items()
                    if selected_filter.startswith(
                        name
                    )
                ),
                ".png",
            )

        try:
            self.figure.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

            QMessageBox.information(
                self,
                "Export completed",
                "Plot successfully saved:"
                f"\n\n{filename}",
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Export error",
                "Unable to save plot."
                f"\n\n{error}",
            )


    def _mvl_icon(self):
        """Resolve the same application logo even from another working directory."""
        for path in (Path(__file__).resolve().parents[2] / "resources" / "Logo.png",
                     Path("resources/Logo.png")):
            if path.is_file():
                return QIcon(str(path))
        getter = getattr(self.main_window, "windowIcon", None)
        return getter() if callable(getter) else QApplication.windowIcon()

    def _axis_keys(self):
        parameter = self.last_plot_parameter if self.last_plot_results is not None else self.x_selector.currentText()
        return parameter, self.y_selector.currentData()

    def _default_axis_html(self, axis):
        parameter, _ = self._axis_keys()
        if axis == "y":
            return self.get_selected_result()["combo"].split(" · ", 1)[-1]
        if parameter == "γ̇ (strain rate)":
            return "γ̇ (s<sup>−1</sup>)"
        return escape(self.EXPORT_AXIS_LABELS.get(parameter, parameter)).replace("H₂O", "H<sub>2</sub>O")

    def _make_label_editor(self, html):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        editor = QTextEdit()
        editor.setAcceptRichText(False)
        editor.setLineWrapMode(QTextEdit.NoWrap)
        editor.setFixedHeight(47)
        editor.setFont(self.font())
        editor.setStyleSheet("QTextEdit { background:white; border:1px solid #cbd1d7; border-radius:5px; padding:3px; }")
        editor.setHtml(html)
        layout.addWidget(editor)
        row = QHBoxLayout()
        for text, alignment in (("Normal", QTextCharFormat.AlignNormal),
                                ("Subscript", QTextCharFormat.AlignSubScript),
                                ("Superscript", QTextCharFormat.AlignSuperScript)):
            def apply_format(_=False, alignment=alignment):
                formatting = QTextCharFormat()
                formatting.setVerticalAlignment(alignment)
                editor.mergeCurrentCharFormat(formatting)
            button = self._button(text, apply_format, "smallButton")
            button.setFocusPolicy(Qt.NoFocus)
            button.setToolTip("Select text, then choose its position.")
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        return container, editor

    @staticmethod
    def _label_to_plot(editor):
        """Convert visible rich text to a plot label, preserving subscripts."""
        parts = []
        block = editor.document().begin()
        while block.isValid():
            if parts:
                parts.append(" ")
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid():
                    text = fragment.text()
                    alignment = fragment.charFormat().verticalAlignment()
                    if alignment in (QTextCharFormat.AlignSubScript, QTextCharFormat.AlignSuperScript):
                        script = text.replace("\\", r"\backslash ").replace("{", r"\{").replace("}", r"\}").replace("$", r"\$").replace(" ", r"\ ")
                        operator = "_" if alignment == QTextCharFormat.AlignSubScript else "^"
                        parts.append("$" + operator + r"{\mathrm{" + script + "}}$")
                    else:
                        parts.append(text.replace("$", r"\$"))
                iterator += 1
            block = block.next()
        return "".join(parts).strip()

    def _apply_axis_options(self):
        parameter, result_id = self._axis_keys()
        self.ax.set_xscale("linear")
        self.ax.set_yscale("linear")
        for axis, key in (("x", parameter), ("y", result_id)):
            options = self._axis_options[axis].get(key, {})
            if options.get("plot") is not None:
                getattr(self.ax, f"set_{axis}label")(options["plot"])
            if options.get("limits") is not None:
                getattr(self.ax, f"set_{axis}lim")(options["limits"])
        self.ax.set_title(self._plot_options["title"])
        self.ax.grid(self._plot_options["grid"])
        legend = self.ax.get_legend()
        if legend is not None:
            legend.set_visible(self._plot_options["legend"])
        if self._layout_margins is not None:
            self.figure.set_layout_engine(None)
            self.figure.subplots_adjust(**self._layout_margins)

    def _make_axes_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit axes — " + self.get_selected_result()["title"])
        dialog.setWindowIcon(self.windowIcon())
        dialog.resize(620, 680)
        outer = QVBoxLayout(dialog)
        outer.addWidget(QLabel(self.get_selected_result()["title"], objectName="plotTitle"))
        tabs = QTabWidget()
        outer.addWidget(tabs, 1)
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        title = QLineEdit(self._plot_options["title"])
        form.addRow("Title", title)
        layout.addLayout(form)
        fields = {}
        for axis, key in zip(("x", "y"), self._axis_keys()):
            options = self._axis_options[axis].get(key, {})
            box = QGroupBox(f"{axis.upper()} axis")
            form = QFormLayout(box)
            container, editor = self._make_label_editor(options.get("html", self._default_axis_html(axis)))
            form.addRow("Label", container)
            automatic = QCheckBox("Automatic limits")
            automatic.setChecked(options.get("limits") is None)
            form.addRow(automatic)
            limits = options.get("limits") or getattr(self.ax, f"get_{axis}lim")()
            minimum, maximum = QLineEdit(f"{limits[0]:.12g}"), QLineEdit(f"{limits[1]:.12g}")
            row = QHBoxLayout()
            row.addWidget(QLabel("Min"))
            row.addWidget(minimum)
            row.addWidget(QLabel("Max"))
            row.addWidget(maximum)
            form.addRow(row)
            for edit in (minimum, maximum):
                edit.setEnabled(not automatic.isChecked())
            automatic.toggled.connect(lambda checked, lo=minimum, hi=maximum: (lo.setEnabled(not checked), hi.setEnabled(not checked)))
            fields[axis] = {"label": editor, "auto": automatic, "min": minimum, "max": maximum,
                            "initial_html": editor.toHtml(), "initial_plot": getattr(self.ax, f"get_{axis}label")()}
            layout.addWidget(box)
        grid = QCheckBox("Show grid")
        grid.setChecked(self._plot_options["grid"])
        legend = QCheckBox("Show sample legend")
        legend.setChecked(self._plot_options["legend"])
        row = QHBoxLayout()
        row.addWidget(grid)
        row.addWidget(legend)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        tabs.addTab(scroll, "Axes")
        curves_page = QWidget()
        curves_layout = QVBoxLayout(curves_page)
        curves_layout.addWidget(self._note("Choose a sample to edit its color, line, symbol and legend name."))
        selector = QComboBox()
        for name, style in self._sample_styles.items():
            label = str(style.get("label", name))
            selector.addItem(label if label == name else f"{label} ({name})", name)
        curves_layout.addWidget(selector)
        def edit_curve():
            name = selector.currentData()
            if name is not None:
                style_dialog = self._make_style_dialog(name)
                style_dialog.setWindowIcon(self.windowIcon())
                style_dialog.exec()
                style_dialog.deleteLater()
        button = self._button("Edit selected curve…", edit_curve)
        button.setEnabled(selector.count() > 0)
        curves_layout.addWidget(button)
        curves_layout.addStretch()
        tabs.addTab(curves_page, "Curves")
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            updates = {}
            try:
                for axis, values in fields.items():
                    editor = values["label"]
                    plot_text = (values["initial_plot"] if editor.toHtml() == values["initial_html"]
                                 else self._label_to_plot(editor))
                    MathTextParser("path").parse(plot_text)
                    limits = None if values["auto"].isChecked() else (self._float(values["min"]), self._float(values["max"]))
                    if limits is not None and limits[0] == limits[1]:
                        raise ValueError(f"{axis.upper()} limits must be different.")
                    updates[axis] = {"html": editor.toHtml(), "plot": plot_text, "limits": limits}
                MathTextParser("path").parse(title.text())
            except (ValueError, TypeError) as error:
                QMessageBox.warning(dialog, "Invalid axis settings", str(error))
                return
            for axis, key in zip(("x", "y"), self._axis_keys()):
                self._axis_options[axis][key] = updates[axis]
            self._plot_options.update(title=title.text(), grid=grid.isChecked(), legend=legend.isChecked())
            self._refresh_result_view()
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)
        dialog.axis_fields, dialog.button_box, dialog.tabs = fields, buttons, tabs
        return dialog

    def _remember_original_plot(self):
        """Keep the initial appearance of the latest calculated plot."""
        self._capture_curve_styles()
        self._original_plot_configuration = deepcopy({
            "axes": self._axis_options,
            "plot": self._plot_options,
            "margins": self._layout_margins,
            "styles": self._sample_styles,
        })

    def restore_original_plot(self):
        """Restore Home even after an editor has cleared navigation history."""
        if self._original_plot_configuration is None:
            return
        original = deepcopy(self._original_plot_configuration)
        self._axis_options = original["axes"]
        self._plot_options = original["plot"]
        self._layout_margins = original["margins"]
        # Detach edited artists before redraw can capture their styles again.
        self._curve_artists.clear()
        self._sample_styles = original["styles"]
        self.figure.set_layout_engine("constrained" if self._layout_margins is None else None)
        self._refresh_result_view()
        self.canvas.draw()
        self.toolbar.push_current()

    def _make_layout_dialog(self):
        """Single-plot borders only; no inter-plot spacing or value export."""
        self.canvas.draw()
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure plot")
        dialog.setWindowIcon(self.windowIcon())
        dialog.resize(410, 350)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note("Set margins as fractions of the figure width or height (0–1). Changing a value selects custom margins."))
        automatic = QCheckBox("Automatic margins")
        automatic.setChecked(self._layout_margins is None)
        layout.addWidget(automatic)
        position = self.ax.get_position()
        initial = self._layout_margins or {"left": position.x0, "right": position.x1,
                                           "bottom": position.y0, "top": position.y1}
        box = QGroupBox("Borders")
        form = QFormLayout(box)
        fields = {}
        step_buttons = {}
        for name in ("left", "right", "bottom", "top"):
            spin = QDoubleSpinBox()
            spin.setRange(0, 1)
            spin.setDecimals(3)
            spin.setSingleStep(.01)
            spin.setValue(initial[name])
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.valueChanged.connect(lambda _value: automatic.setChecked(False))
            decrease = self._button("−", spin.stepDown, "smallButton")
            increase = self._button("+", spin.stepUp, "smallButton")
            for button, verb in ((decrease, "Decrease"), (increase, "Increase")):
                button.setFixedWidth(32)
                button.setToolTip(f"{verb} {name} by 0.01")
            def update_buttons(_=None, spin=spin, decrease=decrease, increase=increase):
                decrease.setEnabled(spin.value() > spin.minimum())
                increase.setEnabled(spin.value() < spin.maximum())
            spin.valueChanged.connect(update_buttons)
            update_buttons()
            row = QHBoxLayout()
            row.addWidget(decrease)
            row.addWidget(spin, 1)
            row.addWidget(increase)
            fields[name] = spin
            step_buttons[name] = decrease, increase
            form.addRow(name.title(), row)
        layout.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(lambda: automatic.setChecked(True))
        def save():
            margins = {name: spin.value() for name, spin in fields.items()}
            if not automatic.isChecked() and not (margins["left"] < margins["right"] and margins["bottom"] < margins["top"]):
                QMessageBox.warning(dialog, "Invalid margins", "Left must be below Right; Bottom must be below Top.")
                return
            self._layout_margins = None if automatic.isChecked() else margins
            self.figure.set_layout_engine("constrained" if automatic.isChecked() else None)
            if self._layout_margins is not None:
                self.figure.subplots_adjust(**self._layout_margins)
            self.canvas.draw_idle()
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.margin_fields, dialog.auto_margins, dialog.button_box = fields, automatic, buttons
        dialog.margin_step_buttons = step_buttons
        return dialog

    @staticmethod
    def _button(text, callback, object_name=None):
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        if object_name:
            button.setObjectName(object_name)
        button.clicked.connect(callback)
        return button


    @staticmethod
    def _note(text):
        label = QLabel(text, objectName="plotSubtitle")
        label.setWordWrap(True)
        return label


    def _inputs_changed(self, *_):
        if self._initializing:
            return
        self._inputs_dirty = True
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        self.status_label.setText("Settings changed — calculate again." if self.last_plot_results is not None else "Ready — select samples and calculate.")
        if self.last_plot_results is not None:
            self.plot_subtitle.setText("Previous calculation · settings have changed.")
        self.point_selector.setEnabled(False)
        self.point_decrease_button.setEnabled(False)
        self.point_increase_button.setEnabled(False)
        self._update_point_table()


    def _sample_for_model_preview(self):
        selected = self.get_selected_samples()
        if not selected:
            raise ValueError("Select a sample to display computed parameters.")
        sample = self.apply_fixed_melt_parameters_to_sample(deepcopy(selected[0]))
        variable = self.x_selector.currentText()
        for parameter, (_, edit) in self.fixed_widgets.items():
            if parameter != variable and edit.text().strip():
                setattr(sample, self.SAMPLE_ATTRIBUTES[parameter], self._float(edit))
        if variable in self.SAMPLE_ATTRIBUTES and self.from_edit.text().strip():
            setattr(sample, self.SAMPLE_ATTRIBUTES[variable], self._float(self.from_edit))
        return sample


    def _make_model_dialog(self, group):
        title, selector_name, _manager = ModelsParametersPanel.MODEL_GROUPS[group]
        source = self.models_parameters_panel
        name = getattr(source, selector_name).currentText()
        parameters = source.get_group_parameters(group)
        if parameters is None:
            raise ValueError("The selected model parameters must be numeric.")
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{name} — parameters")
        dialog.resize(560, 470)
        layout = QVBoxLayout(dialog)
        selected = self.get_selected_samples()
        sample_name = str(selected[0].name) if selected else "no sample selected"
        layout.addWidget(self._note(f"Parameters apply to all selected samples. Computed values refer to {sample_name} at From."))
        panel = ModelsParametersPanel(
            manage_main_inputs=False, viscosity_engine=self.viscosity_engine,
            sample_provider=self._sample_for_model_preview, visible_groups=(group,),
        )
        panel.set_parameters({f"{group}_model": name, f"{group}_parameters": parameters})
        panel.set_model_selection_enabled(group, False)
        if group == "crystal" and self.x_selector.currentText() == "γ̇ (strain rate)":
            strain_name = self.get_strain_rate_parameter_name(parameters)
            widget = panel.crystal_parameters.get(strain_name)
            if widget is not None:
                label = panel.crystal_parameters_layout.labelForField(widget)
                widget.hide()
                if label is not None:
                    label.hide()
            if name == self.CARICCHI_MODEL:
                panel.crystal_parameters_box.hide()
            layout.addWidget(self._note("Strain rate is supplied by the X range."))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            values = panel.get_group_parameters(group)
            if values is not None:
                source.set_parameters({f"{group}_model": name, f"{group}_parameters": values})
                self.sync_variable_parameter_visibility()
                self._inputs_changed()
                dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.parameters_panel = panel
        dialog.button_box = buttons
        return dialog


    def edit_model_parameters(self, group):
        try:
            dialog = self._make_model_dialog(group)
            dialog.exec()
            dialog.deleteLater()
        except Exception as error:
            QMessageBox.warning(self, "Model parameters", str(error))


    def _capture_curve_styles(self, *_):
        """Keep edits made through Matplotlib when changing Y or recalculating."""
        for name, line in self._curve_artists.items():
            self._sample_styles[name] = {
                key: getattr(line, f"get_{key}")()
                for key in ("color", "linestyle", "linewidth", "marker", "markersize",
                            "markerfacecolor", "markeredgecolor", "markeredgewidth", "label", "alpha", "drawstyle")
            }
        for checkbox, sample in self.sample_checks:
            color = to_hex(self._sample_styles[str(sample.name)]["color"])
            checkbox.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
        if self.last_plot_results is not None:
            for row, name in enumerate(self.last_plot_results):
                item = self.results_table.item(row, 0)
                if item is not None:
                    item.setForeground(QColor(to_hex(self._sample_styles[str(name)]["color"])))


    def _make_style_dialog(self, name):
        self._capture_curve_styles()
        style = deepcopy(self._sample_styles[name])
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Curve style — {name}")
        dialog.resize(410, 350)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        label_edit = QLineEdit(str(style.get("label", name)))
        form.addRow("Legend label", label_edit)
        color = to_hex(style["color"])
        color_button = QPushButton(color)
        color_button.setStyleSheet(f"border: 3px solid {color};")
        chosen_color = [color]
        def choose_color():
            selected = QColorDialog.getColor(QColor(chosen_color[0]), dialog, f"Color — {name}")
            if selected.isValid():
                chosen_color[0] = selected.name()
                color_button.setText(selected.name())
                color_button.setStyleSheet(f"border: 3px solid {selected.name()};")
        color_button.clicked.connect(choose_color)
        form.addRow("Color", color_button)
        line_style = QComboBox()
        for label, value in (("Solid", "-"), ("Dashed", "--"), ("Dash-dot", "-."), ("Dotted", ":"), ("No line", "None")):
            line_style.addItem(label, value)
        line_style.setCurrentIndex(max(0, line_style.findData(style.get("linestyle", "-"))))
        form.addRow("Line style", line_style)
        width = QDoubleSpinBox()
        width.setRange(0.1, 12)
        width.setSingleStep(0.2)
        width.setValue(style.get("linewidth", 1.8))
        form.addRow("Line width", width)
        marker = QComboBox()
        for label, value in (("None", "None"), ("Circle", "o"), ("Square", "s"), ("Triangle", "^"), ("Diamond", "D"), ("Point", "."), ("Cross", "x"), ("Plus", "+")):
            marker.addItem(label, value)
        marker.setCurrentIndex(max(0, marker.findData(style.get("marker", "o"))))
        form.addRow("Symbol", marker)
        size = QDoubleSpinBox()
        size.setRange(0.1, 30)
        size.setSingleStep(0.5)
        size.setValue(style.get("markersize", 4.5))
        form.addRow("Symbol size", size)
        layout.addLayout(form)
        layout.addWidget(self._note("Appearance is retained when changing the plotted viscosity or recalculating."))
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            updated = dict(style, color=chosen_color[0], label=label_edit.text().strip() or name,
                           linestyle=line_style.currentData(), linewidth=width.value(),
                           marker=marker.currentData(), markersize=size.value())
            if chosen_color[0] != color:
                updated.update(markerfacecolor=chosen_color[0], markeredgecolor=chosen_color[0])
            self._sample_styles[name] = updated
            if name in self._curve_artists:
                self._curve_artists[name].set(**updated)
                self._update_legend_styles()
            self._capture_curve_styles()
            self.canvas.draw_idle()
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.style_widgets = dict(label=label_edit, color=color_button, linestyle=line_style, linewidth=width, marker=marker, markersize=size)
        dialog.button_box = buttons
        return dialog


    def edit_sample_style(self, name):
        dialog = self._make_style_dialog(name)
        dialog.exec()
        dialog.deleteLater()


    def _update_legend_styles(self):
        legend = self.ax.get_legend()
        if legend is None:
            return
        handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
        for line, handle, text in zip(self._curve_artists.values(), handles, legend.get_texts()):
            for key in ("color", "linestyle", "linewidth", "marker", "markersize", "markerfacecolor", "markeredgecolor", "alpha"):
                getattr(handle, f"set_{key}")(getattr(line, f"get_{key}")())
            text.set_text(line.get_label())


    def _configure_point_selector(self, reset=False):
        if self.last_plot_results is None:
            return
        minimum, maximum = self._parameter_limit(self.last_plot_parameter) or (None, None)
        self.point_selector.blockSignals(True)
        try:
            self.point_selector.setRange(minimum if minimum is not None else -1e100,
                                         maximum if maximum is not None else 1e100)
            self.point_selector.setSingleStep(self.last_plot_step)
            if reset:
                self.point_selector.setValue(float(self.last_plot_values[0]))
        finally:
            self.point_selector.blockSignals(False)
        self.point_selector.setEnabled(not self._inputs_dirty)
        self.point_selector.setToolTip("Enter X and press Enter. Y is calculated using the models and sample inputs saved with this plot.")
        unit = self.PARAMETER_UNITS[self.last_plot_parameter]
        for button, action in ((self.point_decrease_button, "Decrease"), (self.point_increase_button, "Increase")):
            button.setToolTip(f"{action} X by {self.last_plot_step:g} {unit}")
        self._update_point_buttons()

    def _update_point_buttons(self):
        enabled = self.last_plot_results is not None and not self._inputs_dirty
        value = self.point_selector.value()
        self.point_decrease_button.setEnabled(enabled and value > self.point_selector.minimum())
        self.point_increase_button.setEnabled(enabled and value < self.point_selector.maximum())

    def _query_point(self, x):
        """Evaluate the actual model at X, without interpolating the grid."""
        if self._point_cache_x == x:
            return self._point_cache
        index = next((i for i, value in enumerate(self.last_plot_values) if float(value) == x), None)
        answers = {}
        if index is not None:
            for name, results in self.last_plot_results.items():
                answers[name] = results[index], self._error_messages.get((name, self.last_plot_values[index]), "")
        else:
            models = {
                group: getattr(self.viscosity_engine, f"{group}_manager").get_model(self.last_plot_settings[f"{group}_model"])
                for group in ("melt", "crystal", "vesicle")
            }
            for name, source_sample in self.last_plot_samples.items():
                try:
                    sample = deepcopy(source_sample)
                    settings = deepcopy(self.last_plot_settings)
                    parameter = self.last_plot_parameter
                    if parameter in self.SAMPLE_ATTRIBUTES:
                        setattr(sample, self.SAMPLE_ATTRIBUTES[parameter], x)
                    elif parameter == "γ̇ (strain rate)":
                        parameters = settings["crystal_parameters"]
                        strain_name = "strain_rate" if "strain_rate" in parameters else None
                        if strain_name is None and "gamma" in parameters:
                            label = str(getattr(models["crystal"], "parameters", {}).get("gamma", {}).get("label", "")).lower()
                            if any(text in label for text in ("strain", "γ̇", "s⁻¹", "s^-1")):
                                strain_name = "gamma"
                        if strain_name is None:
                            raise ValueError("The selected crystal model does not use strain rate.")
                        parameters[strain_name] = x
                        lo, hi = getattr(models["crystal"], "model_parameter_limits", {}).get(strain_name, (None, None))
                        if (lo is not None and x < lo) or (hi is not None and x > hi):
                            raise ValueError("Strain rate is outside the selected model limits.")
                    for group, model in models.items():
                        limits = dict(getattr(model, "model_physical_limits", {}) or {})
                        dynamic_method = getattr(model, "get_dynamic_physical_limits", None)
                        dynamic = (dynamic_method(settings[f"{group}_parameters"]) or {}) if callable(dynamic_method) else {}
                        limits.update(dynamic)
                        for physical, (lo, hi) in limits.items():
                            if physical not in self.SAMPLE_ATTRIBUTES:
                                continue
                            value = float(getattr(sample, self.SAMPLE_ATTRIBUTES[physical]))
                            if not math.isfinite(value) or not self.value_within_limit(value, lo, hi, physical, physical in dynamic):
                                raise ValueError(f"{settings[f'{group}_model']}: " + self._limit_text(physical, lo, hi, physical in dynamic or physical == "Vesicles"))
                    result = self.viscosity_engine.calculate(sample=sample, **settings)
                    answers[name] = result, ""
                except Exception as error:
                    answers[name] = self._empty_result(), str(error)
        self._point_cache_x, self._point_cache = x, answers
        return answers

    def _update_point_table(self, *_):
        if self.last_plot_results is None or not self.last_plot_values:
            return
        self._update_point_buttons()
        x = self.point_selector.value()
        parameter = self.last_plot_parameter
        self.point_label.setText("Calculate again to query X." if self._inputs_dirty else self.PARAMETER_UNITS[parameter])
        answers = {} if self._inputs_dirty else self._query_point(x)
        info = self.get_selected_result()
        self.results_table.setRowCount(len(self.last_plot_results))
        for row, (name, results) in enumerate(self.last_plot_results.items()):
            result, message = answers.get(name, (self._empty_result(), "Settings changed: calculate the plot again."))
            value = self._result_value(result, info)
            values = [self._result_value(result, info) for result in results]
            text_values = (str(name), f"{value:.6f}" if math.isfinite(value) else "—",
                           f"{int(np.isfinite(values).sum())}/{len(values)}")
            for column, text in enumerate(text_values):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setForeground(QColor(to_hex(self._sample_styles[str(name)]["color"])))
                item.setToolTip(message or f"{parameter} = {x:.12g} {self.PARAMETER_UNITS[parameter]}")
                self.results_table.setItem(row, column, item)


    def _plot_clicked(self, event):
        if (not self.last_plot_values or self._inputs_dirty or event.inaxes is not self.ax
                or event.xdata is None or self.toolbar.mode or not math.isfinite(event.xdata)):
            return
        self.point_selector.setValue(float(event.xdata))


    def show_plot_errors(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("2D Plot — point errors")
        dialog.resize(900, 510)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note(f"{len(self.last_plot_errors)} failed points. Showing up to 2,000; Excel includes all errors."))
        table = QTableWidget(min(2000, len(self.last_plot_errors)), 3)
        table.setHorizontalHeaderLabels(("Sample", self.EXPORT_AXIS_LABELS.get(self.last_plot_parameter, "X"), "Reason"))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        for row, (name, value, error) in enumerate(self.last_plot_errors[:2000]):
            for column, text in enumerate((str(name), f"{value:.8g}", error)):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(row, column, item)
        table.resizeRowsToContents()
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()


    @staticmethod
    def _format_excel_subscripts(filename):
        """Keep η normal and its phase letters subscripted in Excel headers.

        Header-only OOXML formatting also supports openpyxl versions without
        CellRichText. Scientific values and worksheet structure are untouched.
        """
        namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        tag = lambda name: f"{{{namespace}}}{name}"
        pattern = re.compile(r"η(mcb|mc|mb|m|r,[cb])")
        path = Path(filename)
        with NamedTemporaryFile(dir=path.parent, suffix=".xlsx", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with ZipFile(path) as source, ZipFile(temporary_path, "w") as target:
                for entry in source.infolist():
                    content = source.read(entry.filename)
                    if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", entry.filename):
                        root = ET.fromstring(content)
                        header = root.find(f"{tag('sheetData')}/{tag('row')}[@r='1']")
                        for cell in header if header is not None else ():
                            inline = cell.find(tag("is"))
                            if inline is None:
                                continue
                            text = "".join(inline.itertext())
                            matches = list(pattern.finditer(text))
                            if not matches:
                                continue
                            inline.clear()
                            parts, start = [], 0
                            for match in matches:
                                parts.extend(((text[start:match.start(1)], False), (match.group(1), True)))
                                start = match.end()
                            parts.append((text[start:], False))
                            for value, subscript in parts:
                                if not value:
                                    continue
                                run = ET.SubElement(inline, tag("r"))
                                font = ET.SubElement(run, tag("rPr"))
                                ET.SubElement(font, tag("b"))
                                if subscript:
                                    ET.SubElement(font, tag("vertAlign"), {"val": "subscript"})
                                value_element = ET.SubElement(run, tag("t"))
                                value_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                                value_element.text = value
                        content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                    target.writestr(entry, content)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)