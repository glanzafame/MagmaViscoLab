"""MagmaViscoLab: multi-sample 3D sweeps with editable axes and surfaces."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.mathtext import MathTextParser
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QCheckBox, QColorDialog, QComboBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressDialog, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)
from core.parameter_limits import PARAMETER_LIMITS
from gui.panels.models_parameters_panel import ModelsParametersPanel
from gui.plot_window import PlotWindow, Plot2DToolbar, RichTextComboBox, XValueSpinBox


class Plot3DToolbar(NavigationToolbar2QT):
    """Route the toolbar editor to controls that understand three axes."""

    toolitems = tuple((text, "Edit axes and surfaces", icon, callback)
                      if callback == "edit_parameters" else (text, tooltip, icon, callback)
                      for text, tooltip, icon, callback in Plot2DToolbar.toolitems)

    def __init__(self, canvas, parent, plot_window):
        self.plot_window = plot_window
        super().__init__(canvas, parent)

    def edit_parameters(self, *_):
        self.plot_window.edit_axes()

    def configure_subplots(self, *_):
        dialog = self.plot_window._make_layout_dialog()
        dialog.exec()
        dialog.deleteLater()

    def home(self, *_):
        self.plot_window.restore_original_plot()

    def save_figure(self, *_):
        self.plot_window.export_plot()


class Plot3DWindow(QMainWindow):

    # --- Shared 2D/3D definitions ----------------------------------

    CARICCHI_MODEL = "Caricchi et al. (2007)"

    PARAMETERS = list(
        PlotWindow.DEFAULT_RANGES
    )
    DEFAULT_RANGES = PlotWindow.DEFAULT_RANGES
    PARAMETER_UNITS = PlotWindow.PARAMETER_UNITS
    SAMPLE_ATTRIBUTES = PlotWindow.SAMPLE_ATTRIBUTES
    MODEL_FIXED_TO_PLOT_PARAMETER = (
        PlotWindow.MODEL_FIXED_TO_PLOT_PARAMETER
    )
    FIXED_FIELD_STYLE = PlotWindow.FIXED_FIELD_STYLE
    LABELS = PlotWindow.X_LABELS
    EXPORT_AXIS_LABELS = (
        PlotWindow.EXPORT_AXIS_LABELS
    )
    RESULTS = PlotWindow.RESULTS
    STYLE = PlotWindow.STYLE

    PLOT_RESULT_KEYS = tuple(
        result["key"]
        for result in RESULTS.values()
    )

    Z_EXPORT_LABELS = {
        "m":
            "Melt viscosity log₁₀ ηm (Pa·s)",

        "mc":
            "Crystal-bearing magma viscosity "
            "log₁₀ ηmc (Pa·s)",

        "mb":
            "Vesicle-bearing magma viscosity "
            "log₁₀ ηmb (Pa·s)",

        "mcb":
            "Three-phase magma viscosity "
            "log₁₀ ηmcb (Pa·s)",
    }

    # --- Surface appearance and interaction ------------------------

    SURFACE_COLORS = (
        "#b94a36",
        "#2f6f8f",
        "#d49a2a",
        "#5c8f5a",
        "#7b5ea7",
        "#57616b",
    )

    ROTATION_STEP = 5.0
    MIN_ELEVATION = -89.0
    MAX_ELEVATION = 89.0

    PLOT_BOX_ASPECT = (1.0, 1.0, 0.78)
    PLOT_ZOOM = 0.984
    MIN_PLOT_ZOOM = 0.55
    MAX_PLOT_ZOOM = 1.25
    ZOOM_STEP = 0.05

    DEFAULT_PLOT_MARGINS = {"left": 0.015, "right": 0.97, "bottom": 0.10, "top": 0.94}


    def __init__(self, sample=None, main_window=None):
        super().__init__()
        self.main_window = main_window
        source = list(getattr(main_window, "samples", None) or [])
        self.samples = deepcopy(source or ([sample] if sample is not None else []))
        self.viscosity_engine = getattr(main_window, "viscosity_engine", None)
        self.X = self.Y = self.Z = self.result_surfaces = None
        self.last_settings = self.last_ranges = None
        self.last_x_parameter = self.last_y_parameter = None
        self.last_z_parameter = self.last_z_result_id = None
        self.last_plot_samples = {}
        self.last_plot_errors = []
        self._error_messages = {}
        self._initializing = True
        self._inputs_dirty = False
        self._calculating = False
        self.fixed_parameters_dirty = False
        self.axes = self.colorbar = self.color_norm = None
        self.legend_axes = self.sample_legend = None
        self._legend_labels = []
        self._layout_margins = None
        self._original_plot_configuration = None
        self._point_cache_key, self._point_cache = None, {}
        self.plot_zoom = self.PLOT_ZOOM
        self._view = (25.0, -60.0)
        self._range_parameters = {}
        self._axis_options = {}
        self._result_options = {}
        self._display = {"title": "", "font_size": 11, "grid": True,
                         "legend": True, "cmap": "viridis", "projection": "persp"}
        self._sample_styles = {
            str(item.name): {"label": str(item.name),
                             "color": self.SURFACE_COLORS[i % len(self.SURFACE_COLORS)],
                             "alpha": 0.85, "mesh": True, "linewidth": 0.7,
                             "linestyle": "-"}
            for i, item in enumerate(self.samples)
        }
        self.setWindowTitle("MagmaViscoLab – 3D Plot")
        self.setWindowIcon(PlotWindow._mvl_icon(self))
        self.resize(1500, 940)
        self.setMinimumSize(1120, 740)
        self.setStyleSheet(self.STYLE)
        self.create_interface()
        self._initializing = False
        self._remember_original_plot()

    def create_interface(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.addWidget(QLabel("3D Plot", objectName="panelTitle"))
        outer.addWidget(self._note("Compare selected samples across two physical parameters."))
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)
        self.controls = QWidget(objectName="controlPanel")
        layout = QVBoxLayout(self.controls)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._create_samples_box())
        layout.addWidget(self._create_axes_box())
        layout.addWidget(self._create_fixed_box())

        self.models_parameters_panel = ModelsParametersPanel(
            self.main_window, manage_main_inputs=False,
            viscosity_engine=self.viscosity_engine,
            sample_provider=self._sample_for_model_preview,
        )
        self.models_parameters_panel.setParent(self.controls)
        self.models_parameters_panel.hide()
        source = getattr(self.main_window, "models_parameters_panel", None)
        if source is not None:
            settings = source.get_parameters()
            if settings is not None:
                self.models_parameters_panel.set_parameters(deepcopy(settings))
        box = QGroupBox("Models and parameters")
        model_layout = QVBoxLayout(box)
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
        layout.addWidget(box)
        layout.addStretch()
        panel = self.models_parameters_panel
        panel.model_selector.currentTextChanged.connect(self.sync_fixed_melt_physical_parameters)
        panel.crystal_model.currentTextChanged.connect(self.sync_crystal_model_controls)
        panel.parameters_changed.connect(self._inputs_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(520)
        scroll.setWidget(self.controls)
        splitter.addWidget(scroll)

        self.plot_panel = QWidget()
        plot_layout = QVBoxLayout(self.plot_panel)
        plot_layout.setContentsMargins(14, 0, 0, 0)
        self.plot_title = QLabel("Viscosity surfaces", objectName="plotTitle")
        self.plot_subtitle = self._note("Surface colors show viscosity; colored meshes identify samples.")
        plot_layout.addWidget(self.plot_title)
        plot_layout.addWidget(self.plot_subtitle)
        self.figure = Figure(figsize=(9, 6), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.toolbar = Plot3DToolbar(self.canvas, self.plot_panel, self)
        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(self.toolbar, 1)
        plot_layout.addLayout(toolbar_row)
        plot_layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("key_press_event", self.rotate_with_keyboard)
        self.canvas.mpl_connect("scroll_event", self.zoom_with_mouse_wheel)
        self.canvas.mpl_connect("resize_event", lambda _: self.apply_safe_plot_layout())
        self.canvas.mpl_connect("button_press_event", lambda _: self.canvas.setFocus())
        plot_layout.addWidget(self._note("Drag to rotate · arrow keys to adjust the view · mouse wheel to zoom."))
        point_row = QHBoxLayout()
        self.point_step_buttons = {}
        for axis in ("x", "y"):
            column = QVBoxLayout()
            column.setSpacing(3)
            column.addWidget(QLabel(f"Value at {axis.upper()} point"))
            row = QHBoxLayout()
            spin = XValueSpinBox()
            spin.setMinimumWidth(110)
            spin.setEnabled(False)
            spin.valueChanged.connect(self._update_point_table)
            spin.editingFinished.connect(self._update_point_table)
            setattr(self, f"{axis}_point_selector", spin)
            decrease = self._button("−", spin.stepDown, "smallButton")
            increase = self._button("+", spin.stepUp, "smallButton")
            for button in (decrease, increase):
                button.setFixedWidth(32)
                button.setEnabled(False)
            self.point_step_buttons[axis] = decrease, increase
            row.addWidget(decrease)
            row.addWidget(spin, 1)
            row.addWidget(increase)
            unit = QLabel("")
            setattr(self, f"{axis}_point_unit", unit)
            row.addWidget(unit)
            column.addLayout(row)
            point_row.addLayout(column, 1)
        point_row.addStretch()
        self.errors_button = self._button("Point errors…", self.show_plot_errors)
        self.errors_button.setEnabled(False)
        point_row.addWidget(self.errors_button)
        plot_layout.addLayout(point_row)
        self.point_label = self._note("—")
        plot_layout.addWidget(self.point_label)
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(("Sample", "Z", "Valid grid points"))
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().hide()
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.setMinimumHeight(100)
        self.results_table.setMaximumHeight(155)
        plot_layout.addWidget(self.results_table)
        splitter.addWidget(self.plot_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([445, 1020])

        footer = QHBoxLayout()
        self.status_label = self._note("Ready — select samples and calculate.")
        footer.addWidget(self.status_label, 1)
        self.export_excel_button = self._button("Export Excel", self.export_data)
        self.export_plot_button = self._button("Export figure", self.export_plot)
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        self.calculate_button = self._button("Calculate plot", self.calculate_plot, "calculateButton")
        self.calculate_button.setMinimumWidth(210)
        for button in (self.export_excel_button, self.export_plot_button, self.calculate_button):
            footer.addWidget(button)
        outer.addLayout(footer)
        self.sync_fixed_melt_physical_parameters()
        self._draw_empty_plot()

    def _create_axes_box(self):
        box = QGroupBox("Plot axes and ranges")
        layout = QVBoxLayout(box)
        for axis in ("x", "y"):
            selector = QComboBox()
            selector.addItems(self.PARAMETERS)
            if axis == "y":
                selector.setCurrentIndex(1)
            setattr(self, f"{axis}_selector", selector)
            form = QFormLayout()
            form.addRow(f"{axis.upper()} axis", selector)
            layout.addLayout(form)
            row = QHBoxLayout()
            for name, title in (("from", "From"), ("to", "To"), ("step", "Step")):
                column = QVBoxLayout()
                column.setSpacing(3)
                column.addWidget(QLabel(title))
                edit = QLineEdit()
                edit.setMinimumWidth(55)
                edit.textEdited.connect(self._inputs_changed)
                setattr(self, f"{axis}_{name}", edit)
                column.addWidget(edit)
                row.addLayout(column, 1)
            layout.addLayout(row)
            note = self._note("")
            setattr(self, f"{axis}_range_note", note)
            layout.addWidget(note)
        self.z_selector = RichTextComboBox()
        for result_id, result in self.RESULTS.items():
            self.z_selector.addItem(result["combo"], result_id)
        form = QFormLayout()
        form.addRow("Z axis", self.z_selector)
        layout.addLayout(form)
        self.x_selector.currentTextChanged.connect(self.update_y_parameters)
        self.y_selector.currentTextChanged.connect(self.update_defaults)
        self.y_selector.currentTextChanged.connect(self.update_fixed_parameters)
        for selector in (self.x_selector, self.y_selector):
            selector.currentTextChanged.connect(self._inputs_changed)
        self.z_selector.currentIndexChanged.connect(self._refresh_result_view)
        return box

    def _inputs_changed(self, *_):
        if self._initializing:
            return
        self._inputs_dirty = True
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        self.status_label.setText("Settings changed — calculate again.")
        if self.result_surfaces is not None:
            self.plot_subtitle.setText("Previous calculation · settings have changed.")
        for axis in ("x", "y"):
            getattr(self, f"{axis}_point_selector").setEnabled(False)
            for button in self.point_step_buttons[axis]:
                button.setEnabled(False)
        self._update_point_table()

    def get_available_axis_parameters(self):
        fixed = self.get_fixed_melt_plot_parameters()
        model = self.get_selected_model("melt")
        required = getattr(model, "required_melt_physical_parameters", ("temperature", "water")) or ()
        names = {"Temperature": "temperature", "H₂O": "water"}
        available = [p for p in self.PARAMETERS if p not in fixed
                     and (p not in names or names[p] in required)]
        if self.get_strain_rate_parameter_name() is None:
            available = [p for p in available if p != "γ̇ (strain rate)"]
        return available

    def sync_fixed_melt_physical_parameters(self, *_):
        available = self.get_available_axis_parameters()
        current = self.x_selector.currentText()
        self.x_selector.blockSignals(True)
        self.x_selector.clear()
        self.x_selector.addItems(available)
        if current in available:
            self.x_selector.setCurrentText(current)
        self.x_selector.blockSignals(False)
        self.update_y_parameters()

    def update_y_parameters(self, *_):
        available = [p for p in self.get_available_axis_parameters() if p != self.x_selector.currentText()]
        current = self.y_selector.currentText()
        self.y_selector.blockSignals(True)
        self.y_selector.clear()
        self.y_selector.addItems(available)
        if current in available:
            self.y_selector.setCurrentText(current)
        self.y_selector.blockSignals(False)
        self.update_defaults()
        self.update_fixed_parameters()

    def update_defaults(self, *_):
        for axis in ("x", "y"):
            parameter = getattr(self, f"{axis}_selector").currentText()
            if parameter not in self.DEFAULT_RANGES:
                continue
            if self._range_parameters.get(axis) != parameter:
                for name, value in zip(("from", "to", "step"), self.DEFAULT_RANGES[parameter]):
                    getattr(self, f"{axis}_{name}").setText(value)
                self._range_parameters[axis] = parameter
            getattr(self, f"{axis}_range_note").setText(f"{parameter} range in {self.PARAMETER_UNITS[parameter]}")

    def update_fixed_parameters(self, *_):
        if not hasattr(self, "models_parameters_panel"):
            return
        variables = {self.x_selector.currentText(), self.y_selector.currentText()}
        fixed = self.get_fixed_melt_plot_parameters()
        model = self.get_selected_model("melt")
        required = getattr(model, "required_melt_physical_parameters", ("temperature", "water")) or ()
        physical_names = {"Temperature": "temperature", "H₂O": "water"}
        selected = self.get_selected_samples()
        for parameter, (_, edit) in self.fixed_widgets.items():
            unused = parameter in physical_names and physical_names[parameter] not in required
            disabled = parameter in fixed or parameter in variables or unused
            edit.setEnabled(not disabled)
            edit.setReadOnly(disabled)
            edit.setStyleSheet(self.FIXED_FIELD_STYLE if disabled else "")
            tooltip = ("Fixed by the selected melt model." if parameter in fixed else
                       "Values come from the axis range." if parameter in variables else
                       "Not used by the selected melt model." if unused else
                       "Apply to use this value for all selected samples in this window.")
            edit.setToolTip(tooltip)
            value = fixed.get(parameter)
            if value is None and selected:
                value = getattr(selected[0], self.SAMPLE_ATTRIBUTES[parameter])
            edit.setText("" if value is None else f"{float(value):.12g}")
        self.sync_variable_parameter_visibility()
        self.clear_fixed_parameters_dirty()

    def mark_fixed_parameters_dirty(self, *_):
        self.fixed_parameters_dirty = True
        self.apply_button.setText("Apply fixed values to selected samples  ●")
        self.apply_button.setStyleSheet("background:#f4c86b;border:1px solid #d79a25;color:#392b14;")
        self._inputs_changed()

    def apply_fixed_parameters(self):
        selected = self.get_selected_samples()
        if not selected:
            QMessageBox.warning(self, "No samples", "Select at least one sample.")
            return
        try:
            values = {self.SAMPLE_ATTRIBUTES[p]: self._float(edit)
                      for p, (_, edit) in self.fixed_widgets.items() if not edit.isReadOnly()}
            for p, (_, edit) in self.fixed_widgets.items():
                if edit.isReadOnly():
                    continue
                minimum, maximum = self._parameter_limit(p) or (None, None)
                value = self._float(edit)
                if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                    raise ValueError(self.format_limit_text(p, minimum, maximum))
            if not self.validate_model_parameters() or not self.validate_fixed_physical_parameters():
                return
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "Invalid fixed parameters", str(error))
            return
        for sample in selected:
            for attribute, value in values.items():
                setattr(sample, attribute, value)
        self.clear_fixed_parameters_dirty()
        self._inputs_changed()
        self.status_label.setText(f"Fixed values applied to {len(selected)} sample(s) — calculate again.")

    def sync_crystal_model_controls(self, *_):
        self.sync_fixed_melt_physical_parameters()
        self.sync_variable_parameter_visibility()

    def sync_variable_parameter_visibility(self, *_):
        panel = self.models_parameters_panel
        swept = "γ̇ (strain rate)" in {self.x_selector.currentText(), self.y_selector.currentText()}
        name = self.get_strain_rate_parameter_name()
        widget = panel.crystal_parameters.get(name)
        if widget is not None:
            widget.setVisible(not swept)
            label = panel.crystal_parameters_layout.labelForField(widget)
            if label is not None:
                label.setVisible(not swept)
        caricchi = swept and panel.crystal_model.currentText() == self.CARICCHI_MODEL
        panel.crystal_parameters_box.setVisible(bool(panel.crystal_parameters) and not caricchi)
        if hasattr(self, "model_parameter_buttons"):
            button = self.model_parameter_buttons["crystal"]
            button.setEnabled(not caricchi)
            button.setToolTip("Strain rate is supplied by the selected axis range." if caricchi else "Edit crystal-model parameters.")

    def _sample_for_model_preview(self):
        selected = self.get_selected_samples()
        if not selected:
            raise ValueError("Select a sample to display computed parameters.")
        sample = deepcopy(selected[0])
        for p, (_, edit) in self.fixed_widgets.items():
            if edit.text().strip():
                setattr(sample, self.SAMPLE_ATTRIBUTES[p], self._float(edit))
        for axis in ("x", "y"):
            p = getattr(self, f"{axis}_selector").currentText()
            if p in self.SAMPLE_ATTRIBUTES:
                setattr(sample, self.SAMPLE_ATTRIBUTES[p], self._float(getattr(self, f"{axis}_from")))
        return self.apply_fixed_melt_parameters_to_sample(sample)

    def get_combined_physical_limits(self, model_type, parameters=None):
        model = self.get_selected_model(model_type)
        limits = deepcopy(getattr(model, "model_physical_limits", {}) or {})
        method = getattr(model, "get_dynamic_physical_limits", None)
        dynamic = method(parameters or {}) if callable(method) else {}
        limits.update(dynamic or {})
        return limits, set(dynamic or {})

    def create_range(self, start_edit, end_edit, step_edit, parameter, settings):
        try:
            start, end, step = map(self._float, (start_edit, end_edit, step_edit))
            if step <= 0:
                raise ValueError("Step must be greater than zero.")
            limit = self._parameter_limit(parameter)
            if limit is not None:
                minimum, maximum = limit
                if any((minimum is not None and v < minimum) or (maximum is not None and v > maximum) for v in (start, end)):
                    raise ValueError(self.format_limit_text(parameter, minimum, maximum))
            intervals = abs(end - start) / step
            if not math.isfinite(intervals) or intervals >= 2000:
                raise ValueError(f"The {parameter} range exceeds 2,000 points. Increase Step.")
            if not self.validate_axis_model_limits(parameter, start, end, settings):
                return None
            count = int(math.floor(intervals + 1e-12))
            direction = 1 if end >= start else -1
            values = start + direction * step * np.arange(count + 1, dtype=float)
            if not np.isclose(values[-1], end, rtol=1e-12, atol=1e-12):
                values = np.append(values, end)
            else:
                values[-1] = end
            if len(values) > 2000 or (len(values) > 1 and np.any(direction * np.diff(values) <= 0)):
                raise ValueError("Step is too small to produce a valid range.")
            return values
        except (ValueError, TypeError, OverflowError) as error:
            QMessageBox.warning(self, "Invalid range", f"{parameter}\n\n{error}")
            return None

    def _validate_point(self, sample, settings):
        # Validate each sample and the actual swept parameters, including
        # physical limits that change with the crystal strain rate.
        for p, attribute in self.SAMPLE_ATTRIBUTES.items():
            value = float(getattr(sample, attribute))
            minimum, maximum = self._parameter_limit(p) or (None, None)
            if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                raise ValueError(f"{p}: invalid physical value {value:g}.")
        for group in ("melt", "crystal", "vesicle"):
            model = getattr(self.viscosity_engine, f"{group}_manager").get_model(settings[f"{group}_model"])
            limits = deepcopy(getattr(model, "model_physical_limits", {}) or {})
            method = getattr(model, "get_dynamic_physical_limits", None)
            dynamic = (method(settings[f"{group}_parameters"]) or {}) if callable(method) else {}
            limits.update(dynamic)
            for p, (minimum, maximum) in limits.items():
                if p in self.SAMPLE_ATTRIBUTES:
                    value = float(getattr(sample, self.SAMPLE_ATTRIBUTES[p]))
                    if not self.value_within_limit(value, minimum, maximum, p, p in dynamic):
                        raise ValueError(f"{settings[f'{group}_model']}: " + self.format_limit_text(p, minimum, maximum, p in dynamic))

    def calculate_plot(self):
        if self._calculating:
            return
        selected = self.get_selected_samples()
        if not selected or self.viscosity_engine is None:
            QMessageBox.warning(self, "Calculation unavailable", "Select a sample and an available viscosity engine.")
            return
        if self.fixed_parameters_dirty:
            QMessageBox.warning(self, "Fixed values not applied", "Apply the edited fixed values before calculating.")
            return
        try:
            if not self.validate_model_parameters() or not self.validate_fixed_physical_parameters():
                return
            settings = self.models_parameters_panel.get_parameters()
            if settings is None:
                return
            xp, yp = self.x_selector.currentText(), self.y_selector.currentText()
            if not xp or not yp or xp == yp:
                raise ValueError("Select two different physical parameters.")
            xv = self.create_range(self.x_from, self.x_to, self.x_step, xp, settings)
            yv = self.create_range(self.y_from, self.y_to, self.y_step, yp, settings) if xv is not None else None
            if xv is None or yv is None:
                return
            total = len(xv) * len(yv) * len(selected)
            if total > 250000:
                raise ValueError(f"The grid requires {total:,} calculations. Increase one or both steps (maximum 250,000 points).")
        except Exception as error:
            QMessageBox.warning(self, "Calculation settings", str(error))
            return

        X, Y = np.meshgrid(xv, yv)
        surfaces, snapshots, errors, messages = {}, {}, [], {}
        progress = QProgressDialog("Calculating surfaces…", "Cancel", 0, total, self)
        progress.setWindowTitle("3D Plot")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        self._calculating = True
        self.controls.setEnabled(False)
        self.plot_panel.setEnabled(False)
        self.calculate_button.setEnabled(False)
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        completed, cancelled = 0, False
        try:
            for base in selected:
                name = str(base.name)
                model_sample = self.apply_fixed_melt_parameters_to_sample(deepcopy(base))
                snapshots[name] = deepcopy(model_sample)
                arrays = {key: np.full(X.shape, np.nan) for key in self.PLOT_RESULT_KEYS}
                for row, column in np.ndindex(X.shape):
                    if completed % 200 == 0:
                        progress.setLabelText(f"{name} · {completed:,} / {total:,} points")
                        progress.setValue(completed)
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            cancelled = True
                            break
                    sample, point_settings = deepcopy(model_sample), deepcopy(settings)
                    self.apply_variable_parameter(sample, point_settings, xp, float(X[row, column]))
                    self.apply_variable_parameter(sample, point_settings, yp, float(Y[row, column]))
                    try:
                        self._validate_point(sample, point_settings)
                        result = self.viscosity_engine.calculate(sample=sample, **point_settings)
                        values = {key: float(result[key]) for key in self.PLOT_RESULT_KEYS}
                        if not all(math.isfinite(v) for v in values.values()):
                            raise ValueError("The model returned a non-finite viscosity.")
                        for key, value in values.items():
                            arrays[key][row, column] = value
                    except Exception as error:
                        message = str(error)
                        errors.append({"Sample": name, self.EXPORT_AXIS_LABELS[xp]: float(X[row, column]),
                                       self.EXPORT_AXIS_LABELS[yp]: float(Y[row, column]), "Error": message})
                        messages[name, row, column] = message
                    completed += 1
                if cancelled:
                    break
                surfaces[name] = arrays
            if not cancelled:
                progress.setValue(total)
        finally:
            progress.close()
            progress.deleteLater()
            self._calculating = False
            self.controls.setEnabled(True)
            self.plot_panel.setEnabled(True)
            self.calculate_button.setEnabled(True)
            ready = self.result_surfaces is not None and not self._inputs_dirty
            self.export_excel_button.setEnabled(ready)
            self.export_plot_button.setEnabled(ready and self.colorbar is not None)
        if cancelled:
            self.status_label.setText("Calculation cancelled. Previous results retained.")
            return
        self.X, self.Y, self.result_surfaces = X, Y, surfaces
        self.last_settings = deepcopy(settings)
        self.last_x_parameter, self.last_y_parameter = xp, yp
        self.last_ranges = {"x": (float(xv[0]), float(xv[-1]), self._float(self.x_step)),
                            "y": (float(yv[0]), float(yv[-1]), self._float(self.y_step))}
        self.last_plot_samples = snapshots
        self.last_plot_errors, self._error_messages = errors, messages
        self._inputs_dirty = False
        self._point_cache_key, self._point_cache = None, {}
        self._configure_point_controls(reset=True)
        self.errors_button.setEnabled(bool(errors))
        self._refresh_result_view()
        self._remember_original_plot()
        self.export_excel_button.setEnabled(True)
        self.export_plot_button.setEnabled(self.colorbar is not None)
        self.status_label.setText(f"{len(surfaces)} sample(s) · {len(xv)} × {len(yv)} grid · {total - len(errors):,}/{total:,} valid points")
        if errors:
            QMessageBox.warning(self, "Some points could not be calculated",
                                f"{len(errors):,} point(s) have no result. See Point errors for details.\n\n{errors[0]['Sample']}: {errors[0]['Error']}")

    def closeEvent(self, event):
        if self._calculating:
            event.ignore()
        else:
            super().closeEvent(event)

    def _refresh_result_view(self, *_):
        if not hasattr(self, "figure"):
            return
        if self.result_surfaces is None:
            self._draw_empty_plot()
            return
        self.last_z_result_id = self.z_selector.currentData()
        info = self.RESULTS[self.last_z_result_id]
        self.last_z_parameter = info["plot"]
        self.Z = {name: arrays[info["key"]] for name, arrays in self.result_surfaces.items()}
        self.plot_title.setText(info["title"])
        if not self._inputs_dirty:
            self.plot_subtitle.setText(f"{self.last_x_parameter} × {self.last_y_parameter} · shared viscosity color scale")
        self.draw_surfaces()
        self._update_point_table()

    def _plot_parameters(self):
        if self.result_surfaces is not None:
            return self.last_x_parameter, self.last_y_parameter, self.last_z_result_id
        return self.x_selector.currentText(), self.y_selector.currentText(), self.z_selector.currentData()

    def apply_safe_plot_layout(self):
        """Keep a compact scale beside the plot and single-line sample names."""
        if self.axes is None:
            return
        margins = self._layout_margins or self.DEFAULT_PLOT_MARGINS
        left, right = margins["left"], margins["right"]
        bottom, top = margins["bottom"], margins["top"]
        width, height = right - left, top - bottom
        # Move the enlarged box slightly upward to leave room for X/Y labels.
        enlargement = max(0.0, (self.plot_zoom - self.PLOT_ZOOM)
                          / (self.MAX_PLOT_ZOOM - self.PLOT_ZOOM))
        self.axes.set_position([left, bottom + height * .04 * enlargement,
                                width * .76, height])
        self.axes.apply_aspect()
        if self.colorbar is None:
            return
        column_width = width * .16
        legend_height = 0
        if self.sample_legend is not None:
            rows = len(self._legend_labels)
            figure_height = max(1.0, self.figure.get_figheight() * 72)
            available = height * .52
            font_size = min(9.0, max(5.0, (available * figure_height - 24) / max(1, 1.45 * rows)))
            for text in self.sample_legend.get_texts():
                text.set_fontsize(font_size)
            renderer = self.canvas.get_renderer()
            for _ in range(3):
                legend_box = self.sample_legend.get_window_extent(renderer).transformed(
                    self.figure.transFigure.inverted())
                if legend_box.width <= width * .34 or font_size <= 5:
                    break
                font_size = max(5.0, font_size * width * .34 / legend_box.width)
                for text in self.sample_legend.get_texts():
                    text.set_fontsize(font_size)
            legend_box = self.sample_legend.get_window_extent(renderer).transformed(
                self.figure.transFigure.inverted())
            column_width = max(column_width, legend_box.width + width * .015)
            legend_height = max(height * .16, legend_box.height + height * .015)
        # Fit the projected box, allowing for labels whose size is in points.
        # This also keeps maximum zoom usable when the window is made smaller.
        figure_width, figure_height = self.figure.bbox.size
        pixels_per_point = self.figure.dpi / 72
        label_scale = self._display["font_size"] / 11
        compact_padding = max(0.0, 4.65 * self.figure.dpi - figure_height) * .12
        legend_gap = 24 * pixels_per_point + 3 * compact_padding
        corners = np.array(np.meshgrid(self.axes.get_xlim(), self.axes.get_ylim(),
                                       self.axes.get_zlim())).reshape(3, -1)
        projected = np.array(proj3d.proj_transform(*corners, self.axes.get_proj()))[:2].T
        box = self.axes.transData.transform(projected)
        box_low, box_high = box.min(axis=0), box.max(axis=0)
        box_size = box_high - box_low
        lower = np.array([left * figure_width + 12 * pixels_per_point,
                          max(bottom * figure_height,
                              60 * pixels_per_point * label_scale + compact_padding)])
        upper = np.array([(right - column_width) * figure_width - legend_gap,
                          top * figure_height - 3 * pixels_per_point])
        available_size = np.maximum(upper - lower, 1)
        fit = min(1.0, float(np.min(available_size / np.maximum(box_size, 1))))
        fitted_size = box_size * fit
        center_box = np.clip((box_low + box_high) / 2,
                             lower + fitted_size / 2, upper - fitted_size / 2)
        position = self.axes.get_position().transformed(self.figure.transFigure)
        origin = center_box - ((box_low + box_high) / 2 - position.p0) * fit
        self.axes.set_position([origin[0] / figure_width, origin[1] / figure_height,
                                position.width * fit / figure_width,
                                position.height * fit / figure_height])
        plot_right = (center_box[0] + fitted_size[0] / 2) / figure_width
        center = min(right - width * .04,
                     plot_right + 78 * pixels_per_point * label_scale / figure_width)
        # Long names can extend to the right without pushing the scale away.
        column_left = min(right - column_width,
                          max(plot_right + legend_gap / figure_width,
                              center - column_width / 2))
        if self.sample_legend is not None:
            self.legend_axes.set_position([column_left, bottom, column_width, legend_height])
        scale_height = min(height * .52, max(height * .12, height * .88 - legend_height))
        scale_bottom = bottom + legend_height + height * .08
        self.colorbar.ax.set_position([center - width * .006, scale_bottom,
                                      width * .012, scale_height])

    def configure_3d_axes(self, ax):
        self.axes = ax
        self._style_axes(ax)
        xp, yp, result_id = self._plot_parameters()
        info = self.RESULTS[result_id]
        size = self._display["font_size"]
        for direction, parameter in (("x", xp), ("y", yp)):
            options = self._axis_options.get(parameter, {})
            getattr(ax, f"set_{direction}label")(
                options.get("label") or self.LABELS.get(parameter, parameter), fontsize=size, labelpad=10)
            if options.get("limits") is not None:
                getattr(ax, f"set_{direction}lim")(options["limits"])
        result_options = self._result_options.get(result_id, {})
        ax.set_zlabel(result_options.get("label") or info["plot"], fontsize=size, labelpad=12)
        if result_options.get("limits") is not None:
            ax.set_zlim(result_options["limits"])
        ax.set_label(info["title"])
        ax.set_title(self._display["title"], fontsize=size + 1, pad=15)
        ax.tick_params(labelsize=max(7, size - 2))
        ax.set_proj_type(self._display["projection"])
        ax.view_init(elev=self._view[0], azim=self._view[1])
        ax.set_box_aspect(self.PLOT_BOX_ASPECT, zoom=self.plot_zoom)
        ax.grid(self._display["grid"])

    def _draw_empty_plot(self):
        self.draw_surfaces()

    def draw_surfaces(self, capture_view=True):
        if capture_view and self.axes is not None:
            self._view = (float(self.axes.elev), float(self.axes.azim))
        self.figure.clear()
        self.colorbar = self.color_norm = None
        self.legend_axes = self.sample_legend = None
        self._legend_labels = []
        ax = self.figure.add_subplot(111, projection="3d")
        self.axes = ax
        valid = {name: z for name, z in (self.Z or {}).items() if np.any(np.isfinite(z))}
        if not valid:
            self.configure_3d_axes(ax)
            message = ("No valid points — open Point errors for details" if self.result_surfaces is not None
                       else "Select samples and calculate the plot")
            ax.text2D(0.5, 0.5, message, transform=ax.transAxes,
                      ha="center", va="center", color="#83929c", wrap=True)
        else:
            all_z = np.concatenate([z[np.isfinite(z)] for z in valid.values()])
            result_options = self._result_options.get(self.last_z_result_id, {})
            low, high = result_options.get("clim") or (float(all_z.min()), float(all_z.max()))
            if low == high:
                margin = max(abs(low) * 0.01, 0.1)
                low, high = low - margin, high + margin
            norm = Normalize(low, high)
            self.color_norm = norm
            cmap = colormaps[self._display["cmap"]]
            handles = []
            for name, z in valid.items():
                style = self._sample_styles[name]
                finite = np.isfinite(z)
                covered = np.zeros(z.shape, dtype=bool)
                if min(z.shape) >= 2:
                    # Only complete valid cells form surface faces: invalid
                    # calculations leave gaps rather than joined triangles.
                    cells = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, 1:] & finite[1:, :-1]
                    rows, cols = np.nonzero(cells)
                    if len(rows):
                        vertices = np.stack([
                            np.column_stack((self.X[rows + dy, cols + dx],
                                             self.Y[rows + dy, cols + dx],
                                             z[rows + dy, cols + dx]))
                            for dy, dx in ((0, 0), (0, 1), (1, 1), (1, 0))
                        ], axis=1)
                        surface = Poly3DCollection(vertices, cmap=cmap, norm=norm,
                                                   linewidths=0, antialiased=False,
                                                   alpha=style["alpha"])
                        surface.set_array(vertices[:, :, 2].mean(axis=1))
                        ax.add_collection3d(surface)
                        for dy, dx in ((0, 0), (0, 1), (1, 1), (1, 0)):
                            covered[rows + dy, cols + dx] = True
                    if style["mesh"]:
                        ax.plot_wireframe(self.X, self.Y, z,
                                          rstride=max(1, z.shape[0] // 18),
                                          cstride=max(1, z.shape[1] // 18),
                                          color=style["color"], linewidth=style["linewidth"],
                                          linestyle=style["linestyle"], alpha=0.95)
                elif z.size > 1:
                    # A single row/column is a curve, still colored by the
                    # same viscosity scale at its calculated grid points.
                    ax.plot(self.X.ravel(), self.Y.ravel(), z.ravel(),
                            color=style["color"], linewidth=style["linewidth"],
                            linestyle=style["linestyle"])
                isolated = finite & ~covered
                if np.any(isolated):
                    ax.scatter(self.X[isolated], self.Y[isolated], z[isolated],
                               c=z[isolated], cmap=cmap, norm=norm, s=22,
                               edgecolors=style["color"], linewidths=0.7, depthshade=False)
                handles.append(Line2D([], [], color=style["color"], linestyle=style["linestyle"],
                                      linewidth=max(1.5, style["linewidth"]), label=style["label"]))
            ax.auto_scale_xyz(self.X.ravel(), self.Y.ravel(), all_z)
            self.configure_3d_axes(ax)
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(all_z)
            cax = self.figure.add_axes([0.90, 0.23, 0.018, 0.57], label="Viscosity color scale")
            extension = ("both" if all_z.min() < low and all_z.max() > high else
                         "min" if all_z.min() < low else "max" if all_z.max() > high else "neither")
            self.colorbar = self.figure.colorbar(mappable, cax=cax, extend=extension)
            self.colorbar.ax.tick_params(labelsize=8, pad=3)
            if self._display["legend"]:
                self._legend_labels = [" ".join(handle.get_label().splitlines())
                                       for handle in handles]
                self.legend_axes = self.figure.add_axes([.73, .10, .24, .23], label="Sample legend")
                self.legend_axes.set_axis_off()
                self.sample_legend = self.legend_axes.legend(
                    handles, self._legend_labels, title="", loc="upper center",
                    bbox_to_anchor=(.5, 1), borderaxespad=0, ncol=1,
                    frameon=True, fancybox=True, facecolor="white", edgecolor="#cbd4dc",
                    framealpha=1, fontsize=9, title_fontsize=9, borderpad=.8,
                    columnspacing=.8, handlelength=1.8, handletextpad=.6)
        self.apply_safe_plot_layout()
        self.toolbar.update()
        self.canvas.draw_idle()

    def _configure_point_controls(self, reset=False):
        if self.result_surfaces is None:
            return
        for axis, parameter in (("x", self.last_x_parameter), ("y", self.last_y_parameter)):
            spin = getattr(self, f"{axis}_point_selector")
            minimum, maximum = self._parameter_limit(parameter) or (None, None)
            spin.blockSignals(True)
            try:
                spin.setRange(minimum if minimum is not None else -1e100,
                              maximum if maximum is not None else 1e100)
                spin.setSingleStep(self.last_ranges[axis][2])
                if reset:
                    spin.setValue(self.last_ranges[axis][0])
            finally:
                spin.blockSignals(False)
            spin.setEnabled(not self._inputs_dirty)
            spin.setToolTip(f"Enter {parameter} and press Enter to calculate Z with the saved models and sample inputs.")
            unit = self.PARAMETER_UNITS[parameter]
            getattr(self, f"{axis}_point_unit").setText(unit)
            for button, verb in zip(self.point_step_buttons[axis], ("Decrease", "Increase")):
                button.setToolTip(f"{verb} {axis.upper()} by {spin.singleStep():g} {unit}")
        self._update_point_buttons()

    def _update_point_buttons(self):
        enabled = self.result_surfaces is not None and not self._inputs_dirty and not self._calculating
        for axis in ("x", "y"):
            spin = getattr(self, f"{axis}_point_selector")
            decrease, increase = self.point_step_buttons[axis]
            decrease.setEnabled(enabled and spin.value() > spin.minimum())
            increase.setEnabled(enabled and spin.value() < spin.maximum())

    def _apply_query_parameter(self, sample, settings, parameter, value):
        if parameter in self.SAMPLE_ATTRIBUTES:
            setattr(sample, self.SAMPLE_ATTRIBUTES[parameter], value)
            return
        if parameter != "γ̇ (strain rate)":
            raise ValueError(f"Unknown physical parameter: {parameter}.")
        model = self.viscosity_engine.crystal_manager.get_model(settings["crystal_model"])
        parameters = settings["crystal_parameters"]
        name = "strain_rate" if "strain_rate" in parameters else None
        if name is None and "gamma" in parameters:
            label = str(getattr(model, "parameters", {}).get("gamma", {}).get("label", "")).lower()
            if any(text in label for text in ("strain", "γ̇", "s⁻¹", "s^-1")):
                name = "gamma"
        if name is None:
            raise ValueError("The selected crystal model does not use strain rate.")
        minimum, maximum = getattr(model, "model_parameter_limits", {}).get(name, (None, None))
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            raise ValueError("Strain rate is outside the selected crystal model limits.")
        parameters[name] = value

    def _query_point(self, x, y):
        """Calculate Z at the entered coordinates, without grid interpolation."""
        if self._point_cache_key == (x, y):
            return self._point_cache
        columns = np.flatnonzero(self.X[0, :] == x)
        rows = np.flatnonzero(self.Y[:, 0] == y)
        answers = {}
        if len(rows) and len(columns):
            row, column = int(rows[0]), int(columns[0])
            for name, arrays in self.result_surfaces.items():
                result = {key: float(values[row, column]) for key, values in arrays.items()}
                answers[name] = result, self._error_messages.get((name, row, column), "")
        else:
            for name, source in self.last_plot_samples.items():
                try:
                    sample, settings = deepcopy(source), deepcopy(self.last_settings)
                    self._apply_query_parameter(sample, settings, self.last_x_parameter, x)
                    self._apply_query_parameter(sample, settings, self.last_y_parameter, y)
                    self._validate_point(sample, settings)
                    result = self.viscosity_engine.calculate(sample=sample, **settings)
                    values = {key: float(result[key]) for key in self.PLOT_RESULT_KEYS}
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError("The model returned a non-finite viscosity.")
                    answers[name] = values, ""
                except Exception as error:
                    answers[name] = {key: math.nan for key in self.PLOT_RESULT_KEYS}, str(error)
        self._point_cache_key, self._point_cache = (x, y), answers
        return answers

    def _update_point_table(self, *_):
        if self.Z is None:
            return
        self._update_point_buttons()
        x, y = self.x_point_selector.value(), self.y_point_selector.value()
        info = self.RESULTS[self.last_z_result_id]
        self.point_label.setText("Settings changed — calculate again to query X and Y." if self._inputs_dirty
                                 else f"Z: {info['combo']}")
        answers = {} if self._inputs_dirty else self._query_point(x, y)
        self.results_table.setRowCount(len(self.Z))
        for i, (name, grid) in enumerate(self.Z.items()):
            result, message = answers.get(name, ({info["key"]: math.nan}, "Settings changed: calculate again."))
            z = result[info["key"]]
            values = (self._sample_styles[name]["label"], f"{z:.6f}" if math.isfinite(z) else "—",
                      f"{np.count_nonzero(np.isfinite(grid)):,} / {grid.size:,}")
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                if j == 0:
                    item.setForeground(QColor(self._sample_styles[name]["color"]))
                    item.setToolTip(name)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setToolTip(message or f"X = {x:.12g}; Y = {y:.12g}")
                self.results_table.setItem(i, j, item)

    def _remember_original_plot(self):
        self._original_plot_configuration = deepcopy({
            "axes": self._axis_options, "results": self._result_options,
            "display": self._display, "styles": self._sample_styles,
            "margins": self._layout_margins, "zoom": self.plot_zoom,
            "view": (float(self.axes.elev), float(self.axes.azim)),
        })

    def restore_original_plot(self):
        if self._original_plot_configuration is None or self._calculating:
            return
        original = deepcopy(self._original_plot_configuration)
        self._axis_options, self._result_options = original["axes"], original["results"]
        self._display, self._sample_styles = original["display"], original["styles"]
        self._layout_margins = original["margins"]
        self.plot_zoom, self._view = original["zoom"], original["view"]
        self.draw_surfaces(capture_view=False)
        self._update_point_table()
        self.canvas.draw()
        self.toolbar.push_current()

    def show_plot_errors(self):
        if not self.last_plot_errors:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("3D Plot — point errors")
        dialog.resize(850, 480)
        layout = QVBoxLayout(dialog)
        limit = 1000
        layout.addWidget(self._note(f"{len(self.last_plot_errors):,} failed points. Showing the first {min(limit, len(self.last_plot_errors)):,}. All errors are included in Excel."))
        columns = list(self.last_plot_errors[0])
        table = QTableWidget(min(limit, len(self.last_plot_errors)), len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, record in enumerate(self.last_plot_errors[:limit]):
            for column, key in enumerate(columns):
                table.setItem(row, column, QTableWidgetItem(str(record[key])))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def _make_style_dialog(self, name):
        style = deepcopy(self._sample_styles[name])
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{name} — surface style")
        dialog.setWindowIcon(self.windowIcon())
        dialog.resize(470, 370)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note("Sample color applies to the mesh and legend. Surface colors represent viscosity on the shared color scale."))
        form = QFormLayout()
        label = QLineEdit(style["label"])
        color = QPushButton(style["color"])
        color.setStyleSheet(f"color:{style['color']};font-weight:bold;")
        def pick_color():
            value = QColorDialog.getColor(QColor(style["color"]), dialog, "Sample mesh color")
            if value.isValid():
                style["color"] = value.name()
                color.setText(value.name())
                color.setStyleSheet(f"color:{value.name()};font-weight:bold;")
        color.clicked.connect(pick_color)
        alpha = QDoubleSpinBox()
        alpha.setRange(0.10, 1.0)
        alpha.setSingleStep(0.05)
        alpha.setValue(style["alpha"])
        mesh = QCheckBox("Show sample mesh")
        mesh.setChecked(style["mesh"])
        width = QDoubleSpinBox()
        width.setRange(0.1, 4.0)
        width.setSingleStep(0.1)
        width.setValue(style["linewidth"])
        linestyle = QComboBox()
        for text, value in (("Solid", "-"), ("Dashed", "--"), ("Dotted", ":"), ("Dash-dot", "-.")):
            linestyle.addItem(text, value)
        linestyle.setCurrentIndex(linestyle.findData(style["linestyle"]))
        for title, widget in (("Legend name", label), ("Sample color", color), ("Surface opacity", alpha),
                              ("Mesh", mesh), ("Line width", width), ("Line style", linestyle)):
            form.addRow(title, widget)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            style.update(label=label.text().strip() or name, alpha=alpha.value(), mesh=mesh.isChecked(),
                         linewidth=width.value(), linestyle=linestyle.currentData())
            self._sample_styles[name] = style
            self.draw_surfaces()
            self._update_point_table()
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.label_edit, dialog.opacity_edit = label, alpha
        dialog.mesh_check, dialog.button_box = mesh, buttons
        return dialog

    def edit_sample_style(self, name):
        dialog = self._make_style_dialog(name)
        dialog.exec()
        dialog.deleteLater()

    def _make_axes_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit 3D axes and surfaces")
        dialog.setWindowIcon(self.windowIcon())
        dialog.resize(620, 680)
        outer = QVBoxLayout(dialog)
        xp, yp, result_id = self._plot_parameters()
        info = self.RESULTS[result_id]
        outer.addWidget(QLabel(info["title"], objectName="plotTitle"))
        tabs = QTabWidget()
        outer.addWidget(tabs, 1)
        axes_page = QWidget()
        layout = QVBoxLayout(axes_page)
        layout.addWidget(self._note("Leave labels empty to keep the standard scientific notation and subscripts. Limits refer to the displayed plot."))
        fields = {}
        for direction, name, options in (
            ("x", xp, self._axis_options.get(xp, {})),
            ("y", yp, self._axis_options.get(yp, {})),
            ("z", info["title"], self._result_options.get(result_id, {})),
        ):
            box = QGroupBox(f"{direction.upper()} axis — {name}")
            form = QFormLayout(box)
            default_label = self.LABELS.get(name, info["plot"])
            preview = QLabel(info["combo"] if direction == "z" else default_label)
            if name == "γ̇ (strain rate)":
                preview.setText("γ̇ (s<sup>−1</sup>)")
            form.addRow("Default label", preview)
            label = QLineEdit(options.get("label", ""))
            label.setPlaceholderText("Automatic label")
            form.addRow("Custom label", label)
            limits = options.get("limits") or getattr(self.axes, f"get_{direction}lim")()
            auto = QCheckBox("Automatic limits")
            auto.setChecked(options.get("limits") is None)
            form.addRow(auto)
            row = QHBoxLayout()
            lo, hi = QLineEdit(f"{limits[0]:.12g}"), QLineEdit(f"{limits[1]:.12g}")
            row.addWidget(QLabel("Min"))
            row.addWidget(lo)
            row.addWidget(QLabel("Max"))
            row.addWidget(hi)
            for edit in (lo, hi):
                edit.setEnabled(not auto.isChecked())
            auto.toggled.connect(lambda checked, lo=lo, hi=hi: (lo.setEnabled(not checked), hi.setEnabled(not checked)))
            form.addRow(row)
            fields[direction] = {"label": label, "auto": auto, "min": lo, "max": hi}
            layout.addWidget(box)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(axes_page)
        tabs.addTab(scroll, "Axes")
        view_page = QWidget()
        form = QFormLayout(view_page)
        title = QLineEdit(self._display["title"])
        size = QSpinBox()
        size.setRange(8, 24)
        size.setValue(self._display["font_size"])
        elevation, azimuth = QDoubleSpinBox(), QDoubleSpinBox()
        elevation.setRange(-89, 89)
        azimuth.setRange(-360, 360)
        elevation.setValue(self.axes.elev)
        azimuth.setValue(((self.axes.azim + 180) % 360) - 180)
        projection = QComboBox()
        projection.addItem("Perspective", "persp")
        projection.addItem("Orthographic", "ortho")
        projection.setCurrentIndex(projection.findData(self._display["projection"]))
        grid, legend = QCheckBox("Show grid"), QCheckBox("Show sample legend")
        grid.setChecked(self._display["grid"])
        legend.setChecked(self._display["legend"])
        cmap = QComboBox()
        cmap.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm"])
        cmap.setCurrentText(self._display["cmap"])
        color_limits = self._result_options.get(result_id, {}).get("clim")
        automatic = QCheckBox("Automatic viscosity color range")
        automatic.setChecked(color_limits is None)
        current = color_limits or ((self.color_norm.vmin, self.color_norm.vmax) if self.color_norm is not None else (0, 1))
        cmin, cmax = QLineEdit(f"{current[0]:.12g}"), QLineEdit(f"{current[1]:.12g}")
        cmin.setEnabled(not automatic.isChecked())
        cmax.setEnabled(not automatic.isChecked())
        automatic.toggled.connect(lambda checked: (cmin.setEnabled(not checked), cmax.setEnabled(not checked)))
        for text, widget in (("Plot title", title), ("Axis font size", size), ("Elevation (°)", elevation),
                             ("Azimuth (°)", azimuth), ("Projection", projection), ("", grid), ("", legend),
                             ("Color map", cmap), ("", automatic), ("Color minimum", cmin), ("Color maximum", cmax)):
            form.addRow(text, widget)
        form.addRow(self._note("Use Style… beside each sample to edit its mesh color, opacity and legend name. The color range is shared by all samples."))
        tabs.addTab(view_page, "View and surfaces")
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            try:
                options = {}
                for direction, widgets in fields.items():
                    text = widgets["label"].text().strip()
                    if "$" in text:
                        MathTextParser("path").parse(text)
                    limits = None if widgets["auto"].isChecked() else (self._float(widgets["min"]), self._float(widgets["max"]))
                    if limits is not None and limits[0] == limits[1]:
                        raise ValueError(f"{direction.upper()} limits must be different.")
                    options[direction] = {"label": text, "limits": limits}
                clim = None if automatic.isChecked() else (self._float(cmin), self._float(cmax))
                if clim is not None and clim[0] >= clim[1]:
                    raise ValueError("Color minimum must be less than color maximum.")
                if "$" in title.text():
                    MathTextParser("path").parse(title.text())
            except (ValueError, TypeError) as error:
                QMessageBox.warning(dialog, "Invalid axis settings", str(error))
                return
            self._axis_options[xp], self._axis_options[yp] = options["x"], options["y"]
            self._result_options[result_id] = {**options["z"], "clim": clim}
            self._display.update(title=title.text().strip(), font_size=size.value(), grid=grid.isChecked(),
                                 legend=legend.isChecked(), cmap=cmap.currentText(), projection=projection.currentData())
            self._view = (elevation.value(), azimuth.value())
            self.draw_surfaces(capture_view=False)
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)
        dialog.axis_fields, dialog.button_box = fields, buttons
        dialog.cmap_selector, dialog.color_auto = cmap, automatic
        dialog.color_min, dialog.color_max = cmin, cmax
        dialog.elevation_edit, dialog.azimuth_edit = elevation, azimuth
        return dialog

    def edit_axes(self, *_):
        dialog = self._make_axes_dialog()
        dialog.exec()
        dialog.deleteLater()

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
        initial = self._layout_margins or self.DEFAULT_PLOT_MARGINS
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
            self.apply_safe_plot_layout()
            self.canvas.draw_idle()
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.margin_fields, dialog.auto_margins, dialog.button_box = fields, automatic, buttons
        dialog.margin_step_buttons = step_buttons
        return dialog

    def _export_ready(self):
        if self.result_surfaces is None:
            QMessageBox.warning(self, "No results", "Calculate the plot before exporting.")
            return False
        if self._inputs_dirty or self.fixed_parameters_dirty:
            QMessageBox.warning(self, "Settings changed", "Calculate again before exporting the updated settings.")
            return False
        return True

    def _write_excel(self, filename):
        """Export calculation snapshots: one tidy data sheet per sample."""
        used_names = {"parameters", "samples"}
        metadata = [("Displayed viscosity", self.RESULTS[self.last_z_result_id]["title"]),
                    ("X axis", self.EXPORT_AXIS_LABELS[self.last_x_parameter]),
                    ("Y axis", self.EXPORT_AXIS_LABELS[self.last_y_parameter])]
        for axis in ("x", "y"):
            for name, value in zip(("From", "To", "Step"), self.last_ranges[axis]):
                metadata.append((f"{axis.upper()} {name}", value))
        for group in ("melt", "crystal", "vesicle"):
            metadata.append((f"{group.title()} model", self.last_settings[f"{group}_model"]))
            metadata.extend((f"{group.title()} parameter — {name}", value)
                            for name, value in self.last_settings[f"{group}_parameters"].items())
        metadata.append(("Sample inputs", "Samples sheet contains fixed inputs; X/Y columns override the swept values at each grid point."))
        snapshots = []
        oxides = list(getattr(getattr(self.main_window, "composition_panel", None), "OXIDES", ()))
        for name, sample in self.last_plot_samples.items():
            record = {"Sample": name}
            for parameter, attribute in self.SAMPLE_ATTRIBUTES.items():
                record[self.EXPORT_AXIS_LABELS[parameter]] = getattr(sample, attribute)
            attributes = vars(sample) if hasattr(sample, "__dict__") else {}
            for attribute in dict.fromkeys([*oxides, *attributes]):
                if attribute in {"name", *self.SAMPLE_ATTRIBUTES.values(), *self.PLOT_RESULT_KEYS, "eta_r_c", "eta_r_b"}:
                    continue
                value = getattr(sample, attribute, None)
                if value is None or isinstance(value, (str, int, float, bool, np.number)):
                    record[f"{attribute} (wt%)" if attribute in oxides else attribute] = value
            snapshots.append(record)
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for name, arrays in self.result_surfaces.items():
                data = {self.EXPORT_AXIS_LABELS[self.last_x_parameter]: self.X.ravel(),
                        self.EXPORT_AXIS_LABELS[self.last_y_parameter]: self.Y.ravel()}
                for result_id, info in self.RESULTS.items():
                    data[f"log₁₀ η{result_id} (Pa·s)"] = arrays[info["key"]].ravel()
                data["Status"] = ["Error" if (name, row, col) in self._error_messages else "OK" for row, col in np.ndindex(self.X.shape)]
                data["Error"] = [self._error_messages.get((name, row, col), "") for row, col in np.ndindex(self.X.shape)]
                sheet = self.make_sheet_name(name, used_names)
                data = {"Sample": [name] * self.X.size, **data}
                pd.DataFrame(data).to_excel(writer, sheet_name=sheet, index=False)
            pd.DataFrame(metadata, columns=["Parameter", "Value"]).to_excel(writer, sheet_name="Parameters", index=False)
            pd.DataFrame(snapshots).to_excel(writer, sheet_name="Samples", index=False)
            for sheet in writer.book.worksheets:
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for cells in sheet.iter_cols(min_row=1, max_row=1):
                    cell = cells[0]
                    sheet.column_dimensions[cell.column_letter].width = min(40, max(17, len(str(cell.value)) + 3))
        PlotWindow._format_excel_subscripts(filename)

    def export_data(self):
        if not self._export_ready():
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export 3D data", "MVL_3D_plot.xlsx", "Excel Files (*.xlsx)")
        if not filename:
            return
        if not filename.lower().endswith(".xlsx"):
            filename += ".xlsx"
        try:
            self._write_excel(filename)
            QMessageBox.information(self, "Export completed", f"Saved {len(self.result_surfaces)} sample sheet(s), all four viscosities and calculation settings.\n\n{filename}")
        except Exception as error:
            QMessageBox.critical(self, "Export error", str(error))

    def export_plot(self):
        if not self._export_ready():
            return
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export 3D figure", "MVL_3D_plot", "PNG image (*.png);;PDF figure (*.pdf);;SVG figure (*.svg)")
        if not filename:
            return
        extension = {"PNG image (*.png)": ".png", "PDF figure (*.pdf)": ".pdf", "SVG figure (*.svg)": ".svg"}[selected_filter]
        if Path(filename).suffix.lower() not in {".png", ".pdf", ".svg"}:
            filename += extension
        try:
            self.canvas.draw()
            labels = [self.axes.xaxis.label, self.axes.yaxis.label, self.axes.zaxis.label]
            if self.sample_legend is not None:
                labels.append(self.sample_legend)
            self.figure.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.2,
                                bbox_extra_artists=labels, facecolor="white")
            QMessageBox.information(self, "Export completed", f"Figure saved to:\n\n{filename}")
        except Exception as error:
            QMessageBox.critical(self, "Export error", str(error))

    def get_selected_samples(self):
        """Return every sample selected for calculation."""
        return [
            sample
            for checkbox, sample in self.sample_checks
            if checkbox.isChecked()
        ]

    def get_selected_result(self):
        """Return the definition of the selected Z result."""
        return self.RESULTS[
            self.z_selector.currentData()
        ]

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

    def apply_fixed_melt_parameters_to_sample(
        self,
        sample,
    ):
        """Apply physical values imposed by the melt model."""
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
        """Validate every currently displayed model parameter."""
        variables = {
            self.x_selector.currentText(),
            self.y_selector.currentText(),
        }

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
                    and "γ̇ (strain rate)"
                    in variables
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
    def format_limit_text(
        parameter,
        minimum,
        maximum,
        dynamic=False,
    ):
        """Format a physical-limit warning."""
        if (
            (
                dynamic
                or parameter == "Vesicles"
            )
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
        """Validate physical parameters not assigned to X or Y."""
        variables = {
            self.x_selector.currentText(),
            self.y_selector.currentText(),
        }

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
                    parameter in variables
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

                text = self.format_limit_text(
                    parameter,
                    minimum,
                    maximum,
                    dynamic,
                )

                QMessageBox.warning(
                    self,
                    "Outside model limits",
                    f"{model.name}\n\n{text}\n"
                    f"Current fixed value: {value:g}",
                )
                return False

        return True

    def validate_axis_model_limits(
        self,
        parameter,
        start,
        end,
        settings,
    ):
        """Validate one axis range against all selected models."""
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

            text = self.format_limit_text(
                parameter,
                minimum,
                maximum,
                dynamic,
            )

            QMessageBox.warning(
                self,
                "Outside model limits",
                f"{model.name}\n\n{text}",
            )
            return False

        return True

    @staticmethod
    def _parameter_limit(parameter):
        """Return the global limit of a surface parameter."""
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

    def apply_variable_parameter(
        self,
        sample,
        settings,
        parameter,
        value,
    ):
        """Apply one axis value to a sample or model parameter."""
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

    @staticmethod
    def _style_axes(ax):
        """Apply the standard three-dimensional plot style."""
        ax.set_facecolor("#fbfcfd")
        ax.tick_params(colors="#4a535b")

        for axis in (
            ax.xaxis,
            ax.yaxis,
            ax.zaxis,
        ):
            axis.pane.set_facecolor(
                (
                    0.97,
                    0.98,
                    0.99,
                    1.0,
                )
            )
            axis.pane.set_edgecolor(
                "#aeb6bd"
            )
            axis._axinfo["grid"].update(
                color="#d8dde2",
                linestyle="--",
                linewidth=0.7,
            )

    def rotate_with_keyboard(self, event):
        """Rotate the surface using the keyboard arrow keys."""
        if (
            self.axes is None
            or event.key
            not in {
                "left",
                "right",
                "up",
                "down",
            }
        ):
            return

        elevation = float(self.axes.elev)
        azimuth = float(self.axes.azim)

        if event.key == "left":
            azimuth -= self.ROTATION_STEP

        elif event.key == "right":
            azimuth += self.ROTATION_STEP

        elif event.key == "up":
            elevation = min(
                self.MAX_ELEVATION,
                elevation + self.ROTATION_STEP,
            )

        else:
            elevation = max(
                self.MIN_ELEVATION,
                elevation - self.ROTATION_STEP,
            )

        self.axes.view_init(
            elev=elevation,
            azim=azimuth,
        )

        self.apply_safe_plot_layout()
        self.canvas.draw_idle()

    def zoom_with_mouse_wheel(self, event):
        """Zoom the surface using the mouse wheel."""
        if self.axes is None:
            return

        direction = getattr(
            event,
            "step",
            0,
        )

        if event.button == "up" or direction > 0:
            new_zoom = (
                self.plot_zoom
                + self.ZOOM_STEP
            )

        elif (
            event.button == "down"
            or direction < 0
        ):
            new_zoom = (
                self.plot_zoom
                - self.ZOOM_STEP
            )

        else:
            return

        new_zoom = min(
            self.MAX_PLOT_ZOOM,
            max(
                self.MIN_PLOT_ZOOM,
                new_zoom,
            ),
        )

        if new_zoom == self.plot_zoom:
            return

        self.plot_zoom = new_zoom

        self.axes.set_box_aspect(
            self.PLOT_BOX_ASPECT,
            zoom=self.plot_zoom,
        )

        self.apply_safe_plot_layout()
        self.canvas.setFocus()
        self.canvas.draw_idle()

    def clear_fixed_parameters_dirty(self):
        """Clear the unapplied fixed-parameter warning."""
        self.fixed_parameters_dirty = False
        self.apply_button.setText(
            "Apply fixed values to selected samples"
        )
        self.apply_button.setToolTip("")
        self.apply_button.setStyleSheet("")

    @staticmethod
    def make_sheet_name(name, used_names):
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

    @staticmethod
    def _float(widget):
        value = float(widget.text().replace(",", "."))
        if not math.isfinite(value):
            raise ValueError("Values must be finite numbers.")
        return value

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
                values[name] = Plot3DWindow._float(widget)
            except ValueError:
                return None

        return values

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

    def _set_all_samples(self, state):
        for checkbox, _ in self.sample_checks:
            checkbox.blockSignals(True)
            checkbox.setChecked(state == Qt.Checked)
            checkbox.blockSignals(False)
        self.update_fixed_parameters()
        self._inputs_changed()

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

    def edit_model_parameters(self, group):
        try:
            dialog = self._make_model_dialog(group)
            dialog.exec()
            dialog.deleteLater()
        except Exception as error:
            QMessageBox.warning(self, "Model parameters", str(error))

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
        if group == "crystal" and "γ̇ (strain rate)" in {self.x_selector.currentText(), self.y_selector.currentText()}:
            strain_name = self.get_strain_rate_parameter_name(parameters)
            widget = panel.crystal_parameters.get(strain_name)
            if widget is not None:
                label = panel.crystal_parameters_layout.labelForField(widget)
                widget.hide()
                if label is not None:
                    label.hide()
            if name == self.CARICCHI_MODEL:
                panel.crystal_parameters_box.hide()
            layout.addWidget(self._note("Strain rate is supplied by the selected axis range."))
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