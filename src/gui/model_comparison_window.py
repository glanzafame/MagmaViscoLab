"""Compare published MVL models using one common sample and physical sweep
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from html import escape
import math
import os
import re
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QTextDocument
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QAbstractSpinBox, QCheckBox, QColorDialog, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QStyle, QStyledItemDelegate, QStyleOptionComboBox, QTabBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from core.parameter_limits import PARAMETER_LIMITS
from gui.panels.models_parameters_panel import ModelsParametersPanel
from gui.plot_window import PlotWindow, Plot2DToolbar, XValueSpinBox


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


class ModelComparisonWindow(QMainWindow):
    """One model family varies; other required phase models remain fixed."""

    GROUPS = {
        "melt": "Melt viscosity",
        "crystal": "Crystal correction",
        "vesicle": "Vesicle correction",
    }
    AXES = {
        "Temperature": ("temperature", "Temperature (°C)", "°C"),
        "H₂O": ("H2O", "H₂O (wt%)", "wt%"),
        "Crystals": ("crystals", "Crystals (vol%)", "vol%"),
        "Vesicles": ("Vesicles", "Vesicles (vol%)", "vol%"),
        "γ̇ (strain rate)": (None, "γ̇ (s⁻¹)", "s⁻¹"),
    }
    STRAIN = "γ̇ (strain rate)"
    DEFAULT_RANGES = {
        "Temperature": ("900", "1200", "50"),
        "H₂O": ("0", "4", "0.5"),
        "Crystals": ("0", "50", "5"),
        "Vesicles": ("0", "50", "5"),
        "γ̇ (strain rate)": ("0.01", "1", "0.05"),
    }
    OUTPUTS = {
        "log10_eta_m": ("Melt viscosity · log<sub>10</sub> η<sub>m</sub> (Pa·s)",
                        r"$\log_{10}\eta_{\mathrm{m}}\;(\mathrm{Pa\,s})$"),
        "log10_eta_mc": ("Melt + crystals · log<sub>10</sub> η<sub>mc</sub> (Pa·s)",
                         r"$\log_{10}\eta_{\mathrm{mc}}\;(\mathrm{Pa\,s})$"),
        "log10_eta_mb": ("Melt + vesicles · log<sub>10</sub> η<sub>mb</sub> (Pa·s)",
                         r"$\log_{10}\eta_{\mathrm{mb}}\;(\mathrm{Pa\,s})$"),
        "log10_eta_mcb": ("Three phases · log<sub>10</sub> η<sub>mcb</sub> (Pa·s)",
                          r"$\log_{10}\eta_{\mathrm{mcb}}\;(\mathrm{Pa\,s})$"),
        "eta_r_c": ("Crystal factor · η<sub>r,c</sub>",
                    r"$\eta_{\mathrm{r,c}}$"),
        "eta_r_b": ("Vesicle factor · η<sub>r,b</sub>",
                    r"$\eta_{\mathrm{r,b}}$"),
    }
    OUTPUT_TITLES = {
        "log10_eta_m": "Melt viscosity",
        "log10_eta_mc": "Crystal-bearing magma viscosity",
        "log10_eta_mb": "Vesicle-bearing magma viscosity",
        "log10_eta_mcb": "Three-phase magma viscosity",
        "eta_r_c": "Crystal correction factor",
        "eta_r_b": "Vesicle correction factor",
    }
    OUTPUT_SYMBOLS = {
        "log10_eta_m": "m", "log10_eta_mc": "mc", "log10_eta_mb": "mb",
        "log10_eta_mcb": "mcb", "eta_r_c": "r,c", "eta_r_b": "r,b",
    }
    GROUP_OUTPUTS = {
        "melt": ("log10_eta_m", "log10_eta_mc", "log10_eta_mb", "log10_eta_mcb"),
        "crystal": ("eta_r_c", "log10_eta_mc", "log10_eta_mcb"),
        "vesicle": ("eta_r_b", "log10_eta_mb", "log10_eta_mcb"),
    }
    FIXED_NAMES = {
        "temperature": "Temperature", "water": "H₂O",
        "crystals": "Crystals", "vesicles": "Vesicles",
    }
    MAX_POINTS = 20000
    MAX_EVALUATIONS = 200000
    COLORS = ("#007d9d", "#d45a3c", "#7956a3", "#279165", "#c18b16",
              "#4577bd", "#c4578b", "#686b70", "#728835", "#905d41")

    def __init__(self, samples, main_window=None, initial_sample_name=None,
                 initial_settings=None):
        super().__init__()
        self.samples = deepcopy(list(samples))
        if not self.samples:
            raise ValueError("Load at least one sample before comparing models.")
        self.main_window = main_window
        self.viscosity_engine = getattr(main_window, "viscosity_engine", None)
        if not callable(getattr(self.viscosity_engine, "compare_models", None)):
            raise ValueError("Install the updated core/viscosity_engine.py first.")
        self.oxide_names = tuple(getattr(
            getattr(main_window, "composition_panel", None), "OXIDES", (
                "SiO2", "TiO2", "Al2O3", "FeO", "Fe2O3", "MnO", "MgO",
                "CaO", "Na2O", "K2O", "P2O5", "H2O", "F2O_1",
            )
        ))
        self._updating = True
        self._busy = False
        self._dirty = True
        self._group = "melt"
        self._run = None
        self._axis = None
        self._locked = {}
        self._unlocked_text = {}
        self._parameters = {group: {} for group in self.GROUPS}
        self._checked = {}
        self._fixed_names = {}
        self._point_cache = None
        self._axis_options = {"x": {}, "y": {}, "difference": {}}
        self._plot_options = {"title": None, "grid": True, "legend": True}
        self._curve_styles = {group: {} for group in self.GROUPS}
        self._layout_margins = None
        self._original_plot_configuration = None
        self._views = {
            group: {"axis": axis, "output": self.GROUP_OUTPUTS[group][0]}
            for group, axis in zip(self.GROUPS, ("Temperature", "Crystals", "Vesicles"))
        }
        self._ranges = {group: deepcopy(self.DEFAULT_RANGES) for group in self.GROUPS}
        settings = deepcopy(initial_settings or {})
        for group in self.GROUPS:
            names = self.viscosity_engine.get_model_names(group)
            for name in names:
                self._parameters[group][name] = self._model_defaults(group, name)
            initial = settings.get(f"{group}_model")
            if initial not in names:
                initial = next(iter(names), "")
            self._fixed_names[group] = initial
            selected = [initial] if initial else []
            selected.extend(name for name in names if name != initial)
            self._checked[group] = set(selected[:3])
            if initial:
                self._parameters[group][initial].update(
                    settings.get(f"{group}_parameters") or {}
                )

        self.setWindowTitle("MagmaViscoLab – Compare models")
        self.setWindowIcon(PlotWindow._mvl_icon(self))
        self.resize(1500, 920)
        self.setMinimumSize(1120, 740)
        self._build_interface()
        self._apply_style()
        names = [str(sample.name) for sample in self.samples]
        if initial_sample_name is not None and str(initial_sample_name) in names:
            self.sample_selector.setCurrentIndex(names.index(str(initial_sample_name)))
        self._load_sample_values()
        self._load_group()
        self._updating = False
        self._sync_controls()
        self._draw_empty_plot()

    # --- Interface --------------------------------------------------

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
        label = QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("subtleText")
        return label

    def _build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 12, 16, 12)
        title = QLabel("Compare models")
        title.setObjectName("windowTitle")
        outer.addWidget(title)
        outer.addWidget(self._note(
            "Compare one model family using the same sample and physical conditions."
        ))
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        for label in self.GROUPS.values():
            self.tabs.addTab(label)
        self.tabs.currentChanged.connect(self._change_group)
        outer.addWidget(self.tabs)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)
        self.control_panel = QWidget()
        self.control_panel.setObjectName("controlPanel")
        controls = QVBoxLayout(self.control_panel)
        controls.setContentsMargins(10, 8, 10, 10)
        controls.setSpacing(10)

        sample_box = QGroupBox("Sample")
        row = QHBoxLayout(sample_box)
        self.sample_selector = QComboBox()
        self.sample_selector.addItems([str(sample.name) for sample in self.samples])
        self.sample_selector.currentIndexChanged.connect(self._change_sample)
        row.addWidget(self.sample_selector, 1)
        row.addWidget(self._button("Composition…", self.show_composition, "smallButton"))
        controls.addWidget(sample_box)

        self.models_box = QGroupBox("Models to compare")
        models_layout = QVBoxLayout(self.models_box)
        tools_row = QHBoxLayout()
        tools_row.addWidget(self._button("Select all", lambda: self._select_all(True), "smallButton"))
        tools_row.addWidget(self._button("Clear", lambda: self._select_all(False), "smallButton"))
        tools_row.addStretch()
        models_layout.addLayout(tools_row)
        models_scroll = QScrollArea()
        models_scroll.setWidgetResizable(True)
        models_scroll.setMinimumHeight(145)
        models_scroll.setMaximumHeight(230)
        self.models_container = QWidget()
        self.models_layout = QVBoxLayout(self.models_container)
        self.models_layout.setContentsMargins(0, 0, 5, 0)
        self.models_layout.setSpacing(3)
        models_scroll.setWidget(self.models_container)
        models_layout.addWidget(models_scroll)
        controls.addWidget(self.models_box)

        axes_box = QGroupBox("Plot axes and range")
        axes_layout = QVBoxLayout(axes_box)
        form = QFormLayout()
        self.x_selector = QComboBox()
        self.y_selector = RichTextComboBox()
        self.x_selector.currentIndexChanged.connect(self._change_axis)
        self.y_selector.currentIndexChanged.connect(self._change_output)
        form.addRow("X axis", self.x_selector)
        form.addRow("Y axis", self.y_selector)
        axes_layout.addLayout(form)
        range_row = QHBoxLayout()
        self.from_edit, self.to_edit, self.step_edit = (QLineEdit() for _ in range(3))
        for label, edit in zip(("From", "To", "Step"), self.range_edits):
            column = QVBoxLayout()
            column.setSpacing(3)
            column.addWidget(QLabel(label))
            edit.setMinimumWidth(55)
            edit.textChanged.connect(self._range_changed)
            column.addWidget(edit)
            range_row.addLayout(column, 1)
        axes_layout.addLayout(range_row)
        self.range_note = self._note("")
        axes_layout.addWidget(self.range_note)
        controls.addWidget(axes_box)

        shared_box = QGroupBox("Shared physical conditions")
        fixed_form = QFormLayout(shared_box)
        self.physical_edits = {}
        for parameter, (_attribute, label, _unit) in self.AXES.items():
            edit = QLineEdit()
            edit.textChanged.connect(self._mark_dirty)
            self.physical_edits[parameter] = edit
            fixed_form.addRow(label, edit)
        self.conditions_note = self._note("")
        fixed_form.addRow(self.conditions_note)
        controls.addWidget(shared_box)

        self.support_box = QGroupBox("Other phase models")
        support_layout = QVBoxLayout(self.support_box)
        self.support_rows = {}
        self.support_selectors = {}
        for group, label in self.GROUPS.items():
            section = QWidget()
            layout = QVBoxLayout(section)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(3)
            layout.addWidget(QLabel(label))
            row = QHBoxLayout()
            selector = QComboBox()
            selector.addItems(self.viscosity_engine.get_model_names(group))
            selector.setCurrentText(self._fixed_names[group])
            selector.currentTextChanged.connect(
                lambda name, group=group: self._change_support(group, name)
            )
            row.addWidget(selector, 1)
            row.addWidget(self._button(
                "Parameters…", lambda _=False, group=group: self.edit_parameters(
                    group, self._fixed_names[group]
                ), "smallButton"
            ))
            layout.addLayout(row)
            self.support_rows[group] = section
            self.support_selectors[group] = selector
            support_layout.addWidget(section)
        controls.addWidget(self.support_box)
        controls.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(520)
        scroll.setWidget(self.control_panel)
        splitter.addWidget(scroll)
        right = QWidget()
        results = QVBoxLayout(right)
        results.setContentsMargins(14, 0, 0, 0)
        self.plot_title = QLabel("Model comparison")
        self.plot_title.setObjectName("plotTitle")
        results.addWidget(self.plot_title)
        self.plot_subtitle = self._note("Select models and click Calculate comparison.")
        results.addWidget(self.plot_subtitle)
        self.figure = Figure(figsize=(8, 6), facecolor="white", layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(350)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = Plot2DToolbar(self.canvas, right, self)
        results.addWidget(self.toolbar)
        results.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._plot_clicked)

        point_row = QHBoxLayout()
        point_row.addWidget(QLabel("Value at X point"))
        self.point_selector = XValueSpinBox()
        self.point_selector.setEnabled(False)
        self.point_selector.valueChanged.connect(self._update_table)
        self.point_selector.editingFinished.connect(self._update_table)
        decrease = self._button("−", self.point_selector.stepDown, "smallButton")
        increase = self._button("+", self.point_selector.stepUp, "smallButton")
        self.point_step_buttons = (decrease, increase)
        for button in self.point_step_buttons:
            button.setFixedWidth(32)
            button.setEnabled(False)
        point_row.addWidget(decrease)
        point_row.addWidget(self.point_selector)
        point_row.addWidget(increase)
        self.point_label = QLabel("—")
        point_row.addWidget(self.point_label)
        point_row.addStretch()
        self.errors_button = self._button("Point errors…", self.show_errors)
        self.errors_button.setEnabled(False)
        point_row.addWidget(self.errors_button)
        results.addLayout(point_row)
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Difference relative to"))
        self.reference_selector = QComboBox()
        self.reference_selector.setMinimumWidth(220)
        self.reference_selector.setEnabled(False)
        self.reference_selector.currentTextChanged.connect(self._reference_changed)
        ref_row.addWidget(self.reference_selector)
        ref_row.addStretch()
        results.addLayout(ref_row)
        self.results_table = self._table(("Model", "Value", "Difference", "Valid points"))
        self.results_table.setMaximumHeight(175)
        self.results_table.setMinimumHeight(105)
        results.addWidget(self.results_table)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([445, 1020])

        footer = QHBoxLayout()
        self.status_label = self._note("Ready")
        footer.addWidget(self.status_label, 1)
        self.export_excel_button = self._button("Export Excel", self.export_excel)
        self.export_plot_button = self._button("Export figure", self.export_plot)
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        footer.addWidget(self.export_excel_button)
        footer.addWidget(self.export_plot_button)
        self.calculate_button = self._button(
            "Calculate comparison", self.calculate_comparison, "calculateButton"
        )
        self.calculate_button.setMinimumWidth(210)
        footer.addWidget(self.calculate_button)
        outer.addLayout(footer)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f3f5f7; color: #26313a; }
            #controlPanel { background: #eef1f4; }
            #windowTitle { color: #a63b2a; font-size: 18pt; font-weight: 700; }
            #plotTitle { color: #26313a; font-size: 14pt; font-weight: 600; }
            #subtleText { color: #64717b; font-size: 9pt; }
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
            QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background: #eef0f2; color: #78818a; }
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
        """)

    @property
    def range_edits(self):
        return self.from_edit, self.to_edit, self.step_edit

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        return table

    # --- Model selection and local settings --------------------------

    def _model(self, group, name):
        return getattr(self.viscosity_engine, f"{group}_manager").get_model(name)

    def _model_defaults(self, group, name):
        values = {}
        for key, info in (getattr(self._model(group, name), "parameters", {}) or {}).items():
            value = info.get("default", 0.0)
            if info.get("type") == "choice":
                options = info.get("options", [])
                if value not in options and options:
                    value = options[0]
            else:
                value = self._number(value, f"{name}: {key}")
            values[key] = deepcopy(value)
        return values

    def selected_models(self):
        return [name for name in self.viscosity_engine.get_model_names(self._group)
                if name in self._checked[self._group]]

    def _load_group(self):
        self.y_selector.clear()
        for key in self.GROUP_OUTPUTS[self._group]:
            self.y_selector.addItem(self.OUTPUTS[key][0], key)
        self.y_selector.setCurrentIndex(
            self.y_selector.findData(self._views[self._group]["output"])
        )
        self._rebuild_models()
        self._axis = None

    def _rebuild_models(self):
        while self.models_layout.count():
            item = self.models_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self.model_checkboxes = {}
        for name in self.viscosity_engine.get_model_names(self._group):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            checkbox = QCheckBox(name)
            checkbox.setChecked(name in self._checked[self._group])
            checkbox.toggled.connect(lambda checked, name=name: self._toggle_model(name, checked))
            row.addWidget(checkbox, 1)
            row.addWidget(self._button(
                "Parameters…", lambda _=False, name=name: self.edit_parameters(self._group, name),
                "smallButton",
            ))
            self.model_checkboxes[name] = checkbox
            self.models_layout.addWidget(row_widget)
        self.models_layout.addStretch()

    def _change_group(self, index):
        if self._updating or self._busy:
            return
        self._updating = True
        self._save_range()
        self._group = tuple(self.GROUPS)[index]
        self._load_group()
        self._updating = False
        self._run = None
        self._reset_result_controls()
        self._sync_controls()
        self._draw_empty_plot()
        self._mark_dirty()

    def _toggle_model(self, name, checked):
        if self._updating:
            return
        if checked:
            self._checked[self._group].add(name)
        else:
            self._checked[self._group].discard(name)
        self._sync_controls()
        self._mark_dirty()

    def _select_all(self, checked):
        self._updating = True
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(checked)
        self._checked[self._group] = set(self.model_checkboxes) if checked else set()
        self._updating = False
        self._sync_controls()
        self._mark_dirty()

    def _change_support(self, group, name):
        if self._updating:
            return
        self._fixed_names[group] = name
        self._sync_controls()
        self._mark_dirty()

    def _change_output(self, *_):
        if self._updating:
            return
        self._views[self._group]["output"] = self.y_selector.currentData()
        self._sync_controls()
        self._mark_dirty()

    def _save_range(self):
        if self._axis is not None:
            self._ranges[self._group][self._axis] = tuple(edit.text() for edit in self.range_edits)

    def _range_changed(self, *_):
        if not self._updating:
            self._save_range()
            self._mark_dirty()

    def _change_axis(self, *_):
        if self._updating:
            return
        self._save_range()
        self._axis = self.x_selector.currentText()
        self._views[self._group]["axis"] = self._axis
        self._updating = True
        self._load_range()
        self._updating = False
        self._sync_controls()
        self._mark_dirty()

    def _load_range(self):
        for edit, value in zip(self.range_edits, self._ranges[self._group][self._axis]):
            edit.setText(value)
        self.range_note.setText(
            f"Units: {self.AXES[self._axis][2]}. Step is positive; ascending and descending ranges are supported."
        )

    def _change_sample(self, *_):
        if self._updating:
            return
        self._updating = True
        self._locked.clear()
        self._unlocked_text.clear()
        self._load_sample_values()
        self._updating = False
        self._sync_controls()
        self._mark_dirty()

    def _load_sample_values(self):
        sample = self.samples[self.sample_selector.currentIndex()]
        for label, (attribute, _display, _unit) in self.AXES.items():
            if attribute:
                self.physical_edits[label].setText(str(getattr(sample, attribute, 0.0)))
        name = self._fixed_names["crystal"]
        rate_key = self._strain_key(name) if name else None
        value = self._parameters["crystal"].get(name, {}).get(rate_key, 0.1)
        self.physical_edits[self.STRAIN].setText(str(value))

    def _strain_key(self, model_name):
        definitions = getattr(self._model("crystal", model_name), "parameters", {}) or {}
        if "strain_rate" in definitions:
            return "strain_rate"
        if "gamma" in definitions:
            label = str(definitions["gamma"].get("label", "")).lower()
            if any(token in label for token in ("strain", "γ̇", "s⁻¹", "s^-1")):
                return "gamma"
        return None

    def _required_phases(self):
        return self.viscosity_engine.OUTPUT_PHASES[self.y_selector.currentData()]

    def _phase_names(self, group):
        if group == self._group:
            return self.selected_models()
        return [self._fixed_names[group]] if self._fixed_names[group] else []

    def _sync_controls(self):
        previous_guard = self._updating
        self._updating = True
        phases = self._required_phases()
        used = set()
        melt_names = self._phase_names("melt") if "melt" in phases else []
        fixed_definitions = []
        for name in melt_names:
            model = self._model("melt", name)
            physical = getattr(model, "required_melt_physical_parameters", ("temperature", "water")) or ()
            used.update(self.FIXED_NAMES[p] for p in physical if p in self.FIXED_NAMES)
            fixed = getattr(model, "fixed_melt_physical_parameters", {}) or {}
            fixed_definitions.append(fixed)
            used.update(self.FIXED_NAMES[p] for p in fixed if p in self.FIXED_NAMES)
        if "crystal" in phases:
            used.add("Crystals")
            if any(self._strain_key(name) for name in self._phase_names("crystal")):
                used.add(self.STRAIN)
        if "vesicle" in phases:
            used.add("Vesicles")
        locked = {}
        if fixed_definitions:
            for key, label in self.FIXED_NAMES.items():
                values = [definition.get(key) for definition in fixed_definitions]
                if all(value is not None for value in values) and len(set(values)) == 1:
                    locked[label] = float(values[0])
        for label in self._locked:
            if label not in locked:
                self.physical_edits[label].setText(self._unlocked_text.pop(label, "0"))
        for label, value in locked.items():
            if label not in self._locked:
                self._unlocked_text[label] = self.physical_edits[label].text()
            self.physical_edits[label].setText(f"{value:g}")
        self._locked = locked
        self._used = used

        available_axes = [label for label in self.AXES if label in used and label not in locked]
        if not available_axes:
            available_axes = [{"melt": "Temperature", "crystal": "Crystals", "vesicle": "Vesicles"}[self._group]]
        desired = self._views[self._group]["axis"]
        if desired not in available_axes:
            desired = available_axes[0]
        self._save_range()
        self.x_selector.clear()
        self.x_selector.addItems(available_axes)
        self.x_selector.setCurrentText(desired)
        self._axis = desired
        self._views[self._group]["axis"] = desired
        self._load_range()
        for label, edit in self.physical_edits.items():
            enabled = label in used and label != self._axis and label not in locked
            edit.setEnabled(enabled)
            if label in locked:
                tip = "Fixed by every melt model used in this comparison."
            elif label == self._axis:
                tip = "Values are supplied by From, To and Step."
            elif label not in used:
                tip = "Not required by the selected output and models."
            else:
                tip = "This value is shared by all compared models."
            edit.setToolTip(tip)
        support_count = 0
        for group, section in self.support_rows.items():
            visible = group in phases and group != self._group
            section.setVisible(visible)
            support_count += int(visible)
        self.support_box.setVisible(bool(support_count))
        constraints = []
        for name, fixed in zip(melt_names, fixed_definitions):
            for key, value in fixed.items():
                if key in self.FIXED_NAMES and self.FIXED_NAMES[key] not in locked:
                    label = self.FIXED_NAMES[key]
                    constraints.append(f"{name}: {label} = {value:g} {self.AXES[label][2]}")
        message = "Gray fields are swept, fixed by a model, or unused."
        if constraints:
            message += "\nRequired conditions: " + "; ".join(constraints) + ". Incompatible points will be reported."
        self.conditions_note.setText(message)
        self._updating = previous_guard

    def _mark_dirty(self, *_):
        if self._updating or self._busy:
            return
        self._dirty = True
        self._point_cache = None
        self._set_point_controls_enabled(False)
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        message = "Settings changed — calculate again. Previous curves are still shown." if self._run else "Ready — select models and calculate."
        self.status_label.setText(message)
        if self._run:
            self.plot_subtitle.setText("Previous calculation · settings have changed.")
            self._update_table()

    def _reset_result_controls(self):
        self.reference_selector.blockSignals(True)
        self.reference_selector.clear()
        self.reference_selector.blockSignals(False)
        self.reference_selector.setEnabled(False)
        self._set_point_controls_enabled(False)
        self._point_cache = None
        self.point_label.setText("—")
        self.results_table.setRowCount(0)
        self.errors_button.setEnabled(False)
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)

    # --- Parameter dialogs and numerical inputs ---------------------

    @staticmethod
    def _number(text, label):
        try:
            value = float(str(text).replace(",", "."))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} must be numeric.") from error
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value

    @staticmethod
    def _global_limit(label):
        aliases = {
            "H₂O": ("H₂O", "H2O"),
            "Vesicles": ("Vesicles", "Bubbles"),
            "γ̇ (strain rate)": ("γ̇ (strain rate)", "Strain rate", "strain_rate", "γ̇"),
        }
        return next((PARAMETER_LIMITS[key] for key in aliases.get(label, (label,))
                     if key in PARAMETER_LIMITS), (None, None))

    @classmethod
    def _check_global_value(cls, label, value):
        minimum, maximum = cls._global_limit(label)
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} must be ≥ {minimum:g} {cls.AXES[label][2]}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{label} must be ≤ {maximum:g} {cls.AXES[label][2]}.")
        if label == cls.STRAIN and value <= 0:
            raise ValueError("Strain rate must be greater than zero.")

    def get_range_values(self):
        start, end, step = [self._number(edit.text(), label) for edit, label in zip(
            self.range_edits, ("From", "To", "Step")
        )]
        if step <= 0:
            raise ValueError("Step must be greater than zero.")
        for value in (start, end):
            self._check_global_value(self._axis, value)
        count_float = abs(end - start) / step
        if not math.isfinite(count_float) or count_float >= self.MAX_POINTS:
            raise ValueError(f"Choose a larger step: at most {self.MAX_POINTS:,} points are allowed.")
        count = math.floor(count_float + 1e-12)
        direction = 1 if end >= start else -1
        values = start + direction * step * np.arange(count + 1, dtype=float)
        if math.isclose(float(values[-1]), end, rel_tol=1e-12, abs_tol=1e-12):
            values[-1] = end
        else:
            values = np.append(values, end)
        if len(values) > self.MAX_POINTS:
            raise ValueError(f"The range exceeds {self.MAX_POINTS:,} points.")
        return values

    def _read_shared_inputs(self, first_x, validate=True):
        sample = deepcopy(self.samples[self.sample_selector.currentIndex()])
        conditions = {}
        for label, (attribute, _display, _unit) in self.AXES.items():
            if label == self._axis:
                value = first_x
            elif label in self._locked:
                value = self._locked[label]
            elif label in self._used:
                value = self._number(self.physical_edits[label].text(), label)
            elif attribute:
                value = self._number(getattr(sample, attribute, 0.0), label)
            else:
                value = None
            if validate and label in self._used:
                self._check_global_value(label, value)
            conditions[label] = value
            if attribute:
                setattr(sample, attribute, value)
        if validate and "melt" in self._required_phases():
            for oxide in self.oxide_names:
                if hasattr(sample, oxide):
                    value = self._number(getattr(sample, oxide), oxide)
                    if value < 0:
                        raise ValueError(f"{oxide} cannot be negative.")
                    setattr(sample, oxide, value)
        return sample, conditions

    def _dialog_sample(self):
        first = self._number(self.from_edit.text(), "From")
        return self._read_shared_inputs(first, validate=False)[0]

    def _make_parameter_dialog(self, group, name):
        """Return an isolated dialog; settings are committed only by Save."""
        if not name:
            raise ValueError(f"No {group} model is available.")
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{name} — parameters")
        dialog.resize(560, 460)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note(
            "Parameters are saved for this model. Computed composition values use the From point."
        ))
        panel = ModelsParametersPanel(
            manage_main_inputs=False, viscosity_engine=self.viscosity_engine,
            sample_provider=self._dialog_sample, visible_groups=(group,),
        )
        values = deepcopy(self._parameters[group][name])
        rate_key = self._strain_key(name) if group == "crystal" else None
        if rate_key:
            value_text = self.from_edit.text() if self._axis == self.STRAIN else self.physical_edits[self.STRAIN].text()
            values[rate_key] = self._number(value_text, self.STRAIN)
        panel.set_parameters({f"{group}_model": name, f"{group}_parameters": values})
        panel.set_model_selection_enabled(group, False)
        if rate_key:
            widget = panel.crystal_parameters.get(rate_key)
            label = panel.crystal_parameters_layout.labelForField(widget)
            if widget is not None:
                widget.hide()
            if label is not None:
                label.hide()
            if self._axis == self.STRAIN and name == "Caricchi et al. (2007)":
                panel.crystal_parameters_box.hide()
            layout.addWidget(self._note(
                "Strain rate is supplied by the X range." if self._axis == self.STRAIN
                else f"Shared strain rate: {values[rate_key]:g} s⁻¹. Set it in Shared physical conditions."
            ))
        model = self._model(group, name)
        fixed = getattr(model, "fixed_melt_physical_parameters", {}) or {}
        if fixed:
            layout.addWidget(self._note("Required physical values: " + "; ".join(
                f"{self.FIXED_NAMES.get(key, key)} = {value:g}" for key, value in fixed.items()
            )))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)

        def save():
            updated = panel.get_group_parameters(group)
            if updated is None:
                return
            # The common strain rate is injected for each calculation point.
            if rate_key:
                updated[rate_key] = self._parameters[group][name][rate_key]
            self._parameters[group][name] = deepcopy(updated)
            self._mark_dirty()
            dialog.accept()

        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.parameters_panel = panel
        dialog.button_box = buttons
        return dialog

    def edit_parameters(self, group, name):
        try:
            dialog = self._make_parameter_dialog(group, name)
            dialog.exec()
            dialog.deleteLater()
        except Exception as error:
            QMessageBox.warning(self, "Model parameters", str(error))

    def show_composition(self):
        try:
            sample = self._dialog_sample()
        except ValueError as error:
            QMessageBox.warning(self, "Composition", str(error))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Composition — {sample.name}")
        dialog.resize(370, 560)
        layout = QVBoxLayout(dialog)
        layout.addWidget(self._note(
            "Composition used at the From point. H₂O follows the shared value or sweep; other oxides remain as loaded."
        ))
        table = self._table(("Component", "wt%"))
        entries = [(name, getattr(sample, name)) for name in self.oxide_names if hasattr(sample, name)]
        table.setRowCount(len(entries))
        for row, (name, value) in enumerate(entries):
            table.setItem(row, 0, QTableWidgetItem("F₂O₋₁" if name == "F2O_1" else name))
            table.setItem(row, 1, QTableWidgetItem(f"{float(value):.6g}"))
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    # --- Per-model checks and calculation snapshots ------------------

    def _build_run(self):
        names = self.selected_models()
        if not names:
            raise ValueError("Select at least one model to compare.")
        values = self.get_range_values()
        if len(values) * len(names) > self.MAX_EVALUATIONS:
            raise ValueError("Too many model evaluations. Increase Step or select fewer models.")
        base_sample, conditions = self._read_shared_inputs(float(values[0]))
        phases = tuple(self._required_phases())
        base_settings = {}
        for group in self.GROUPS:
            name = self._fixed_names[group]
            if group in phases and group != self._group and not name:
                raise ValueError(f"Select a fixed {group} model for this output.")
            base_settings[f"{group}_model"] = name
            base_settings[f"{group}_parameters"] = deepcopy(self._parameters[group].get(name, {}))
        crystal_names = names if self._group == "crystal" else [self._fixed_names["crystal"]]
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "group": self._group, "sample": deepcopy(base_sample),
            "source_sample": deepcopy(self.samples[self.sample_selector.currentIndex()]),
            "axis": self._axis, "output": self.y_selector.currentData(),
            "phases": phases, "models": tuple(names), "requested_x": values,
            "x": [], "conditions": [], "shared": conditions,
            "range": tuple(self._number(edit.text(), "Range") for edit in self.range_edits),
            "base_settings": base_settings,
            "parameters": {name: deepcopy(self._parameters[self._group][name]) for name in names},
            "strain_keys": {name: self._strain_key(name) for name in crystal_names if name},
            "raw": {name: [] for name in names},
            "plotted": {name: [] for name in names},
            "all_results": {name: [] for name in names},
            "all_result_errors": {name: [] for name in names},
            "errors": {name: [] for name in names}, "cancelled": False,
        }

    @staticmethod
    def _point_settings(run, name, conditions):
        settings = deepcopy(run["base_settings"])
        group = run["group"]
        settings[f"{group}_model"] = name
        settings[f"{group}_parameters"] = deepcopy(run["parameters"][name])
        if "crystal" in run["phases"]:
            rate_key = run["strain_keys"].get(settings["crystal_model"])
            if rate_key:
                settings["crystal_parameters"][rate_key] = conditions[ModelComparisonWindow.STRAIN]
        return settings

    @classmethod
    def _physical_label(cls, name):
        aliases = {"H2O": "H₂O", "Bubbles": "Vesicles", "Strain rate": cls.STRAIN,
                   "strain_rate": cls.STRAIN, "γ̇": cls.STRAIN}
        return cls.FIXED_NAMES.get(name, aliases.get(name, name))

    @staticmethod
    def _within_bounds(value, bounds, label, strict_upper=False):
        minimum, maximum = bounds
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} = {value:g}; minimum is {minimum:g}.")
        if maximum is not None and (value >= maximum if strict_upper else value > maximum):
            relation = "<" if strict_upper else "≤"
            raise ValueError(f"{label} = {value:g}; required {relation} {maximum:g}.")

    def _validate_models_at_point(self, phases, settings, conditions):
        """Use the limit metadata already respected by MVL's 2D Plot."""
        for group in phases:
            name = settings[f"{group}_model"]
            model = self._model(group, name)
            parameters = settings[f"{group}_parameters"]
            try:
                for key, bounds in (getattr(model, "model_parameter_limits", {}) or {}).items():
                    if key in parameters and isinstance(parameters[key], (int, float, np.number)):
                        value = self._number(parameters[key], key)
                        self._within_bounds(value, bounds, key)
                limits = dict(getattr(model, "model_physical_limits", {}) or {})
                dynamic_method = getattr(model, "get_dynamic_physical_limits", None)
                dynamic = dynamic_method(deepcopy(parameters)) or {} if callable(dynamic_method) else {}
                limits.update(dynamic)
                for key, bounds in limits.items():
                    label = self._physical_label(key)
                    if label in conditions and conditions[label] is not None:
                        self._within_bounds(
                            conditions[label], bounds, label,
                            strict_upper=(key in dynamic or label == "Vesicles"),
                        )
            except Exception as error:
                raise ValueError(f"{name}: {error}") from error

    def _calculate_point(self, run, x_value):
        """Evaluate the selected comparison and retain all six MVL outputs."""
        x_value = self._number(x_value, run["axis"])
        self._check_global_value(run["axis"], x_value)

        sample = deepcopy(run["sample"])
        conditions = dict(run["shared"])
        conditions[run["axis"]] = float(x_value)

        attribute = self.AXES[run["axis"]][0]
        if attribute:
            setattr(sample, attribute, float(x_value))

        candidates, errors = {}, {}
        point_settings = {}

        base_settings = self._point_settings(
            run,
            run["models"][0],
            conditions,
        )

        for name in run["models"]:
            settings = self._point_settings(
                run,
                name,
                conditions,
            )
            point_settings[name] = settings

            try:
                self._validate_models_at_point(
                    run["phases"],
                    settings,
                    conditions,
                )
                candidates[name] = settings[
                    f"{run['group']}_parameters"
                ]
            except Exception as error:
                errors[name] = str(error)

        results = {}

        if candidates:
            comparison = self.viscosity_engine.compare_models(
                sample=sample,
                group=run["group"],
                models=candidates,
                base_settings=base_settings,
                outputs=(run["output"],),
            )
            results = comparison["results"]
            errors.update(comparison["errors"])

        values = {}

        for name in run["models"]:
            raw = float(
                results.get(name, {}).get(
                    run["output"],
                    float("nan"),
                )
            )

            if not math.isfinite(raw):
                errors.setdefault(
                    name,
                    "No finite result at this X value.",
                )

            values[name] = raw

        # The plotted comparison above remains unchanged.  In parallel,
        # calculate the complete canonical MVL result so Excel can always
        # contain ηm, ηr,c, ηmc, ηr,b, ηmb and ηmcb.
        all_results = {}
        all_result_errors = {}

        for name in run["models"]:
            if name in errors:
                all_results[name] = {
                    key: float("nan")
                    for key in self.OUTPUTS
                }
                all_result_errors[name] = errors[name]
                continue

            try:
                full_result = self.viscosity_engine.calculate(
                    sample=deepcopy(sample),
                    **point_settings[name],
                )

                all_results[name] = {
                    key: float(
                        full_result.get(
                            key,
                            float("nan"),
                        )
                    )
                    for key in self.OUTPUTS
                }

                if not all(
                    math.isfinite(value)
                    for value in all_results[name].values()
                ):
                    all_result_errors[name] = (
                        "One or more complete viscosity "
                        "outputs are not finite."
                    )
                else:
                    all_result_errors[name] = ""

            except Exception as error:
                all_results[name] = {
                    key: float("nan")
                    for key in self.OUTPUTS
                }
                all_result_errors[name] = str(error)

        return (
            conditions,
            values,
            errors,
            all_results,
            all_result_errors,
        )

    def _evaluate_point(self, run, x_value):
        (
            conditions,
            values,
            errors,
            all_results,
            all_result_errors,
        ) = self._calculate_point(run, x_value)

        run["x"].append(float(x_value))
        run["conditions"].append(conditions)

        for name in run["models"]:
            raw = values[name]
            run["raw"][name].append(raw)
            run["plotted"][name].append(raw)
            run["all_results"][name].append(
                all_results[name]
            )
            run["all_result_errors"][name].append(
                all_result_errors.get(name, "")
            )
            run["errors"][name].append(
                errors.get(name, "")
            )

    def calculate_comparison(self):
        if self._busy:
            return
        try:
            run = self._build_run()
        except Exception as error:
            QMessageBox.warning(self, "Comparison inputs", str(error))
            return
        self._busy = True
        self._set_point_controls_enabled(False)
        self.control_panel.setEnabled(False)
        self.tabs.setEnabled(False)
        self.calculate_button.setEnabled(False)
        self.reference_selector.setEnabled(False)
        self.export_excel_button.setEnabled(False)
        self.export_plot_button.setEnabled(False)
        progress = QProgressDialog("Comparing models…", "Cancel", 0, len(run["requested_x"]), self)
        progress.setWindowTitle("Compare models")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        fatal_error = None
        try:
            for index, value in enumerate(run["requested_x"]):
                if index % 10 == 0:
                    progress.setLabelText(
                        f"{run['axis']} = {value:g} · point {index + 1}/{len(run['requested_x'])}"
                    )
                    progress.setValue(index)
                    QApplication.processEvents()
                    if progress.wasCanceled():
                        run["cancelled"] = True
                        break
                self._evaluate_point(run, float(value))
        except Exception as error:
            fatal_error = error
        finally:
            progress.close()
            progress.deleteLater()
            self._busy = False
            self.control_panel.setEnabled(True)
            self.tabs.setEnabled(True)
            self.calculate_button.setEnabled(True)
        if fatal_error is not None or not run["x"]:
            self._mark_dirty()
            self.reference_selector.setEnabled(self._run is not None)
            if fatal_error is not None:
                QMessageBox.critical(self, "Comparison error", str(fatal_error))
            else:
                self.status_label.setText("Cancelled before the first point was completed.")
            return
        self._run = run
        self._dirty = False
        self._point_cache = None
        self.reference_selector.blockSignals(True)
        previous_reference = self.reference_selector.currentText()
        self.reference_selector.clear()
        self.reference_selector.addItems(run["models"])
        valid = [name for name in run["models"] if np.isfinite(run["plotted"][name]).any()]
        self.reference_selector.setCurrentText(
            previous_reference if previous_reference in valid else next(iter(valid), run["models"][0])
        )
        self.reference_selector.blockSignals(False)
        self.reference_selector.setEnabled(True)
        minimum, maximum = self._global_limit(run["axis"])
        self.point_selector.blockSignals(True)
        self.point_selector.setRange(minimum if minimum is not None else -1e100,
                                     maximum if maximum is not None else 1e100)
        self.point_selector.setSingleStep(run["range"][2])
        self.point_selector.setValue(run["x"][0])
        self.point_selector.setToolTip("Enter a physical X value. Each model is evaluated directly at that value.")
        self.point_selector.blockSignals(False)
        self._set_point_controls_enabled(True)
        error_count = sum(bool(error) for entries in run["errors"].values() for error in entries)
        self.errors_button.setText(f"Point errors ({error_count})…")
        self.errors_button.setEnabled(bool(error_count))
        self.export_excel_button.setEnabled(True)
        self.export_plot_button.setEnabled(bool(valid))
        successful = len(run["x"]) * len(run["models"]) - error_count
        prefix = "Cancelled · partial results. " if run["cancelled"] else "Completed. "
        self.status_label.setText(
            f"{prefix}{len(run['x'])} X points · {successful} valid model results · {error_count} errors."
        )
        self._render_plot()
        self._remember_original_plot()

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
        else:
            super().closeEvent(event)

    # --- Curves, reference differences and point values --------------

    def _new_axes(self, output=None):
        self.figure.clear()
        self.main_axes, self.difference_axes = self.figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": (3, 1.35)}
        )
        self.ax = self.main_axes  # Shared toolbar helpers use this name.
        output = output or self.y_selector.currentData()
        # Matplotlib's Edit axes dialog uses these names before axis formulas.
        self.main_axes.set_label(self.OUTPUT_TITLES[output])
        self.difference_axes.set_label("Difference relative to reference")
        for axes in (self.main_axes, self.difference_axes):
            axes.set_xscale("linear")
            axes.set_yscale("linear")
            axes.grid(True, color="#e6ebef", linewidth=0.7)
            axes.spines[["top", "right"]].set_visible(False)
            for spine in ("bottom", "left"):
                axes.spines[spine].set_color("#adb9c1")
            axes.tick_params(labelsize=9, colors="#44545f")
        self.difference_axes.axhline(0, color="#798894", linewidth=0.9, linestyle="--")
        self.difference_axes.set_ylabel(self._difference_label(output), fontsize=11)

    def _draw_empty_plot(self):
        self._new_axes()
        self.plot_title.setText(self.GROUPS[self._group])
        self.plot_subtitle.setText("Select models and click Calculate comparison.")
        self.main_axes.text(
            0.5, 0.5, "Your selected models will appear here",
            ha="center", va="center", transform=self.main_axes.transAxes, color="#83929c",
        )
        self.main_axes.set_ylabel(self.OUTPUTS[self.y_selector.currentData()][1])
        self.difference_axes.set_xlabel(self.AXES[self._axis][1])
        self._apply_axis_options()
        self.canvas.draw_idle()

    def _reference_changed(self, *_):
        if not self._busy and self._run is not None:
            self._render_plot()

    def _render_plot(self):
        run = self._run
        if run is None:
            return
        self._new_axes(run["output"])
        reference = self.reference_selector.currentText()
        self.difference_axes.set_label(f"Difference relative to {reference}")
        ref_values = np.asarray(run["plotted"][reference], dtype=float)
        x = np.asarray(run["x"], dtype=float)
        any_curve = False
        for index, name in enumerate(run["models"]):
            y = np.asarray(run["plotted"][name], dtype=float)
            if not np.isfinite(y).any():
                continue
            any_curve = True
            kwargs = self._style_for(name, run["group"])
            self.main_axes.plot(x, y, **kwargs)
            self.difference_axes.plot(x, y - ref_values, **kwargs)
        if any_curve:
            self.main_axes.legend(
                fontsize=8, frameon=True, facecolor="white", edgecolor="#d5dfe5",
                ncol=2 if len(run["models"]) > 6 else 1, loc="best",
            )
        else:
            self.main_axes.text(
                0.5, 0.5, "No valid results. Open Point errors for details.",
                ha="center", va="center", transform=self.main_axes.transAxes, color="#a63b2a",
            )
        if not np.isfinite(ref_values).any():
            self.difference_axes.text(
                0.5, 0.6, "The reference has no valid points; choose another model.",
                ha="center", va="center", transform=self.difference_axes.transAxes,
                color="#a63b2a", fontsize=9,
            )
        self.main_axes.set_ylabel(self.OUTPUTS[run["output"]][1], fontsize=11)
        self.difference_axes.set_xlabel(self.AXES[run["axis"]][1], fontsize=11)
        self.difference_axes.set_title(f"Model − {reference}", loc="left", fontsize=9, color="#536674")
        if len(x) > 1 and x[0] != x[-1]:
            self.main_axes.set_xlim(float(x[0]), float(x[-1]))
        self.figure.suptitle(
            f"{run['sample'].name} · {self.GROUPS[run['group']]}" + (" · partial range" if run["cancelled"] else ""),
            fontsize=12, color="#263e4c",
        )
        self.plot_title.setText(f"{run['sample'].name} · {self.GROUPS[run['group']]}")
        subtitle = f"{len(run['models'])} models · {len(x)} X points · "
        subtitle += ("dimensionless factors · differences in factor values"
                     if run["output"].startswith("eta_r_") else "differences in log₁₀ units")
        if run["cancelled"]:
            subtitle += " · partial results"
        if self._dirty:
            subtitle = "Previous calculation · settings have changed. " + subtitle
        self.plot_subtitle.setText(subtitle)
        self._apply_axis_options()
        self.toolbar.update()
        self.canvas.draw_idle()
        self._update_table()

    def _update_table(self, *_):
        run = self._run
        if run is None:
            return
        relative = run["output"].startswith("eta_r_")
        self.results_table.setHorizontalHeaderLabels((
            "Model", "Correction factor" if relative else "log₁₀ η (Pa·s)",
            "Δ factor" if relative else "Δ log₁₀ η", "Valid points"))
        self.point_label.setText(self.AXES[run["axis"]][2] if not self._dirty else "Calculate again to query X.")
        values, errors = ({}, {}) if self._dirty else self._values_at_x(self.point_selector.value())
        reference = self.reference_selector.currentText()
        ref_value = values.get(reference, float("nan"))
        self.results_table.setRowCount(len(run["models"]))
        for row, name in enumerate(run["models"]):
            value = values.get(name, float("nan"))
            delta = value - ref_value
            valid_count = int(np.isfinite(run["plotted"][name]).sum())
            texts = (name, f"{value:.6f}" if math.isfinite(value) else "—",
                     f"{delta:+.6f}" if math.isfinite(delta) else "—",
                     f"{valid_count}/{len(run['x'])}")
            for column, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setForeground(QColor(self._style_for(name, run["group"])["color"]))
                item.setToolTip(errors.get(name, "") or (
                    "Reference unavailable at this point." if column == 2 and not math.isfinite(delta) else ""
                ))
                self.results_table.setItem(row, column, item)

    def _plot_clicked(self, event):
        if (self._run is None or self._dirty or self._busy or event.xdata is None or self.toolbar.mode
                or event.inaxes not in (self.main_axes, self.difference_axes)):
            return
        self.point_selector.setValue(float(event.xdata))

    def _set_point_controls_enabled(self, enabled):
        self.point_selector.setEnabled(enabled)
        for button in self.point_step_buttons:
            button.setEnabled(enabled)

    def _values_at_x(self, x):
        """Reuse an exact sweep point or calculate X directly; never interpolate."""
        if self._point_cache is not None and self._point_cache[0] == x:
            return self._point_cache[1:]
        run = self._run
        matches = np.flatnonzero(np.asarray(run["x"], dtype=float) == x)
        if len(matches):
            index = int(matches[0])
            values = {name: run["plotted"][name][index] for name in run["models"]}
            errors = {name: run["errors"][name][index] for name in run["models"]}
        else:
            try:
                (
                    _conditions,
                    values,
                    errors,
                    _all_results,
                    _all_result_errors,
                ) = self._calculate_point(run, x)
            except Exception as error:
                values = {name: float("nan") for name in run["models"]}
                errors = {name: str(error) for name in run["models"]}
        self._point_cache = (x, values, errors)
        return values, errors

    # --- Plot appearance: shared axis controls and per-model curves ---

    def _axis_keys(self):
        return ((self._run["axis"], self._run["output"]) if self._run is not None
                else (self._axis, self.y_selector.currentData()))

    @classmethod
    def _difference_label(cls, output, rich=False):
        symbol = cls.OUTPUT_SYMBOLS[output]
        relative = output.startswith("eta_r_")
        if rich:
            return "Δ " + ("" if relative else "log<sub>10</sub> ") + f"η<sub>{symbol}</sub>"
        return "$\\Delta " + ("" if relative else r"\log_{10}") + r"\eta_{\mathrm{" + symbol + "}}$"

    def _default_axis_html(self, direction):
        parameter, output = self._axis_keys()
        if direction == "difference":
            return self._difference_label(output, rich=True)
        if direction == "y":
            return self.OUTPUTS[output][0].split(" · ", 1)[-1]
        if parameter == self.STRAIN:
            return "γ̇ (s<sup>−1</sup>)"
        return escape(self.AXES[parameter][1]).replace("H₂O", "H<sub>2</sub>O")

    def _style_for(self, name, group=None):
        group = group or (self._run["group"] if self._run is not None else self._group)
        styles = self._curve_styles[group]
        if name not in styles:
            index = list(self.viscosity_engine.get_model_names(group)).index(name)
            styles[name] = dict(label=name, color=self.COLORS[index % len(self.COLORS)],
                                linestyle=("-", "--", "-.")[index // len(self.COLORS) % 3],
                                linewidth=1.8, marker=".", markersize=3)
        return styles[name]

    def _apply_axis_options(self):
        parameter, output = self._axis_keys()
        for axes in (self.main_axes, self.difference_axes):
            axes.set_xscale("linear")
            axes.set_yscale("linear")
            axes.grid(self._plot_options["grid"])
        for direction, key, axes, coordinate in (
            ("x", parameter, self.difference_axes, "x"),
            ("y", output, self.main_axes, "y"),
            ("difference", output, self.difference_axes, "y"),
        ):
            options = self._axis_options[direction].get(key, {})
            if options.get("plot") is not None:
                getattr(axes, f"set_{coordinate}label")(options["plot"])
            if options.get("limits") is not None:
                getattr(axes, f"set_{coordinate}lim")(options["limits"])
        if self._plot_options["title"] is not None:
            self.figure.suptitle(self._plot_options["title"], fontsize=12, color="#263e4c")
        legend = self.main_axes.get_legend()
        if legend is not None:
            legend.set_visible(self._plot_options["legend"])
        self.figure.set_layout_engine("constrained" if self._layout_margins is None else None)
        if self._layout_margins is not None:
            self.figure.subplots_adjust(**self._layout_margins)

    def _make_axes_dialog(self):
        parameter, output = self._axis_keys()
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit axes — " + self.OUTPUT_TITLES[output])
        dialog.setWindowIcon(self.windowIcon())
        dialog.resize(650, 720)
        outer = QVBoxLayout(dialog)
        outer.addWidget(QLabel(self.OUTPUT_TITLES[output], objectName="plotTitle"))
        tabs = QTabWidget()
        outer.addWidget(tabs, 1)
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        initial_title = self.figure._suptitle.get_text() if self.figure._suptitle is not None else ""
        title = QLineEdit(initial_title)
        form.addRow("Title", title)
        layout.addLayout(form)
        fields = {}
        definitions = (
            ("x", parameter, self.difference_axes, "x", "X axis — shared by both plots"),
            ("y", output, self.main_axes, "y", "Y axis — " + self.OUTPUT_TITLES[output]),
            ("difference", output, self.difference_axes, "y", "Y axis — Difference relative to reference"),
        )
        for direction, key, axes, coordinate, heading in definitions:
            options = self._axis_options[direction].get(key, {})
            box = QGroupBox(heading)
            form = QFormLayout(box)
            container, editor = PlotWindow._make_label_editor(
                self, options.get("html", self._default_axis_html(direction)))
            form.addRow("Label", container)
            automatic = QCheckBox("Automatic limits")
            automatic.setChecked(options.get("limits") is None)
            form.addRow(automatic)
            limits = options.get("limits") or getattr(axes, f"get_{coordinate}lim")()
            minimum, maximum = QLineEdit(f"{limits[0]:.12g}"), QLineEdit(f"{limits[1]:.12g}")
            row = QHBoxLayout()
            for label, edit in (("Min", minimum), ("Max", maximum)):
                row.addWidget(QLabel(label))
                row.addWidget(edit)
                edit.setEnabled(not automatic.isChecked())
            form.addRow(row)
            automatic.toggled.connect(lambda checked, lo=minimum, hi=maximum:
                                      (lo.setEnabled(not checked), hi.setEnabled(not checked)))
            fields[direction] = dict(label=editor, auto=automatic, min=minimum, max=maximum,
                                     initial_html=editor.toHtml(),
                                     initial_plot=getattr(axes, f"get_{coordinate}label")())
            layout.addWidget(box)
        grid, legend = QCheckBox("Show grid"), QCheckBox("Show model legend")
        grid.setChecked(self._plot_options["grid"])
        legend.setChecked(self._plot_options["legend"])
        row = QHBoxLayout()
        row.addWidget(grid)
        row.addWidget(legend)
        layout.addLayout(row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        tabs.addTab(scroll, "Axes")
        curves = QWidget()
        curve_layout = QVBoxLayout(curves)
        curve_layout.addWidget(self._note("Choose a model to edit its color, line, symbol and legend name. Changes apply to both plots."))
        selector = QComboBox()
        names = self._run["models"] if self._run is not None else self.selected_models()
        for name in names:
            label = self._style_for(name)["label"]
            selector.addItem(label if label == name else f"{label} ({name})", name)
        curve_layout.addWidget(selector)
        def edit_curve():
            name = selector.currentData()
            if name is not None:
                editor = self._make_style_dialog(name)
                editor.exec()
                editor.deleteLater()
                label = self._style_for(name)["label"]
                selector.setItemText(selector.currentIndex(), label if label == name else f"{label} ({name})")
        button = self._button("Edit selected curve…", edit_curve)
        button.setEnabled(selector.count() > 0)
        curve_layout.addWidget(button)
        curve_layout.addStretch()
        tabs.addTab(curves, "Curves")
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            updates = {}
            try:
                for direction, values in fields.items():
                    editor = values["label"]
                    plot_text = (values["initial_plot"] if editor.toHtml() == values["initial_html"]
                                 else PlotWindow._label_to_plot(editor))
                    MathTextParser("path").parse(plot_text)
                    limits = None if values["auto"].isChecked() else tuple(
                        self._number(values[key].text(), f"{direction} {key}") for key in ("min", "max"))
                    if limits is not None and limits[0] == limits[1]:
                        raise ValueError("Axis limits must be different.")
                    updates[direction] = dict(html=editor.toHtml(), plot=plot_text, limits=limits)
                MathTextParser("path").parse(title.text())
            except (ValueError, TypeError) as error:
                QMessageBox.warning(dialog, "Invalid axis settings", str(error))
                return
            for direction, key, _axes, _coordinate, _heading in definitions:
                self._axis_options[direction][key] = updates[direction]
            if title.text() != initial_title:
                self._plot_options["title"] = title.text()
            self._plot_options.update(grid=grid.isChecked(), legend=legend.isChecked())
            self._render_plot() if self._run is not None else self._draw_empty_plot()
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)
        dialog.axis_fields, dialog.button_box, dialog.tabs = fields, buttons, tabs
        dialog.curve_selector, dialog.title_edit = selector, title
        return dialog

    def _make_style_dialog(self, name):
        group = self._run["group"] if self._run is not None else self._group
        style = deepcopy(self._style_for(name, group))
        dialog = QDialog(self)
        dialog.setWindowTitle("Curve style — " + name)
        dialog.setWindowIcon(self.windowIcon())
        dialog.resize(440, 380)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        label = QLineEdit(style["label"])
        form.addRow("Legend label", label)
        color = [style["color"]]
        color_button = QPushButton(color[0])
        color_button.setStyleSheet(f"border: 3px solid {color[0]};")
        def choose_color():
            selected = QColorDialog.getColor(QColor(color[0]), dialog, "Color — " + name)
            if selected.isValid():
                color[0] = selected.name()
                color_button.setText(color[0])
                color_button.setStyleSheet(f"border: 3px solid {color[0]};")
        color_button.clicked.connect(choose_color)
        form.addRow("Color", color_button)
        line, marker = QComboBox(), QComboBox()
        for text, value in (("Solid", "-"), ("Dashed", "--"), ("Dash-dot", "-."), ("Dotted", ":"), ("No line", "None")):
            line.addItem(text, value)
        for text, value in (("None", "None"), ("Circle", "o"), ("Square", "s"), ("Triangle", "^"), ("Diamond", "D"), ("Point", "."), ("Cross", "x"), ("Plus", "+")):
            marker.addItem(text, value)
        line.setCurrentIndex(max(0, line.findData(style["linestyle"])))
        marker.setCurrentIndex(max(0, marker.findData(style["marker"])))
        form.addRow("Line style", line)
        fields = {}
        for key, text, upper, step in (("linewidth", "Line width", 12, .2), ("markersize", "Symbol size", 30, .5)):
            spin = QDoubleSpinBox()
            spin.setRange(.1, upper)
            spin.setSingleStep(step)
            spin.setValue(style[key])
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            row = QHBoxLayout()
            for caption, callback in (("−", spin.stepDown), ("+", spin.stepUp)):
                button = self._button(caption, callback, "smallButton")
                button.setFixedWidth(32)
                if caption == "+":
                    row.addWidget(spin)
                row.addWidget(button)
            form.addRow(text, row)
            fields[key] = spin
        form.addRow("Symbol", marker)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        def save():
            self._curve_styles[group][name] = dict(
                style, label=label.text().strip() or name, color=color[0],
                linestyle=line.currentData(), marker=marker.currentData(),
                linewidth=fields["linewidth"].value(), markersize=fields["markersize"].value())
            if self._run is not None:
                self._render_plot()
            dialog.accept()
        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.style_widgets = dict(fields, label=label, color=color_button, linestyle=line, marker=marker)
        dialog.button_box = buttons
        return dialog

    def _make_layout_dialog(self):
        dialog = PlotWindow._make_layout_dialog(self)
        if self._layout_margins is None:
            main, difference = self.main_axes.get_position(), self.difference_axes.get_position()
            for name, value in dict(left=min(main.x0, difference.x0), right=max(main.x1, difference.x1),
                                    bottom=difference.y0, top=main.y1).items():
                spin = dialog.margin_fields[name]
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        return dialog

    def _remember_original_plot(self):
        self._original_plot_configuration = deepcopy(dict(
            axes=self._axis_options, plot=self._plot_options, styles=self._curve_styles,
            margins=self._layout_margins))

    def restore_original_plot(self):
        if self._original_plot_configuration is None:
            return
        original = deepcopy(self._original_plot_configuration)
        self._axis_options, self._plot_options = original["axes"], original["plot"]
        self._curve_styles, self._layout_margins = original["styles"], original["margins"]
        self._render_plot() if self._run is not None else self._draw_empty_plot()
        self.canvas.draw()
        self.toolbar.push_current()

    def show_errors(self):
        if self._run is None:
            return
        rows = [
            (name, self._run["x"][index], error)
            for name in self._run["models"]
            for index, error in enumerate(self._run["errors"][name]) if error
        ]
        dialog = QDialog(self)
        dialog.setWindowTitle("Comparison point errors")
        dialog.resize(950, 520)
        layout = QVBoxLayout(dialog)
        limit = 2000
        layout.addWidget(self._note(
            f"{len(rows)} failed model points. Curves have gaps at these positions. "
            + (f"Showing the first {limit}; Export Excel includes all errors." if len(rows) > limit else "")
        ))
        table = self._table(("Model", self.AXES[self._run["axis"]][1], "Reason"))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.setWordWrap(True)
        table.setRowCount(min(len(rows), limit))
        for row, (name, x, error) in enumerate(rows[:limit]):
            for column, text in enumerate((name, f"{x:.9g}", error)):
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

    # --- Export exactly the displayed calculation --------------------

    def export_frames(self):
        """Export comparison data plus a standard MVL Samples sheet."""
        if self._run is None or self._dirty:
            raise ValueError(
                "Calculate the current settings before exporting."
            )

        run = self._run
        reference = self.reference_selector.currentText()
        output = run["output"]
        symbol = f"η{self.OUTPUT_SYMBOLS[output]}"
        relative = output.startswith("eta_r_")
        notation = (
            symbol
            if relative
            else f"log₁₀ {symbol}"
        )

        result_column = (
            f"{self.OUTPUT_TITLES[output]} {notation}"
            + (
                ""
                if relative
                else " (Pa·s)"
            )
        )

        has_errors = any(
            error
            for entries in run["errors"].values()
            for error in entries
        )

        has_full_errors = any(
            error
            for entries in run["all_result_errors"].values()
            for error in entries
        )

        model_columns = {
            "melt": "Liquid viscosity model",
            "crystal": "Crystal correction model",
            "vesicle": "Vesicle correction model",
        }

        parameter_prefixes = {
            "melt": "Liquid parameter",
            "crystal": "Crystal parameter",
            "vesicle": "Vesicle parameter",
        }

        rows = []

        for name in run["models"]:
            for index, x in enumerate(run["x"]):
                conditions = run["conditions"][index]
                plotted = run["plotted"][name][index]
                delta = (
                    plotted
                    - run["plotted"][reference][index]
                )

                settings = self._point_settings(
                    run,
                    name,
                    conditions,
                )

                complete = run["all_results"][
                    name
                ][index]

                row = {
                    "Sample":
                        str(run["sample"].name),

                    "Compared model":
                        name,

                    self.AXES[run["axis"]][1]:
                        x,

                    "Melt viscosity "
                    "log₁₀ ηm (Pa·s)":
                        (
                            complete["log10_eta_m"]
                            if math.isfinite(
                                complete["log10_eta_m"]
                            )
                            else None
                        ),

                    "Crystal correction "
                    "factor ηr,c":
                        (
                            complete["eta_r_c"]
                            if math.isfinite(
                                complete["eta_r_c"]
                            )
                            else None
                        ),

                    "Crystal-bearing magma "
                    "viscosity log₁₀ ηmc (Pa·s)":
                        (
                            complete["log10_eta_mc"]
                            if math.isfinite(
                                complete["log10_eta_mc"]
                            )
                            else None
                        ),

                    "Vesicle correction "
                    "factor ηr,b":
                        (
                            complete["eta_r_b"]
                            if math.isfinite(
                                complete["eta_r_b"]
                            )
                            else None
                        ),

                    "Vesicle-bearing magma "
                    "viscosity log₁₀ ηmb (Pa·s)":
                        (
                            complete["log10_eta_mb"]
                            if math.isfinite(
                                complete["log10_eta_mb"]
                            )
                            else None
                        ),

                    "Three-phase magma "
                    "viscosity log₁₀ ηmcb (Pa·s)":
                        (
                            complete["log10_eta_mcb"]
                            if math.isfinite(
                                complete["log10_eta_mcb"]
                            )
                            else None
                        ),

                    result_column:
                        (
                            plotted
                            if math.isfinite(plotted)
                            else None
                        ),
                }

                row[
                    f"Δ {notation} "
                    "(model − reference)"
                ] = (
                    delta
                    if math.isfinite(delta)
                    else None
                )

                row["Reference model"] = reference

                row.update(
                    {
                        self.AXES[label][1]:
                            value

                        for label, value
                        in conditions.items()

                        if (
                            label != run["axis"]
                            and value is not None
                        )
                    }
                )

                for group in run["phases"]:
                    if group != run["group"]:
                        row[
                            model_columns[group]
                        ] = settings[
                            f"{group}_model"
                        ]

                    rate_key = (
                        run["strain_keys"].get(
                            settings["crystal_model"]
                        )
                        if group == "crystal"
                        else None
                    )

                    for key, value in settings[
                        f"{group}_parameters"
                    ].items():
                        # Strain rate is already a physical/X column.
                        if key != rate_key:
                            row[
                                f"{parameter_prefixes[group]} "
                                f"- {key}"
                            ] = value

                if has_errors:
                    row["Error"] = run["errors"][
                        name
                    ][index]

                if has_full_errors:
                    row[
                        "Complete outputs error"
                    ] = run[
                        "all_result_errors"
                    ][name][index]

                if run["cancelled"]:
                    row["Calculation"] = (
                        "Partial: "
                        f"{len(run['x'])}/"
                        f"{len(run['requested_x'])} "
                        "X points (cancelled)"
                    )

                rows.append(row)

        sheet_name = re.sub(
            r"[\\/?*\[\]:\x00-\x1f]",
            "_",
            str(run["sample"].name),
        )
        sheet_name = (
            sheet_name[:31].strip("'")
            or "Sample"
        )

        # Same sample/composition concept used in the other MVL exports.
        sample = run["sample"]

        sample_row = {
            "Sample":
                str(sample.name),

            "Temperature (°C)":
                getattr(
                    sample,
                    "temperature",
                    None,
                ),

            "H₂O (wt%)":
                getattr(
                    sample,
                    "H2O",
                    None,
                ),

            "Crystals (vol%)":
                getattr(
                    sample,
                    "crystals",
                    None,
                ),

            "Vesicles (vol%)":
                getattr(
                    sample,
                    "Vesicles",
                    None,
                ),
        }

        for oxide in self.oxide_names:
            if not hasattr(sample, oxide):
                continue

            label = (
                "F₂O₋₁ (wt%)"
                if oxide == "F2O_1"
                else f"{oxide} (wt%)"
            )

            sample_row[label] = getattr(
                sample,
                oxide,
            )

        return {
            sheet_name:
                pd.DataFrame(rows),

            "Samples":
                pd.DataFrame(
                    [sample_row]
                ),
        }

    def write_excel(self, filename):
        from openpyxl.styles import Alignment

        frames = self.export_frames()
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for sheet_name, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                sheet = writer.sheets[sheet_name]
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for cells in sheet.columns:
                    header = cells[0]
                    sheet.column_dimensions[header.column_letter].width = min(
                        38, max(14, len(str(header.value)) + 2)
                    )
                    header.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                sheet.row_dimensions[1].height = 44
                # Sample identifiers and messages are literal text, including '='.
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.data_type == "f":
                            cell.data_type = "s"
        self._format_excel_subscripts(filename)

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

    def export_excel(self):
        if self._run is None or self._dirty:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export model comparison", "MVL_model_comparison.xlsx", "Excel files (*.xlsx)"
        )
        if not filename:
            return
        if not filename.lower().endswith(".xlsx"):
            filename += ".xlsx"
        try:
            self.write_excel(filename)
            self.status_label.setText(f"Exported comparison: {filename}")
        except Exception as error:
            QMessageBox.critical(self, "Excel export error", str(error))

    def export_plot(self):
        if self._run is None or self._dirty:
            return
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export comparison figure", "MVL_model_comparison.png",
            "PNG image (*.png);;PDF figure (*.pdf);;SVG figure (*.svg)",
        )
        if not filename:
            return
        extension = {"PDF figure (*.pdf)": ".pdf", "SVG figure (*.svg)": ".svg"}.get(selected_filter, ".png")
        path = Path(filename)
        if path.suffix.lower() not in (".png", ".pdf", ".svg"):
            filename += extension
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
            self.status_label.setText(f"Exported figure: {filename}")
        except Exception as error:
            QMessageBox.critical(self, "Figure export error", str(error))
