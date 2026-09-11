from collections.abc import Mapping
from copy import deepcopy
import math

from PySide6.QtCore import Qt, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


class ModelsParametersPanel(QGroupBox):
    """Model controls shared by MainWindow, plots and comparison dialogs.

    The existing calls ModelsParametersPanel(main_window) and
    ModelsParametersPanel(main_window, manage_main_inputs=False) remain valid.
    A comparison can additionally provide a viscosity_engine, a callable
    sample_provider and visible_groups, for example ("crystal",).

    When sample_provider is supplied, the panel reads a copy of its sample
    and never manages or connects to the MainWindow input widgets. The caller
    should call refresh_computed_fields() when its local inputs change.
    """

    parameters_changed = Signal()

    # --- Model-panel definitions ------------------------------------

    MODEL_GROUPS = {
        "melt": ("Melt viscosity", "model_selector", "melt_manager"),
        "crystal": ("Crystal correction", "crystal_model", "crystal_manager"),
        "vesicle": ("Vesicle correction", "vesicle_model", "vesicle_manager"),
    }

    PHYSICAL_PARAMETER_WIDGETS = {
        "temperature": "temperature_edit",
        "water": "water_edit",
        "crystals": "crystals_edit",
        "vesicles": "vesicles_edit",
    }

    SAMPLE_PHYSICAL_ATTRIBUTES = {
        "temperature": "temperature",
        "water": "H2O",
        "crystals": "crystals",
        "vesicles": "Vesicles",
    }

    FIXED_FIELD_STYLE = """
        QLineEdit,
        QLineEdit:disabled {
            background-color: #e1e4e7;
            color: #66717b;
            border: 1px solid #b9c0c6;
        }
    """

    COMPUTED_FIELD_STYLE = """
        QLabel {
            padding: 7px 9px;
            background-color: #f1f3f5;
            border: 1px solid #d4d8dc;
            border-radius: 4px;
            color: #333333;
            font-size: 10.5pt;
            font-weight: normal;
        }
    """

    INVALID_FIELD_STYLE = """
        border: 2px solid red;
        background-color: #ffe6e6;
    """

    # --- Panel initialization ---------------------------------------

    def __init__(
        self,
        main_window=None,
        manage_main_inputs=True,
        *,
        viscosity_engine=None,
        sample_provider=None,
        visible_groups=None,
    ):
        """Create selectors and optionally bind computed fields to local data."""
        if sample_provider is not None and not callable(sample_provider):
            raise TypeError("sample_provider must be callable or None.")
        visible_groups = self._validated_groups(visible_groups)
        super().__init__("Models and Parameters")

        self.main_window = main_window
        self.viscosity_engine = viscosity_engine
        self.sample_provider = sample_provider
        self.manage_main_inputs = bool(manage_main_inputs) and sample_provider is None

        self._fixed_melt_physical_parameters = set()
        self._applying_fixed_parameters = False
        self._refreshing_computed_fields = False
        self._loading_settings = False
        self._active_model_names = {}
        self._parameter_cache = {group: {} for group in self.MODEL_GROUPS}

        self.computed_fields_boxes = {}
        self.computed_fields_layouts = {}
        self.computed_fields_widgets = {}
        self.selector_managers = {}

        layout = QHBoxLayout(self)
        for group, (title, selector_name, manager_name) in self.MODEL_GROUPS.items():
            box = self._create_model_box(group, title, selector_name, manager_name)
            setattr(self, f"{group}_box", box)
            layout.addWidget(box)

        self.set_visible_groups(visible_groups)
        self._connect_computed_field_updates()
        QTimer.singleShot(0, self.refresh_computed_fields)

    # --- Public configuration ---------------------------------------

    @classmethod
    def _validated_groups(cls, groups):
        """Normalize the visible model-group selection."""
        if groups is None:
            return tuple(cls.MODEL_GROUPS)
        if isinstance(groups, str):
            groups = (groups,)
        else:
            groups = tuple(groups)
        unknown = set(groups).difference(cls.MODEL_GROUPS)
        if unknown:
            raise ValueError("Unknown model group(s): " + ", ".join(sorted(unknown)))
        return groups

    def set_visible_groups(self, groups=None):
        """Show all groups or only the group used by a parameter dialog."""
        groups = self._validated_groups(groups)
        for group in self.MODEL_GROUPS:
            getattr(self, f"{group}_box").setVisible(group in groups)

    def set_model_selection_enabled(self, group, enabled):
        """Lock a dialog to the model chosen in the comparison list."""
        if group not in self.MODEL_GROUPS:
            raise ValueError(f"Unknown model group: {group}")
        selector_name = self.MODEL_GROUPS[group][1]
        getattr(self, selector_name).setEnabled(bool(enabled))

    def set_parameters(self, settings):
        """Restore a complete or partial dictionary from get_parameters().

        Only groups present in settings are changed. A model name alone
        restores its cached values. An explicit *_parameters mapping resets
        that model to its declared defaults and then applies the supplied
        values. Unknown names, choices and non-finite numbers raise an error
        before any controls are changed. Scientific limits remain the
        responsibility of the model and calculation engine.
        """
        if not isinstance(settings, Mapping):
            raise TypeError("Model settings must be a mapping.")

        allowed_keys = {
            f"{group}_{suffix}"
            for group in self.MODEL_GROUPS
            for suffix in ("model", "parameters")
        }
        unknown_keys = set(settings).difference(allowed_keys)
        if unknown_keys:
            raise ValueError(
                "Unknown model setting(s): " + ", ".join(sorted(unknown_keys))
            )

        plans = []
        for group, (_title, selector_name, manager_name) in self.MODEL_GROUPS.items():
            model_key, values_key = f"{group}_model", f"{group}_parameters"
            if model_key not in settings and values_key not in settings:
                continue

            selector = getattr(self, selector_name)
            model_name = settings.get(model_key, selector.currentText())
            if not isinstance(model_name, str) or selector.findText(model_name) < 0:
                raise ValueError(f"Unknown {group} model: {model_name}")

            values = None
            if values_key in settings:
                supplied = settings[values_key]
                if supplied is None:
                    supplied = {}
                if not isinstance(supplied, Mapping):
                    raise TypeError(f"{values_key} must be a mapping.")
                manager = self._get_manager(manager_name)
                if manager is None:
                    raise ValueError(f"No manager available for {group} models.")
                model = manager.get_model(model_name)
                definitions = getattr(model, "parameters", {}) or {}
                unknown = set(supplied).difference(definitions)
                if unknown:
                    raise ValueError(
                        f"Unknown parameter(s) for {model_name}: "
                        + ", ".join(sorted(unknown))
                    )

                values = {}
                for name, info in definitions.items():
                    value = supplied.get(name, info.get("default", 0.0))
                    if info.get("type", "float") == "choice":
                        options = [str(option) for option in info.get("options", [])]
                        if name not in supplied and str(value) not in options and options:
                            value = options[0]
                        value = str(value)
                        if value not in options:
                            raise ValueError(f"Invalid choice for {model_name}: {name}")
                        values[name] = value
                    else:
                        try:
                            values[name] = self._finite_number(value)
                        except (TypeError, ValueError, OverflowError) as error:
                            raise ValueError(
                                f"{model_name}: {name} must be a finite number."
                            ) from error
            plans.append((group, selector_name, model_name, values))

        if not plans:
            return

        was_loading = self._loading_settings
        self._loading_settings = True
        try:
            for group, selector_name, model_name, values in plans:
                getattr(self, selector_name).setCurrentText(model_name)
                if values is not None:
                    widgets = getattr(self, f"{group}_parameters")
                    for name, value in values.items():
                        widget = widgets[name]
                        with QSignalBlocker(widget):
                            if isinstance(widget, QComboBox):
                                widget.setCurrentText(value)
                            else:
                                widget.setText(str(value))
                                if not widget.isReadOnly():
                                    widget.setStyleSheet("")
                    if group == "melt" and model_name == "Vetere et al. (2008)":
                        self._update_vetere_ratio()
                self._remember_current_parameters(group)
        finally:
            self._loading_settings = was_loading

        self.refresh_computed_fields()
        if not self._loading_settings:
            self.parameters_changed.emit()

    # --- Model-box construction -------------------------------------

    def _create_model_box(self, group, title, selector_name, manager_name):
        """Create one model selector and its parameter controls."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(QLabel("Model"))

        selector = QComboBox()
        setattr(self, selector_name, selector)
        self.selector_managers[selector_name] = manager_name
        layout.addWidget(selector)

        computed_box = QWidget()
        computed_layout = QVBoxLayout(computed_box)
        computed_layout.setContentsMargins(0, 4, 0, 2)
        computed_layout.setSpacing(4)
        computed_box.hide()
        self.computed_fields_boxes[selector_name] = computed_box
        self.computed_fields_layouts[selector_name] = computed_layout
        self.computed_fields_widgets[selector_name] = {}
        layout.addWidget(computed_box)

        parameters_box = QGroupBox()
        parameters_layout = QFormLayout(parameters_box)
        setattr(self, f"{group}_parameters", {})
        setattr(self, f"{group}_parameters_box", parameters_box)
        setattr(self, f"{group}_parameters_layout", parameters_layout)
        layout.addWidget(parameters_box)

        manager = self._get_manager(manager_name)
        if manager is not None:
            selector.addItems(manager.models.keys())
        selector.currentTextChanged.connect(
            lambda _text, group=group, selector_name=selector_name,
            manager_name=manager_name:
                self._update_parameters(group, selector_name, manager_name)
        )
        self._update_parameters(group, selector_name, manager_name)
        return box

    # --- Model access -----------------------------------------------

    def _get_manager(self, manager_name):
        """Return a manager from the explicit engine or the MainWindow engine."""
        engine = self.viscosity_engine
        if engine is None and self.main_window is not None:
            engine = getattr(self.main_window, "viscosity_engine", None)
        return getattr(engine, manager_name, None) if engine is not None else None

    def _get_model(self, selector_name, manager_name):
        """Return the model selected in one model box."""
        model_name = getattr(self, selector_name).currentText()
        manager = self._get_manager(manager_name)
        if not model_name or manager is None:
            return None
        try:
            return manager.get_model(model_name)
        except Exception:
            return None

    # --- Dynamic parameter controls --------------------------------

    def _remember_current_parameters(self, group):
        """Keep each model's raw entries, including unfinished numeric edits."""
        model_name = self._active_model_names.get(group)
        if not model_name:
            return
        widgets = getattr(self, f"{group}_parameters", {})
        self._parameter_cache[group][model_name] = {
            name: widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            for name, widget in widgets.items()
        }

    def _notify_parameter_change(self, *_):
        """Notify the host after an ordinary parameter edit."""
        if not self._loading_settings:
            self.parameters_changed.emit()

    def _update_parameters(self, group, selector_name, manager_name):
        """Rebuild controls and restore the selected model's previous entries."""
        self._remember_current_parameters(group)
        parameters_box = getattr(self, f"{group}_parameters_box")
        parameters_layout = getattr(self, f"{group}_parameters_layout")
        self._clear_layout(parameters_layout)

        parameter_widgets = {}
        setattr(self, f"{group}_parameters", parameter_widgets)
        self._clear_computed_fields(selector_name)

        model_name = getattr(self, selector_name).currentText()
        self._active_model_names[group] = model_name
        model = self._get_model(selector_name, manager_name)
        if model is None:
            parameters_box.hide()
            self._notify_parameter_change()
            return

        if group == "melt" and self.main_window is not None and self.manage_main_inputs:
            self.main_window.composition_panel.set_enabled_oxides(
                getattr(model, "required_oxides", [])
            )
            self.main_window.physical_panel.set_enabled_parameters(
                getattr(model, "required_melt_physical_parameters", [])
            )
            self._apply_fixed_melt_physical_parameters(model)

        self._setup_computed_fields(selector_name, model)
        self._refresh_selector_computed_fields(selector_name, model)

        definitions = getattr(model, "parameters", {}) or {}
        saved_values = self._parameter_cache[group].get(model_name, {})
        for name, info in definitions.items():
            default = info.get("default", 0.0)
            value = saved_values.get(name, default)
            if info.get("type", "float") == "choice":
                widget = QComboBox()
                options = [str(option) for option in info.get("options", [])]
                widget.addItems(options)
                if str(value) in options:
                    widget.setCurrentText(str(value))
                elif str(default) in options:
                    widget.setCurrentText(str(default))
                widget.currentTextChanged.connect(self._notify_parameter_change)
            else:
                widget = QLineEdit(str(value))
                widget.textChanged.connect(self._notify_parameter_change)

            parameters_layout.addRow(info.get("label", name), widget)
            parameter_widgets[name] = widget

        parameters_box.setTitle(f"{model_name} parameters")
        parameters_box.setVisible(bool(definitions))
        if group == "melt" and model_name == "Vetere et al. (2008)":
            self._setup_vetere_parameters()
        self._notify_parameter_change()

    # --- Fixed melt-model parameters --------------------------------

    def _apply_fixed_melt_physical_parameters(self, model):
        """Apply model-fixed physical values only to the managed MainWindow."""
        if (
            not self.manage_main_inputs
            or self.main_window is None
            or self._applying_fixed_parameters
        ):
            return
        physical_panel = getattr(self.main_window, "physical_panel", None)
        if physical_panel is None:
            return

        self._applying_fixed_parameters = True
        try:
            for parameter in self._fixed_melt_physical_parameters:
                widget_name = self.PHYSICAL_PARAMETER_WIDGETS.get(parameter)
                if widget_name is None:
                    continue
                widget = getattr(physical_panel, widget_name, None)
                if widget is not None:
                    widget.setEnabled(True)
                    widget.setReadOnly(False)
                    widget.setStyleSheet("")
                    widget.setToolTip("")

            physical_panel.apply_enabled_state()
            fixed_parameters = dict(
                getattr(model, "fixed_melt_physical_parameters", {}) or {}
            )
            self._fixed_melt_physical_parameters = set(fixed_parameters)
            current_sample = getattr(self.main_window, "current_sample", None)

            for parameter, value in fixed_parameters.items():
                widget_name = self.PHYSICAL_PARAMETER_WIDGETS.get(parameter)
                if widget_name is None:
                    continue
                widget = getattr(physical_panel, widget_name, None)
                if widget is None:
                    continue

                numeric_value = self._finite_number(value)
                text = f"{numeric_value:g}"
                if widget.text() != text:
                    widget.setText(text)
                widget.setReadOnly(True)
                widget.setEnabled(False)
                widget.setStyleSheet(self.FIXED_FIELD_STYLE)
                widget.setToolTip("Fixed by the selected melt-viscosity model.")

                sample_attribute = self.SAMPLE_PHYSICAL_ATTRIBUTES.get(parameter)
                if current_sample is not None and sample_attribute:
                    setattr(current_sample, sample_attribute, numeric_value)
                if parameter == "water":
                    composition_panel = getattr(self.main_window, "composition_panel", None)
                    if composition_panel is not None:
                        composition_panel.set_h2o(text)
        finally:
            self._applying_fixed_parameters = False

    def refresh_fixed_melt_physical_parameters(self):
        """Reapply fixed physical values to the MainWindow's selected sample."""
        if not self.manage_main_inputs or self.main_window is None:
            return
        model = self._get_model("model_selector", "melt_manager")
        if model is not None:
            self._apply_fixed_melt_physical_parameters(model)

    # --- Computed fields --------------------------------------------

    def _setup_computed_fields(self, selector_name, model):
        """Create read-only values calculated directly by a model."""
        definitions = getattr(model, "computed_fields", {}) or {}
        computed_box = self.computed_fields_boxes[selector_name]
        if not definitions:
            computed_box.hide()
            return

        layout = self.computed_fields_layouts[selector_name]
        widgets = {}
        for name, info in definitions.items():
            widget = QLabel()
            widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            widget.setStyleSheet(self.COMPUTED_FIELD_STYLE)
            font = widget.font()
            font.setBold(False)
            font.setWeight(QFont.Weight.Normal)
            widget.setFont(font)
            widget.setToolTip(info.get("tooltip", ""))
            layout.addWidget(widget)
            widgets[name] = widget
        self.computed_fields_widgets[selector_name] = widgets
        computed_box.show()

    def _clear_computed_fields(self, selector_name):
        """Remove computed fields belonging to a previous model."""
        self._clear_layout(self.computed_fields_layouts[selector_name])
        self.computed_fields_widgets[selector_name] = {}
        self.computed_fields_boxes[selector_name].hide()

    def refresh_computed_fields(self, *_):
        """Refresh fields from the local provider or the current GUI inputs."""
        if self._refreshing_computed_fields:
            return
        self._refreshing_computed_fields = True
        try:
            self.refresh_fixed_melt_physical_parameters()
            for selector_name, manager_name in self.selector_managers.items():
                model = self._get_model(selector_name, manager_name)
                if model is not None:
                    self._refresh_selector_computed_fields(selector_name, model)
        finally:
            self._refreshing_computed_fields = False

    def _refresh_selector_computed_fields(self, selector_name, model):
        """Refresh the selected model's read-only values."""
        definitions = getattr(model, "computed_fields", {}) or {}
        widgets = self.computed_fields_widgets.get(selector_name, {})
        if not definitions or not widgets:
            return

        try:
            sample = self._sample_from_interface()
        except Exception as error:
            self._set_computed_fields_unavailable(selector_name, definitions, str(error))
            return

        for name, info in definitions.items():
            widget = widgets.get(name)
            if widget is None:
                continue
            display_label = info.get("label", name)
            try:
                method_name = info.get("method")
                if not method_name:
                    raise ValueError("Computed-field method is missing.")
                method = getattr(model, method_name)
                decimals = int(info.get("decimals", 3))
                value = self._finite_number(method(deepcopy(sample)))
                widget.setText(f"{display_label} = {value:.{decimals}f}")
                widget.setToolTip(info.get("tooltip", ""))
            except Exception as error:
                widget.setText(f"{display_label} = —")
                widget.setToolTip(str(error))

    def _set_computed_fields_unavailable(self, selector_name, computed_fields, message):
        """Mark computed fields unavailable after an input error."""
        widgets = self.computed_fields_widgets.get(selector_name, {})
        for name, info in computed_fields.items():
            widget = widgets.get(name)
            if widget is not None:
                widget.setText(f"{info.get('label', name)} = —")
                widget.setToolTip(message)

    def _sample_from_interface(self):
        """Return an independent sample from the provider or current GUI."""
        if self.sample_provider is not None:
            sample = self.sample_provider()
            if sample is None:
                raise ValueError("No sample selected.")
            return deepcopy(sample)

        if self.main_window is None or self.main_window.current_sample is None:
            raise ValueError("No sample selected.")
        sample = deepcopy(self.main_window.current_sample)
        composition_panel = self.main_window.composition_panel
        for row, oxide in enumerate(composition_panel.OXIDES):
            item = composition_panel.table.item(row, 1)
            if item is None:
                raise ValueError(f"Missing value for {oxide}.")
            setattr(sample, oxide, self._finite_number(item.text()))

        physical_panel = self.main_window.physical_panel
        for parameter, widget_name in self.PHYSICAL_PARAMETER_WIDGETS.items():
            widget = getattr(physical_panel, widget_name)
            attribute = self.SAMPLE_PHYSICAL_ATTRIBUTES[parameter]
            setattr(sample, attribute, self._finite_number(widget.text()))
        return sample

    def _connect_computed_field_updates(self):
        """Connect MainWindow edits only when it supplies the computed sample."""
        if self.main_window is None or self.sample_provider is not None:
            return
        composition_panel = getattr(self.main_window, "composition_panel", None)
        physical_panel = getattr(self.main_window, "physical_panel", None)
        sample_panel = getattr(self.main_window, "sample_panel", None)

        if composition_panel is not None:
            composition_panel.table.itemChanged.connect(self.refresh_computed_fields)
        if physical_panel is not None:
            for widget_name in self.PHYSICAL_PARAMETER_WIDGETS.values():
                widget = getattr(physical_panel, widget_name, None)
                if widget is not None:
                    widget.textChanged.connect(self.refresh_computed_fields)
        if sample_panel is not None:
            sample_panel.samples_list.itemClicked.connect(
                lambda *_: QTimer.singleShot(0, self.refresh_computed_fields)
            )

    # --- Vetere iron-ratio control ----------------------------------

    def _vetere_widgets(self):
        """Return the legacy Vetere iron controls if the model exposes them."""
        widgets = tuple(
            self.melt_parameters.get(name) for name in ("fe2", "fe3", "fe_ratio")
        )
        return widgets if all(isinstance(widget, QLineEdit) for widget in widgets) else None

    def _setup_vetere_parameters(self):
        """Connect automatic calculation when the iron controls are available."""
        widgets = self._vetere_widgets()
        if widgets is None:
            return
        fe2_widget, fe3_widget, _ = widgets
        fe2_widget.textChanged.connect(self._update_vetere_ratio)
        fe3_widget.textChanged.connect(self._update_vetere_ratio)
        self._update_vetere_ratio()

    def _update_vetere_ratio(self):
        """Preserve automatic/manual behavior of the legacy iron-ratio control."""
        widgets = self._vetere_widgets()
        if widgets is None:
            return
        fe2_widget, fe3_widget, ratio_widget = widgets
        try:
            fe2 = self._finite_number(fe2_widget.text())
        except (TypeError, ValueError, OverflowError):
            fe2 = 0.0
        try:
            fe3 = self._finite_number(fe3_widget.text())
        except (TypeError, ValueError, OverflowError):
            fe3 = 0.0

        if fe2 > 0 and fe3 > 0:
            ratio_widget.setText(f"{fe2 / (fe2 + fe3):.4f}")
            ratio_widget.setReadOnly(True)
            ratio_widget.setStyleSheet("background-color: #eeeeee;color: #555555;")
            ratio_widget.setToolTip("Automatically calculated from Fe²⁺ and Fe³⁺.")
        else:
            ratio_widget.setReadOnly(False)
            ratio_widget.setStyleSheet("")
            ratio_widget.setToolTip(
                "Enter Fe²⁺/Feₜₒₜ manually when Fe²⁺ and Fe³⁺ are not both available."
            )

    # --- Parameter collection ---------------------------------------

    def get_parameters(self, show_errors=True):
        """Return the same six setting keys used by ViscosityEngine.calculate()."""
        settings = {}
        for group, (_title, selector_name, _manager_name) in self.MODEL_GROUPS.items():
            values = self.get_group_parameters(group, show_errors=show_errors)
            if values is None:
                return None
            settings[f"{group}_model"] = getattr(self, selector_name).currentText()
            settings[f"{group}_parameters"] = values
        return settings

    def get_group_parameters(self, group, show_errors=True):
        """Read only one group's parameter values, for a single-model dialog."""
        if group not in self.MODEL_GROUPS:
            raise ValueError(f"Unknown model group: {group}")
        values = self._read_parameters(
            getattr(self, f"{group}_parameters"), group, show_errors=show_errors
        )
        if values is not None:
            self._remember_current_parameters(group)
        return values

    @staticmethod
    def _finite_number(value):
        """Accept decimal commas and reject non-finite numerical inputs."""
        number = float(str(value).replace(",", "."))
        if not math.isfinite(number):
            raise ValueError("Value must be a finite number.")
        return number

    def _read_parameters(self, parameter_widgets, parameter_group, show_errors=True):
        """Read choices and finite numeric values from one model's controls."""
        parameters = {}
        for name, widget in parameter_widgets.items():
            if isinstance(widget, QComboBox):
                parameters[name] = widget.currentText()
                continue
            try:
                parameters[name] = self._finite_number(widget.text())
                if not widget.isReadOnly():
                    widget.setStyleSheet("")
            except (TypeError, ValueError, OverflowError):
                widget.setStyleSheet(self.INVALID_FIELD_STYLE)
                if show_errors:
                    widget.setFocus()
                    widget.selectAll()
                    QMessageBox.warning(
                        self, "Invalid model parameter",
                        f"Invalid {parameter_group} parameter: {name}.\n"
                        "Enter a finite numeric value.",
                    )
                return None
        return parameters

    # --- Layout cleanup ---------------------------------------------

    def _clear_layout(self, layout):
        """Delete every widget and nested layout from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget, child_layout = item.widget(), item.layout()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)