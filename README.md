# MagmaViscoLab 1.0

MagmaViscoLab (MVL) is a Python desktop application for calculating and visualizing magma and lava viscosity. It combines selectable melt-viscosity models with crystal and vesicle corrections, using chemical compositions and physical conditions supplied through an Excel workbook or edited in the graphical interface.

## Features

- 22 model implementations: 10 melt models, 8 crystal corrections and 4 vesicle corrections.
- Single-sample and batch calculations.
- Melt, crystal-bearing, vesicle-bearing and three-phase viscosity outputs.
- Two-dimensional curves and three-dimensional surfaces, including multiple samples.
- Same-sample comparisons between models.
- Excel input and result export; figure export in PNG, PDF and SVG formats.
- Model-dependent parameters, input controls and computed fields.

## Install and run from source

The instructions below use Python 3.12. Install Python before continuing. Keep `src`, `resources` and `examples` together in the project folder.

### Windows

Extract the downloaded source archive. Open PowerShell in the `MagmaViscoLab` folder containing this README, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/main.py
```

These commands call the virtual environment's Python directly, so activating the environment is unnecessary. If your installed Python 3.12 is available as `python` instead of `py`, use `python -m venv .venv` for the first command.

To start MVL again, open PowerShell in the same project folder and run:

```powershell
.\.venv\Scripts\python.exe src/main.py
```

### Linux and macOS

With Python 3.12 and its virtual-environment support installed, run these commands from the project folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/main.py
```

An interactive graphical session and the operating-system libraries required by Qt are needed. The preparation checks described below used Linux with Qt's offscreen platform; interactive Linux and macOS operation has not been verified as part of these checks.

**Always launch from the project folder**, because several image paths are relative to the working directory.

## First calculation

1. Click **Open Excel** and select `examples/Default_compositions.xlsx`.
2. Select a sample from the list.
3. Select a melt model, a crystal correction and a vesicle correction. For a first run with the supplied workbook, use **Giordano et al. (2008)**, **Roscoe (1952)** and **Mackenzie (1950)**.
4. Review the composition, temperature, water, crystal and vesicle contents, and model-specific parameters.
5. Click **Calculate** for the selected sample or **Calculate all samples** for the workbook.
6. Use **Export Excel** to save results, or open **2D Plot**, **3D Plot** or **Compare models** to explore the calculations.

The **Normalize** button rescales the displayed composition to 100 wt%. Use it only when this treatment is appropriate for your input data.

## Excel input

Use `examples/Default_compositions.xlsx` as the input template. The application reads the first worksheet, with one sample per row and headers in the first row. Keep the exact column names below and use unique, non-empty sample names.

| Columns | Meaning and units |
| --- | --- |
| `Sample` | Sample identifier |
| `SiO2`, `TiO2`, `Al2O3`, `FeO`, `Fe2O3`, `MnO`, `MgO`, `CaO`, `Na2O`, `K2O`, `P2O5`, `Cr2O3` | Oxide concentrations, wt% |
| `H2O` | Water content, wt% |
| `F2O_1` | Fluorine-related composition field in the model's F₂O₋₁ convention, wt% |
| `Temperature` | Temperature, °C |
| `Crystals` | Crystal content, vol% |
| `Vesicles` | Vesicle content, vol% |

Enter numeric cells without unit suffixes. Enter 30 for 30 vol%, rather than 0.30 or an Excel percentage-formatted value. Use explicit numeric values for composition and physical fields; select models appropriate for the sample and their calibration domains. A model may use only a subset of the supplied oxides or impose fixed physical conditions.

The example workbook contains BRO1 (Etna), MONT153 (Soufriere Hill), M103 (Mayotte) and SMI1 (Santorini), using the identifiers stored in the supplied file.

## Results and units

| Output key | Quantity |
| --- | --- |
| `log10_eta_m` | Base-10 logarithm of melt viscosity in Pa·s |
| `log10_eta_mc` | Base-10 logarithm of melt + crystals viscosity in Pa·s |
| `log10_eta_mb` | Base-10 logarithm of melt + vesicles viscosity in Pa·s |
| `log10_eta_mcb` | Base-10 logarithm of three-phase viscosity in Pa·s |
| `eta_r_c` | Dimensionless crystal relative-viscosity factor |
| `eta_r_b` | Dimensionless vesicle relative-viscosity factor |

The engine combines the factors in linear viscosity space:

`eta_mcb = eta_m * eta_r_c * eta_r_b`

Equivalently, their logarithms are added. Recover viscosity in Pa·s from a logarithmic output with `10 ** log10_eta`. Strain-rate inputs, where present, use s⁻¹.

## Implemented models

| Group | Implementations |
| --- | --- |
| Melt viscosity | Shaw (1972); Hui & Zhang (2007); Giordano et al. (2008); Vetere et al. (2008); Di Genova et al. (2023); Valdivia et al. (2023); Fanesi et al. (2025); Valdivia et al. (2025); Dominijanni et al. (2026); Stopponi et al. (2026) |
| Crystal corrections | Roscoe (1952); Krieger & Dougherty (1959); Caricchi et al. (2007); Costa et al. (2009); Mueller et al. (2011); Vona et al. (2011); Faroughi & Huber (2015); Liu et al. (2017) |
| Vesicle corrections | Mackenzie (1950); Pal (2003); Mader et al. (2013); Vona et al. (2016) |

Refer to the original publications for assumptions, calibration ranges and parameter definitions. Model selection and valid numerical results do not by themselves establish that a model is applicable to a particular magma or flow regime.

## Project organization

| Path | Contents |
| --- | --- |
| `src/main.py` | Application entry point |
| `src/core/` | Sample objects, sample management and viscosity engine |
| `src/data_io/` | Excel reader and data files |
| `src/gui/` | Main window, panels, plotting and comparison windows |
| `src/models/melt/` | Melt-viscosity implementations |
| `src/models/corrections/` | Crystal and vesicle corrections and model managers |
| `resources/` | Logos and interface icons |
| `examples/` | Example input workbook |
| `requirements.txt` | Direct Python runtime dependencies |

## Dependencies

MVL uses PySide6, Matplotlib, NumPy, pandas and openpyxl. `requirements.txt` lists direct runtime dependencies without version pins; it is not a reproducible environment lock file. For a research calculation, record the MVL release or commit, dependency versions, input workbook, model selections and parameters.

## Preparation checks

Source-package smoke checks used Python 3.12.14 on Linux with PySide6 6.11.2, Matplotlib 3.10.8, NumPy 2.3.5, pandas 2.2.3 and openpyxl 3.1.5. They covered loading all 22 models, importing the four example samples, calculating all six outputs with the Giordano–Roscoe–Mackenzie combination, and opening the main, 2D, 3D and comparison windows using Qt's offscreen platform. Logo loading was also checked.

These are functional smoke checks, not an independent validation of all scientific equations or an interactive test of every plotting and export operation. The package preparation corrected logo filename capitalization in `main_window.py` and `plot_window.py`, and the project-root lookup for the 2D window icon. Model equations were not changed.

## Acknowledgements

MagmaViscoLab was developed within the KLARA – Kinetics of Lava Flows Crystallization research project, funded under the FIS2 – Fondo Italiano per la Scienza programme (Project No. FIS-03129), at the Department of Biological, Geological and Environmental Sciences, University of Catania.

## Citation and license

When reporting calculations, identify the MVL version or commit and cite the original publications for the selected models. A software citation and archival DOI have not yet been added to this source package.

The MagmaViscoLab source code in `src/` is licensed under the **GNU General Public License, version 3 only** (`GPL-3.0-only`). See [LICENSE](LICENSE) for the complete license text.

MagmaViscoLab is free software: you can redistribute it and/or modify it under the terms of version 3 of the GNU General Public License as published by the Free Software Foundation.

MagmaViscoLab is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Third-party dependencies retain their own licenses. This source-code license statement does not assign a license to the accompanying documentation, logos, icons or example datasets.
