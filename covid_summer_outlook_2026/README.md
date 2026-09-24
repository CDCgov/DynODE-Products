# COVID-19 Scenario Modeling for the 2026 Summer Outlook

[Getting Started](#getting-started) | [Model](#model) | [Data](#data) | [Workflow Reference](#workflow-reference)


This codebase contains a scenario modeling workflow using
[DynODE](https://github.com/cdcgov/DynODE), CFA's ODE modeling
framework for infectious disease scenario analysis. The project in this folder is `covid_summer_outlook_2026`, which includes the SEIS compartmental
model we used to generate COVID-19 scenarios for the
[CDC 2026 COVID-19 Summer Outlook](https://www.cdc.gov/cfa-qualitative-assessments/php/data-research/2026covid-19outlook.html).

This codebase implements two scenarios: one assuming no emergence of a new immune-escape variant, and one assuming
emergence of a variant with moderate immune-escape properties. Projections
are produced by first fitting the model to observed state-level hospitalization
data, and then running the calibrated model under each scenario assumption to predict future dynamics.

To produce the results in the Outlook, we used a cloud-based, high-performance computing environment to fit and produce projections for 50 state-level models, which we then aggregate to produce US-level projections.  The code provided here is bundled with data for an example population (`EX`) that can be run on a typical computer.

## Getting Started

### Requirements

- Python 3.10 ([python.org](https://www.python.org/))
- uv 0.11.27 ([docs.astral.sh/uv/](https://docs.astral.sh/uv/))
- Git ([git-scm.org](https://git-scm.com/))
- R ([r-project.org](https://www.r-project.org/))
  - Pacman package ([available from CRAN](https://cran.r-project.org/web/packages/pacman/index.html))

### Installation

1) Clone the repository

```bash
git clone https://github.com/cdcgov/DynODE-Products.git
cd DynODE-Products
```

2) Change to the folder for the 2026 COVID Summer Outlook.

```bash
cd covid_summer_outlook_2026
```

## End-to-End Example

### Generating Compartment Initialization File (Optional)
Disease dynamics are sensitive to population age structure and the level and type of immunity present when a pathogen is introduced. We provide a precomputed initialization file that specifies initial model compartment sizes in the `data/initialization-data` folder that can be used for the model [calibration and projection](#calibrate-the-transmission-model-and-produce-scenario-projections) steps below.

If you wish to run the code that was used to produce the precomputed file, you may do so from this directory with the following command:

```bash
uv run python initializers/initialization_generation/run.py
```

Once that command is run, the folder `initializers/output` will be created if it does not already exist, and a json file for the states selected will be present. If you wish to use an initialization file you have generated, please copy or move it from `initializers/output` to `data/initialization-data`.

The current repository only has data for the example population, but if state data in the correct formats are added to `initializers/demographic-data` and `initializers/initialization_generation/data`, then additional states can be run by changing the `state_sample` input in the `initializers/initialization_generation/initialization_config.json` file. This json file is also where most changes to outputs would occur including start date, variants tracked, and more.

### Calibrate the Transmission Model and Produce Scenario Projections

From `covid_summer_outlook_2026/`, the calibration and projection workflows run in five steps. Fitting and projection each require their own unique job ID (`-j`), which is used in output file paths and should contain only alphanumeric characters, hyphens, and underscores.

1) Fit the model to observed data.

```bash
uv run python run_fitting.py -s EX -j your_fit_job_id
```

2) Visualize fit results.

```bash
uv run python postprocessing_scripts/1_plot_obs_vs_fit.py -j your_fit_job_id
```

3) Run projection under `intro` and `nointro` scenarios. See [Scenarios](#scenarios) for explanations. Set the previous fit argument `-pf` to the fitting job ID used in step 1.

```bash
uv run python run_projection.py -s EX -sc intro -j your_projection_job_id -pf your_fit_job_id
uv run python run_projection.py -s EX -sc nointro -j your_projection_job_id -pf your_fit_job_id
```

4) Postprocess projection data. (Note that we are using `pacman` to load and install R packages here. If your system is missing required packages, installation may take a long time while dependencies are compiled from source code. You might wish to consider [configuring your R environment up to install precompiled binary packages from Posit Package Manager](https://packagemanager.posit.co/client/#/repos/cran/setup), which avoids the compilation step).

```bash
Rscript postprocessing_scripts/collate_results.R -j your_projection_job_id
```

5) Visualize projection results.

```bash
Rscript postprocessing_scripts/visualize_projection.R -j your_projection_job_id
```

## Model

The COVID model used in this workflow is a compartmental deterministic SEIS transmission model
implemented in DynODE and
stratified by age, vaccination status, immune state, and SARS-CoV-2 variant.
Immunity is represented as a dynamic state that is boosted by infection or
vaccination and wanes over time, with protection levels that are variant-specific,
such that past infection with a more similar variant provides higher protection against
a challenging variant. Model-generated infections are then
translated to hospitalizations using age- and immunity-informed
infection-to-hospitalization rates with an assumed time lag from infection to observed
hospitalization. Parameters are fit to observed state-level data and then used
to generate scenario trajectories under alternative variant-emergence
assumptions.

For a more detailed model description, see the CDC
[Scenario Modeling Methods - COVID-19](https://www.cdc.gov/cfa-qualitative-assessments/php/about/scenario-models-methods.html)
document.

### Scenarios

This codebase includes two scenario definitions used for the 2026 summer
outlook workflow:

- `nointro`: no emergence of a new variant during the summer outlook period.
- `intro`: emergence of a variant with moderate immune-escape properties,
  implemented by introducing a new strain that is parameterized based on how the model has calibrated actual recently circulating strains.

This repository contains the code and workflow that produced the results in the
[2026 COVID-19 Summer Outlook](https://www.cdc.gov/cfa-qualitative-assessments/php/data-research/2026covid-19outlook.html)
(published June 1, 2026).



## Data

### Data Sources

| Data | Description | Source |
|---|---|---|
| Hospital Admissions | Hospitalization rates are drawn from National Healthcare Safety Network (NHSN) surveillance data, except between April 2024 and Dec 21, 2024 (most states) or May 2025 (Pennsylvania and Arizona), where gaps in NHSN reporting required substituting projected rates from internal National Syndromic Surveillance Program (NSSP) emergency department visit data. Wyoming used NHSN data for all time periods.| [CDC NHSN HRD Metrics](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr/about_data) |
| SARS-CoV-2 Variants | Variant data is grouped into a smaller number of modeled strains as described in [`variant_lookup_2602_5s.csv`](data/variant-data/variant_lookup_2602_5s.csv). | [CDC Variant Proportions](https://data.cdc.gov/Laboratory-Surveillance/SARS-CoV-2-Variant-Proportions/jr58-6ysp/about_data) |
| Contact Matrices | Age-structured social mixing patterns are used to model disease transmission. For more details, please refer to [Mistry, D., Litvinova, M., Pastore y Piontti, A. et al. "Inferring high-resolution human mixing patterns for disease modeling." Nat Commun 12, 323 (2021).](https://www.nature.com/articles/s41467-020-20544-y) |[Contact matrices](https://github.com/mobs-lab/mixing-patterns/tree/main)|
| Vaccination Coverage (Adults & Children) | Differentiable functions (splines and Hill functions) are fit to National Immunization Survey vaccination rates for each age group. | [Adult data](https://data.cdc.gov/Vaccinations/Weekly-Cumulative-COVID-19-Vaccination-Coverage-an/ksfb-ug5d/about_data)  [Pediatric data](https://data.cdc.gov/Child-Vaccinations/Weekly-Parental-Intent-for-Vaccination-and-Cumulat/ithv-4e9m/about_data) |

### Data processing and availability

Example population (`EX`) data is provided to make it possible to run the code as-provided.  To add additional states or populations, the corresponding hospitalization,
variant proportion, vaccination, and demographic inputs (e.g. from the data sources above) need to be added in `data/`, along with matching changes to the configuration files.

## Workflow Reference

### Repository Layout
- `run_fitting.py`: fitting entrypoint for state-level calibration
- `run_projection.py`: projection entrypoint for scenario runs
- `ode.py`: core ODE model dynamics used by workflows
- `pyproject.toml`: project package metadata (dynode installed from CDCgov/DynODE repository)
- **`configs/`**
  - `config_fit.py`: fit configuration
  - `config_proj.py`: projection configuration
  - `params.py`: parameter definitions
- **`data/`**: model inputs for demographics, hospitalization,
vaccination, and variant proportions
- **`data_scripts/`**: scripts used to prepare and transform raw input data into model-ready files
- **`initializers/`**: initialization logic for immune and strain-state setup used in the fit and projection workflows
- **`initializers/intialization_generation`**: code that generates initial infection and vaccination history used in fitting process
- **`postprocessing_scripts/`**: scripts to postprocess and visualize fit and projection outputs
- **`utils/`**: helper utilities used by fit and projection runs


### Fitting, Projection, and Postprocessing

| Stage | Script | Description | Inputs | Outputs |
|---|---|---|---|---|
| Fitting | `run_fitting.py` | Calibrates model parameters to observed state-level data | State hospitalization, variant, vaccination, demographics, model config | `output/<fit-jobid>/<state>/` |
| Visualize fit | `postprocessing_scripts/1_plot_obs_vs_fit.py` | Generates observed-vs-fitted plots | Fit outputs (`predictive_output.json`, `samples.json`) | `obs_vs_fitted_<job_id>.pdf` in `output/<fit-jobid>/` |
| Projection | `run_projection.py` | Generates forward scenario trajectories using previously fitted samples | Fit samples (`-pf`), scenario, model config | `output/<proj-jobid>/<state>/<scenario>/` |
| Postprocess projection | `postprocessing_scripts/collate_results.R` | Collates projection outputs across scenarios into analysis-ready files | Projection outputs under `output/<proj-jobid>/` | `projection_sim.parquet`, `obs.parquet` in `output/<proj-jobid>/` |
| Visualize projection | `postprocessing_scripts/visualize_projection.R` | Generates projection visualization from collated parquet files | `projection_sim.parquet`, `obs.parquet` | Projection figures in `output/<proj-jobid>/` |

### Model Initialization

Since the model is initialized on Feb 11, 2022, two years into the pandemic, the immune profile of the population at initialization is complicated.
We created an ad hoc pipeline to reconstruct the population’s immune history based on serological, hospitalization, vaccination, and viral-strain data.
This pipeline is available in the subfolder [`initializers/initialization_generation/`](initializers/initialization_generation/).
For demonstration purpose, code in the subfolder can be used to generate the initial state used in the example here.
For convenience, the output from this process for the example population is already provided in the [`data/initialization-data/`](data/initialization-data/) folder.

### Common arguments

- `-s, --state`: USPS state code (or `EX` to use the example population bundled here).
- `-j, --jobid`: job identifier used in output paths.
- `-sc, --scenario`: projection scenario (`intro` or `nointro`), projection only.
- `-pf, --prev-fit`: previous fitting job ID used as projection input, must match the fitting job ID.


------------------------------------------------------------------------------------

## Project Contributors
- Thomas Hladish, CDC/CFA
- Kok Ben Toh, CDC/CFA
- Michael Batista, CDC/CFA
- Elisha B. Are, CDC/CFA
- Laura Albrecht, CDC/CFA
- Nicholas Adams, CDC/CFA

## Contact Us

Email: CFAScenarios@cdc.gov
