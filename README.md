# Basketball Analytics & Data Mining Dashboard

ASBD laboratory project using the original UCI Basketball Dataset archive and a local nine-page Streamlit application.

## Start on Windows

```powershell
cd D:\ASBD_lab
# Run this only if .venv does not exist.
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [the dashboard](http://localhost:8501). Run commands from the project root. Python 3.12 is the development environment.

## Data and methods

- **Source.** [UCI Basketball Dataset 587](https://archive.ics.uci.edu/dataset/587/basketball+dataset), DOI [10.24432/C56G77](https://doi.org/10.24432/C56G77), CC BY 4.0. The loader downloads the official ZIP into `data/raw/` on first use. Recording, participant code, activity, trial, and original member name are preserved.
- **Sensor interpretation.** Each file contains elapsed seconds, Cartesian acceleration, reported acceleration magnitude, and two angles. X-prefixed trials use g as specified by UCI and are converted to m/s². The mixed recordings have unverified units and no ground-truth activity timeline. The archive supplies no validated angular velocity vector.
- **Preprocessing.** Cleaning coerces numerics, rejects invalid required readings and timestamps, and resolves redundant or conflicting packets within each recording. The rolling median and residual audit work separately per recording. Pooled outlier flags are diagnostic. Fixed, non-overlapping one-second windows on labeled trials yield 24 features. Downloadable exports offer optional scaling and one-hot encoding.
- **Classifier.** The class-balanced decision tree and ANOVA feature selection fit on the training partition of a seeded stratified recording-group split. The first split containing all five classes on both sides is used. Participant overlap is possible, so scores estimate generalization to held-out trials from this collection. The default report uses depth 5, minimum leaf 4 and 12 selected features.
- **Mining.** Each window contributes six items. Pooled empirical tertiles are used for descriptive association mining. Apriori uses MLxtend. A-CLOSE independently searches frequent minimal generators, intersects matching transactions, and verifies closedness. It reports approximate rules over nested closed sets and exact implications from generators. The mining UI bounds thresholds to keep computations interactive. Full CSVs are available in the app. Defaults are 0.15 support, 0.55 confidence and 1.0 lift.
- **Reports.** Open Project Report and generate HTML or PDF. The HTML contains interactive, self-contained plots. The PDF uses static charts. Both include 12 visualizations, numerical inferences, and held-out model results. Report settings are reproducible defaults, independent of individual page controls.

To regenerate the default reports, processed datasets, CSV tables, and fitted model:

```powershell
.\.venv\Scripts\python.exe -m utils.export
```

The two mixed recordings differ in header metadata and contain identical sensor payloads. Both remain visible in the raw explorer and one copy is retained in the cleaned samples. The recording archive is read directly because the tabular UCI API does not supply the original per-trial files.

`app.py` and `pages/` contain Streamlit entry points, `utils/` contains the analysis modules, and `tests/` exercises the original dataset, algorithms, chart and report exports, and multipage views.


## Deploy

Push the source to GitHub. In Streamlit Community Cloud, select the repository, its branch, and `app.py`. Choose Python 3.12 to match local development.
