# Stack-Tox: Toxicity Prediction from SMILES

Predicts molecular toxicity (toxic = positive class) from SMILES strings using the
best-fold models saved by `test_individual_models.py`. The prediction script needs
**only the saved `.joblib` model files**. Each file is a self-contained bundle holding
the fitted model, the exact feature names/order, the TRAIN-fold medians used for
imputation, class labels and the library versions used for training.

## Installation

Python 3.11 or 3.12 (the models were trained with scikit-learn 1.6.1 / NumPy 2.0.2, which do not support Python 3.13+).

```bash
pip install -r requirements.txt
```

Use the pinned versions for exact reproducibility. The script warns if the installed
scikit-learn / LightGBM / NumPy / pandas differ from those the model was trained with.

## Run with Docker (recommended, no Python setup)

The image reproduces the training environment (Python 3.12, scikit-learn 1.6.1, NumPy 2.0.2).
Models and data are **not** baked into the image: your current folder is mounted at `/data`.

Folder layout (same as above): `test_prediction.py`, `requirements.txt`, `Dockerfile`,
`sample_smiles.csv` and `saved_models/` in one folder.

```bash
# build once (from that folder)
docker build -t stacktox .
```

Run (the current folder is mounted at `/data`; use the line for your shell):

| Shell | Command |
|---|---|
| Windows Command Prompt | `docker run --rm -v "%cd%:/data" stacktox --input sample_smiles.csv --output toxicity_predictions.csv` |
| Windows PowerShell | `docker run --rm -v "${PWD}:/data" stacktox --input sample_smiles.csv --output toxicity_predictions.csv` |
| Linux / macOS | `docker run --rm -v "$(pwd):/data" stacktox --input sample_smiles.csv --output toxicity_predictions.csv` |

`--input` and `--output` are required; further options go after them, for example (Command Prompt):

```bat
docker run --rm -v "%cd%:/data" stacktox --input my_molecules.csv --output my_predictions.csv --model all
```

Notes:
- `${PWD}` only works in PowerShell. In Command Prompt it fails with
  `"${PWD}" includes invalid characters for a local volume name`; use `%cd%` instead.
- Output CSV and `_run_info.json` appear in the same folder.
- To use the exact patch release the models were trained with:
  `docker build --build-arg PYTHON_VERSION=3.12.13 -t stacktox .`

## Required files

```
project/
├── test_prediction.py
├── requirements.txt
├── sample_smiles.csv                 # your input
└── saved_models/
    └── ToxBits/
        ├── toxbits_balanced_all_three/best_models/*.joblib
        ├── toxbits_balanced_mollogp/best_models/*.joblib
        ├── toxbits_balanced_molwt/best_models/*.joblib
        └── toxbits_balanced_polarity/best_models/*.joblib
```

Only the `best_models/` folders are used (one best-fold model per learner and training
set, selected on the validation set only).

## Input

A CSV with a `SMILES` column and, optionally, an `ID` column:

```csv
ID,SMILES
Molecule_1,CCO
Molecule_2,c1ccccc1
```

## Usage

```bash
python test_prediction.py --input sample_smiles.csv --output toxicity_predictions.csv   # ensemble, all_three
python test_prediction.py --input my_molecules.csv --output my_predictions.csv
python test_prediction.py --input my_molecules.csv --output out.csv --training-set molwt --model lightgbm
python test_prediction.py --input my_molecules.csv --output out.csv --model all   # 4 learners side by side
python test_prediction.py --input my_molecules.csv --output out.csv --training-set all --model all   # all 16 models
python test_prediction.py --input my_molecules.csv --output out.csv --model-file path/to/model.joblib
```

`--input` and `--output` are mandatory, must be `.csv` files, and must be written in full (abbreviated
options such as `--in` are rejected).

| Option | Values | Default |
|---|---|---|
| `--input` | **required**, input `.csv` with a SMILES column | none |
| `--output` | **required**, output `.csv` file | none |
| `--models-dir` | root of saved models | `saved_models` |
| `--training-set` | `all_three`, `mollogp`, `molwt`, `polarity`, `all` | `all_three` |
| `--model` | `rf`, `lightgbm`, `svm`, `ensemble`, `all` | `ensemble` |
| `--model-file` | path to a `.joblib` bundle | none |
| `--threshold` | probability cut-off | none (model's own rule) |

Learners: `rf` = RF_TreeReg_Strict_L50, `lightgbm` = LightGBM_Default,
`svm` = SVM_Linear, `ensemble` = Ensemble_Stack_RF_LGBM_SVMLinear
(stack of the three, logistic-regression meta-learner).

## Output

`toxicity_predictions.csv`:

```csv
ID,SMILES,Toxicity_Probability,Prediction
Molecule_1,CCO,0.0774,Non-Toxic
Molecule_2,c1ccccc1,0.7359,Toxic
Molecule_3,notasmiles,,Invalid_SMILES
```

- `Toxicity_Probability`: predicted probability of the toxic class.
- `Prediction`: `Toxic` / `Non-Toxic`, or `Invalid_SMILES` for empty or unparseable SMILES
  (these rows are kept in place and are not predicted).
- With `--model all` / `--training-set all`, one `Toxicity_Probability__<set>__<model>` and
  `Prediction__<set>__<model>` column pair is written per model.
- A provenance file `<output>_run_info.json` is written next to the CSV (model files with
  SHA-256, fold, library versions, decision rule).

## Method

1. **Descriptors**: 61 features. RDKit descriptors (60) plus `FpDensityMorgan3`
   (radius 3, 2048 bits), computed as in `generate_features.py`.
2. **Preprocessing**: columns are aligned to the model's stored feature list; +/-inf become
   NaN; NaN are imputed with the TRAIN-fold medians stored in the model file.
   No collinearity filter or ablation is applied (all features were used in training).
3. **Decision rule**: identical to the benchmark. The label is `model.predict` (each model's
   own decision rule; no tuned threshold was used in training). `--threshold` is optional and
   overrides this with `probability >= threshold`. Note that for `svm`, `predict` uses the SVM
   decision function, so it can differ from a 0.5 cut-off on the Platt-scaled probability.

## Troubleshooting

- `No best-fold model found ...`: check `--models-dir` and that
  `saved_models/ToxBits/<tag>/best_models/` contains the `.joblib` files.
- Library version warning: install the versions in `requirements.txt`.
- Invalid molecules appear as `Invalid_SMILES` with an empty probability.