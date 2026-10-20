#!/usr/bin/env python3
"""
test_prediction.py
==================

Stack-Tox: predict molecular toxicity from SMILES with the best-fold models
saved by `test_individual_models.py`.

This script is self-contained. It needs ONLY the saved `.joblib` model bundles
(no config.py, loader.py, generate_features.py or feature_list.json). Each
bundle stores the fitted model together with everything required to apply it:
the exact feature names/order, the TRAIN-fold medians used for imputation, the
class labels and the library versions it was trained with.

Pipeline
--------
  1. Read SMILES from a CSV file.
  2. Compute the RDKit descriptors used in training (same definitions as
     generate_features.py: 60 RDKit descriptors + FpDensityMorgan3 = 61 features).
  3. Align to the feature names stored in the model bundle, replace +/-inf by
     NaN and impute NaN with the TRAIN medians stored in the bundle (identical
     to the benchmark protocol).
  4. Predict with the saved best-fold model and write a CSV.

Decision rule
-------------
The benchmark scored every model with `model.predict` (the model's own decision
rule; no tuned threshold was ever used), so the label in the output is
`model.predict`. `Toxicity_Probability` is `predict_proba` for the toxic class.
An optional `--threshold` overrides the label with `probability >= threshold`.

Saved-model layout expected under --models-dir (written by test_individual_models.py):

    saved_models/ToxBits/<experiment_tag>/best_models/
        <tag>__<Model>__best_fold<k>.joblib

Usage
-----
    python test_prediction.py --input sample_smiles.csv --output toxicity_predictions.csv
    python test_prediction.py --input my.csv --output out.csv --training-set molwt --model lightgbm
    python test_prediction.py --input my.csv --output out.csv --model all     # 4 models side by side
    python test_prediction.py --input my.csv --output out.csv --model-file path/to/model.joblib

--input and --output are REQUIRED, must be given in full (no abbreviations such as --in / --out)
and must be .csv files.
"""

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, RDLogger, __version__ as RDKIT_VERSION
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_MODELS_DIR = "saved_models"
DEFAULT_TRAINING_SET = "all_three"
DEFAULT_MODEL = "ensemble"

# Training sets (balancing of the ToxBits training data) -> experiment tag on disk
TRAINING_SETS = {
    "all_three": "toxbits_balanced_all_three",
    "mollogp": "toxbits_balanced_mollogp",
    "molwt": "toxbits_balanced_molwt",
    "polarity": "toxbits_balanced_polarity",
}

# Models -> name used in the saved file names
MODELS = {
    "rf": "RF_TreeReg_Strict_L50",
    "lightgbm": "LightGBM_Default",
    "svm": "SVM_Linear",
    "ensemble": "Ensemble_Stack_RF_LGBM_SVMLinear",
}

# Same positive-class definition as in training/evaluation (toxic = positive class)
POSITIVE_LABELS = ["Toxic", "1", "True", "active", "true"]

# Descriptors exactly as in generate_features.py (60 via RDKit + FpDensityMorgan3 = 61)
NEEDED_DESCRIPTORS = [
    "SMR_VSA5", "SlogP_VSA5", "Chi2n", "Chi1n", "PEOE_VSA6",
    "NumRotatableBonds", "MolLogP", "MolMR", "VSA_EState7", "qed",
    "Chi3n", "Chi4n", "LabuteASA", "MolWt", "BalabanJ", "Kappa2",
    "BCUT2D_MRLOW", "Chi0n", "Kappa1", "Chi1", "Chi0", "ExactMolWt",
    "fr_ester", "Kappa3", "PEOE_VSA10",
    "fr_Al_OH_noTert", "fr_allylic_oxid", "Chi1v", "VSA_EState3",
    "Chi0v", "fr_ether", "fr_Al_OH", "fr_unbrch_alkane",
    "HeavyAtomMolWt", "BCUT2D_MWHI", "EState_VSA5", "BCUT2D_CHGHI",
    "fr_NH0", "MaxAbsEStateIndex", "MaxAbsPartialCharge",
    "MinPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge",
    "SlogP_VSA2", "TPSA", "SMR_VSA1", "EState_VSA1", "EState_VSA2",
    "SMR_VSA10", "BCUT2D_MRHI", "BCUT2D_LOGPLOW", "PEOE_VSA1",
    "EState_VSA4", "PEOE_VSA8", "SlogP_VSA3", "BCUT2D_MWLOW",
    "PEOE_VSA7", "BCUT2D_CHGLO", "EState_VSA8", "EState_VSA3",
]
MORGAN_COLUMN = "FpDensityMorgan3"
_MORGAN_GEN = GetMorganGenerator(radius=3, fpSize=2048)
_RDKIT_NAMES = {name for name, _ in Descriptors._descList}


# ==============================================================================
# DESCRIPTORS
# ==============================================================================

def compute_descriptors(smiles_list, feature_names):
    """
    Computes the descriptors in `feature_names` for every SMILES.

    Returns (DataFrame indexed 0..n-1, boolean array `valid`). Rows whose SMILES
    cannot be parsed are all-NaN and flagged invalid (they are not predicted).
    Descriptors are computed exactly as in generate_features.py. A feature that
    neither this script nor RDKit can compute stays NaN and is imputed with the
    stored TRAIN median (a warning is printed).
    """
    rd_names = [f for f in feature_names if f != MORGAN_COLUMN and f in _RDKIT_NAMES]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(rd_names) if rd_names else None
    unknown = [f for f in feature_names if f != MORGAN_COLUMN and f not in _RDKIT_NAMES]
    if unknown:
        print(f"[Warn] {len(unknown)} model feature(s) cannot be computed with RDKit and will be "
              f"imputed with the TRAIN median: {unknown}")

    rows, valid = [], np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None
        if mol is None:
            rows.append({f: np.nan for f in feature_names})
            continue
        try:
            row = dict(zip(rd_names, calc.CalcDescriptors(mol))) if calc else {}
            if MORGAN_COLUMN in feature_names:
                row[MORGAN_COLUMN] = _MORGAN_GEN.GetFingerprint(mol).GetNumOnBits() / 2048.0
            rows.append({f: row.get(f, np.nan) for f in feature_names})
            valid[i] = True
        except Exception as exc:  # keep the row, flag it
            print(f"[Warn] descriptor calculation failed for '{str(smi)[:40]}': {exc}")
            rows.append({f: np.nan for f in feature_names})
    return pd.DataFrame(rows, columns=list(feature_names)), valid


# ==============================================================================
# MODEL BUNDLES
# ==============================================================================

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_best_model_file(models_dir, set_key, model_key):
    """Locates <models_dir>/ToxBits/<tag>/best_models/<tag>__<Model>__best_fold*.joblib."""
    tag, model_name = TRAINING_SETS[set_key], MODELS[model_key]
    folder = Path(models_dir) / "ToxBits" / tag / "best_models"
    hits = sorted(folder.glob(f"{tag}__{model_name}__best_fold*.joblib"))
    if not hits:
        raise FileNotFoundError(
            f"No best-fold model found for training set '{set_key}', model '{model_key}' in {folder}")
    if len(hits) > 1:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"[Warn] {len(hits)} candidate files in {folder}; using the newest: {hits[0].name}")
    return hits[0]


def load_bundle(path):
    """Loads a saved bundle (dict: model, feature_names, train_medians, metadata)."""
    bundle = joblib.load(path)
    missing = [k for k in ("model", "feature_names", "train_medians") if k not in bundle]
    if missing:
        raise ValueError(f"{path} is not a valid Stack-Tox bundle (missing keys: {missing})")
    return bundle


def check_versions(bundle, label):
    """Warns when the runtime libraries differ from those the model was trained with."""
    saved = bundle.get("library_versions", {}) or {}
    current = {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
               "joblib": joblib.__version__}
    try:
        import lightgbm
        current["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
    diff = {k: (saved[k], v) for k, v in current.items() if k in saved and saved[k] != v}
    if diff:
        msg = ", ".join(f"{k}: trained {a} / installed {b}" for k, (a, b) in diff.items())
        print(f"[Warn] {label}: library version mismatch ({msg}). "
              f"Install the versions in requirements.txt for exact reproducibility.")


def toxic_label_to_binary(values):
    return pd.Series(np.asarray(values)).astype(str).isin(POSITIVE_LABELS).to_numpy().astype(np.int8)


def toxic_probability(model, X):
    """Probability of the toxic class (same convention as the benchmark)."""
    classes = [str(c) for c in model.classes_]
    pos = [i for i, c in enumerate(classes) if c in POSITIVE_LABELS]
    if not pos:
        raise ValueError(f"No toxic class found among model classes {classes}")
    return model.predict_proba(X)[:, pos[0]]


def predict_with_bundle(bundle, descriptors, valid, threshold=None):
    """
    Applies one bundle to a descriptor table. Returns (probability, label) arrays
    of length n (NaN / 'Invalid_SMILES' for rows that could not be parsed).
    """
    feats = list(bundle["feature_names"])
    medians = pd.Series(bundle["train_medians"], dtype=float)

    X = descriptors.reindex(columns=feats)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians).fillna(0.0)

    prob = np.full(len(X), np.nan)
    label = np.full(len(X), "Invalid_SMILES", dtype=object)
    if valid.any():
        Xv = X.loc[valid]
        model = bundle["model"]
        prob[valid] = toxic_probability(model, Xv)
        if threshold is None:
            toxic = toxic_label_to_binary(model.predict(Xv)) == 1     # model's own decision rule
        else:
            toxic = prob[valid] >= threshold
        label[valid] = np.where(toxic, "Toxic", "Non-Toxic")
    return prob, label


# ==============================================================================
# INPUT / OUTPUT
# ==============================================================================

def csv_filename(value):
    """argparse type: only .csv file names are accepted."""
    if not value.lower().endswith(".csv"):
        raise argparse.ArgumentTypeError(f"'{value}' is not a .csv file name (only .csv files are accepted)")
    return value


def read_input(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    col = next((c for c in df.columns if c.strip().lower() in ("smiles", "canonical_smiles")), None)
    if col is None:
        raise ValueError(f"No SMILES column in {path}. Columns found: {list(df.columns)}")
    out = pd.DataFrame({"SMILES": df[col].astype(str).str.strip().replace({"nan": "", "None": ""})})
    id_col = next((c for c in df.columns if c.strip().lower() == "id"), None)
    out.insert(0, "ID", df[id_col].astype(str).to_numpy() if id_col
               else [f"Molecule_{i + 1}" for i in range(len(df))])
    return out


def resolve_models(args):
    """Returns a list of (set_key, model_key, path)."""
    if args.model_file:
        return [("custom", "custom", Path(args.model_file))]
    sets = list(TRAINING_SETS) if args.training_set == "all" else [args.training_set]
    models = list(MODELS) if args.model == "all" else [args.model]
    return [(s, m, find_best_model_file(args.models_dir, s, m)) for s in sets for m in models]


def describe_bundle(bundle, path):
    vm = bundle.get("val_metrics") or {}
    auc = (vm.get("auc") or {}).get("value")
    return {
        "file": str(path),
        "sha256": sha256_of(path),
        "model_name": bundle.get("model_name"),
        "experiment_tag": bundle.get("experiment_tag"),
        "train_balance": bundle.get("train_balance"),
        "fold": bundle.get("fold"),
        "n_folds": bundle.get("n_folds"),
        "selection_metric": bundle.get("selection_metric"),
        "n_features": len(bundle["feature_names"]),
        "n_train": bundle.get("n_train"),
        "validation_auc": auc,
        "saved_at": bundle.get("saved_at"),
        "trained_with": bundle.get("library_versions"),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Predict toxicity from SMILES with the best-fold Stack-Tox models (.joblib only).",
        allow_abbrev=False)   # full option names only
    p.add_argument("--input", required=True, type=csv_filename,
                   help="REQUIRED. Input .csv file with a SMILES column (and optional ID column).")
    p.add_argument("--output", required=True, type=csv_filename,
                   help="REQUIRED. Output .csv file for the predictions.")
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                   help=f"Root of the saved models (default: ./{DEFAULT_MODELS_DIR}).")
    p.add_argument("--training-set", default=DEFAULT_TRAINING_SET, choices=list(TRAINING_SETS) + ["all"],
                   help=f"Balancing of the training set the model was trained on (default: {DEFAULT_TRAINING_SET}).")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS) + ["all"],
                   help=f"Which learner to use (default: {DEFAULT_MODEL}).")
    p.add_argument("--model-file", default=None,
                   help="Path to one specific .joblib bundle (overrides --training-set / --model).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Optional: label = probability >= threshold. Default: the model's own predict().")
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)   # sklearn feature-name chatter

    if Path(args.input).resolve() == Path(args.output).resolve():
        p.error("--output must be a different file from --input (the input would be overwritten)")

    mols = read_input(args.input)
    selected = resolve_models(args)
    print(f"Input: {args.input}  ({len(mols)} molecules)")

    # Load every requested bundle; compute the union of required descriptors once.
    bundles = []
    for set_key, model_key, path in selected:
        b = load_bundle(path)
        check_versions(b, path.name)
        bundles.append((set_key, model_key, path, b))
    all_features = list(dict.fromkeys(f for *_, b in bundles for f in b["feature_names"]))
    descriptors, valid = compute_descriptors(mols["SMILES"].tolist(), all_features)

    n_invalid = int((~valid).sum())
    if n_invalid:
        print(f"[Warn] {n_invalid} molecule(s) have an empty/invalid SMILES and are reported as 'Invalid_SMILES'.")

    out = mols.copy()
    info = []
    for set_key, model_key, path, b in bundles:
        try:
            prob, label = predict_with_bundle(b, descriptors, valid, args.threshold)
        except AttributeError as exc:
            tw = b.get("library_versions", {})
            raise RuntimeError(
                f"{path.name} could not be applied ({exc}). This almost always means the installed "
                f"scikit-learn/numpy differ from the versions the model was trained with "
                f"(trained: scikit-learn {tw.get('sklearn')}, numpy {tw.get('numpy')}, "
                f"lightgbm {tw.get('lightgbm')}; installed: scikit-learn {sklearn.__version__}, "
                f"numpy {np.__version__}). Create a fresh environment with the versions in "
                f"requirements.txt (Python 3.11 or 3.12).") from exc
        if len(bundles) == 1:
            out["Toxicity_Probability"], out["Prediction"] = prob, label
        else:
            suffix = f"{set_key}__{model_key}" if args.model_file is None else path.stem
            out[f"Toxicity_Probability__{suffix}"] = prob
            out[f"Prediction__{suffix}"] = label
        d = describe_bundle(b, path)
        info.append(d)
        auc = d["validation_auc"]
        print(f"Model: {d['model_name']} | train set: {d['train_balance']} | best fold {d['fold']}/{d['n_folds']} "
              f"| features: {d['n_features']} | val AUC: {'n/a' if auc is None else f'{auc:.4f}'} | {path.name}")

    out.to_csv(args.output, index=False)

    # Console preview (single-model mode)
    if len(bundles) == 1:
        print("\nToxicity predictions")
        print("=" * 80)
        for _, r in out.iterrows():
            pr = "   n/a " if pd.isna(r["Toxicity_Probability"]) else f"{r['Toxicity_Probability']:.4f}"
            print(f"{str(r['ID']):<20s} | P(toxic) = {pr} | {r['Prediction']}")
        print("=" * 80)
    print(f"\nPredictions saved to: {args.output}")

    # Provenance sidecar (model files, hashes, library versions)
    meta_path = Path(args.output).with_name(Path(args.output).stem + "_run_info.json")
    meta = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(args.input),
        "n_molecules": int(len(mols)),
        "n_invalid_smiles": n_invalid,
        "decision_rule": ("probability >= %g" % args.threshold) if args.threshold is not None
                         else "model.predict (model's own decision rule, as in the benchmark)",
        "runtime": {"python": sys.version.split()[0], "rdkit": RDKIT_VERSION, "sklearn": sklearn.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "models": info,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"Run information saved to: {meta_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)