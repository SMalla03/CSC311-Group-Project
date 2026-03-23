from __future__ import annotations

"""Train and run a scikit-learn random forest for painting prediction.

Usage:
    python models/random_forest/rf_model.py --train
    python models/random_forest/rf_model.py --predict data/training_data_202601.csv
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import (  # noqa: E402
    apply_feature_subset,
    classification_report,
    dense_load,
    load_feature_names,
    load_preprocessor,
    preprocess_frame,
    read_rows,
    select_feature_indices,
)


DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "models" / "random_forest"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "rf_model.joblib"
DEFAULT_INFERENCE_PARAMS_PATH = DEFAULT_OUTPUT_DIR / "rf_inference_params.pkl"
DEFAULT_FEATURE_SET = "full"
DEFAULT_N_ESTIMATORS = 50
DEFAULT_MAX_DEPTH: int | None = 5
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_MAX_FEATURES = "sqrt"
DEFAULT_TOP_FEATURES = 25
DEFAULT_N_JOBS = 1
RUN_LOG_PATH = DEFAULT_OUTPUT_DIR / "rf_training_log.csv"


def load_processed_splits(processed_dir: Path, feature_set: str) -> tuple[np.ndarray, ...]:
    feature_names_all = load_feature_names(processed_dir)
    feature_indices = select_feature_indices(feature_names_all, feature_set)
    feature_names = [feature_names_all[index] for index in feature_indices.tolist()]
    x_train = apply_feature_subset(dense_load(processed_dir / "train_X.npz"), feature_indices)
    y_train = np.load(processed_dir / "train_y.npy")
    x_val = apply_feature_subset(dense_load(processed_dir / "val_X.npz"), feature_indices)
    y_val = np.load(processed_dir / "val_y.npy")
    x_test = apply_feature_subset(dense_load(processed_dir / "test_X.npz"), feature_indices)
    y_test = np.load(processed_dir / "test_y.npy")
    return x_train, y_train, x_val, y_val, x_test, y_test, feature_names


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_training_log(path: Path, run_summary: dict[str, object], metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature_set",
        "selected_feature_count",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "seed",
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
    ]
    row = {**run_summary, **metrics}
    frame = pd.DataFrame([{column: row[column] for column in columns}])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def summarize_feature_groups(feature_names: list[str], importances: np.ndarray) -> list[tuple[str, float]]:
    grouped: dict[str, float] = {}
    for name, importance in zip(feature_names, importances):
        if name.startswith("text:"):
            group = "text"
        elif name.startswith("season="):
            group = "season"
        elif name.startswith("room="):
            group = "room"
        elif name.startswith("view_with="):
            group = "view_with"
        elif name.startswith("feel_"):
            group = "feel_numeric"
        elif "=" in name:
            group = name.split("=", 1)[0]
        else:
            group = name.split("__", 1)[0]
        grouped[group] = grouped.get(group, 0.0) + float(importance)
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def top_feature_rows(feature_names: list[str], importances: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    indices = np.argsort(importances)[::-1][:top_k]
    return [(feature_names[index], float(importances[index])) for index in indices]


def write_training_report(path: Path, run_summary: dict[str, object], split_reports: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Random Forest Training Report",
        "",
        "## Run Summary",
        "",
        f"- Feature subset: `{run_summary['feature_set']}`",
        f"- Selected feature count: `{run_summary['selected_feature_count']}`",
        f"- Number of trees: `{run_summary['n_estimators']}`",
        f"- Max depth: `{run_summary['max_depth']}`",
        f"- Min samples split: `{run_summary['min_samples_split']}`",
        f"- Min samples leaf: `{run_summary['min_samples_leaf']}`",
        f"- Max features per split: `{run_summary['max_features']}`",
        f"- Seed: `{run_summary['seed']}`",
        "",
    ]
    for split_name, report in split_reports.items():
        lines.extend(
            [
                f"## {split_name.title()} Metrics",
                "",
                f"- Accuracy: `{report['accuracy']:.4f}`",
                f"- Macro F1: `{report['macro_avg']['f1']:.4f}`",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["per_class"]:
            lines.append(
                f"| {row['label']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
            )
        lines.extend(["", "Confusion matrix (`rows=true`, `cols=predicted`):", ""])
        header = " | ".join(["true \\ pred", *[row["label"] for row in report["per_class"]]])
        divider = " | ".join(["---"] * (len(report["per_class"]) + 1))
        lines.append(f"| {header} |")
        lines.append(f"| {divider} |")
        for label, row_values in zip([row["label"] for row in report["per_class"]], report["confusion_matrix"]):
            lines.append(f"| {label} | {' | '.join(str(value) for value in row_values)} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_summary(path: Path, feature_names: list[str], importances: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Random Forest Feature Summary",
        "",
        "This file summarizes the learned feature importance values from the random forest.",
        "",
        "## Top Individual Features",
        "",
        "| Feature | Importance |",
        "| --- | ---: |",
    ]
    for feature_name, importance in top_feature_rows(feature_names, importances, DEFAULT_TOP_FEATURES):
        lines.append(f"| {feature_name} | {importance:.6f} |")
    lines.extend(["", "## Importance By Feature Group", "", "| Group | Total Importance |", "| --- | ---: |"])
    for group_name, importance in summarize_feature_groups(feature_names, importances):
        lines.append(f"| {group_name} | {importance:.6f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_performance_summary(path: Path, run_summary: dict[str, object], split_reports: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test_report = split_reports["test"]
    best_test_class = max(test_report["per_class"], key=lambda row: row["f1"])
    weakest_test_class = min(test_report["per_class"], key=lambda row: row["f1"])
    lines = [
        "# Random Forest Performance Summary",
        "",
        (
            f"The random forest was trained using the `{run_summary['feature_set']}` feature set with "
            f"`{run_summary['n_estimators']}` trees. On the latest run, it reached "
            f"`{split_reports['train']['accuracy']:.4f}` training accuracy, "
            f"`{split_reports['validation']['accuracy']:.4f}` validation accuracy, and "
            f"`{split_reports['test']['accuracy']:.4f}` test accuracy."
        ),
        "",
        (
            f"On the test set, the strongest class by F1 was `{best_test_class['label']}` "
            f"with precision `{best_test_class['precision']:.4f}`, recall `{best_test_class['recall']:.4f}`, "
            f"and F1 `{best_test_class['f1']:.4f}`. The weakest class was `{weakest_test_class['label']}` "
            f"with precision `{weakest_test_class['precision']:.4f}`, recall `{weakest_test_class['recall']:.4f}`, "
            f"and F1 `{weakest_test_class['f1']:.4f}`."
        ),
        "",
        "Use the full training report for confusion matrices and the feature summary to see which inputs the forest relied on most.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_inference_params(path: Path, model: RandomForestClassifier, labels: list[str], feature_set: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": labels,
        "feature_set": feature_set,
        "trees": [],
    }
    for estimator in model.estimators_:
        tree = estimator.tree_
        payload["trees"].append(
            {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": tree.value[:, 0, :].tolist(),
            }
        )
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def train_model(
    preprocessed_dir: Path,
    output_dir: Path,
    model_path: Path,
    feature_set: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str | int | float | None,
    seed: int,
) -> dict[str, float]:
    x_train, y_train, x_val, y_val, x_test, y_test, feature_names = load_processed_splits(preprocessed_dir, feature_set)
    preprocessor = load_preprocessor(preprocessed_dir)
    label_names = preprocessor["label_encoder"].classes_
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        n_jobs=DEFAULT_N_JOBS,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    split_reports = {
        "train": classification_report(y_train, train_pred, label_names),
        "validation": classification_report(y_val, val_pred, label_names),
        "test": classification_report(y_test, test_pred, label_names),
    }
    metrics = {
        "train_accuracy": split_reports["train"]["accuracy"],
        "val_accuracy": split_reports["validation"]["accuracy"],
        "test_accuracy": split_reports["test"]["accuracy"],
    }
    run_summary = {
        "feature_set": feature_set,
        "selected_feature_count": len(feature_names),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": label_names, "feature_set": feature_set}, model_path)
    export_inference_params(output_dir / DEFAULT_INFERENCE_PARAMS_PATH.name, model, label_names, feature_set)
    write_json(output_dir / "rf_metrics.json", split_reports)
    write_training_report(output_dir / "rf_training_report.md", run_summary, split_reports)
    write_feature_summary(output_dir / "rf_feature_summary.md", feature_names, model.feature_importances_)
    write_performance_summary(output_dir / "rf_performance_summary.md", run_summary, split_reports)
    append_training_log(output_dir / RUN_LOG_PATH.name, run_summary, metrics)
    return metrics


def predict_all(
    filename: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    preprocessed_dir: Path = DEFAULT_PREPROCESSED_DIR,
) -> list[str]:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    labels = bundle["labels"]
    feature_set = bundle["feature_set"]
    preprocessor = load_preprocessor(preprocessed_dir)
    frame = read_rows(filename, preprocessor)
    x = preprocess_frame(frame, preprocessor)
    feature_indices = select_feature_indices(load_feature_names(preprocessed_dir), feature_set)
    x = apply_feature_subset(x, feature_indices)
    predictions = model.predict(x)
    return [labels[index] for index in predictions]


def parse_max_depth(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


def parse_max_features(value: str) -> str | int | float | None:
    lowered = value.lower()
    if lowered in {"sqrt", "log2", "none"}:
        return None if lowered == "none" else lowered
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run a random forest for painting prediction.")
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train", action="store_true", help="Train the model on processed train/val/test splits.")
    parser.add_argument("--predict", type=Path, help="Predict painting labels for a CSV file.")
    parser.add_argument("--feature-set", choices=["full", "text_season_feel"], default=DEFAULT_FEATURE_SET)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=parse_max_depth, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--min-samples-split", type=int, default=DEFAULT_MIN_SAMPLES_SPLIT)
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--max-features", type=parse_max_features, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()

    if args.train:
        metrics = train_model(
            preprocessed_dir=args.preprocessed_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            feature_set=args.feature_set,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            seed=args.seed,
        )
        print(json.dumps(metrics, indent=2))

    if args.predict is not None:
        predictions = predict_all(
            filename=str(args.predict),
            model_path=args.model_path,
            preprocessed_dir=args.preprocessed_dir,
        )
        print(json.dumps(predictions, indent=2))

    if not args.train and args.predict is None:
        parser.error("Choose at least one of --train or --predict.")


if __name__ == "__main__":
    main()
