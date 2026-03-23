from __future__ import annotations

"""Train and run a scikit-learn logistic regression model for painting prediction.

Usage:
    python models/logistic/logistic_model.py --train
    python models/logistic/logistic_model.py --predict data/training_data_202601.csv
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "models" / "logistic"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "logistic_model.joblib"
DEFAULT_FEATURE_SET = "full"
DEFAULT_C = 2.0
DEFAULT_MAX_ITER = 2000
DEFAULT_TOP_FEATURES = 12
RUN_LOG_PATH = DEFAULT_OUTPUT_DIR / "logistic_training_log.csv"


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
        "c",
        "max_iter",
        "class_weight",
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


def top_feature_rows(weights: np.ndarray, feature_names: list[str], top_k: int) -> list[dict[str, float | str]]:
    indices = np.argsort(np.abs(weights))[::-1][:top_k]
    return [
        {"feature": feature_names[index], "weight": float(weights[index]), "abs_weight": float(abs(weights[index]))}
        for index in indices
    ]


def write_training_report(path: Path, run_summary: dict[str, object], split_reports: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Logistic Regression Training Report",
        "",
        "## Run Summary",
        "",
        f"- Feature subset: `{run_summary['feature_set']}`",
        f"- Selected feature count: `{run_summary['selected_feature_count']}`",
        f"- Regularization strength C: `{run_summary['c']}`",
        f"- Max iterations: `{run_summary['max_iter']}`",
        f"- Class weight: `{run_summary['class_weight']}`",
        f"- Seed: `{run_summary['seed']}`",
        f"- Solver: `{run_summary['solver']}`",
        f"- Multi-class: `{run_summary['multi_class']}`",
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


def write_weight_summary(path: Path, model: LogisticRegression, feature_names: list[str], labels: list[str], top_features: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Logistic Regression Weight Summary",
        "",
        "This file shows the learned class weights from scikit-learn's `LogisticRegression`.",
        "",
    ]
    for class_index, label in enumerate(labels):
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Intercept: `{float(model.intercept_[class_index]):.6f}`",
                "",
                "| Feature | Weight | abs(weight) |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in top_feature_rows(model.coef_[class_index], feature_names, top_features):
            lines.append(
                f"| {row['feature']} | {row['weight']:.6f} | {row['abs_weight']:.6f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_performance_summary(path: Path, run_summary: dict[str, object], split_reports: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test_report = split_reports["test"]
    best_test_class = max(test_report["per_class"], key=lambda row: row["f1"])
    weakest_test_class = min(test_report["per_class"], key=lambda row: row["f1"])
    lines = [
        "# Logistic Regression Performance Summary",
        "",
        (
            f"The logistic regression model was trained using the `{run_summary['feature_set']}` feature set with "
            f"`C={run_summary['c']}` and reached `{split_reports['train']['accuracy']:.4f}` training accuracy, "
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
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_model(
    preprocessed_dir: Path,
    output_dir: Path,
    model_path: Path,
    feature_set: str,
    c: float,
    max_iter: int,
    class_weight: str | None,
    seed: int,
) -> dict[str, float]:
    x_train, y_train, x_val, y_val, x_test, y_test, feature_names = load_processed_splits(preprocessed_dir, feature_set)
    preprocessor = load_preprocessor(preprocessed_dir)
    label_names = preprocessor["label_encoder"].classes_

    model = LogisticRegression(
        C=c,
        max_iter=max_iter,
        class_weight=class_weight,
        solver="lbfgs",
        multi_class="multinomial",
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
        "c": c,
        "max_iter": max_iter,
        "class_weight": class_weight or "none",
        "seed": seed,
        "solver": "lbfgs",
        "multi_class": "multinomial",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": label_names, "feature_set": feature_set}, model_path)
    write_json(output_dir / "logistic_metrics.json", split_reports)
    write_training_report(output_dir / "logistic_training_report.md", run_summary, split_reports)
    write_weight_summary(output_dir / "logistic_weight_summary.md", model, feature_names, label_names, DEFAULT_TOP_FEATURES)
    write_performance_summary(output_dir / "logistic_performance_summary.md", run_summary, split_reports)
    append_training_log(output_dir / RUN_LOG_PATH.name, run_summary, metrics)
    return metrics


def predict_all(
    filename: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    preprocessed_dir: Path = DEFAULT_PREPROCESSED_DIR,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> list[str]:
    preprocessor = load_preprocessor(preprocessed_dir)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    labels = bundle["labels"]
    effective_feature_set = bundle.get("feature_set", feature_set)
    frame = read_rows(filename, preprocessor)
    feature_indices = select_feature_indices(load_feature_names(preprocessed_dir), effective_feature_set)
    x = apply_feature_subset(preprocess_frame(frame, preprocessor), feature_indices)
    predictions = model.predict(x)
    return [labels[index] for index in predictions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run a scikit-learn logistic regression model for painting prediction.")
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train", action="store_true", help="Train the model on processed train/val/test splits.")
    parser.add_argument("--predict", type=Path, help="Predict painting labels for a CSV file.")
    parser.add_argument("--feature-set", choices=["full", "text_season_feel"], default=DEFAULT_FEATURE_SET)
    parser.add_argument("--c", type=float, default=DEFAULT_C, help="Inverse regularization strength.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--class-weight", choices=["balanced"], default=None)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()

    if args.train:
        metrics = train_model(
            preprocessed_dir=args.preprocessed_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            feature_set=args.feature_set,
            c=args.c,
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            seed=args.seed,
        )
        print(json.dumps(metrics, indent=2))

    if args.predict is not None:
        predictions = predict_all(
            filename=str(args.predict),
            model_path=args.model_path,
            preprocessed_dir=args.preprocessed_dir,
            feature_set=args.feature_set,
        )
        print(json.dumps(predictions, indent=2))

    if not args.train and args.predict is None:
        parser.error("Choose at least one of --train or --predict.")


if __name__ == "__main__":
    main()
