from __future__ import annotations

"""Train and run a NumPy MLP for painting prediction.

Usage:
    python mlp_model.py --train
    python mlp_model.py --predict data/example.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import preprocess_dataset as prep


DEFAULT_PROCESSED_DIR = Path("processed")
DEFAULT_MODEL_PATH = DEFAULT_PROCESSED_DIR / "mlp_model.npz"
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 25
DEFAULT_TOP_FEATURES = 12
DEFAULT_FEATURE_SET = "full"
RUN_LOG_PATH = DEFAULT_PROCESSED_DIR / "mlp_training_log.csv"


def ensure_preprocessor_pickle_compatibility() -> None:
    """Expose preprocessing classes under __main__ so the existing pickle loads."""
    import __main__

    __main__.NumericColumnSpec = prep.NumericColumnSpec
    __main__.NumericPreprocessor = prep.NumericPreprocessor
    __main__.SimpleTfidfVectorizer = prep.SimpleTfidfVectorizer
    __main__.SimpleMultiLabelBinarizer = prep.SimpleMultiLabelBinarizer
    __main__.SimpleLabelEncoder = prep.SimpleLabelEncoder


def load_preprocessor(processed_dir: Path) -> dict:
    ensure_preprocessor_pickle_compatibility()
    with (processed_dir / "preprocessor.pkl").open("rb") as handle:
        return pickle.load(handle)


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


class NumpyMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = 311,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.w1 = (rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_linear = x @ self.w1 + self.b1
        hidden = relu(hidden_linear)
        logits = hidden @ self.w2 + self.b2
        return hidden_linear, hidden, logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, _, logits = self.forward(x)
        return softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[np.ndarray, ...]]:
        hidden_linear, hidden, logits = self.forward(x)
        probs = softmax(logits)
        targets = one_hot_encode(y, probs.shape[1])
        batch_size = x.shape[0]

        data_loss = -np.sum(targets * np.log(probs + 1e-12)) / batch_size
        reg_loss = 0.5 * self.weight_decay * (np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2))
        loss = float(data_loss + reg_loss)

        dlogits = (probs - targets) / batch_size
        dw2 = hidden.T @ dlogits + self.weight_decay * self.w2
        db2 = dlogits.sum(axis=0)

        dhidden = dlogits @ self.w2.T
        dhidden[hidden_linear <= 0] = 0.0
        dw1 = x.T @ dhidden + self.weight_decay * self.w1
        db1 = dhidden.sum(axis=0)
        return loss, (dw1, db1, dw2, db2)

    def step(self, grads: tuple[np.ndarray, ...]) -> None:
        dw1, db1, dw2, db2 = grads
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def save(self, path: Path, labels: list[str]) -> None:
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            labels=np.array(labels, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> tuple["NumpyMLP", list[str]]:
        saved = np.load(path, allow_pickle=True)
        model = cls(
            input_dim=saved["w1"].shape[0],
            hidden_dim=saved["w1"].shape[1],
            output_dim=saved["w2"].shape[1],
        )
        model.w1 = saved["w1"].astype(np.float32)
        model.b1 = saved["b1"].astype(np.float32)
        model.w2 = saved["w2"].astype(np.float32)
        model.b2 = saved["b2"].astype(np.float32)
        labels = saved["labels"].tolist()
        return model, labels


def accuracy(model: NumpyMLP, x: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(x) == y).mean())


def dense_load(path: Path) -> np.ndarray:
    return sparse.load_npz(path).astype(np.float32).toarray()


def load_feature_names(processed_dir: Path | str) -> list[str]:
    processed_dir = Path(processed_dir)
    with (processed_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_feature_indices(feature_names: list[str], feature_set: str) -> np.ndarray:
    if feature_set == "full":
        selected = list(range(len(feature_names)))
    elif feature_set == "text_season_feel":
        selected = [
            index
            for index, name in enumerate(feature_names)
            if name.startswith("text:") or name.startswith("season=") or name.startswith("feel_")
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if not selected:
        raise ValueError("No features matched the configured subset.")
    return np.array(selected, dtype=np.int64)


def apply_feature_subset(x: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    return x[:, feature_indices]


def load_processed_splits(processed_dir: Path) -> tuple[np.ndarray, ...]:
    x_train = dense_load(processed_dir / "train_X.npz")
    y_train = np.load(processed_dir / "train_y.npy")
    x_val = dense_load(processed_dir / "val_X.npz")
    y_val = np.load(processed_dir / "val_y.npy")
    x_test = dense_load(processed_dir / "test_X.npz")
    y_test = np.load(processed_dir / "test_y.npy")
    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_frame(frame: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    resolved = preprocessor["resolved_columns"]
    text_columns = preprocessor["text_columns"]
    categorical_columns = preprocessor["categorical_columns"]
    vectorizer = preprocessor["vectorizer"]
    categorical_encoders = preprocessor["categorical_encoders"]
    numeric_processor = preprocessor["numeric_processor"]

    text_documents = prep.combine_text_columns(frame, text_columns)
    text_matrix = vectorizer.transform(text_documents)

    categorical_parts = []
    for feature_name, source_column in categorical_columns.items():
        values = frame[source_column].map(prep.split_multivalue_cell)
        categorical_parts.append(categorical_encoders[feature_name].transform(values))
    categorical_matrix = sparse.hstack(categorical_parts, format="csr", dtype=np.float32)

    numeric_matrix = sparse.csr_matrix(numeric_processor.transform(frame))
    full_matrix = sparse.hstack([text_matrix, categorical_matrix, numeric_matrix], format="csr", dtype=np.float32)
    return full_matrix.toarray().astype(np.float32)


def read_rows(csv_path: Path, preprocessor: dict) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = [column for column in preprocessor["resolved_columns"].values() if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing}")
    return frame


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[true_value, pred_value] += 1
    return matrix


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    per_class: list[dict[str, object]] = []

    for class_index, label in enumerate(labels):
        tp = int(matrix[class_index, class_index])
        fp = int(matrix[:, class_index].sum() - tp)
        fn = int(matrix[class_index, :].sum() - tp)
        support = int(matrix[class_index, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    accuracy_value = float((y_true == y_pred).mean())
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))
    weighted_f1 = float(
        sum(row["f1"] * row["support"] for row in per_class) / max(1, len(y_true))
    )

    return {
        "accuracy": accuracy_value,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "support": int(len(y_true)),
        },
        "weighted_avg": {
            "f1": weighted_f1,
            "support": int(len(y_true)),
        },
        "confusion_matrix": matrix.tolist(),
    }


def top_feature_rows(weights: np.ndarray, feature_names: list[str], top_k: int) -> list[dict[str, float | str]]:
    indices = np.argsort(np.abs(weights))[::-1][:top_k]
    return [
        {
            "feature": feature_names[index],
            "weight": float(weights[index]),
            "abs_weight": float(abs(weights[index])),
        }
        for index in indices
    ]


def summarize_hidden_units(
    model: NumpyMLP,
    feature_names: list[str],
    labels: list[str],
    top_units: int = 6,
    top_features: int = 8,
) -> list[dict[str, object]]:
    hidden_strength = np.max(np.abs(model.w2), axis=1)
    unit_indices = np.argsort(hidden_strength)[::-1][:top_units]
    rows: list[dict[str, object]] = []

    for unit_index in unit_indices:
        class_index = int(np.argmax(np.abs(model.w2[unit_index])))
        rows.append(
            {
                "hidden_unit": int(unit_index),
                "strongest_output_class": labels[class_index],
                "output_weight": float(model.w2[unit_index, class_index]),
                "top_input_features": top_feature_rows(model.w1[:, unit_index], feature_names, top_features),
            }
        )
    return rows


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_training_log(path: Path, run_summary: dict[str, object], metrics: dict[str, float]) -> None:
    columns = [
        "feature_set",
        "selected_feature_count",
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "patience",
        "seed",
        "best_epoch",
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


def write_training_report(
    path: Path,
    run_summary: dict[str, object],
    split_reports: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# MLP Training Report",
        "",
        "## Run Summary",
        "",
        f"- Feature subset: `{run_summary['feature_subset']}`",
        f"- Selected feature count: `{run_summary['selected_feature_count']}`",
        f"- Hidden size: `{run_summary['hidden_dim']}`",
        f"- Learning rate: `{run_summary['learning_rate']}`",
        f"- Weight decay: `{run_summary['weight_decay']}`",
        f"- Batch size: `{run_summary['batch_size']}`",
        f"- Max epochs: `{run_summary['epochs']}`",
        f"- Patience: `{run_summary['patience']}`",
        f"- Seed: `{run_summary['seed']}`",
        f"- Best epoch: `{run_summary['best_epoch']}`",
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
        for label, row_values in zip(
            [row["label"] for row in report["per_class"]],
            report["confusion_matrix"],
        ):
            row_text = " | ".join(str(value) for value in row_values)
            lines.append(f"| {label} | {row_text} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_weight_summary(
    path: Path,
    model: NumpyMLP,
    feature_names: list[str],
    labels: list[str],
    top_features: int,
) -> None:
    lines = [
        "# MLP Weight Summary",
        "",
        "This file is a readable summary of the learned weights.",
        "The full numeric arrays are still stored in `processed/mlp_model.npz`, but this report highlights the strongest patterns.",
        "",
        "## Output Layer",
        "",
        "Each class score is computed from the hidden layer through `W2` and `b2`.",
        "",
    ]

    for class_index, label in enumerate(labels):
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Output bias: `{model.b2[class_index]:.6f}`",
                "",
                "| Hidden Unit | Weight To Class |",
                "| --- | ---: |",
            ]
        )
        top_hidden = np.argsort(np.abs(model.w2[:, class_index]))[::-1][:top_features]
        for unit_index in top_hidden:
            lines.append(f"| {unit_index} | {model.w2[unit_index, class_index]:.6f} |")
        lines.append("")

    lines.extend(
        [
            "## Hidden Units And Their Strongest Input Features",
            "",
            "These are the hidden units with the strongest downstream effect on at least one class.",
            "",
        ]
    )

    for row in summarize_hidden_units(model, feature_names, labels, top_features=top_features):
        lines.extend(
            [
                f"### Hidden Unit {row['hidden_unit']}",
                "",
                f"- Strongest output class: `{row['strongest_output_class']}`",
                f"- Output weight to that class: `{row['output_weight']:.6f}`",
                "",
                "| Input Feature | Weight | abs(weight) |",
                "| --- | ---: | ---: |",
            ]
        )
        for feature_row in row["top_input_features"]:
            lines.append(
                f"| {feature_row['feature']} | {feature_row['weight']:.6f} | {feature_row['abs_weight']:.6f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def save_readable_artifacts(
    processed_dir: Path,
    model: NumpyMLP,
    labels: list[str],
    feature_names: list[str],
    run_summary: dict[str, object],
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, object]]:
    split_reports: dict[str, dict[str, object]] = {}
    for split_name, (y_true, y_pred) in split_predictions.items():
        split_reports[split_name] = classification_report(y_true, y_pred, labels)

    write_json(processed_dir / "mlp_metrics.json", split_reports)
    write_training_report(processed_dir / "mlp_training_report.md", run_summary, split_reports)
    write_weight_summary(
        processed_dir / "mlp_weight_summary.md",
        model,
        feature_names,
        labels,
        top_features=DEFAULT_TOP_FEATURES,
    )
    return split_reports


def train_model(
    processed_dir: Path,
    model_path: Path,
    feature_set: str,
    hidden_dim: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    weight_decay: float,
    seed: int,
) -> dict[str, float]:
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_splits(processed_dir)
    preprocessor = load_preprocessor(processed_dir)
    label_names = preprocessor["label_encoder"].classes_
    full_feature_names = load_feature_names(processed_dir)
    feature_indices = select_feature_indices(full_feature_names, feature_set)
    feature_names = [full_feature_names[index] for index in feature_indices.tolist()]
    x_train = apply_feature_subset(x_train, feature_indices)
    x_val = apply_feature_subset(x_val, feature_indices)
    x_test = apply_feature_subset(x_test, feature_indices)

    model = NumpyMLP(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=len(label_names),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    best_val = -np.inf
    best_weights: tuple[np.ndarray, ...] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        x_train_epoch = x_train[order]
        y_train_epoch = y_train[order]
        batch_losses: list[float] = []

        for start in range(0, len(x_train_epoch), batch_size):
            end = start + batch_size
            x_batch = x_train_epoch[start:end]
            y_batch = y_train_epoch[start:end]
            loss, grads = model.loss_and_gradients(x_batch, y_batch)
            model.step(grads)
            batch_losses.append(loss)

        train_acc = accuracy(model, x_train, y_train)
        val_acc = accuracy(model, x_val, y_val)
        avg_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        print(
            f"epoch={epoch:03d} loss={avg_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_weights = (
                model.w1.copy(),
                model.b1.copy(),
                model.w2.copy(),
                model.b2.copy(),
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_weights is None:
        raise RuntimeError("Training did not produce any model checkpoints.")

    model.w1, model.b1, model.w2, model.b2 = best_weights
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    train_acc = float((train_pred == y_train).mean())
    val_acc = float((val_pred == y_val).mean())
    test_acc = float((test_pred == y_test).mean())
    model.save(model_path, label_names)
    run_summary = {
        "feature_subset": feature_set,
        "feature_set": feature_set,
        "selected_feature_count": len(feature_names),
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "seed": seed,
        "best_epoch": best_epoch,
    }
    metrics = {"train_accuracy": train_acc, "val_accuracy": val_acc, "test_accuracy": test_acc}
    save_readable_artifacts(
        processed_dir=processed_dir,
        model=model,
        labels=label_names,
        feature_names=feature_names,
        run_summary=run_summary,
        split_predictions={
            "train": (y_train, train_pred),
            "validation": (y_val, val_pred),
            "test": (y_test, test_pred),
        },
    )
    append_training_log(processed_dir / RUN_LOG_PATH.name, run_summary, metrics)
    return metrics


def predict_all(
    filename: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> list[str]:
    preprocessor = load_preprocessor(processed_dir)
    model, labels = NumpyMLP.load(model_path)
    frame = read_rows(Path(filename), preprocessor)
    feature_indices = select_feature_indices(load_feature_names(processed_dir), feature_set)
    x = apply_feature_subset(preprocess_frame(frame, preprocessor), feature_indices)
    predictions = model.predict(x)
    return [labels[index] for index in predictions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run a NumPy MLP for painting prediction.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train", action="store_true", help="Train the model on processed train/val/test splits.")
    parser.add_argument("--predict", type=Path, help="Predict painting labels for a CSV file.")
    parser.add_argument(
        "--feature-set",
        choices=["full", "text_season_feel"],
        default=DEFAULT_FEATURE_SET,
        help="Choose which input feature subset to use.",
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()

    if args.train:
        metrics = train_model(
            processed_dir=args.processed_dir,
            model_path=args.model_path,
            feature_set=args.feature_set,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        print(json.dumps(metrics, indent=2))

    if args.predict is not None:
        predictions = predict_all(
            filename=str(args.predict),
            model_path=args.model_path,
            processed_dir=args.processed_dir,
            feature_set=args.feature_set,
        )
        print(json.dumps(predictions, indent=2))

    if not args.train and args.predict is None:
        parser.error("Choose at least one of --train or --predict.")


if __name__ == "__main__":
    main()
