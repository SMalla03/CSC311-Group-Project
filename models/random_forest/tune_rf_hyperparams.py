from __future__ import annotations

"""Run a random-forest hyperparameter grid and plot train/validation accuracy trends.

Usage:
    python models/random_forest/tune_rf_hyperparams.py
    python models/random_forest/tune_rf_hyperparams.py --n-estimators 25,50,100 --max-depth 5,10,15,none
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest.rf_model import load_processed_splits, parse_max_depth, parse_max_features  # noqa: E402


DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "models" / "random_forest" / "tuning"
DEFAULT_FEATURE_SET = "full"
DEFAULT_SEED = 311


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_depth_list(raw: str) -> list[int | None]:
    return [parse_max_depth(value.strip()) for value in raw.split(",") if value.strip()]


def parse_max_features_list(raw: str) -> list[str | int | float | None]:
    return [parse_max_features(value.strip()) for value in raw.split(",") if value.strip()]


def format_param_value(value: object) -> str:
    if value is None:
        return "none"
    return str(value)


def display_max_features_value(value: object, selected_feature_count: int) -> str:
    if value == "sqrt":
        return f"{int(np.sqrt(selected_feature_count))} (sqrt)"
    return format_param_value(value)


def train_grid(
    preprocessed_dir: Path,
    feature_set: str,
    seed: int,
    n_estimators_values: list[int],
    max_depth_values: list[int | None],
    min_samples_split_values: list[int],
    min_samples_leaf_values: list[int],
    max_features_values: list[str | int | float | None],
) -> pd.DataFrame:
    x_train, y_train, x_val, y_val, _x_test, _y_test, feature_names = load_processed_splits(
        preprocessed_dir,
        feature_set,
    )
    selected_feature_count = len(feature_names)

    rows: list[dict[str, object]] = []
    combinations = list(
        itertools.product(
            n_estimators_values,
            max_depth_values,
            min_samples_split_values,
            min_samples_leaf_values,
            max_features_values,
        )
    )

    for run_index, (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features) in enumerate(
        combinations,
        start=1,
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=True,
            n_jobs=1,
            random_state=seed,
        )
        model.fit(x_train, y_train)

        train_accuracy = float((model.predict(x_train) == y_train).mean())
        val_accuracy = float((model.predict(x_val) == y_val).mean())
        rows.append(
            {
                "run_index": run_index,
                "feature_set": feature_set,
                "seed": seed,
                "n_estimators": n_estimators,
                "max_depth": format_param_value(max_depth),
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": format_param_value(max_features),
                "max_features_display": display_max_features_value(max_features, selected_feature_count),
                "selected_feature_count": selected_feature_count,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "generalization_gap": train_accuracy - val_accuracy,
            }
        )
        print(
            f"[{run_index}/{len(combinations)}] "
            f"trees={n_estimators}, depth={format_param_value(max_depth)}, "
            f"min_split={min_samples_split}, min_leaf={min_samples_leaf}, "
            f"max_features={format_param_value(max_features)} -> "
            f"train={train_accuracy:.4f}, val={val_accuracy:.4f}"
        )

    return pd.DataFrame(rows)


def summarize_by_hyperparameter(results: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    grouped = (
        results.groupby(column, dropna=False)[["train_accuracy", "val_accuracy", "generalization_gap"]]
        .mean()
        .reset_index()
    )
    grouped[column] = grouped[column].astype(str)
    grouped[column] = pd.Categorical(grouped[column], categories=order, ordered=True)
    grouped = grouped.sort_values(column).reset_index(drop=True)
    return grouped


def plot_hyperparameter(
    results: pd.DataFrame,
    column: str,
    order: list[str],
    output_path: Path,
    display_column: str | None = None,
) -> None:
    summary = summarize_by_hyperparameter(results, column, order)
    x = np.arange(len(summary))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x, summary["train_accuracy"], marker="o", linewidth=2, label="Train Accuracy")
    axis.plot(x, summary["val_accuracy"], marker="o", linewidth=2, label="Validation Accuracy")
    axis.set_xticks(x)
    if display_column is None:
        tick_labels = summary[column].astype(str).tolist()
    else:
        display_map = (
            results[[column, display_column]]
            .drop_duplicates()
            .assign(**{column: lambda frame: frame[column].astype(str)})
            .set_index(column)[display_column]
            .to_dict()
        )
        tick_labels = [display_map.get(value, value) for value in summary[column].astype(str).tolist()]
    axis.set_xticklabels(tick_labels, rotation=20, ha="right")
    axis.set_xlabel(column)
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"Random Forest Accuracy vs {column}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(path: Path, results: pd.DataFrame) -> None:
    best_row = results.sort_values(["val_accuracy", "train_accuracy"], ascending=[False, False]).iloc[0]
    lines = [
        "# Random Forest Hyperparameter Tuning Summary",
        "",
        f"- Total runs: `{len(results)}`",
        f"- Best validation accuracy: `{best_row['val_accuracy']:.4f}`",
        f"- Train accuracy at best validation run: `{best_row['train_accuracy']:.4f}`",
        f"- Generalization gap at best validation run: `{best_row['generalization_gap']:.4f}`",
        "",
        "## Best Run",
        "",
        f"- n_estimators: `{best_row['n_estimators']}`",
        f"- max_depth: `{best_row['max_depth']}`",
        f"- min_samples_split: `{best_row['min_samples_split']}`",
        f"- min_samples_leaf: `{best_row['min_samples_leaf']}`",
        f"- max_features: `{best_row['max_features']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search and visualize random forest hyperparameters.")
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-set", choices=["full", "text_season_feel"], default=DEFAULT_FEATURE_SET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-estimators", type=str, default="5,10,25,50,100,200")
    parser.add_argument("--max-depth", type=str, default="3,5,10,15,25,none")
    parser.add_argument("--min-samples-split", type=str, default="2,5,10,15")
    parser.add_argument("--min-samples-leaf", type=str, default="1,5,10,15")
    parser.add_argument("--max-features", type=str, default="sqrt,2,5,10,25,75")
    args = parser.parse_args()

    n_estimators_values = parse_int_list(args.n_estimators)
    max_depth_values = parse_depth_list(args.max_depth)
    min_samples_split_values = parse_int_list(args.min_samples_split)
    min_samples_leaf_values = parse_int_list(args.min_samples_leaf)
    max_features_values = parse_max_features_list(args.max_features)

    results = train_grid(
        preprocessed_dir=args.preprocessed_dir,
        feature_set=args.feature_set,
        seed=args.seed,
        n_estimators_values=n_estimators_values,
        max_depth_values=max_depth_values,
        min_samples_split_values=min_samples_split_values,
        min_samples_leaf_values=min_samples_leaf_values,
        max_features_values=max_features_values,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rf_hyperparameter_grid_results.csv"
    config_path = args.output_dir / "rf_hyperparameter_grid_config.json"
    summary_path = args.output_dir / "rf_hyperparameter_summary.md"
    results.to_csv(csv_path, index=False)

    config = {
        "feature_set": args.feature_set,
        "seed": args.seed,
        "n_estimators": [int(value) for value in n_estimators_values],
        "max_depth": [format_param_value(value) for value in max_depth_values],
        "min_samples_split": [int(value) for value in min_samples_split_values],
        "min_samples_leaf": [int(value) for value in min_samples_leaf_values],
        "max_features": [format_param_value(value) for value in max_features_values],
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    plot_hyperparameter(
        results,
        "n_estimators",
        [format_param_value(value) for value in n_estimators_values],
        args.output_dir / "rf_tuning_n_estimators.png",
    )
    plot_hyperparameter(
        results,
        "max_depth",
        [format_param_value(value) for value in max_depth_values],
        args.output_dir / "rf_tuning_max_depth.png",
    )
    plot_hyperparameter(
        results,
        "min_samples_split",
        [format_param_value(value) for value in min_samples_split_values],
        args.output_dir / "rf_tuning_min_samples_split.png",
    )
    plot_hyperparameter(
        results,
        "min_samples_leaf",
        [format_param_value(value) for value in min_samples_leaf_values],
        args.output_dir / "rf_tuning_min_samples_leaf.png",
    )
    plot_hyperparameter(
        results,
        "max_features",
        [format_param_value(value) for value in max_features_values],
        args.output_dir / "rf_tuning_max_features.png",
        display_column="max_features_display",
    )
    write_summary(summary_path, results)

    best_row = results.sort_values(["val_accuracy", "train_accuracy"], ascending=[False, False]).iloc[0]
    print(f"Saved grid results to {csv_path}")
    print(f"Saved plots to {args.output_dir}")
    print(
        "Best validation run: "
        f"val={best_row['val_accuracy']:.4f}, train={best_row['train_accuracy']:.4f}, "
        f"trees={best_row['n_estimators']}, depth={best_row['max_depth']}, "
        f"min_split={best_row['min_samples_split']}, min_leaf={best_row['min_samples_leaf']}, "
        f"max_features={best_row['max_features']}"
    )


if __name__ == "__main__":
    main()
