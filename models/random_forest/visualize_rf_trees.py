from __future__ import annotations

"""Save a few random-forest trees as image files.

Usage:
    python models/random_forest/visualize_rf_trees.py
    python models/random_forest/visualize_rf_trees.py --num-trees 4 --max-depth 5
"""

import argparse
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import load_feature_names, select_feature_indices  # noqa: E402


DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "processed" / "models" / "random_forest" / "rf_model.joblib"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "models" / "random_forest" / "tree_images"


def save_tree_images(
    model_path: Path,
    preprocessed_dir: Path,
    output_dir: Path,
    num_trees: int,
    max_depth: int,
    dpi: int,
) -> list[Path]:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_set = bundle["feature_set"]
    labels = list(bundle["labels"])

    all_feature_names = load_feature_names(preprocessed_dir)
    feature_indices = select_feature_indices(all_feature_names, feature_set)
    feature_names = [all_feature_names[index] for index in feature_indices.tolist()]

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    tree_count = min(num_trees, len(model.estimators_))

    for tree_index in range(tree_count):
        estimator = model.estimators_[tree_index]
        figure = plt.figure(figsize=(28, 16))
        plot_tree(
            estimator,
            feature_names=feature_names,
            class_names=labels,
            filled=True,
            rounded=True,
            impurity=True,
            proportion=False,
            max_depth=max_depth,
            fontsize=8,
        )
        plt.tight_layout()
        image_path = output_dir / f"tree_{tree_index:02d}.png"
        figure.savefig(image_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        image_paths.append(image_path)

    return image_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a few trees from the trained random forest as images.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-trees", type=int, default=3, help="How many trees to render.")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum printed depth for each tree.")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    args = parser.parse_args()

    image_paths = save_tree_images(
        model_path=args.model_path,
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        num_trees=args.num_trees,
        max_depth=args.max_depth,
        dpi=args.dpi,
    )
    print(f"Wrote {len(image_paths)} tree image(s) to {args.output_dir}")
    print(f"Rendered trees with printed max depth {args.max_depth}")


if __name__ == "__main__":
    main()
