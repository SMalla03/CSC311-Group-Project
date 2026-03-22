from __future__ import annotations

"""Preprocess survey responses into train/val/test artifacts for painting prediction.

Usage:
    python preprocessing/preprocess_dataset.py
    python preprocessing/preprocess_dataset.py --csv data/training_data_202601.csv --out processed/preprocessing

Outputs:
    - sparse feature matrices: train_X.npz / val_X.npz / test_X.npz
    - encoded labels: train_y.npy / val_y.npy / test_y.npy
    - row tracking files: train_rows.csv / val_rows.csv / test_rows.csv
    - metadata, feature names, and the fitted preprocessing objects
"""

import argparse
import json
import pickle
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "training_data_202601.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_RANDOM_SEED = 311
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.20
DEFAULT_TEST_RATIO = 0.10
DEFAULT_TEXT_MAX_FEATURES = 5000
DEFAULT_TEXT_MIN_DF = 2
DEFAULT_TEXT_MAX_DF = 0.90
DEFAULT_CLIP_QUANTILE = 0.99
# Internal separator inserted between text answers so bigrams stay within
# a single survey answer instead of leaking across different questions.
TEXT_BOUNDARY_TOKEN = "__sep__"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "no",
    "not", "of", "on", "or", "our", "out", "so", "that", "the", "their", "them", "there", "they",
    "this", "to", "too", "was", "we", "were", "what", "when", "which", "who", "with", "would", "you",
    "your", "very", "really", "like", "feel", "feels", "feeling", "painting", "art", "piece", "makes",
    "make", "made", "im", "ive", "cant", "dont", "didnt", "itself", "also", "about", "because", "than",
    "then", "while", "where", "after", "before", "during", "over", "under", "up", "down", "off", "all",
    "some", "any", "much", "many", "more", "most", "less", "few", "bit", "little", "one", "two",
    "three", "can", "could", "should", "will", "shall", "might", "may",
}


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def resolve_columns(columns: list[str]) -> dict[str, str]:
    # Column names in the CSV contain punctuation/encoding quirks, so we
    # resolve them through a normalized form rather than matching literally.
    normalized_to_original = {normalize_name(column): column for column in columns}

    expected = {
        "group_id": "unique id",
        "target": "painting",
        "intensity": "on a scale of 110 how intense is the emotion conveyed by the artwork",
        "feel_text": "describe how this painting makes you feel",
        "sombre": "this art piece makes me feel sombre",
        "content": "this art piece makes me feel content",
        "calm": "this art piece makes me feel calm",
        "uneasy": "this art piece makes me feel uneasy",
        "colours": "how many prominent colours do you notice in this painting",
        "objects": "how many objects caught your eye in the painting",
        "price": "how much in canadian dollars would you be willing to pay for this painting",
        "room": "if you could purchase this painting which room would you put that painting in",
        "view_with": "if you could view this art in person who would you want to view it with",
        "season": "what season does this art piece remind you of",
        "food_text": "if this painting was a food what would be",
        "soundtrack_text": "imagine a soundtrack for this painting describe that soundtrack without naming any objects in the painting",
    }

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key, normalized_name in expected.items():
        match = normalized_to_original.get(normalized_name)
        if match is None:
            missing.append(key)
        else:
            resolved[key] = match

    if missing:
        raise ValueError(f"Could not resolve expected columns: {missing}")

    return resolved


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Drop apostrophes without inserting spaces so possessives/contractions
    # do not become clipped bigrams like "shepherd s" or "van gogh s".
    text = text.replace("'", "").replace("`", "")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def clean_currency(value: object) -> float | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("dollars", "")
    text = text.replace("dollar", "")
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match is None:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def clean_likert(value: object) -> float | None:
    if pd.isna(value):
        return None

    match = re.match(r"^\s*([1-5])", str(value))
    if match is None:
        return None
    return float(match.group(1))


def clean_numeric(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_multivalue_cell(value: object) -> list[str]:
    if pd.isna(value):
        return ["UNK"]

    parts = [part.strip() for part in str(value).split(",")]
    cleaned = [part.title() for part in parts if part.strip()]
    if not cleaned:
        return ["UNK"]
    return cleaned


@dataclass
class NumericColumnSpec:
    name: str
    source_column: str
    parser: str
    min_value: float | None = None
    max_value: float | None = None
    clip_upper: bool = False
    log1p: bool = False
    zero_indicator: bool = False


class NumericPreprocessor:
    def __init__(self, specs: list[NumericColumnSpec], clip_quantile: float) -> None:
        self.specs = specs
        self.clip_quantile = clip_quantile
        self.stats: dict[str, dict[str, float | None]] = {}
        self.feature_names: list[str] = []

    def _parse_series(self, frame: pd.DataFrame, spec: NumericColumnSpec) -> pd.Series:
        series = frame[spec.source_column]
        if spec.parser == "currency":
            parsed = series.map(clean_currency)
        elif spec.parser == "likert":
            parsed = series.map(clean_likert)
        else:
            parsed = series.map(clean_numeric)

        parsed = pd.to_numeric(parsed, errors="coerce")
        if spec.min_value is not None:
            parsed = parsed.where(parsed >= spec.min_value)
        if spec.max_value is not None:
            parsed = parsed.where(parsed <= spec.max_value)
        return parsed.astype(float)

    def fit(self, frame: pd.DataFrame) -> None:
        self.feature_names = []
        for spec in self.specs:
            parsed = self._parse_series(frame, spec)
            clip_upper = None
            if spec.clip_upper and parsed.notna().any():
                # Learn the clipping threshold on the training split only,
                # then reuse the same cutoff for val/test.
                clip_upper = float(parsed.quantile(self.clip_quantile))
                parsed = parsed.clip(upper=clip_upper)

            median = float(parsed.median()) if parsed.notna().any() else 0.0
            filled = parsed.fillna(median)

            if spec.zero_indicator:
                # Used only for price so models can distinguish true zeros
                # from other small values after the log transform.
                _ = (filled == 0).astype(float)
            transformed = np.log1p(filled) if spec.log1p else filled
            mean = float(transformed.mean()) if len(transformed) else 0.0
            std = float(transformed.std(ddof=0)) if len(transformed) else 1.0
            if std == 0:
                std = 1.0

            self.stats[spec.name] = {
                "median": median,
                "clip_upper": clip_upper,
                "mean": mean,
                "std": std,
            }

            self.feature_names.append(spec.name)
            self.feature_names.append(f"{spec.name}__missing")
            if spec.zero_indicator:
                self.feature_names.append(f"{spec.name}__is_zero")

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []

        for spec in self.specs:
            parsed = self._parse_series(frame, spec)
            stats = self.stats[spec.name]
            clip_upper = stats["clip_upper"]
            if clip_upper is not None:
                parsed = parsed.clip(upper=float(clip_upper))

            missing = parsed.isna().astype(float).to_numpy().reshape(-1, 1)
            filled = parsed.fillna(float(stats["median"]))

            if spec.zero_indicator:
                zero_values = (filled == 0).astype(float).to_numpy().reshape(-1, 1)
            else:
                zero_values = None

            transformed = np.log1p(filled) if spec.log1p else filled
            scaled = ((transformed - float(stats["mean"])) / float(stats["std"])).to_numpy().reshape(-1, 1)

            columns.append(scaled)
            columns.append(missing)
            if zero_values is not None:
                columns.append(zero_values)

        return np.hstack(columns).astype(np.float32)


def combine_text_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    cleaned = [frame[column].map(clean_text) for column in columns]
    combined = pd.concat(cleaned, axis=1).fillna("")
    # Keep the answers in one text field for simpler vectorization, but mark
    # question boundaries so the TF-IDF bigrams do not span across columns.
    return combined.apply(
        lambda row: f" {TEXT_BOUNDARY_TOKEN} ".join(value for value in row if value).strip(),
        axis=1,
    )


class SimpleTfidfVectorizer:
    def __init__(self, max_features: int, min_df: int, max_df: float) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.feature_names_: list[str] = []

    @staticmethod
    def _extract_terms(document: str) -> list[str]:
        # We emit unigrams and bigrams, but only within a single answer segment.
        raw_tokens = [token for token in document.split() if token]
        terms: list[str] = []
        current_segment: list[str] = []

        for token in raw_tokens:
            if token == TEXT_BOUNDARY_TOKEN:
                if current_segment:
                    current_segment = [
                        token for token in current_segment
                        if len(token) >= 2 and token not in STOPWORDS
                    ]
                    terms.extend(current_segment)
                    terms.extend(
                        f"{current_segment[i]} {current_segment[i + 1]}"
                        for i in range(len(current_segment) - 1)
                    )
                    current_segment = []
                continue
            current_segment.append(token)

        if current_segment:
            current_segment = [
                token for token in current_segment
                if len(token) >= 2 and token not in STOPWORDS
            ]
            terms.extend(current_segment)
            terms.extend(
                f"{current_segment[i]} {current_segment[i + 1]}"
                for i in range(len(current_segment) - 1)
            )
        return terms

    def fit(self, documents: pd.Series) -> None:
        doc_count = len(documents)
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()

        for document in documents.tolist():
            terms = self._extract_terms(document)
            if not terms:
                continue
            document_frequency.update(set(terms))
            term_frequency.update(terms)

        # Keep only terms seen often enough to be useful, but not so often
        # that they appear in almost every training row.
        max_df_count = self.max_df * doc_count if self.max_df <= 1 else self.max_df
        eligible_terms = [
            term
            for term, df_value in document_frequency.items()
            if df_value >= self.min_df and df_value <= max_df_count
        ]
        eligible_terms.sort(key=lambda term: (-term_frequency[term], term))
        if self.max_features > 0:
            eligible_terms = eligible_terms[: self.max_features]

        self.feature_names_ = eligible_terms
        self.vocabulary_ = {term: index for index, term in enumerate(self.feature_names_)}
        self.idf_ = np.array(
            [
                np.log((1 + doc_count) / (1 + document_frequency[term])) + 1.0
                for term in self.feature_names_
            ],
            dtype=np.float32,
        )

    def transform(self, documents: pd.Series) -> sparse.csr_matrix:
        if self.idf_ is None:
            raise ValueError("Vectorizer must be fit before transform.")

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for row_index, document in enumerate(documents.tolist()):
            term_counts = Counter(
                term for term in self._extract_terms(document) if term in self.vocabulary_
            )
            if not term_counts:
                continue

            total_terms = float(sum(term_counts.values()))
            for term, count in term_counts.items():
                col_index = self.vocabulary_[term]
                tf = count / total_terms
                rows.append(row_index)
                cols.append(col_index)
                data.append(float(tf * self.idf_[col_index]))

        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(documents), len(self.feature_names_)),
            dtype=np.float32,
        )
        return matrix

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self.feature_names_, dtype=object)


class SimpleMultiLabelBinarizer:
    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.class_to_index: dict[str, int] = {}

    def fit(self, values: pd.Series) -> None:
        classes = sorted({item for labels in values.tolist() for item in labels})
        self.classes_ = classes
        self.class_to_index = {label: index for index, label in enumerate(classes)}

    def transform(self, values: pd.Series) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for row_index, labels in enumerate(values.tolist()):
            for label in labels:
                col_index = self.class_to_index.get(label)
                if col_index is None:
                    continue
                rows.append(row_index)
                cols.append(col_index)
                data.append(1.0)

        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(values), len(self.classes_)),
            dtype=np.float32,
        )


class SimpleLabelEncoder:
    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.class_to_index: dict[str, int] = {}

    def fit(self, values: pd.Series) -> None:
        self.classes_ = sorted(pd.unique(values).tolist())
        self.class_to_index = {label: index for index, label in enumerate(self.classes_)}

    def transform(self, values: pd.Series) -> np.ndarray:
        return np.array([self.class_to_index[label] for label in values.tolist()], dtype=np.int64)


def build_split_assignments(group_ids: np.ndarray, seed: int) -> tuple[set[int], set[int], set[int]]:
    if len(group_ids) == 0:
        raise ValueError("No group ids found for splitting.")

    # Split at the respondent level so one person's answers for the three
    # paintings never get spread across train/val/test.
    unique_groups = np.array(sorted(pd.unique(group_ids)))
    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    total = len(shuffled_groups)
    train_count = int(round(total * DEFAULT_TRAIN_RATIO))
    val_count = int(round(total * DEFAULT_VAL_RATIO))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        raise ValueError("Split sizes do not leave any groups for the test set.")

    train_groups = set(shuffled_groups[:train_count].tolist())
    val_groups = set(shuffled_groups[train_count : train_count + val_count].tolist())
    test_groups = set(shuffled_groups[train_count + val_count :].tolist())
    return train_groups, val_groups, test_groups


def write_sparse_split(output_dir: Path, split_name: str, x_matrix: sparse.csr_matrix, y_array: np.ndarray, rows: pd.DataFrame) -> None:
    sparse.save_npz(output_dir / f"{split_name}_X.npz", x_matrix)
    np.save(output_dir / f"{split_name}_y.npy", y_array)
    rows.loc[:, ["unique_id", "Painting"]].to_csv(output_dir / f"{split_name}_rows.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess survey data for painting prediction.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV file.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for grouped split.")
    parser.add_argument("--text-max-features", type=int, default=DEFAULT_TEXT_MAX_FEATURES, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--text-min-df", type=int, default=DEFAULT_TEXT_MIN_DF, help="Minimum document frequency for TF-IDF.")
    parser.add_argument("--text-max-df", type=float, default=DEFAULT_TEXT_MAX_DF, help="Maximum document frequency for TF-IDF.")
    parser.add_argument("--clip-quantile", type=float, default=DEFAULT_CLIP_QUANTILE, help="Upper quantile used for clipping outliers.")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if not (0 < args.clip_quantile <= 1):
        raise ValueError("--clip-quantile must be in the range (0, 1].")

    frame = pd.read_csv(args.csv)
    resolved = resolve_columns(list(frame.columns))

    rename_map = {
        resolved["group_id"]: "unique_id",
        resolved["target"]: "Painting",
    }
    frame = frame.rename(columns=rename_map).copy()

    text_columns = [
        resolved["feel_text"],
        resolved["food_text"],
        resolved["soundtrack_text"],
    ]
    categorical_columns = {
        "room": resolved["room"],
        "view_with": resolved["view_with"],
        "season": resolved["season"],
    }
    numeric_specs = [
        NumericColumnSpec("emotion_intensity", resolved["intensity"], parser="numeric", min_value=1, max_value=10),
        NumericColumnSpec("feel_sombre", resolved["sombre"], parser="likert", min_value=1, max_value=5),
        NumericColumnSpec("feel_content", resolved["content"], parser="likert", min_value=1, max_value=5),
        NumericColumnSpec("feel_calm", resolved["calm"], parser="likert", min_value=1, max_value=5),
        NumericColumnSpec("feel_uneasy", resolved["uneasy"], parser="likert", min_value=1, max_value=5),
        NumericColumnSpec("prominent_colours", resolved["colours"], parser="numeric", min_value=0, clip_upper=True),
        NumericColumnSpec("objects_caught_eye", resolved["objects"], parser="numeric", min_value=0, clip_upper=True),
        NumericColumnSpec("price_cad", resolved["price"], parser="currency", min_value=0, clip_upper=True, log1p=True, zero_indicator=True),
    ]

    train_groups, val_groups, test_groups = build_split_assignments(frame["unique_id"].to_numpy(), args.seed)

    split_masks = {
        "train": frame["unique_id"].isin(train_groups),
        "val": frame["unique_id"].isin(val_groups),
        "test": frame["unique_id"].isin(test_groups),
    }

    splits = {name: frame.loc[mask].reset_index(drop=True) for name, mask in split_masks.items()}

    # Fit text vocabulary on train only, then transform all splits with the
    # frozen training vocabulary and IDF weights.
    vectorizer = SimpleTfidfVectorizer(
        max_features=args.text_max_features,
        min_df=args.text_min_df,
        max_df=args.text_max_df,
    )
    train_text = combine_text_columns(splits["train"], text_columns)
    vectorizer.fit(train_text)

    text_matrices = {
        name: vectorizer.transform(combine_text_columns(split_frame, text_columns))
        for name, split_frame in splits.items()
    }

    categorical_encoders: dict[str, SimpleMultiLabelBinarizer] = {}
    categorical_matrices: dict[str, sparse.csr_matrix] = {}
    categorical_feature_names: list[str] = []

    for feature_name, column_name in categorical_columns.items():
        # Multi-select responses like "Bedroom,Office" become one binary column
        # per category, learned from the training split.
        encoder = SimpleMultiLabelBinarizer()
        encoder.fit(splits["train"][column_name].map(split_multivalue_cell))
        categorical_encoders[feature_name] = encoder
        categorical_feature_names.extend([f"{feature_name}={label}" for label in encoder.classes_])

    for split_name, split_frame in splits.items():
        encoded_parts = []
        for feature_name, column_name in categorical_columns.items():
            encoder = categorical_encoders[feature_name]
            encoded_parts.append(encoder.transform(split_frame[column_name].map(split_multivalue_cell)))
        categorical_matrices[split_name] = sparse.hstack(encoded_parts, format="csr")

    numeric_processor = NumericPreprocessor(numeric_specs, clip_quantile=args.clip_quantile)
    numeric_processor.fit(splits["train"])
    numeric_matrices = {
        name: sparse.csr_matrix(numeric_processor.transform(split_frame))
        for name, split_frame in splits.items()
    }

    # Encode the target labels once so later training scripts can use numeric y.
    label_encoder = SimpleLabelEncoder()
    label_encoder.fit(splits["train"]["Painting"])
    y_arrays = {
        name: label_encoder.transform(split_frame["Painting"])
        for name, split_frame in splits.items()
    }

    feature_names = (
        [f"text:{name}" for name in vectorizer.get_feature_names_out().tolist()]
        + categorical_feature_names
        + numeric_processor.feature_names
    )

    args.out.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        # Final feature matrix = [text TF-IDF | categorical multi-hot | numeric]
        x_matrix = sparse.hstack(
            [text_matrices[split_name], categorical_matrices[split_name], numeric_matrices[split_name]],
            format="csr",
        )
        write_sparse_split(args.out, split_name, x_matrix, y_arrays[split_name], splits[split_name])

    metadata = {
        "source_csv": str(args.csv),
        "random_seed": args.seed,
        "group_column": "unique_id",
        "target_column": "Painting",
        "split_ratios": {
            "train": DEFAULT_TRAIN_RATIO,
            "val": DEFAULT_VAL_RATIO,
            "test": DEFAULT_TEST_RATIO,
        },
        "split_sizes": {
            split_name: {
                "rows": int(len(splits[split_name])),
                "groups": int(splits[split_name]["unique_id"].nunique()),
            }
            for split_name in ("train", "val", "test")
        },
        "label_mapping": {
            label: int(index) for index, label in enumerate(label_encoder.classes_)
        },
        "feature_count": len(feature_names),
        "text_feature_count": int(text_matrices["train"].shape[1]),
        "categorical_feature_count": int(categorical_matrices["train"].shape[1]),
        "numeric_feature_count": int(numeric_matrices["train"].shape[1]),
        "feature_names_path": "feature_names.json",
    }

    with (args.out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with (args.out / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    with (args.out / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(
            {
                "resolved_columns": resolved,
                "text_columns": text_columns,
                "categorical_columns": categorical_columns,
                "numeric_specs": numeric_specs,
                "vectorizer": vectorizer,
                "categorical_encoders": categorical_encoders,
                "numeric_processor": numeric_processor,
                "label_encoder": label_encoder,
            },
            handle,
        )

    for split_name in ("train", "val", "test"):
        print(
            f"{split_name}: rows={len(splits[split_name])}, "
            f"groups={splits[split_name]['unique_id'].nunique()}, "
            f"features={text_matrices[split_name].shape[1] + categorical_matrices[split_name].shape[1] + numeric_matrices[split_name].shape[1]}"
        )


if __name__ == "__main__":
    main()
