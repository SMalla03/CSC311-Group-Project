from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


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


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
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
        raw_tokens = [token for token in document.split() if token]
        terms: list[str] = []
        current_segment: list[str] = []
        for token in raw_tokens:
            if token == TEXT_BOUNDARY_TOKEN:
                if current_segment:
                    current_segment = [
                        item for item in current_segment if len(item) >= 2 and item not in STOPWORDS
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
            current_segment = [item for item in current_segment if len(item) >= 2 and item not in STOPWORDS]
            terms.extend(current_segment)
            terms.extend(
                f"{current_segment[i]} {current_segment[i + 1]}"
                for i in range(len(current_segment) - 1)
            )
        return terms

    def transform(self, documents: pd.Series) -> np.ndarray:
        if self.idf_ is None:
            raise ValueError("Vectorizer must be fit before transform.")
        matrix = np.zeros((len(documents), len(self.feature_names_)), dtype=np.float32)
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
                matrix[row_index, col_index] = float(tf * self.idf_[col_index])
        return matrix


class SimpleMultiLabelBinarizer:
    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.class_to_index: dict[str, int] = {}

    def transform(self, values: pd.Series) -> np.ndarray:
        matrix = np.zeros((len(values), len(self.classes_)), dtype=np.float32)
        for row_index, labels in enumerate(values.tolist()):
            for label in labels:
                col_index = self.class_to_index.get(label)
                if col_index is None:
                    continue
                matrix[row_index, col_index] = 1.0
        return matrix


class SimpleLabelEncoder:
    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.class_to_index: dict[str, int] = {}

