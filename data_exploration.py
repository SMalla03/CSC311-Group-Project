from __future__ import annotations

import argparse
from collections import Counter
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEXT_RESPONSE_COLUMNS = {
	"Describe how this painting makes you feel.",
	"If this painting was a food, what would be?",
	"Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
}

LIKERT_COLUMNS = {
	"This art piece makes me feel sombre.",
	"This art piece makes me feel content.",
	"This art piece makes me feel calm.",
	"This art piece makes me feel uneasy.",
}

CURRENCY_COLUMN = "How much (in Canadian dollars) would you be willing to pay for this painting?"
INTENSITY_COLUMN = "On a scale of 1–10, how intense is the emotion conveyed by the artwork?"
COLOURS_COLUMN = "How many prominent colours do you notice in this painting?"
OBJECTS_COLUMN = "How many objects caught your eye in the painting?"

SCALE_COLUMN_RANGES: dict[str, tuple[int, int]] = {
	INTENSITY_COLUMN: (1, 10),
	**{column: (1, 5) for column in LIKERT_COLUMNS},
}

OUTLIER_FILTER_COLUMNS = {
	COLOURS_COLUMN,
	OBJECTS_COLUMN,
	CURRENCY_COLUMN,
}

DISCRETE_COUNT_COLUMNS = {
	COLOURS_COLUMN,
	OBJECTS_COLUMN,
}

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

MIN_VALUES_FOR_NUMERIC = 5


def slugify(value: str) -> str:
	value = value.strip().lower()
	value = re.sub(r"[^a-z0-9]+", "_", value)
	return value.strip("_") or "unknown"


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
	if not match:
		return None

	try:
		return float(match.group(0))
	except ValueError:
		return None


def clean_likert(value: object) -> float | None:
	if pd.isna(value):
		return None

	text = str(value).strip()
	if not text:
		return None

	match = re.match(r"^([1-5])", text)
	if not match:
		return None

	return float(match.group(1))


def to_numeric_series(series: pd.Series, column_name: str) -> pd.Series:
	if column_name == CURRENCY_COLUMN:
		return series.map(clean_currency)
	if column_name in LIKERT_COLUMNS:
		return series.map(clean_likert)
	return pd.to_numeric(series, errors="coerce")


def filter_middle_percent(series: pd.Series, middle_percent: float) -> pd.Series:
	if middle_percent >= 100:
		return series

	numeric = series.dropna()
	if numeric.empty:
		return series

	tail_percent = (100 - middle_percent) / 2
	lower = numeric.quantile(tail_percent / 100)
	upper = numeric.quantile(1 - (tail_percent / 100))
	return numeric[(numeric >= lower) & (numeric <= upper)]


def draw_numeric_histogram(
	ax: plt.Axes,
	series: pd.Series,
	discrete_range: tuple[int, int] | None = None,
) -> None:
	data = series.dropna()
	if data.empty:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		ax.set_xlabel("Value")
		ax.set_ylabel("Count")
		return

	if discrete_range is not None:
		min_value, max_value = discrete_range
		bins = [value - 0.5 for value in range(min_value, max_value + 2)]
		ax.hist(data, bins=bins, edgecolor="black")
		ax.set_xticks(range(min_value, max_value + 1))
	else:
		ax.hist(data, bins=12, edgecolor="black")

	ax.set_xlabel("Value")
	ax.set_ylabel("Count")


def draw_price_log_histogram(ax: plt.Axes, series: pd.Series) -> None:
	data = series.dropna()
	non_negative = data[data >= 0]
	negative_count = int((data < 0).sum())

	if non_negative.empty:
		ax.text(0.5, 0.5, "No non-negative prices", ha="center", va="center", transform=ax.transAxes)
		ax.set_xlabel("log(1 + Price CAD)")
		ax.set_ylabel("Count")
		if negative_count > 0:
			ax.text(0.98, 0.02, f"Ignored negative prices: {negative_count}", ha="right", va="bottom", transform=ax.transAxes)
		return

	transformed = np.log1p(non_negative)
	ax.hist(transformed, bins=12, edgecolor="black")

	ax.set_xlabel("log(1 + Price CAD)")
	ax.set_ylabel("Count")

	if negative_count > 0:
		ax.text(0.98, 0.02, f"Ignored negative prices: {negative_count}", ha="right", va="bottom", transform=ax.transAxes)


def draw_categorical_bars(ax: plt.Axes, series: pd.Series) -> None:
	cleaned = series.dropna().astype(str).str.strip()
	cleaned = cleaned[cleaned != ""]
	if cleaned.empty:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		ax.set_xlabel("Count")
		ax.set_ylabel("Category")
		return

	exploded = cleaned.str.split(",").explode().str.strip()
	exploded = exploded[exploded != ""]
	if exploded.empty:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		ax.set_xlabel("Count")
		ax.set_ylabel("Category")
		return

	counts = exploded.value_counts().head(12)
	ordered = counts.sort_values()
	ax.barh(ordered.index, ordered.values)
	ax.set_xlabel("Count")
	ax.set_ylabel("Category")


def extract_tokens(text_series: pd.Series) -> list[str]:
	text = " ".join(text_series.dropna().astype(str).tolist()).lower()
	tokens = re.findall(r"[a-z']+", text)
	cleaned_tokens: list[str] = []

	for token in tokens:
		token = token.strip("'")
		if len(token) < 3:
			continue
		if token in STOPWORDS:
			continue
		cleaned_tokens.append(token)

	return cleaned_tokens


def draw_top_words_plot(ax: plt.Axes, series: pd.Series, top_n: int = 15) -> None:
	tokens = extract_tokens(series)
	if not tokens:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		ax.set_xlabel("Word Frequency")
		ax.set_ylabel("Word")
		return

	word_counts = Counter(tokens).most_common(top_n)
	if not word_counts:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		ax.set_xlabel("Word Frequency")
		ax.set_ylabel("Word")
		return

	words = [word for word, _ in word_counts]
	counts = [count for _, count in word_counts]
	ax.barh(words[::-1], counts[::-1])
	ax.set_xlabel("Word Frequency")
	ax.set_ylabel("Word")


def build_non_text_columns(columns: list[str]) -> list[str]:
	excluded = {"unique_id", "Painting", *TEXT_RESPONSE_COLUMNS}
	return [column for column in columns if column not in excluded]


def build_feature_columns(columns: list[str]) -> list[str]:
	excluded = {"unique_id", "Painting"}
	return [column for column in columns if column not in excluded]


def missing_percentage(series: pd.Series) -> float:
	if len(series) == 0:
		return 0.0

	missing_mask = series.isna()
	if series.dtype == object:
		empty_string_mask = series.fillna("").astype(str).str.strip().eq("")
		missing_mask = missing_mask | empty_string_mask

	return float(missing_mask.mean() * 100)


def print_missing_data_summary(df: pd.DataFrame) -> None:
	summary_rows: list[tuple[str, int, float]] = []
	row_count = len(df)

	for column in df.columns:
		percent_missing = missing_percentage(df[column])
		if percent_missing <= 0:
			continue
		missing_count = int(round((percent_missing / 100) * row_count))
		summary_rows.append((column, missing_count, percent_missing))

	summary_rows.sort(key=lambda item: item[2], reverse=True)

	print("\nMissing data summary (ranked by % missing):")
	if not summary_rows:
		print("  No missing data found.")
		return

	for column, missing_count, percent_missing in summary_rows:
		print(f"  - {column}: {missing_count}/{row_count} ({percent_missing:.2f}%)")


def get_discrete_range(column: str, numeric: pd.Series) -> tuple[int, int] | None:
	if column in SCALE_COLUMN_RANGES:
		return SCALE_COLUMN_RANGES[column]

	if column in DISCRETE_COUNT_COLUMNS:
		data = numeric.dropna()
		if data.empty:
			return None
		min_value = int(np.floor(data.min()))
		max_value = int(np.ceil(data.max()))
		if min_value == max_value:
			min_value -= 1
			max_value += 1
		return (min_value, max_value)

	return None


def plot_feature_across_paintings(
	df: pd.DataFrame,
	feature: str,
	paintings: list[str],
	output_root: Path,
	middle_percent: float,
) -> bool:
	if not paintings:
		return False

	fig, axes = plt.subplots(1, len(paintings), figsize=(6 * len(paintings), 5))
	if len(paintings) == 1:
		axes = [axes]

	is_text_feature = feature in TEXT_RESPONSE_COLUMNS

	for ax, painting in zip(axes, paintings):
		subset = df[df["Painting"] == painting].copy()
		series = subset[feature] if feature in subset.columns else pd.Series(dtype=object)

		if is_text_feature:
			draw_top_words_plot(ax, series)
		else:
			numeric = to_numeric_series(series, feature)
			if feature in OUTLIER_FILTER_COLUMNS:
				numeric = filter_middle_percent(numeric, middle_percent)
			numeric_count = numeric.notna().sum()

			if numeric_count >= MIN_VALUES_FOR_NUMERIC:
				if feature == CURRENCY_COLUMN:
					draw_price_log_histogram(ax, numeric)
				else:
					discrete_range = get_discrete_range(feature, numeric)
					draw_numeric_histogram(ax, numeric, discrete_range=discrete_range)
			else:
				draw_categorical_bars(ax, series)

		ax.set_title(painting)

	fig.suptitle(feature)
	fig.tight_layout(rect=[0, 0, 1, 0.95])

	output_path = output_root / f"{slugify(feature)}.png"
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return True


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate one plot per feature, with subplots for each painting.",
	)
	parser.add_argument(
		"--csv",
		type=Path,
		default=Path("training_data_202601.csv"),
		help="Path to the training CSV file.",
	)
	parser.add_argument(
		"--middle-percent",
		type=float,
		default=100,
		help="Keep only the middle X percent for selected numeric columns (0-100].",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path("plots"),
		help="Directory where plot images will be saved.",
	)
	args = parser.parse_args()

	if not args.csv.exists():
		raise FileNotFoundError(f"CSV file not found: {args.csv}")

	df = pd.read_csv(args.csv)

	if "Painting" not in df.columns:
		raise ValueError("Expected a 'Painting' column in the CSV.")
	if not (0 < args.middle_percent <= 100):
		raise ValueError("--middle-percent must be in the range (0, 100].")

	print_missing_data_summary(df)

	args.out.mkdir(parents=True, exist_ok=True)

	paintings = sorted(df["Painting"].dropna().unique())
	feature_columns = build_feature_columns(list(df.columns))

	total_files = 0
	for feature in feature_columns:
		written = plot_feature_across_paintings(
			df,
			feature,
			paintings,
			args.out,
			args.middle_percent,
		)
		if written:
			total_files += 1
			print(f"Wrote: {feature}")

	print(f"Done. Total plots written: {total_files}")


if __name__ == "__main__":
	main()
