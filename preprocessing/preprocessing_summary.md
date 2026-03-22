# Preprocessing Summary

## Split
- Grouped `70/20/10` train/validation/test split by `unique_id` to avoid respondent leakage across splits.
- Final sizes:
  - Train: `1179` rows, `393` respondents
  - Validation: `336` rows, `112` respondents
  - Test: `171` rows, `57` respondents

## Target
- Label: `Painting`
- Encoding:
  - `0` = `The Persistence of Memory`
  - `1` = `The Starry Night`
  - `2` = `The Water Lily Pond`

## Text Features
- Used 3 free-response columns:
  - feeling description
  - food analogy
  - soundtrack description
- Cleaning:
  - lowercase
  - ASCII normalization
  - apostrophes removed without splitting words
  - punctuation removed
  - whitespace collapsed
  - stopwords removed before TF-IDF
  - tokens shorter than 2 characters removed before TF-IDF
- The 3 text answers are joined with an internal boundary token so bigrams do not cross from one question into another.
- TF-IDF vocabulary is fit on the training split only.
- Uses unigrams + bigrams with:
  - `max_features = 5000`
  - `min_df = 2`
  - `max_df = 0.90`
- Current text feature count: `2688`

## Categorical Features
- Multi-select columns:
  - room
  - who to view with
  - season
- Comma-separated responses were split and multi-hot encoded.
- Blank responses were mapped to `UNK`.

## Numeric / Ordinal Features
- Included:
  - emotion intensity
  - 4 Likert items
  - colour count
  - object count
  - price
- Invalid or missing numeric values were imputed with the training-set median.
- Added a missing-indicator feature for each numeric column.
- Count and price outliers were clipped at the training-set 99th percentile.
- Price was `log1p`-transformed and also got a zero-price indicator.
- Numeric features were standardized using training-set statistics.

## Outputs
- Sparse matrices:
  - `processed/train_X.npz`
  - `processed/val_X.npz`
  - `processed/test_X.npz`
- Labels:
  - `processed/train_y.npy`
  - `processed/val_y.npy`
  - `processed/test_y.npy`
- Row tracking:
  - `processed/train_rows.csv`
  - `processed/val_rows.csv`
  - `processed/test_rows.csv`
- Metadata / vocabulary / fitted preprocessor:
  - `processed/metadata.json`
  - `processed/feature_names.json`
  - `processed/preprocessor.pkl`

## Final Feature Totals
- Text: `2688`
- Categorical: `17`
- Numeric: `17`
- Total: `2722`

## Feature Value Guide
- Text features: all `text:...` columns are TF-IDF weights.
  - Range: nonnegative real values
  - `0` means the term is absent in that row
  - positive value means the term/bigram is present, weighted by TF-IDF

- `room=Bathroom`, `room=Bedroom`, `room=Dining Room`, `room=Living Room`, `room=Office`, `room=UNK`
  - Values: `0` or `1`

- `view_with=By Yourself`, `view_with=Coworkers/Classmates`, `view_with=Family Members`, `view_with=Friends`, `view_with=Strangers`, `view_with=UNK`
  - Values: `0` or `1`

- `season=Fall`, `season=Spring`, `season=Summer`, `season=Winter`, `season=UNK`
  - Values: `0` or `1`

- `emotion_intensity`, `feel_sombre`, `feel_content`, `feel_calm`, `feel_uneasy`, `prominent_colours`, `objects_caught_eye`, `price_cad`
  - Values: standardized numeric values
  - These are real numbers centered/scaled using train-set statistics, so they can be negative, `0`, or positive
  - `price_cad` is standardized after `log1p(price)`

- `emotion_intensity__missing`, `feel_sombre__missing`, `feel_content__missing`, `feel_calm__missing`, `feel_uneasy__missing`, `prominent_colours__missing`, `objects_caught_eye__missing`, `price_cad__missing`
  - Values: `0` or `1`
  - `1` means the original value was missing or invalid before imputation

- `price_cad__is_zero`
  - Values: `0` or `1`
  - `1` means the cleaned original price was exactly `0`
