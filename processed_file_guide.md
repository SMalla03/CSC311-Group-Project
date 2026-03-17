# Guide to the Processed Files

This project saves the preprocessed dataset into a few Python-friendly formats.

## Main Files
- `processed/train_X.npz`, `processed/val_X.npz`, `processed/test_X.npz`
  - Sparse feature matrices
  - Shape: `(num_rows, num_features)`
  - Use these as the input `X` for models

- `processed/train_y.npy`, `processed/val_y.npy`, `processed/test_y.npy`
  - Encoded label arrays
  - Shape: `(num_rows,)`
  - Use these as the target `y` for models

- `processed/preprocessor.pkl`
  - Pickled Python dictionary containing the fitted preprocessing objects
  - Includes:
    - resolved column mapping
    - text column list
    - categorical column mapping
    - numeric specs
    - fitted TF-IDF vectorizer
    - fitted categorical encoders
    - fitted numeric preprocessor
    - fitted label encoder
  - Use this if you want to inspect or reuse the same preprocessing logic later

- `processed/feature_names.json`
  - Ordered list of feature names
  - The index in this file matches the column index in `X`

- `processed/metadata.json`
  - Summary information about splits, label mapping, and feature counts

- `processed/train_rows.csv`, `processed/val_rows.csv`, `processed/test_rows.csv`
  - Row tracking files
  - These tell you which original `unique_id` and `Painting` label correspond to each row in the saved matrices

## How To Load `.npz` and `.npy`

```python
import numpy as np
from scipy import sparse

X_train = sparse.load_npz("processed/train_X.npz")
y_train = np.load("processed/train_y.npy")

print(X_train.shape)
print(y_train.shape)
```

Notes:
- `X_train` is sparse, not dense
- If you print it directly, you will not see a nice table
- For most ML libraries, keeping it sparse is preferred

## How To Inspect One Row

```python
import json
import numpy as np
from scipy import sparse

feature_names = json.load(open("processed/feature_names.json", "r", encoding="utf-8"))
X_train = sparse.load_npz("processed/train_X.npz").tocsr()
y_train = np.load("processed/train_y.npy")

row = X_train.getrow(0)
for idx, value in zip(row.indices, row.data):
    print(feature_names[idx], value)

print("label:", y_train[0])
```

This prints only the nonzero features in that row.

## How To Load The `.pkl`

```python
import pickle

with open("processed/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

print(preprocessor.keys())
```

Expected keys:
- `resolved_columns`
- `text_columns`
- `categorical_columns`
- `numeric_specs`
- `vectorizer`
- `categorical_encoders`
- `numeric_processor`
- `label_encoder`

## How To Map Numeric Labels Back To Painting Names

```python
import json
import numpy as np

metadata = json.load(open("processed/metadata.json", "r", encoding="utf-8"))
y_train = np.load("processed/train_y.npy")

index_to_label = {v: k for k, v in metadata["label_mapping"].items()}
print(index_to_label[y_train[0]])
```

## Typical Training Workflow

```python
import numpy as np
from scipy import sparse

X_train = sparse.load_npz("processed/train_X.npz")
y_train = np.load("processed/train_y.npy")

X_val = sparse.load_npz("processed/val_X.npz")
y_val = np.load("processed/val_y.npy")

X_test = sparse.load_npz("processed/test_X.npz")
y_test = np.load("processed/test_y.npy")
```

Then:
- fit your model on `(X_train, y_train)`
- tune/check performance on `(X_val, y_val)`
- report final performance on `(X_test, y_test)`

## Note About Preview CSVs
- In preview files, blank cells usually mean the processed value is `0`
- That is because sparse matrices only explicitly store nonzero entries
