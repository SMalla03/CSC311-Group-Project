# Random Forest Painting Classifier Guide

This document explains how the random forest model in this project works, how it connects to the preprocessing pipeline, and what files it produces after training.

## Goal

The task is to predict which of three paintings a survey response refers to:

- `The Persistence of Memory`
- `The Starry Night`
- `The Water Lily Pond`

This is a multiclass classification problem, so the random forest learns to separate the three classes directly from the processed survey features.

## Main Files

- [rf_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/models/random_forest/rf_model.py)
  Trains the random forest, saves the model, and handles prediction on raw CSV files.

- [pred_rf.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/models/random_forest/pred_rf.py)
  Thin wrapper exposing `predict_all`, which is useful for a submission-style interface.

- [preprocess_dataset.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/preprocessing/preprocess_dataset.py)
  Builds the processed train/validation/test matrices and saves the preprocessing objects.

- [processed/rf_model.joblib](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/random_forest/rf_model.joblib)
  Saved trained random forest bundle, including the fitted model and label names.

- [processed/rf_training_report.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/random_forest/rf_training_report.md)
  Human-readable report with train, validation, and test metrics.

- [processed/rf_feature_summary.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/random_forest/rf_feature_summary.md)
  Human-readable summary of feature importances.

- [processed/rf_performance_summary.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/random_forest/rf_performance_summary.md)
  Short plain-language summary of the model's performance.

- [processed/rf_metrics.json](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/random_forest/rf_metrics.json)
  Structured metrics for train, validation, and test splits.

- [processed/rf_training_log.csv](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/rf_training_log.csv)
  Append-only log of training runs and hyperparameters.

## How The Random Forest Works

A random forest is an ensemble of many decision trees.

Each tree:

- is trained on a bootstrap sample of the training data
- sees only a random subset of features at each split
- learns a sequence of questions such as “is this feature above some threshold?”

For prediction:

- each tree predicts a class
- the forest aggregates the trees' outputs
- the final class is the majority vote across the forest

This randomization is what makes a random forest more stable than relying on a single decision tree.

## Why Random Forests Can Help Here

This project mixes:

- sparse text-like TF-IDF features
- one-hot categorical features
- numeric survey features

Random forests are useful because they can model nonlinear relationships and feature interactions without requiring a lot of manual feature engineering after preprocessing.

## Training Flow

The training script:

1. loads the processed train, validation, and test matrices
2. optionally selects a feature subset such as `full` or `text_season_feel`
3. fits a `RandomForestClassifier`
4. evaluates the model on train, validation, and test splits
5. saves the fitted model
6. writes readable markdown reports and a CSV log

## Important Hyperparameters

- `n_estimators`
  The number of trees in the forest. More trees usually improve stability, but increase training time.

- `max_depth`
  The maximum depth of each tree. Smaller depths regularize the model. `None` means trees can keep growing until other stopping rules stop them.

- `min_samples_split`
  The minimum number of samples required to split an internal node.

- `min_samples_leaf`
  The minimum number of samples required in a leaf node.

- `max_features`
  How many features each tree considers when looking for the best split. `sqrt` is a common default for classification.

- `feature_set`
  Controls whether the model uses the full processed feature set or a restricted subset like `text_season_feel`.

## Default Hyperparameter Values

The defaults in [rf_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/rf_model.py) are:

- `feature_set = "full"`
- `n_estimators = 300`
- `max_depth = None`
- `min_samples_split = 2`
- `min_samples_leaf = 1`
- `max_features = "sqrt"`
- `seed = 311`

These defaults mean:

- the model uses all processed features unless you choose another feature set
- the forest contains 300 trees
- tree depth is not capped directly, so the other stopping rules control how far trees grow
- a split needs at least 2 samples
- each leaf must contain at least 1 sample
- each split considers roughly the square root of the total feature count
- the run is reproducible because the random seed is fixed

## Feature Importance

Scikit-learn's random forest exposes `feature_importances_`, which estimates how much each feature contributed to reducing impurity across the trees.

This project writes those values into [processed/rf_feature_summary.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/rf_feature_summary.md), where they are shown as:

- top individual features
- grouped importance totals by feature family

## Prediction On New CSV Files

The `predict_all` function:

1. loads the saved model bundle from `rf_model.joblib`
2. loads the saved preprocessor
3. preprocesses the new CSV into the same feature space used during training
4. applies the same feature subset used by the saved model
5. predicts the painting label for each row

## How To Train

```powershell
python models/random_forest/rf_model.py --train
```

Example with explicit settings:

```powershell
python models/random_forest/rf_model.py --train --feature-set full --n-estimators 300 --max-depth none
```

## How To Predict

```powershell
python models/random_forest/rf_model.py --predict data/training_data_202601.csv
```

## Summary

This random forest pipeline gives you:

- a trained random forest classifier
- readable markdown evaluation reports
- readable feature-importance summaries
- a plain-language performance summary
- a CSV log of all random-forest training runs

That makes it easier to compare the random forest directly against the MLP and the other models in your group project.
