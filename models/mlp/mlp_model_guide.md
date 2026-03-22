# MLP Painting Classifier Guide

This document explains how the MLP model in this project works after switching to scikit-learn's `MLPClassifier`.

## Goal

The task is to predict which of three paintings a survey response refers to:

- `The Persistence of Memory`
- `The Starry Night`
- `The Water Lily Pond`

This is a multiclass classification problem.

## Main Files

- [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/models/mlp/mlp_model.py)
  Trains the MLP, saves the model, and handles prediction on raw CSV files.

- [pred_mlp.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/models/mlp/pred_mlp.py)
  Thin wrapper exposing `predict_all`.

- [preprocess_dataset.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/preprocessing/preprocess_dataset.py)
  Builds the processed feature matrices and saves the preprocessing objects.

- [processed/mlp_model.joblib](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/mlp/mlp_model.joblib)
  Saved trained scikit-learn MLP model bundle.

- [processed/mlp_training_report.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/mlp/mlp_training_report.md)
  Human-readable train, validation, and test metrics.

- [processed/mlp_weight_summary.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/mlp/mlp_weight_summary.md)
  Human-readable summary of learned hidden-unit and output-layer weights.

- [processed/mlp_metrics.json](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/mlp/mlp_metrics.json)
  Machine-readable metrics for all splits.

- [processed/mlp_training_log.csv](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/models/mlp/mlp_training_log.csv)
  Append-only log of sklearn MLP training runs and hyperparameters.

## How The Pipeline Works

The full pipeline is:

1. preprocess raw survey responses into text, categorical, and numeric features
2. load the processed train, validation, and test matrices
3. optionally choose a feature subset such as `full` or `text_season_feel`
4. train a scikit-learn `MLPClassifier`
5. evaluate on train, validation, and test splits
6. save the trained model and write readable reports
7. reuse the saved preprocessor and model to predict on new CSV rows

## Preprocessing

The preprocessing step is still handled by [preprocess_dataset.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/preprocess_dataset.py).

It creates:

- TF-IDF text features from free-response questions
- one-hot or multi-hot categorical features
- standardized numeric features with missing-value indicators

Those are stacked into a single feature vector per row.

## What The MLP Is

The model is a feedforward neural network implemented through scikit-learn's `MLPClassifier`.

In the current setup it uses:

- one hidden layer
- ReLU activation
- Adam optimization
- built-in multiclass classification

The hidden layer lets the model learn nonlinear combinations of the input features, which makes it more flexible than a purely linear classifier.

## Why scikit-learn Instead Of The Old NumPy Version

Using `MLPClassifier` is helpful because it:

- is more standard and easier to justify in a report
- handles optimization internally
- supports built-in early stopping
- stores weights and metadata in a clean fitted estimator
- reduces the amount of custom neural-network code in the project

## Default Hyperparameter Values

The defaults in [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py) are:

- `feature_set = "full"`
- `hidden_dim = 128`
- `learning_rate = 0.001`
- `weight_decay = 1e-4`
- `batch_size = 64`
- `epochs = 300`
- `patience = 25`
- `seed = 311`

The model also uses:

- `solver = "adam"`
- `activation = "relu"`
- `early_stopping = True`

## What The Hyperparameters Mean

- `hidden_dim`
  The number of neurons in the hidden layer. Larger values increase model capacity.

- `learning_rate`
  The initial step size used by Adam.

- `weight_decay`
  L2 regularization strength. In scikit-learn this is the `alpha` parameter.

- `batch_size`
  The number of training examples used per gradient update.

- `epochs`
  The maximum number of training iterations.

- `patience`
  How many epochs the model will wait without improvement before stopping.

- `feature_set`
  Chooses whether to use the full processed feature set or a smaller subset.

## Training

Training uses the processed training split only. The model's internal early stopping uses a held-out fraction from the training data, while the external validation split remains untouched for honest evaluation afterward.

After fitting, the script evaluates on:

- training split
- validation split
- test split

It then writes:

- [processed/mlp_training_report.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_training_report.md)
- [processed/mlp_weight_summary.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_weight_summary.md)
- [processed/mlp_metrics.json](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_metrics.json)
- [processed/mlp_training_log.csv](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_training_log.csv)

## Weight Summary

The saved weight summary is based on the trained scikit-learn model's learned parameters:

- `coefs_[0]` for input-to-hidden weights
- `intercepts_[0]` for hidden biases
- `coefs_[1]` for hidden-to-output weights
- `intercepts_[1]` for output biases

The report highlights:

- the strongest hidden units for each class
- the strongest input features connected to those hidden units

## Prediction On New CSV Files

`predict_all` works like this:

1. load the saved MLP bundle from `processed/mlp_model.joblib`
2. load the saved preprocessing pipeline
3. transform the new CSV into the same feature space
4. apply the same feature subset used during training
5. run `model.predict(...)`
6. map numeric predictions back to painting labels

## Example Commands

Train:

```powershell
python models/mlp/mlp_model.py --train
```

Train with explicit settings:

```powershell
python models/mlp/mlp_model.py --train --hidden-dim 128 --feature-set full
```

Predict:

```powershell
python models/mlp/mlp_model.py --predict data/training_data_202601.csv
```

## Summary

The MLP pipeline now uses a standard scikit-learn neural-network implementation while keeping the same general workflow:

- preprocess the data
- train the MLP
- evaluate on held-out splits
- save the model
- write readable reports
- predict on new CSV files

That makes it easier to maintain, easier to explain, and easier to compare fairly against the random forest and any other models your group builds.
