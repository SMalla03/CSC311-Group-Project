# MLP Painting Classifier Guide

This document explains how the MLP model in this project works, how it connects to the preprocessing pipeline, and how to train and use it for painting prediction.

## Goal

The task is to predict which of three paintings a survey response refers to:

- `The Persistence of Memory`
- `The Starry Night`
- `The Water Lily Pond`

The model uses the survey answers as input features and outputs one of those three painting labels.

Because there are three mutually exclusive classes, this is a **multiclass classification** problem. That is why the model uses a **softmax output layer** rather than binary logistic regression.

## Files Involved

- [preprocess_dataset.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/preprocess_dataset.py)
  Builds the processed feature matrices and saves the fitted preprocessing objects.

- [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py)
  Contains the NumPy implementation of the MLP, training code, evaluation code, and prediction logic.

- [pred_mlp.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/pred_mlp.py)
  Thin wrapper exposing `predict_all`, which is the expected submission-style interface.

- [processed/preprocessor.pkl](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/preprocessor.pkl)
  Saved preprocessing objects used to transform raw CSV rows into the same feature space used during training.

- [processed/train_X.npz](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/train_X.npz), [processed/val_X.npz](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/val_X.npz), [processed/test_X.npz](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/test_X.npz)
  Saved input feature matrices for the train, validation, and test splits.

- [processed/train_y.npy](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/train_y.npy), [processed/val_y.npy](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/val_y.npy), [processed/test_y.npy](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/test_y.npy)
  Integer-encoded labels corresponding to the three paintings.

- [processed/mlp_model.npz](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_model.npz)
  Saved trained weights and label names for the MLP.

- [processed/mlp_training_report.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_training_report.md)
  Human-readable report with accuracy, per-class scores, and confusion matrices.

- [processed/mlp_weight_summary.md](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_weight_summary.md)
  Human-readable summary of the learned weights and the strongest feature patterns.

- [processed/mlp_metrics.json](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/processed/mlp_metrics.json)
  Machine-readable evaluation metrics for train, validation, and test splits.

## High-Level Pipeline

The complete pipeline works in this order:

1. Raw survey CSV data is cleaned and converted into numeric features.
2. Those features are split into training, validation, and test sets.
3. The MLP is trained on the training set.
4. The validation set is used to choose when to stop training.
5. The test set is used for final evaluation.
6. For future predictions, new CSV rows are preprocessed in exactly the same way as training data.
7. The trained MLP predicts one of the three painting labels.

## Step 1: Preprocessing

The preprocessing logic lives in [preprocess_dataset.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/preprocess_dataset.py). Its job is to take survey responses and convert them into machine-learning features.

### 1. Text Features

Several survey questions are free-response text. These are cleaned and combined into one text field per row.

Important steps:

- Text is lowercased.
- Non-ASCII characters are normalized away.
- Punctuation is removed.
- Some very common words are filtered out through a stopword list.
- Unigrams and bigrams are extracted.
- TF-IDF style weighting is applied.

This produces a sparse matrix where each column corresponds to a text feature such as a word or two-word phrase.

### 2. Categorical Features

Some survey answers are categories, such as season or room. These are split into label values and converted into one-hot or multi-hot encoded columns.

That means:

- each possible category becomes its own feature column
- if a row contains that category, the column gets a `1`
- otherwise it gets a `0`

### 3. Numeric Features

Numeric-like columns such as intensity or price are cleaned and standardized.

This includes:

- converting strings into numbers
- handling missing values
- clipping extreme values when appropriate
- applying `log1p` to skewed quantities such as price
- standardizing features so they have a more stable numeric range
- adding missing-value indicators

### 4. Combining Features

The final feature matrix is built by horizontally stacking:

- text features
- categorical features
- numeric features

The result is a single feature vector per survey response.

## Step 2: Train / Validation / Test Splits

The processed data is split into three parts:

- training split
- validation split
- test split

This project splits by `unique_id`, not just by row. That matters because each person may have answered questions for multiple paintings. Splitting by person prevents leakage where the model sees part of one person’s answers during training and another part during testing.

The approximate split ratios are:

- train: `70%`
- validation: `20%`
- test: `10%`

## Why The MLP Uses Softmax

The output is one of three paintings, so the model must choose among three classes.

A softmax layer is used because:

- it produces one score for each painting
- it converts those scores into probabilities
- the probabilities sum to `1`
- the highest probability becomes the predicted class

This is the multiclass extension of logistic regression. In other words, for this project:

- binary logistic regression would be for two classes
- softmax is the appropriate version for three classes

## MLP Architecture

The model in [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py) is a small feedforward neural network with one hidden layer.

Its structure is:

1. Input layer
   Takes in the full preprocessed feature vector.

2. Hidden linear layer
   Multiplies the input by `W1` and adds `b1`.

3. ReLU activation
   Replaces negative hidden values with zero.

4. Output linear layer
   Multiplies the hidden representation by `W2` and adds `b2`.

5. Softmax
   Converts the output logits into class probabilities.

Mathematically:

```text
h_linear = XW1 + b1
h = ReLU(h_linear)
logits = hW2 + b2
probs = softmax(logits)
```

### Parameter Shapes

If:

- `D` = number of input features
- `H` = hidden layer size
- `C` = number of classes

then:

- `W1` has shape `(D, H)`
- `b1` has shape `(H,)`
- `W2` has shape `(H, C)`
- `b2` has shape `(C,)`

For this project:

- `C = 3`
- the default hidden size is `128`

## Why ReLU Is Used

ReLU is the hidden activation:

```text
ReLU(x) = max(0, x)
```

It is a common choice because it:

- is simple and fast
- helps the network learn nonlinear patterns
- avoids some of the optimization problems of older activations like sigmoid or tanh

Without a nonlinear activation like ReLU, stacking linear layers would still behave like a single linear model.

## Loss Function

The training objective is **cross-entropy loss** for multiclass classification, plus **L2 regularization**.

### Cross-Entropy

Cross-entropy compares:

- the predicted probability distribution from softmax
- the true class label

If the model assigns high probability to the correct painting, the loss is low. If it assigns low probability to the correct painting, the loss is high.

### L2 Regularization

The loss also includes weight decay:

```text
0.5 * lambda * (||W1||^2 + ||W2||^2)
```

This discourages overly large weights and helps reduce overfitting.

## Training Process

The training loop in [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py) is implemented manually using NumPy.

### 1. Load Processed Data

The script loads:

- `train_X.npz`, `train_y.npy`
- `val_X.npz`, `val_y.npy`
- `test_X.npz`, `test_y.npy`

The saved matrices are sparse, but the MLP converts them to dense arrays using `.toarray()` because the network uses dense matrix multiplication.

### 2. Initialize Parameters

Weights are initialized randomly with scale based on layer size:

- `W1` uses a He-style scaling based on the input dimension
- `W2` uses a similar scaling based on the hidden dimension
- biases start at zero

This helps keep activations numerically stable at the beginning of training.

### 3. Shuffle Training Data Each Epoch

At the start of each epoch, training rows are shuffled. This prevents batches from always appearing in the same order.

### 4. Mini-Batch Gradient Descent

Training is done in mini-batches, not one example at a time.

For each batch:

1. Run forward propagation.
2. Compute softmax probabilities.
3. Compute cross-entropy loss.
4. Compute gradients using backpropagation.
5. Update the parameters with gradient descent.

### 5. Backpropagation

The code manually computes gradients for:

- `W2`, `b2`
- hidden activations
- `W1`, `b1`

This is the learning step that tells the model how to adjust weights so predictions improve over time.

### 6. Validation Accuracy

After each epoch, the code computes:

- training accuracy
- validation accuracy

The validation set is not used to update weights. It is only used to estimate how well the model generalizes.

### 7. Early Stopping

The training loop uses early stopping:

- if validation accuracy improves, the current weights are saved as the best checkpoint
- if validation accuracy does not improve for a fixed number of epochs, training stops

This prevents training for too long once the model stops improving on unseen validation data.

## Evaluation

After training finishes, the best validation checkpoint is restored and evaluated on:

- training set
- validation set
- test set

This gives three accuracy values:

- training accuracy measures fit on seen data
- validation accuracy measures tuning performance
- test accuracy estimates final performance on held-out data

In the tested run, the model reached approximately:

- train accuracy: `0.9271`
- validation accuracy: `0.8750`
- test accuracy: `0.8596`

Those numbers can change somewhat depending on hyperparameters and randomness, but they show the MLP is learning meaningful patterns.

## Prediction Flow For New CSV Files

The function `predict_all` in [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py) handles predictions on raw CSV files.

### What Happens Inside `predict_all`

1. Load the saved preprocessor from `processed/preprocessor.pkl`.
2. Load the saved MLP weights from `processed/mlp_model.npz`.
3. Read the input CSV with pandas.
4. Recreate the same text, categorical, and numeric features used during training.
5. Convert the row into a dense input matrix.
6. Run the MLP forward pass.
7. Take the class with highest softmax probability.
8. Convert the class index back into a painting name.

The return value is a Python list of painting labels, one for each row in the CSV.

## Why `pred_mlp.py` Exists

Some class projects or autograders expect a file exposing a function named `predict_all`.

[pred_mlp.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/pred_mlp.py) exists for that reason. It simply imports and re-exports the `predict_all` function from [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py).

That keeps:

- the full training logic in one place
- the submission interface simple

## Why The Script Handles Pickle Compatibility

The saved `preprocessor.pkl` contains custom classes defined in [preprocess_dataset.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/preprocess_dataset.py), such as:

- `NumericColumnSpec`
- `NumericPreprocessor`
- `SimpleTfidfVectorizer`
- `SimpleMultiLabelBinarizer`
- `SimpleLabelEncoder`

Depending on how pickle was created, loading it from another script can fail unless those class names are available under `__main__`.

That is why [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py) includes `ensure_preprocessor_pickle_compatibility()`. It registers those preprocessing classes before calling `pickle.load`.

Without that step, the model could fail while trying to reuse the saved preprocessing pipeline.

## Hyperparameters

Default values in [mlp_model.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/mlp_model.py):

- hidden layer size: `128`
- learning rate: `0.05`
- weight decay: `1e-4`
- batch size: `64`
- max epochs: `300`
- early stopping patience: `25`
- random seed: `311`

These are not guaranteed to be optimal. They are just a sensible starting point.

### What Each Hyperparameter Does

- `hidden layer size = 128`
  This is how many neurons are in the hidden layer. A larger hidden layer can learn more complex relationships, but it also increases the chance of overfitting and makes training heavier. `128` is a balanced starting point for this dataset size.

- `learning rate = 0.05`
  This controls how far the weights move on each update. Too small makes learning very slow. Too large can make training unstable. `0.05` worked as a practical fixed step size for this NumPy implementation.

- `weight decay = 1e-4`
  This is L2 regularization. It discourages very large weights, which helps reduce overfitting without overpowering the main learning objective.

- `batch size = 64`
  This is the number of examples used per gradient step. `64` is a common middle ground: stable enough to train smoothly, but still small enough to add some useful randomness.

- `max epochs = 300`
  This is the training ceiling. The model usually stops earlier because of early stopping, but this gives it enough room to improve if needed.

- `early stopping patience = 25`
  This tells the script to stop if validation accuracy does not improve for 25 epochs in a row. It helps avoid wasting time once the model plateaus.

- `random seed = 311`
  This keeps weight initialization and data shuffling reproducible so runs are easier to compare.

## How To Train The Model

Run:

```powershell
python mlp_model.py --train
```

This will:

- load the processed splits
- train the MLP
- print epoch-by-epoch metrics
- save the best model to `processed/mlp_model.npz`
- save a readable report to `processed/mlp_training_report.md`
- save a readable weight summary to `processed/mlp_weight_summary.md`
- save structured metrics to `processed/mlp_metrics.json`
- print final accuracy metrics as JSON

## How To Make Predictions

Run:

```powershell
python mlp_model.py --predict data/training_data_202601.csv
```

This will:

- load the trained MLP
- preprocess the CSV rows
- output a JSON list of painting predictions

If another script or grader needs a `predict_all(filename)` function, it can import from [pred_mlp.py](c:/Users/Franc/OneDrive/Documents/UofT/Winter%202026/CSC311/CSC311-Group-Project/pred_mlp.py).

## Strengths Of This MLP Approach

- It can model nonlinear relationships between survey features and painting labels.
- It combines text, categorical, and numeric information in one classifier.
- It does not rely on external ML libraries like scikit-learn.
- It supports both training and raw-CSV prediction in one pipeline.

## Limitations

- The input matrix is converted from sparse to dense, which may become expensive if the feature space grows much larger.
- It uses only one hidden layer, so capacity is limited compared with deeper networks.
- Optimization is basic gradient descent with fixed learning rate.
- There is no dropout, batch normalization, or more advanced regularization.
- It depends on preprocessing quality; if the engineered features are weak, the MLP can only do so much.

## Ideas For Future Improvement

- tune hidden size, learning rate, batch size, and regularization
- add a second hidden layer
- add dropout
- track confusion matrix and per-class accuracy
- compare against logistic regression, naive Bayes, and random forest on the same splits
- build an ensemble by averaging probabilities from multiple models

## Summary

This project’s MLP pipeline works like this:

1. Convert raw survey responses into text, categorical, and numeric features.
2. Train a one-hidden-layer neural network on the processed training data.
3. Use softmax to predict one of the three paintings.
4. Save the trained model.
5. Reuse the saved preprocessing pipeline and model weights to predict labels for new CSV rows.

Conceptually, this is a multiclass neural-network classifier built on top of your preprocessing system. It is more flexible than a purely linear model, while still being small enough to understand and explain clearly in a group project.
