# Majority Vote Ensemble Report

Models included: `naive_bayes`, `random_forest`, `logistic`.
Tie-breaking rule: sum the tied classes' predicted probabilities across the included models and choose the larger total.

## Train Results

- Ensemble accuracy: `0.9796`
- Ties resolved by summed probabilities: `1`

### Per-Model Accuracy

| Model | Accuracy |
| --- | ---: |
| naive_bayes | 0.9389 |
| random_forest | 0.9796 |
| logistic | 0.9729 |

### Ensemble Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 1.0000 | 0.9746 | 0.9871 | 393 |
| The Starry Night | 1.0000 | 0.9644 | 0.9819 | 393 |
| The Water Lily Pond | 0.9424 | 1.0000 | 0.9704 | 393 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 383 | 0 | 10 |
| The Starry Night | 0 | 379 | 14 |
| The Water Lily Pond | 0 | 0 | 393 |


## Validation Results

- Ensemble accuracy: `0.8899`
- Ties resolved by summed probabilities: `2`

### Per-Model Accuracy

| Model | Accuracy |
| --- | ---: |
| naive_bayes | 0.8542 |
| random_forest | 0.9018 |
| logistic | 0.8720 |

### Ensemble Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9252 | 0.8839 | 0.9041 | 112 |
| The Starry Night | 0.9000 | 0.8036 | 0.8491 | 112 |
| The Water Lily Pond | 0.8527 | 0.9821 | 0.9129 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 99 | 9 | 4 |
| The Starry Night | 7 | 90 | 15 |
| The Water Lily Pond | 1 | 1 | 110 |


## Test Results

- Ensemble accuracy: `0.9064`
- Ties resolved by summed probabilities: `1`

### Per-Model Accuracy

| Model | Accuracy |
| --- | ---: |
| naive_bayes | 0.8655 |
| random_forest | 0.9006 |
| logistic | 0.8830 |

### Ensemble Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9322 | 0.9649 | 0.9483 | 57 |
| The Starry Night | 0.9388 | 0.8070 | 0.8679 | 57 |
| The Water Lily Pond | 0.8571 | 0.9474 | 0.9000 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 55 | 1 | 1 |
| The Starry Night | 3 | 46 | 8 |
| The Water Lily Pond | 1 | 2 | 54 |

