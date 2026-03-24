# Majority Vote Ensemble Report

Models included: `naive_bayes`, `random_forest`, `logistic`.
Tie-breaking rule: sum the tied classes' predicted probabilities across the included models and choose the larger total.

## Train Results

- Ensemble accuracy: `0.9796`
- Ties resolved by summed probabilities: `0`

### Per-Model Accuracy

| Model | Accuracy |
| --- | ---: |
| naive_bayes | 0.9483 |
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

- Ensemble accuracy: `0.8958`
- Ties resolved by summed probabilities: `3`

### Per-Model Accuracy

| Model | Accuracy |
| --- | ---: |
| naive_bayes | 0.8155 |
| random_forest | 0.9018 |
| logistic | 0.8720 |

### Ensemble Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9182 | 0.9018 | 0.9099 | 112 |
| The Starry Night | 0.9109 | 0.8214 | 0.8638 | 112 |
| The Water Lily Pond | 0.8640 | 0.9643 | 0.9114 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 101 | 7 | 4 |
| The Starry Night | 7 | 92 | 13 |
| The Water Lily Pond | 2 | 2 | 108 |


## Test Results

- Ensemble accuracy: `0.9006`
- Ties resolved by summed probabilities: `1`

### Per-Model Accuracy

| Model | Accuracy |
| --- | ---: |
| naive_bayes | 0.8129 |
| random_forest | 0.9006 |
| logistic | 0.8830 |

### Ensemble Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9167 | 0.9649 | 0.9402 | 57 |
| The Starry Night | 0.9200 | 0.8070 | 0.8598 | 57 |
| The Water Lily Pond | 0.8689 | 0.9298 | 0.8983 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 55 | 1 | 1 |
| The Starry Night | 4 | 46 | 7 |
| The Water Lily Pond | 1 | 3 | 53 |

