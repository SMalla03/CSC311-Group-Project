# Logistic Regression Training Report

## Run Summary

- Feature subset: `full`
- Selected feature count: `2722`
- Regularization strength C: `2.0`
- Max iterations: `2000`
- Class weight: `none`
- Seed: `311`
- Solver: `lbfgs`
- Multi-class: `multinomial`

## Train Metrics

- Accuracy: `0.9729`
- Macro F1: `0.9730`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9948 | 0.9695 | 0.9820 | 393 |
| The Starry Night | 0.9921 | 0.9542 | 0.9728 | 393 |
| The Water Lily Pond | 0.9354 | 0.9949 | 0.9642 | 393 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 381 | 1 | 11 |
| The Starry Night | 2 | 375 | 16 |
| The Water Lily Pond | 0 | 2 | 391 |

## Validation Metrics

- Accuracy: `0.8720`
- Macro F1: `0.8707`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9065 | 0.8661 | 0.8858 | 112 |
| The Starry Night | 0.8713 | 0.7857 | 0.8263 | 112 |
| The Water Lily Pond | 0.8438 | 0.9643 | 0.9000 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 97 | 11 | 4 |
| The Starry Night | 8 | 88 | 16 |
| The Water Lily Pond | 2 | 2 | 108 |

## Test Metrics

- Accuracy: `0.8830`
- Macro F1: `0.8812`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.8852 | 0.9474 | 0.9153 | 57 |
| The Starry Night | 0.8980 | 0.7719 | 0.8302 | 57 |
| The Water Lily Pond | 0.8689 | 0.9298 | 0.8983 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 54 | 2 | 1 |
| The Starry Night | 6 | 44 | 7 |
| The Water Lily Pond | 1 | 3 | 53 |
