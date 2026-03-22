# Random Forest Training Report

## Run Summary

- Feature subset: `full`
- Selected feature count: `2722`
- Number of trees: `50`
- Max depth: `5`
- Min samples split: `2`
- Min samples leaf: `1`
- Max features per split: `sqrt`
- Seed: `311`

## Train Metrics

- Accuracy: `0.8830`
- Macro F1: `0.8816`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.8839 | 0.9491 | 0.9153 | 393 |
| The Starry Night | 0.8757 | 0.7888 | 0.8300 | 393 |
| The Water Lily Pond | 0.8883 | 0.9109 | 0.8995 | 393 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 373 | 15 | 5 |
| The Starry Night | 43 | 310 | 40 |
| The Water Lily Pond | 6 | 29 | 358 |

## Validation Metrics

- Accuracy: `0.8690`
- Macro F1: `0.8676`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.8667 | 0.9286 | 0.8966 | 112 |
| The Starry Night | 0.8431 | 0.7679 | 0.8037 | 112 |
| The Water Lily Pond | 0.8947 | 0.9107 | 0.9027 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 104 | 8 | 0 |
| The Starry Night | 14 | 86 | 12 |
| The Water Lily Pond | 2 | 8 | 102 |

## Test Metrics

- Accuracy: `0.8713`
- Macro F1: `0.8671`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.8710 | 0.9474 | 0.9076 | 57 |
| The Starry Night | 0.9091 | 0.7018 | 0.7921 | 57 |
| The Water Lily Pond | 0.8462 | 0.9649 | 0.9016 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 54 | 3 | 0 |
| The Starry Night | 7 | 40 | 10 |
| The Water Lily Pond | 1 | 1 | 55 |
