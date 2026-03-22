# MLP Training Report

## Run Summary

- Feature subset: `full`
- Selected feature count: `2722`
- Hidden size: `128`
- Learning rate: `0.05`
- Weight decay: `0.0001`
- Batch size: `64`
- Max epochs: `80`
- Patience: `12`
- Seed: `311`
- Best epoch: `43`

## Train Metrics

- Accuracy: `0.9271`
- Macro F1: `0.9271`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9683 | 0.9338 | 0.9508 | 393 |
| The Starry Night | 0.9328 | 0.8830 | 0.9072 | 393 |
| The Water Lily Pond | 0.8855 | 0.9644 | 0.9233 | 393 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 367 | 12 | 14 |
| The Starry Night | 11 | 347 | 35 |
| The Water Lily Pond | 1 | 13 | 379 |

## Validation Metrics

- Accuracy: `0.8750`
- Macro F1: `0.8739`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9167 | 0.8839 | 0.9000 | 112 |
| The Starry Night | 0.8713 | 0.7857 | 0.8263 | 112 |
| The Water Lily Pond | 0.8425 | 0.9554 | 0.8954 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 99 | 9 | 4 |
| The Starry Night | 8 | 88 | 16 |
| The Water Lily Pond | 1 | 4 | 107 |

## Test Metrics

- Accuracy: `0.8596`
- Macro F1: `0.8585`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.8644 | 0.8947 | 0.8793 | 57 |
| The Starry Night | 0.8462 | 0.7719 | 0.8073 | 57 |
| The Water Lily Pond | 0.8667 | 0.9123 | 0.8889 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 51 | 5 | 1 |
| The Starry Night | 6 | 44 | 7 |
| The Water Lily Pond | 2 | 3 | 52 |
