# Random Forest Training Report

## Run Summary

- Feature subset: `full`
- Selected feature count: `2722`
- Number of trees: `50`
- Max depth: `None`
- Min samples split: `2`
- Min samples leaf: `1`
- Max features per split: `sqrt`
- Seed: `311`

## Train Metrics

- Accuracy: `0.9796`
- Macro F1: `0.9798`

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

## Validation Metrics

- Accuracy: `0.9018`
- Macro F1: `0.9011`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9273 | 0.9107 | 0.9189 | 112 |
| The Starry Night | 0.9029 | 0.8304 | 0.8651 | 112 |
| The Water Lily Pond | 0.8780 | 0.9643 | 0.9191 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 102 | 7 | 3 |
| The Starry Night | 7 | 93 | 12 |
| The Water Lily Pond | 1 | 3 | 108 |

## Test Metrics

- Accuracy: `0.9006`
- Macro F1: `0.8996`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9322 | 0.9649 | 0.9483 | 57 |
| The Starry Night | 0.9200 | 0.8070 | 0.8598 | 57 |
| The Water Lily Pond | 0.8548 | 0.9298 | 0.8908 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 55 | 1 | 1 |
| The Starry Night | 3 | 46 | 8 |
| The Water Lily Pond | 1 | 3 | 53 |
