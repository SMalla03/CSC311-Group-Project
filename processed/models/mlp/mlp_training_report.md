# MLP Training Report

## Run Summary

- Feature subset: `full`
- Selected feature count: `2722`
- Hidden size: `128`
- Learning rate: `0.01`
- Weight decay: `0.0001`
- Batch size: `64`
- Max epochs: `300`
- Patience: `25`
- Seed: `311`
- Best epoch: `27`
- Solver: `adam`
- Activation: `relu`

## Train Metrics

- Accuracy: `0.9296`
- Macro F1: `0.9297`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9657 | 0.9313 | 0.9482 | 393 |
| The Starry Night | 0.9177 | 0.9084 | 0.9130 | 393 |
| The Water Lily Pond | 0.9075 | 0.9491 | 0.9279 | 393 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 366 | 14 | 13 |
| The Starry Night | 11 | 357 | 25 |
| The Water Lily Pond | 2 | 18 | 373 |

## Validation Metrics

- Accuracy: `0.8810`
- Macro F1: `0.8806`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9252 | 0.8839 | 0.9041 | 112 |
| The Starry Night | 0.8762 | 0.8214 | 0.8479 | 112 |
| The Water Lily Pond | 0.8468 | 0.9375 | 0.8898 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 99 | 7 | 6 |
| The Starry Night | 7 | 92 | 13 |
| The Water Lily Pond | 1 | 6 | 105 |

## Test Metrics

- Accuracy: `0.9006`
- Macro F1: `0.9001`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9167 | 0.9649 | 0.9402 | 57 |
| The Starry Night | 0.8750 | 0.8596 | 0.8673 | 57 |
| The Water Lily Pond | 0.9091 | 0.8772 | 0.8929 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 55 | 1 | 1 |
| The Starry Night | 4 | 49 | 4 |
| The Water Lily Pond | 1 | 6 | 50 |
