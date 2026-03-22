# MLP Training Report

## Run Summary

- Feature subset: `full`
- Selected feature count: `2722`
- Hidden size: `128`
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Batch size: `64`
- Max epochs: `300`
- Patience: `25`
- Seed: `311`
- Best epoch: `31`
- Solver: `adam`
- Activation: `relu`

## Train Metrics

- Accuracy: `0.9500`
- Macro F1: `0.9500`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9791 | 0.9542 | 0.9665 | 393 |
| The Starry Night | 0.9703 | 0.9135 | 0.9410 | 393 |
| The Water Lily Pond | 0.9061 | 0.9822 | 0.9426 | 393 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 375 | 5 | 13 |
| The Starry Night | 7 | 359 | 27 |
| The Water Lily Pond | 1 | 6 | 386 |

## Validation Metrics

- Accuracy: `0.8780`
- Macro F1: `0.8766`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.9174 | 0.8929 | 0.9050 | 112 |
| The Starry Night | 0.8878 | 0.7768 | 0.8286 | 112 |
| The Water Lily Pond | 0.8372 | 0.9643 | 0.8963 | 112 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 100 | 8 | 4 |
| The Starry Night | 8 | 87 | 17 |
| The Water Lily Pond | 1 | 3 | 108 |

## Test Metrics

- Accuracy: `0.8889`
- Macro F1: `0.8866`

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| The Persistence of Memory | 0.8889 | 0.9825 | 0.9333 | 57 |
| The Starry Night | 0.9556 | 0.7544 | 0.8431 | 57 |
| The Water Lily Pond | 0.8413 | 0.9298 | 0.8833 | 57 |

Confusion matrix (`rows=true`, `cols=predicted`):

| true \ pred | The Persistence of Memory | The Starry Night | The Water Lily Pond |
| --- | --- | --- | --- |
| The Persistence of Memory | 56 | 0 | 1 |
| The Starry Night | 5 | 43 | 9 |
| The Water Lily Pond | 2 | 2 | 53 |
