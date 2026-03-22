# MLP Weight Summary

This file is a readable summary of the learned weights from scikit-learn's `MLPClassifier`.
The full trained model is stored in `processed/models/mlp/mlp_model.joblib`, but this report highlights the strongest patterns.

## Output Layer

Each class score is computed from the hidden layer through the second weight matrix and output biases.

### The Persistence of Memory

- Output bias: `0.008585`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 93 | -0.271620 |
| 23 | 0.266124 |
| 5 | -0.261505 |
| 90 | 0.261393 |
| 61 | 0.261125 |
| 120 | -0.260748 |
| 111 | -0.260518 |
| 91 | -0.257131 |
| 28 | 0.254500 |
| 6 | 0.252913 |
| 29 | -0.252871 |
| 115 | -0.251629 |

### The Starry Night

- Output bias: `0.059494`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 3 | 0.279492 |
| 72 | -0.277073 |
| 9 | -0.273161 |
| 64 | -0.266746 |
| 81 | -0.265117 |
| 8 | -0.263738 |
| 57 | -0.263716 |
| 37 | -0.263334 |
| 99 | -0.263020 |
| 114 | 0.261186 |
| 85 | -0.260549 |
| 21 | 0.260350 |

### The Water Lily Pond

- Output bias: `0.049299`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 24 | -0.286922 |
| 30 | -0.281593 |
| 73 | -0.266324 |
| 25 | -0.261632 |
| 124 | 0.254025 |
| 77 | -0.253224 |
| 60 | -0.250959 |
| 127 | 0.248426 |
| 0 | -0.247839 |
| 85 | 0.245738 |
| 82 | -0.244132 |
| 119 | -0.240647 |


## Hidden Units And Their Strongest Input Features

These are the hidden units with the strongest downstream effect on at least one class.

### Hidden Unit 24

- Hidden bias: `0.056570`
- Strongest output class: `The Water Lily Pond`
- Output weight to that class: `-0.286922`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:how time | -0.099056 | 0.099056 |
| text:stars | 0.088719 | 0.088719 |
| text:ice cream | 0.084407 | 0.084407 |
| text:emotional | 0.081995 | 0.081995 |
| text:night | 0.081140 | 0.081140 |
| text:hum | -0.080955 | 0.080955 |
| text:awe | 0.080530 | 0.080530 |
| text:cream | 0.079462 | 0.079462 |
| text:think how | -0.079456 | 0.079456 |
| text:nature | -0.078843 | 0.078843 |
| text:salad | -0.078181 | 0.078181 |
| text:warped clocks | -0.077847 | 0.077847 |

### Hidden Unit 30

- Hidden bias: `0.068552`
- Strongest output class: `The Water Lily Pond`
- Output weight to that class: `-0.281593`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:ice cream | 0.086939 | 0.086939 |
| room=Bedroom | 0.084634 | 0.084634 |
| text:happy | -0.084315 | 0.084315 |
| text:stars | 0.083554 | 0.083554 |
| text:city | 0.079875 | 0.079875 |
| text:sad | 0.078662 | 0.078662 |
| text:patterns | 0.075708 | 0.075708 |
| text:icecream | 0.075697 | 0.075697 |
| text:rising falling | 0.075526 | 0.075526 |
| text:same time | 0.075456 | 0.075456 |
| text:birds chirping | -0.074600 | 0.074600 |
| text:soup | 0.073798 | 0.073798 |

### Hidden Unit 3

- Hidden bias: `-0.002892`
- Strongest output class: `The Starry Night`
- Output weight to that class: `0.279492`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Winter | 0.106951 | 0.106951 |
| text:night sky | 0.084183 | 0.084183 |
| text:swirling | 0.082247 | 0.082247 |
| text:ice | 0.081673 | 0.081673 |
| text:sky | 0.081208 | 0.081208 |
| text:cheerful | -0.079538 | 0.079538 |
| text:cold | 0.078418 | 0.078418 |
| text:due | 0.077695 | 0.077695 |
| text:nature | -0.075691 | 0.075691 |
| text:blueberry pie | 0.073330 | 0.073330 |
| text:cream | 0.073271 | 0.073271 |
| text:lively | 0.073254 | 0.073254 |

### Hidden Unit 72

- Hidden bias: `0.044266`
- Strongest output class: `The Starry Night`
- Output weight to that class: `-0.277073`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Winter | -0.097681 | 0.097681 |
| text:without | 0.085946 | 0.085946 |
| text:here | 0.084635 | 0.084635 |
| text:garden | 0.083441 | 0.083441 |
| text:sky | -0.080284 | 0.080284 |
| text:gentle | 0.076208 | 0.076208 |
| text:moments | 0.076194 | 0.076194 |
| text:sand | 0.076112 | 0.076112 |
| text:empty | 0.075067 | 0.075067 |
| text:beneath | 0.074824 | 0.074824 |
| text:only | 0.074164 | 0.074164 |
| text:ticking | 0.073721 | 0.073721 |

### Hidden Unit 9

- Hidden bias: `0.017100`
- Strongest output class: `The Starry Night`
- Output weight to that class: `-0.273161`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Fall | 0.100134 | 0.100134 |
| text:time | 0.096294 | 0.096294 |
| text:sky | -0.091227 | 0.091227 |
| text:melting | 0.085648 | 0.085648 |
| text:how time | 0.084504 | 0.084504 |
| text:everything | 0.081641 | 0.081641 |
| text:passage | 0.078700 | 0.078700 |
| text:night | -0.078124 | 0.078124 |
| text:somewhat | 0.077064 | 0.077064 |
| text:concept | 0.075308 | 0.075308 |
| text:space | 0.075144 | 0.075144 |
| text:hope | 0.074334 | 0.074334 |

### Hidden Unit 93

- Hidden bias: `0.026132`
- Strongest output class: `The Persistence of Memory`
- Output weight to that class: `-0.271620`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:wonder | 0.097982 | 0.097982 |
| text:stars | 0.092083 | 0.092083 |
| text:garden | -0.089053 | 0.089053 |
| text:awe | 0.085903 | 0.085903 |
| text:sky | 0.085374 | 0.085374 |
| text:swirling | 0.082294 | 0.082294 |
| text:blueberry | 0.081728 | 0.081728 |
| room=Bedroom | 0.081211 | 0.081211 |
| season=Spring | -0.079157 | 0.079157 |
| text:chocolate | 0.078553 | 0.078553 |
| text:ice cream | 0.077669 | 0.077669 |
| view_with=Friends | 0.077392 | 0.077392 |
