# MLP Weight Summary

This file is a readable summary of the learned weights from scikit-learn's `MLPClassifier`.
The full trained model is stored in `processed/models/mlp/mlp_model.joblib`, but this report highlights the strongest patterns.

## Output Layer

Each class score is computed from the hidden layer through the second weight matrix and output biases.

### The Persistence of Memory

- Output bias: `0.001623`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 5 | -0.320909 |
| 91 | -0.310106 |
| 115 | -0.303113 |
| 120 | -0.298700 |
| 111 | -0.285900 |
| 93 | -0.283629 |
| 42 | -0.279845 |
| 6 | 0.276366 |
| 118 | 0.275521 |
| 90 | 0.272021 |
| 65 | 0.271634 |
| 81 | -0.270496 |

### The Starry Night

- Output bias: `0.080792`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 21 | 0.293778 |
| 72 | -0.285843 |
| 9 | -0.278950 |
| 107 | 0.275549 |
| 30 | 0.269959 |
| 57 | -0.267646 |
| 99 | -0.266676 |
| 2 | -0.261986 |
| 58 | 0.258731 |
| 114 | 0.257917 |
| 81 | -0.257757 |
| 93 | 0.254936 |

### The Water Lily Pond

- Output bias: `0.018782`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 30 | -0.308622 |
| 73 | -0.290055 |
| 60 | -0.284201 |
| 25 | -0.281667 |
| 122 | -0.269079 |
| 24 | -0.267530 |
| 94 | 0.265124 |
| 44 | 0.253449 |
| 13 | -0.247954 |
| 26 | 0.245922 |
| 62 | -0.243749 |
| 104 | 0.243685 |


## Hidden Units And Their Strongest Input Features

These are the hidden units with the strongest downstream effect on at least one class.

### Hidden Unit 5

- Hidden bias: `0.111563`
- Strongest output class: `The Persistence of Memory`
- Output weight to that class: `-0.320909`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:time | -0.158760 | 0.158760 |
| text:soundtrack | 0.155844 | 0.155844 |
| prominent_colours | -0.141605 | 0.141605 |
| text:sky | 0.138800 | 0.138800 |
| text:green | 0.138039 | 0.138039 |
| text:full | 0.135781 | 0.135781 |
| text:jazz | 0.135019 | 0.135019 |
| room=Office | 0.134423 | 0.134423 |
| text:looking | 0.134145 | 0.134145 |
| text:ice cream | 0.131462 | 0.131462 |
| text:something | 0.131285 | 0.131285 |
| text:light | 0.131066 | 0.131066 |

### Hidden Unit 91

- Hidden bias: `0.084561`
- Strongest output class: `The Persistence of Memory`
- Output weight to that class: `-0.310106`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:calm | 0.157068 | 0.157068 |
| text:wind | 0.155410 | 0.155410 |
| text:time | -0.149774 | 0.149774 |
| text:joyful | 0.148996 | 0.148996 |
| text:awe | 0.143505 | 0.143505 |
| season=Winter | 0.143134 | 0.143134 |
| room=Bedroom | 0.136369 | 0.136369 |
| text:calm relaxed | 0.135700 | 0.135700 |
| text:hopeful | 0.134012 | 0.134012 |
| text:sounds | 0.132516 | 0.132516 |
| room=Bathroom | 0.132361 | 0.132361 |
| view_with=Family Members | 0.132272 | 0.132272 |

### Hidden Unit 30

- Hidden bias: `0.120042`
- Strongest output class: `The Water Lily Pond`
- Output weight to that class: `-0.308622`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| room=Bedroom | 0.158059 | 0.158059 |
| text:happy | -0.144865 | 0.144865 |
| text:dreamy | 0.141925 | 0.141925 |
| text:stars | 0.141212 | 0.141212 |
| text:ice cream | 0.140814 | 0.140814 |
| text:sad | 0.135264 | 0.135264 |
| text:rising falling | 0.134602 | 0.134602 |
| text:blueberry | 0.132385 | 0.132385 |
| text:nature | -0.130482 | 0.130482 |
| text:rising | 0.128139 | 0.128139 |
| text:instrumental | 0.127797 | 0.127797 |
| season=Winter | 0.127322 | 0.127322 |

### Hidden Unit 115

- Hidden bias: `0.077655`
- Strongest output class: `The Persistence of Memory`
- Output weight to that class: `-0.303113`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:time | -0.166004 | 0.166004 |
| text:low | -0.161057 | 0.161057 |
| text:light | 0.152993 | 0.152993 |
| text:bright | 0.144102 | 0.144102 |
| text:strawberry | 0.142000 | 0.142000 |
| text:upbeat | 0.141948 | 0.141948 |
| text:flute | 0.140189 | 0.140189 |
| room=Bedroom | 0.138072 | 0.138072 |
| text:salad | 0.136303 | 0.136303 |
| feel_sombre | -0.131756 | 0.131756 |
| text:happy | 0.128168 | 0.128168 |
| text:well | 0.128085 | 0.128085 |

### Hidden Unit 120

- Hidden bias: `0.037848`
- Strongest output class: `The Persistence of Memory`
- Output weight to that class: `-0.298700`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:joyful | 0.159622 | 0.159622 |
| season=Spring | 0.153618 | 0.153618 |
| text:flowing | 0.151371 | 0.151371 |
| view_with=Family Members | 0.150041 | 0.150041 |
| season=Winter | 0.146234 | 0.146234 |
| room=Bedroom | 0.142367 | 0.142367 |
| text:salad | 0.142114 | 0.142114 |
| text:flute | 0.139718 | 0.139718 |
| text:dreamy | 0.137685 | 0.137685 |
| text:night | 0.133809 | 0.133809 |
| text:peaceful | 0.133333 | 0.133333 |
| text:strawberry | 0.131017 | 0.131017 |

### Hidden Unit 21

- Hidden bias: `0.103037`
- Strongest output class: `The Starry Night`
- Output weight to that class: `0.293778`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| text:jazz | 0.156143 | 0.156143 |
| text:stars | 0.148130 | 0.148130 |
| text:cream | 0.147270 | 0.147270 |
| text:time | -0.143596 | 0.143596 |
| season=Winter | 0.143043 | 0.143043 |
| text:sky | 0.139534 | 0.139534 |
| text:tempo | 0.134780 | 0.134780 |
| text:background | 0.134306 | 0.134306 |
| text:see | 0.133481 | 0.133481 |
| room=Living Room | 0.133441 | 0.133441 |
| text:wonder | 0.129905 | 0.129905 |
| text:small | 0.128478 | 0.128478 |
