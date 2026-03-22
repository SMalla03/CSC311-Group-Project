# MLP Weight Summary

This file is a readable summary of the learned weights.
The full numeric arrays are still stored in `processed/mlp_model.npz`, but this report highlights the strongest patterns.

## Output Layer

Each class score is computed from the hidden layer through `W2` and `b2`.

### The Persistence of Memory

- Output bias: `0.049119`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 41 | 0.578274 |
| 80 | -0.396016 |
| 88 | 0.390716 |
| 55 | -0.388484 |
| 92 | -0.382723 |
| 5 | 0.347740 |
| 2 | 0.337816 |
| 33 | -0.336035 |
| 97 | -0.329294 |
| 40 | 0.327220 |
| 109 | -0.324318 |
| 72 | 0.321161 |

### The Starry Night

- Output bias: `0.154100`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 109 | 0.512159 |
| 64 | -0.500653 |
| 117 | -0.480070 |
| 11 | 0.450142 |
| 85 | 0.444474 |
| 28 | 0.443162 |
| 33 | 0.440833 |
| 37 | 0.400400 |
| 79 | -0.373553 |
| 24 | -0.368878 |
| 62 | -0.338928 |
| 121 | -0.336324 |

### The Water Lily Pond

- Output bias: `-0.203219`

| Hidden Unit | Weight To Class |
| --- | ---: |
| 92 | 0.463933 |
| 55 | 0.446984 |
| 26 | -0.435578 |
| 43 | 0.421151 |
| 53 | 0.375574 |
| 86 | -0.362672 |
| 84 | 0.362266 |
| 108 | -0.357075 |
| 58 | 0.348138 |
| 71 | -0.331025 |
| 83 | 0.329889 |
| 100 | 0.325678 |

## Hidden Units And Their Strongest Input Features

These are the hidden units with the strongest downstream effect on at least one class.

### Hidden Unit 41

- Strongest output class: `The Persistence of Memory`
- Output weight to that class: `0.578274`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Fall | 0.254937 | 0.254937 |
| feel_content | -0.159830 | 0.159830 |
| text:strong | -0.128481 | 0.128481 |
| room=Office | 0.117294 | 0.117294 |
| text:time | 0.117255 | 0.117255 |
| view_with=By Yourself | 0.112312 | 0.112312 |
| view_with=Coworkers/Classmates | -0.101376 | 0.101376 |
| season=Winter | 0.098019 | 0.098019 |
| feel_uneasy | 0.097695 | 0.097695 |
| text:dreaming | -0.096439 | 0.096439 |
| text:gentle soft | 0.095932 | 0.095932 |
| room=Living Room | 0.093283 | 0.093283 |

### Hidden Unit 109

- Strongest output class: `The Starry Night`
- Output weight to that class: `0.512159`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Winter | 0.300785 | 0.300785 |
| room=Bedroom | 0.212132 | 0.212132 |
| season=Fall | -0.146452 | 0.146452 |
| text:time | -0.103713 | 0.103713 |
| text:garlic | 0.101434 | 0.101434 |
| text:dreamlike atmosphere | -0.098752 | 0.098752 |
| view_with=Friends | 0.095970 | 0.095970 |
| text:awe | 0.094660 | 0.094660 |
| text:blueberries | 0.090837 | 0.090837 |
| text:sad yet | 0.090442 | 0.090442 |
| feel_uneasy | -0.089959 | 0.089959 |
| text:view | 0.089090 | 0.089090 |

### Hidden Unit 64

- Strongest output class: `The Starry Night`
- Output weight to that class: `-0.500653`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Winter | -0.134716 | 0.134716 |
| view_with=By Yourself | 0.127469 | 0.127469 |
| room=Bedroom | -0.109293 | 0.109293 |
| season=Spring | 0.096957 | 0.096957 |
| text:blueberry | -0.091438 | 0.091438 |
| text:clock | 0.088597 | 0.088597 |
| room=Bathroom | 0.088102 | 0.088102 |
| text:unsettled | 0.087914 | 0.087914 |
| text:reflects how | -0.086529 | 0.086529 |
| text:mystical | -0.083302 | 0.083302 |
| text:soft calm | -0.081900 | 0.081900 |
| text:type | -0.080356 | 0.080356 |

### Hidden Unit 117

- Strongest output class: `The Starry Night`
- Output weight to that class: `-0.480070`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Spring | 0.185308 | 0.185308 |
| feel_content | 0.178945 | 0.178945 |
| season=Winter | -0.156949 | 0.156949 |
| season=Fall | -0.134417 | 0.134417 |
| objects_caught_eye | -0.127523 | 0.127523 |
| text:salad | 0.097228 | 0.097228 |
| room=Dining Room | 0.095995 | 0.095995 |
| text:happy lively | 0.090534 | 0.090534 |
| text:beauty | -0.087684 | 0.087684 |
| text:wind blowing | -0.086709 | 0.086709 |
| text:insects | -0.086187 | 0.086187 |
| text:music violins | -0.085692 | 0.085692 |

### Hidden Unit 92

- Strongest output class: `The Water Lily Pond`
- Output weight to that class: `0.463933`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| objects_caught_eye | -0.281326 | 0.281326 |
| season=Spring | 0.204257 | 0.204257 |
| feel_content | 0.149105 | 0.149105 |
| season=Fall | -0.131126 | 0.131126 |
| room=Bedroom | 0.115357 | 0.115357 |
| view_with=Family Members | 0.103139 | 0.103139 |
| text:bells | -0.096091 | 0.096091 |
| text:happy | 0.095016 | 0.095016 |
| room=Dining Room | 0.094802 | 0.094802 |
| text:anxiety | 0.091621 | 0.091621 |
| text:salad | 0.091585 | 0.091585 |
| text:soft gentle | 0.091230 | 0.091230 |

### Hidden Unit 11

- Strongest output class: `The Starry Night`
- Output weight to that class: `0.450142`

| Input Feature | Weight | abs(weight) |
| --- | ---: | ---: |
| season=Winter | 0.186016 | 0.186016 |
| room=Bedroom | 0.175598 | 0.175598 |
| text:strongly | 0.105214 | 0.105214 |
| view_with=Friends | 0.097981 | 0.097981 |
| text:vocal | -0.087475 | 0.087475 |
| text:envision | 0.084592 | 0.084592 |
| text:melts | 0.081083 | 0.081083 |
| text:maybe upbeat | 0.080909 | 0.080909 |
| text:unease | 0.078124 | 0.078124 |
| text:conflicted | 0.076622 | 0.076622 |
| text:melodies | -0.076405 | 0.076405 |
| text:giving | -0.076288 | 0.076288 |
