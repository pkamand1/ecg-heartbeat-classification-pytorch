| Model | Macro F1 | Accuracy | N Se | N PPV | S Se | S PPV | V Se | V PPV | F Se | F PPV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression, RR ratios only | 0.461 | 0.770 | 0.787 | 0.982 | 0.513 | 0.263 | 0.781 | 0.516 | 0.018 | 0.001 |
| Gradient boosting, RR + beat shape | 0.493 | 0.892 | 0.924 | 0.986 | 0.208 | 0.179 | 0.931 | 0.544 | 0.175 | 0.117 |
| Gradient boosting, RR + beat shape + patient template | 0.662 | 0.949 | 0.974 | 0.973 | 0.538 | 0.501 | 0.931 | 0.920 | 0.175 | 0.342 |
| BeatNet, beat shape only (1 seed) | 0.377 | 0.777 | 0.799 | 0.976 | 0.102 | 0.047 | 0.939 | 0.377 | 0.062 | 0.016 |
| BeatNet, beat shape + RR (1 seed) | 0.467 | 0.852 | 0.872 | 0.978 | 0.363 | 0.205 | 0.942 | 0.507 | 0.041 | 0.016 |
| BeatNet, beat shape + RR + patient template (3 seeds, mean ± SD) | 0.525 ± 0.015 | 0.914 ± 0.004 | 0.950 ± 0.005 | 0.957 ± 0.001 | 0.155 ± 0.023 | 0.176 ± 0.011 | 0.958 ± 0.003 | 0.814 ± 0.012 | 0.095 ± 0.046 | 0.119 ± 0.069 |

Random beat split, for contrast only (same patients in train and test):

| Model | Macro F1 | Accuracy | N Se | N PPV | S Se | S PPV | V Se | V PPV | F Se | F PPV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BeatNet, beat shape + RR + patient template (1 seed) | 0.929 | 0.991 | 0.994 | 0.998 | 0.943 | 0.893 | 0.979 | 0.973 | 0.912 | 0.753 |
