# Validation of the updated presets (held-out 30%)

| group | model | n | median rchi2 | 90% rchi2 | limits per fit | evaluations | time (s) | BIC better than old | median dBIC |
|---|---|---|---|---|---|---|---|---|---|
| t1_low | old default | 89 | 1.346 | 2.10 | 2.85 | 31 | 18.1 | 0% | 0.0 |
| t1_low | new default | 89 | 1.249 | 1.89 | 3.07 | 35 | 24.9 | 82% | -83.5 |
| t1_low | classify -> recommended | 68 | 1.103 | 1.53 | 1.76 | 34 | 27.8 | 82% | -176.4 |
| t1_hb | old default | 61 | 1.022 | 1.31 | 1.05 | 24 | 3.3 | 0% | 0.0 |
| t1_hb | new default | 61 | 0.964 | 1.27 | 1.03 | 27 | 3.1 | 66% | -21.1 |
| t1_mgii | old default | 65 | 0.899 | 1.61 | 1.09 | 25 | 0.2 | 0% | 0.0 |
| t1_mgii | new default | 65 | 0.895 | 1.62 | 0.29 | 20 | 0.2 | 77% | -12.2 |
| t1_civ | old default | 49 | 1.757 | 7.86 | 1.43 | 13 | 0.1 | 0% | 0.0 |
| t1_civ | new default | 49 | 1.523 | 7.32 | 0.73 | 24 | 0.2 | 98% | -171.4 |
| sy2 | old default | 46 | 1.010 | 1.32 | 0.59 | 14 | 9.6 | 0% | 0.0 |
| sy2 | new default | 46 | 1.010 | 1.32 | 0.59 | 14 | 11.8 | 0% | 0.0 |
| liner | old default | 44 | 0.976 | 1.22 | 0.00 | 7 | 2.8 | 0% | 0.0 |
| liner | new default | 44 | 0.976 | 1.22 | 0.00 | 7 | 3.2 | 0% | 0.0 |
| comp | old default | 24 | 0.998 | 1.33 | 1.12 | 12 | 10.1 | 0% | 0.0 |
| comp | new default | 24 | 1.010 | 1.37 | 0.04 | 8 | 3.7 | 83% | -38.2 |
| sf | old default | 46 | 1.083 | 1.24 | 0.98 | 18 | 6.9 | 0% | 0.0 |
| sf | new default | 46 | 1.095 | 1.29 | 0.00 | 8 | 2.7 | 76% | -37.9 |
| passive | old default | 25 | 0.977 | 1.10 | 0.80 | 9 | 2.0 | 0% | 0.0 |
| passive | new default | 25 | 0.977 | 1.10 | 0.80 | 9 | 2.5 | 0% | 0.0 |
