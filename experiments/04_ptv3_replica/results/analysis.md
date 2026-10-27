# Instance structure of PTv3 tokens and features

Input: 2 cm grid sample of the 1 cm Replica clouds. Stage s pools by 2^s (stage 4 ~ 32 cm cells). Purity / coverage depend only on the pooling (identical for both models); AUC compares features.

| scene | s0 tokens / purity / coverage | s1 tokens / purity / coverage | s2 tokens / purity / coverage | s3 tokens / purity / coverage | s4 tokens / purity / coverage |
|---|---|---|---|---|---|
| room_0 | 444,744 / 1.000 / 1.00 | 112,882 / 0.982 / 0.99 | 27,916 / 0.960 / 0.98 | 6,705 / 0.915 / 0.81 | 1,496 / 0.824 / 0.62 |
| office_3 | 547,032 / 1.000 / 1.00 | 143,117 / 0.987 / 0.96 | 35,591 / 0.964 / 0.82 | 8,454 / 0.911 / 0.70 | 1,954 / 0.838 / 0.53 |
| frl_apartment_0 | 809,718 / 1.000 / 1.00 | 206,847 / 0.984 / 1.00 | 51,107 / 0.963 / 0.95 | 12,362 / 0.910 / 0.76 | 2,797 / 0.820 / 0.46 |
| hotel_0 | 513,830 / 1.000 / 1.00 | 129,711 / 0.979 / 0.96 | 31,331 / 0.948 / 0.92 | 7,548 / 0.906 / 0.71 | 1,652 / 0.791 / 0.54 |
| apartment_1 | 890,201 / 1.000 / 1.00 | 225,067 / 0.986 / 0.95 | 54,923 / 0.957 / 0.92 | 13,255 / 0.907 / 0.81 | 2,969 / 0.789 / 0.72 |

## Same-instance AUC of per-point features (pairs < 0.5 m apart)

Distance-controlled: AUC within distance bins of 5-10 cm, averaged (raw AUC in brackets is confounded by distance).

| scene | sonata | ptv3_scratch |
|---|---|---|
| room_0 | 0.725 (0.751) | 0.724 (0.750) |
| office_3 | 0.703 (0.738) | 0.682 (0.715) |
| frl_apartment_0 | 0.739 (0.765) | 0.707 (0.737) |
| hotel_0 | 0.740 (0.766) | 0.729 (0.758) |
| apartment_1 | 0.684 (0.715) | 0.755 (0.773) |
