# SDF speed benchmark

NVIDIA GeForce RTX 5090, torch 2.14.1+cu130, float32, dense M SQs x N points, chunks of 4,194,304 pairs. Median time in ms; forward / forward+backward (gradients w.r.t. all SQ parameters). Throughput in the last column is for the forward pass at the largest size.


## Forward (ms)

| method | M=10 N=1,000 | M=10 N=10,000 | M=10 N=100,000 | M=100 N=1,000 | M=100 N=10,000 | M=100 N=100,000 | M=1000 N=1,000 | M=1000 N=10,000 | M=1000 N=100,000 |
|---|---|---|---|---|---|---|---|---|---|
| radial | 0.15 | 0.13 | 0.20 | 0.13 | 0.20 | 1.48 | 0.21 | 1.70 | 16.05 |
| gauge_lb | 0.35 | 0.25 | 0.27 | 0.33 | 0.26 | 1.53 | 0.28 | 1.69 | 17.45 |
| taubin | 0.21 | 0.21 | 0.37 | 0.21 | 0.33 | 2.80 | 0.31 | 2.68 | 28.73 |
| newton_radial | 5.00 | 5.08 | 10.44 | 4.97 | 10.07 | 117.55 | 10.21 | 117.02 | 1233.79 |
| newton_grid | 5.85 | 5.76 | 12.76 | 5.75 | 12.83 | 142.59 | 12.51 | 141.30 | 1459.40 |

## Forward + backward (ms)

| method | M=10 N=1,000 | M=10 N=10,000 | M=10 N=100,000 | M=100 N=1,000 | M=100 N=10,000 | M=100 N=100,000 | M=1000 N=1,000 | M=1000 N=10,000 | M=1000 N=100,000 |
|---|---|---|---|---|---|---|---|---|---|
| radial | 0.82 | 1.10 | 3.35 | 0.73 | 0.97 | 8.82 | 0.91 | 7.52 | 78.15 |
| gauge_lb | 1.11 | 1.24 | 4.14 | 1.07 | 1.87 | 8.52 | 1.47 | 7.98 | 80.72 |
| taubin | 0.96 | 1.19 | 3.94 | 1.02 | 1.48 | 13.00 | 1.24 | 11.86 | 118.57 |
| newton_radial | 6.04 | 6.36 | 13.67 | 6.38 | 11.19 | 126.51 | 10.98 | 123.98 | 1293.84 |
| newton_grid | 7.02 | 7.07 | 15.98 | 6.79 | 13.53 | 150.67 | 13.54 | 148.53 | 1530.00 |

## Throughput at M=1000, N=100,000 (forward)

- radial: 6.23 G evals/s, peak 0.3 GB (fwd+bwd peak 0.6 GB)
- gauge_lb: 5.73 G evals/s, peak 0.3 GB (fwd+bwd peak 0.5 GB)
- taubin: 3.48 G evals/s, peak 0.5 GB (fwd+bwd peak 0.8 GB)
- newton_radial: 0.08 G evals/s, peak 1.4 GB (fwd+bwd peak 1.5 GB)
- newton_grid: 0.07 G evals/s, peak 2.4 GB (fwd+bwd peak 2.4 GB)
