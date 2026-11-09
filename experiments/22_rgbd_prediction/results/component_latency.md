
# Component latency 2026-11-06 18:06, NVIDIA GeForce RTX 5090, batch 8, 60 steps

| variant | step ms | Δ vs default | peak GB | parameters |
|---|---|---|---|---|
| default | 66 | +0 | 3.3 | 1.15 M |
| dense only (no decoder) | 50 | -16 | 3.2 | 0.84 M |
| + volumetric pair loss (--loss both) | 90 | +24 | 5.2 | 1.15 M |
| + union vs targets (--union target) | 90 | +24 | 4.3 | 1.15 M |
| + union vs depth (--union observed) | 86 | +20 | 4.2 | 1.15 M |
| decoder 3 layers | 93 | +27 | 3.5 | 1.77 M |
| decoder 6 layers | 127 | +61 | 4.2 | 2.68 M |
| K 100 | 70 | +4 | 3.3 | 1.15 M |
| K 900 | 75 | +9 | 3.4 | 1.15 M |
| encoder 2 layers | 84 | +18 | 3.5 | 1.35 M |
| encoder 4 layers | 127 | +61 | 3.9 | 1.75 M |
| width 256 | 74 | +8 | 3.6 | 2.32 M |
| width 256, encoder 2 layers | 96 | +30 | 3.9 | 2.84 M |
| no masks | 59 | -7 | 1.7 | 1.15 M |
