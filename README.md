# TRITON: Automated True P-wave Identification in Marine DAS

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of:

> **TRITON: Automated True P-wave Identification in Marine Distributed Acoustic Sensing Shot Gathers Using Deep Learning**  
> Isao Kurosawa, IVXA (2026)  
> *Seismological Research Letters — Electronic Seismologist*

---

## Overview

TRITON solves the problem of automated P-wave identification in marine DAS active-source shot gathers, where water-column acoustic arrivals (1,500 m/s) precede and outpower crustal P-waves (5,000–6,500 m/s) by factors of 8–30.

| Method      | MAE (ms) | R²      | Moveout err (ms) | Dir. err (%) |
|-------------|----------|---------|-----------------|-------------|
| FK-Pick     | 1,720    | −8.36   | 206             | 45.1        |
| STA/LTA     | 1,948    | −11.82  | 202             | 51.4        |
| **CNN_v2**  | **169**   | **0.925** | 126            | 1.9        |
| **MIRROR**  | **193**  | **0.902** | **33**         | **0.0**     |

---

## Installation

```bash
pip install triton-das
```

or from source:

```bash
git clone https://github.com/ivxa-ai/TRITON
cd TRITON
pip install -e .
```

---

## Quick Start

```python
import torch
from triton.models.cnn_v2 import CNNv2
from triton.models.mirror import MIRROR
from triton.data.features import extract_features

# Load model
model = CNNv2()
model.load_state_dict(torch.load("checkpoints/cnn_v2_final.pt"))
model.eval()

# Prepare gather (128 channels x 7500 samples, 500 Hz)
gather = ...  # numpy array (128, 7500)
features = extract_features(gather)

# Predict P-wave arrival times
with torch.no_grad():
    out = model(features["gather_norm"].unsqueeze(0))
    t_p = out["t_p"][0].numpy() * 15.0  # seconds
```

---

## Project Structure

```
triton-das/
├── src/triton/
│   ├── data/
│   │   ├── 001_generator.py     # Synthetic data generation
│   │   └── 002_features.py      # Feature extraction pipeline
│   ├── models/
│   │   ├── 003_cnn_v2.py        # CNN_v2 architecture
│   │   └── 004_mirror.py        # MIRROR architecture
│   └── 005_losses.py            # Loss functions (eq. 5–8)
├── scripts/
│   ├── 006_train.py             # Training script
│   ├── 007_evaluate.py          # Evaluation (Table 1)
│   └── 008_inference.py         # Single-gather inference
└── notebooks/
    └── 009_reproduce_figures.ipynb  # Reproduce Fig. 1–5
```

---

## Reproducing Paper Results

```bash
# 1. Generate synthetic dataset
python scripts/006_train.py --generate-data

# 2. Train CNN_v2
python scripts/006_train.py --model cnn_v2

# 3. Train MIRROR
python scripts/006_train.py --model mirror

# 4. Evaluate (reproduce Table 1)
python scripts/007_evaluate.py --model cnn_v2 --checkpoint models/cnn_v2_best.pt
python scripts/007_evaluate.py --model mirror  --checkpoint models/mirror_best.pt
```

---

## Citation

```bibtex
@article{kurosawa2026triton,
  title   = {TRITON: Automated True P-wave Identification in Marine 
             Distributed Acoustic Sensing Shot Gathers Using Deep Learning},
  author  = {Kurosawa, Isao},
  journal = {Seismological Research Letters},
  year    = {2026},
  note    = {Electronic Seismologist}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
