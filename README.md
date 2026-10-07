# DSGNN

**Dynamic Saturation Graph Neural Network for Microbial Interaction Forecasting**

A dynamic graph neural network for forecasting the future directed interaction trajectories of a microbial strain from its interaction profile observed at a single initial time point.

<p align="center">
  <img src="/dsgnn_architecture.png" alt="DSGNN architecture" width="100%"> 
</p>


**Paper:** *Dynamic Saturation Graph Neural Network for Microbial Interaction Forecasting*  
**Authors:** Mohammed Khatbane, Cécile Mangavel, Frédéric Borges, Marie Filteau, Sabeur Aridhi, Yannick Toussaint  
(COMPLEX NETWORKS 2026)

## Overview

DSGNN combines three main components:

- **Directed edge-aware graph attention** to propagate information through the microbial interaction network while preserving sender/receiver asymmetry.
- **GRU-based temporal updates** to evolve strain representations through time.
- **Dual-saturation dynamics** to model bounded interaction trajectories while allowing both monotonic and non-monotonic behavior.

## Repository structure

```text
DSGNN/
├── model.py                         # Main DSGNN model and architecture components
├── variants/
│   └── models.py                    # DS and DGNN ablation variants
├── dataset.py                       # Data loading and leave-one-strain-out splitting
├── utils.py                         # Masks, losses, metrics, and training utilities
├── train.py                         # Training and evaluation for one held-out strain
├── results/
│   └── per_strain_results.csv
│                                      # Per-variant, per-strain, per-seed results
├── requirements.txt                 # Python dependencies
└── LICENSE                          # Apache License 2.0
```

## Installation

```bash
git clone https://github.com/Medkne/DSGNN.git
cd DSGNN
pip install -r requirements.txt
```

The experiments reported in the paper were run with:

- Python 3.10.13
- PyTorch 2.1.1
- CUDA 12.1
- NVIDIA RTX 3500 Ada GPU

## Data

The experiments use the microbial interaction dataset introduced in:

> Ndiaye, A., Coulombe, K., Fliss, I., & Filteau, M. (2025). *High-throughput ecological interaction mapping of dairy microorganisms*. **International Journal of Food Microbiology, 427**, 110965. https://doi.org/10.1016/j.ijfoodmicro.2024.110965

## Usage

Train and evaluate DSGNN for one leave-one-strain-out fold:

```bash
python train.py --holdout-strain AN44 --seed 42
```

`--holdout-strain` specifies the target strain whose future interactions are hidden during evaluation.

## Contact

mohammed.khatbane@inria.fr
