# DSGNN
 
Code for our paper **Dynamic Saturation Graph Neural Network for Microbial
Interaction Forecasting** (Khatbane, Mangavel, Borges, Filteau, Aridhi,
Toussaint).
 
DSGNN forecasts how a microbial strain's directed pairwise interactions
evolve over time from a single initial observation, using directed
edge-aware graph attention, GRU-based recurrence, and a dual-saturation
decoder.
 
## Contents
 
- `model.py` — the DSGNN model
- `dataset.py` — data loading and train/test/holdout splitting
- `utils.py` — training utilities (masks, loss, metrics)
- `train.py` — trains one leave-one-strain-out fold
## Requirements
 
```bash
pip install -r requirements.txt
```
 
Paper results used Python 3.10.13, PyTorch 2.1.1, CUDA 12.1.
 
## Data
Data is from: *High-throughput ecological interaction mapping of dairy microorganisms* Ndiaye et al. 2025
 
## Usage
 
Train one fold:
 
```bash
python train.py --holdout-strain AN44 --seed 42
``` 
Table 1 in the paper is this run repeated over all 64 evaluation strains and 5 seeds.