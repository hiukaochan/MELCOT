# MELCOT: A Hybrid Learning Architecture with Marginal Preservation for Matrix-Valued Regression

**Paper:** *MELCOT: A Hybrid Learning Architecture with Marginal Preservation for Matrix-Valued Regression*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
5. [Usage](#usage)
6. [Component Breakdown](#component-breakdown)
7. [Experiment Datasets](#experiment-datasets)

---

## 🔍 Introduction

MELCOT is a two-stage hybrid architecture for **matrix-valued regression** that preserves marginal distributions while learning a coupling via optimal transport. The model:

1. **Marginal Estimation (ME):** Predicts row- and column-marginals using Random Forests or SVMs.
2. **Learnable Coupling OT (LCOT):** Learns a mapping function *f* (via DNN, TabNet, FT-Transformer, or Linear Regression) and a Sinkhorn-based transport layer.

Our experiments on three benchmark datasets demonstrate how MELCOT achieves state-of-the-art performance in preserving marginals and overall reconstruction error.

## 📂 Repository Structure

```
├── Baselines/             # Scheme 1 & Scheme 2 baseline models (_S2 suffix = scheme 2)
├── Cost_Models/           # LCOT block candidates: LR, TabNet, FTTransformer, DNN (paper uses DNN)
├── Marginal/              # ME block: RF and SVM variants for two marginals
├── data/                  # Raw data for three datasets
├── OptimalTransport.py    # Learnable OT layer implemented with Sinkhorn
├── MainAlgo.py            # Main script: trains LCOT and evaluates full MELCOT
├── MAIN.ipynb             # Jupyter walkthrough: data prep + demo on medal data (MELCOT[A2])
├── requirements.txt       # Python dependencies
└── .gitignore             # Exclude venv/, __pycache__/, large data
```


## 🚀 Usage

### 1. Run the notebook demo

```bash
jupyter notebook MAIN.ipynb
```

Walks through data cleaning, feature selection, and a sample MELCOT\[A2] run.

### 2. Train & Evaluate from CLI

```bash
python MainAlgo.py \
  --dataset medal \
  --marginal_model RF \
  --cost_model DNN \
  --epochs 2000 \
  --lr 1e-4
```

Key flags:

* `--dataset`: one of `medal`, `elec`, `tourism`
* `--marginal_model`: `RF` or `SVM`
* `--cost_model`: `LR`, `TabNet`, `FTTransformer`, or `DNN`
* `--epochs`, `--lr`: training hyperparameters

## 📦 Component Breakdown

* **Baselines/**: Implement classical baselines under two schemes; suffix `_S2` denotes scheme 2.
* **Marginal/**: Contains `RF.py` and `SVM.py` with two versions each, corresponding to row/column marginals.
* **Cost\_Models/**: Implements four variants for the LCOT *f*-block; only the DNN structure is described in the paper.
* **OptimalTransport.py**: Encodes the differentiable Sinkhorn algorithm to learn couplings.
* **MainAlgo.py**:  LCOT training (calls `Cost_Models/` + `OptimalTransport.py`), and testing of MELCOT.
* **MAIN.ipynb**: A Jupyter notebook for quick prototyping: data loading, preprocessing, feature checks, and an example run.

## 🗃️ Experiment Datasets

1. **Olympic Medal Counts** (2004–2024)
2. **Electricy Usage Prediction** (2009–2019)
3. **Tourism Prediction** (2010–2020)

All datasets are included in the `data/` folder, with raw versions.




