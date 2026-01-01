# MELCOT: Marginal-Preserving Matrix-Valued Regression

📄 **Paper**: *MELCOT: A Hybrid Learning Architecture with Marginal Preservation for Matrix-Valued Regression*  
🎤 **Accepted at**: **WSDM 2026** (ACM Web Search and Data Mining)  
📍 **Location**: Idaho, USA  

---

## 🔍 Overview

MELCOT is a two-stage hybrid architecture for **matrix-valued regression** that preserves marginal distributions while learning a coupling via optimal transport. The model:

1. **Marginal Estimation (ME):** Predicts row- and column-marginals using Random Forests or SVMs.
2. **Learnable Coupling OT (LCOT):** Learns a mapping function *f* (via DNN, TabNet, FT-Transformer, or Linear Regression) and a Sinkhorn-based transport layer.

Our experiments on three benchmark datasets demonstrate how MELCOT achieves state-of-the-art performance in preserving marginals and overall reconstruction error.

## 📂 Repository Structure

```
├── Cost_Functions/        # LCOT block candidates: LR, TabNet, FTTransformer, DNN (paper uses DNN)
├── Marginal/              # ME block: RF and SVM variants for two marginals
├── data/                  # Raw data for three datasets
├── OptimalTransport.py    # Learnable OT layer implemented with Sinkhorn
├── MainAlgo.py            # Main script: trains LCOT and evaluates full MELCOT
```

## 📦 Component Breakdown

* **Marginal/**: Contains `RF.py` and `SVM.py` with two versions each, corresponding to row/column marginals.
* **Cost\_Functions/**: Implements four variants for the LCOT *f*-block; only the DNN structure is described in the paper.
* **OptimalTransport.py**: Encodes the differentiable Sinkhorn algorithm to learn couplings.
* **MainAlgo.py**:  LCOT training (calls `Cost_Models/` + `OptimalTransport.py`), and testing of MELCOT.

## 🗃️ Experiment Datasets

1. **Olympic Medal** (2004–2024)  
   Compiled from multiple authoritative sources:
   - World Bank (2025), *World Bank Open Data* — economic indicators  
     https://data.worldbank.org/
   - Our World in Data (2024), *Life Expectancy vs Health Expenditure* — life expectancy  
     https://ourworldindata.org/grapher/life-expectancy-vs-health-expenditure
   - COMAP (2025), *2025 MCM Problem C Data Archive* — population (1970–2022) and historical Olympic medal counts  
     https://www.immchallenge.org/mcm/2025_Problem_C_Data.zip

2. **Electricity Production** (2009–2019)  
   Kaggle: *Global Data on Sustainable Energy*  
   https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy/data

3. **Tourism** (2010–2020)  
   Kaggle: *Tourism and Economic Impact*  
   https://www.kaggle.com/datasets/bushraqurban/tourism-and-economic-impact


