# Cross-Domain Product Recommendation: Software to Video Games

**2026 Machine Learning Term Project · Team 4**

An end-to-end recommendation system implemented from scratch using NumPy and SciPy on the Amazon Reviews 2023 (5-core) dataset. This project evaluates memory-based collaborative filtering (user-based kNN), model-based collaborative filtering (matrix factorization), and hybrid filtering across in-domain (**Software**, **Video Games**) and cross-domain (**Software → Video Games**) settings.

---

## Overview

The system addresses two tasks across three evaluation settings:
1. **Rating Prediction (RMSE)**: Predicting explicit 1–5 star ratings on held-out user-item interactions.
2. **Top-K Recommendation (F1@10, Recall@10, Precision@10)**: Recommending the top 10 unseen items a user is most likely to rate 4 stars or higher.

### Implemented Models
- **Baselines & Ablations**:
  - `GlobalMean`: Predicts the global average training rating.
  - `Popularity`: Ranks items by training interaction count.
  - `Bias only (MSE)`: Regularized user and item bias model ($\mu + b_u + b_i$).
  - `BPR (b_i only)`: Item-bias ranking baseline trained with Bayesian Personalized Ranking loss.
- **Memory-Based Collaborative Filtering (`memory.py`)**:
  - `UserKNN`: Mean-centered adjusted cosine similarity user-kNN.
  - `ImprovedUserKNN`: User-kNN with alternating regularized bias estimation and binary positive-feedback ($r \ge 4$) cosine similarity, providing both rating predictions (`scores`) and neighbor-vote ranking scores (`ranking_scores`).
- **Model-Based Collaborative Filtering (`mf.py`)**:
  - `MatrixFactorization` / `FixedMF`: Biased matrix factorization trained via mini-batch SGD with zeroed latent vectors for unseen entities.
  - `SplitRegMF`: Biased MF with decoupled $L_2$ regularization for biases (`reg_bias`) and latent factors (`reg_factor`).
  - `BPRMF`: Matrix factorization trained with pairwise Bayesian Personalized Ranking (BPR) loss for top-K recommendation.
- **Hybrid Filtering (`hybrid.py`)**:
  - `Hybrid` (Rating): Linear blend of `SplitRegMF` and `ImprovedUserKNN` rating predictions ($\alpha_{\text{mf}} \cdot \hat{r}_{\text{MF}} + (1 - \alpha_{\text{mf}}) \cdot \hat{r}_{\text{kNN}}$), tuned on validation RMSE.
  - `RankingHybrid` (Top-K): Linear blend of per-user z-score standardized `BPRMF` and `ImprovedUserKNN` vote scores, tuned on validation F1@10.

---

## Project Structure

```text
.
├── main.py                # Unified training, validation tuning, and evaluation pipeline
├── recsys.py              # Global constants, chronological splitting, ID encoding, baselines
├── splits.py              # In-domain and cross-domain data partitioning and leakage checks
├── memory.py              # UserKNN and ImprovedUserKNN implementations
├── mf.py                  # MatrixFactorization, FixedMF, SplitRegMF, and BPRMF implementations
├── hybrid.py              # Rating Hybrid, RankingHybrid, and per-user z-score standardization
├── evaluation.py          # Shared RMSE and Top-K ranking evaluator (MetricAccumulator)
├── requirements.txt       # Python package dependencies
├── tests/
│   └── test_evaluation.py # Unit tests for metrics, masking, tie-breaking, and cross-domain isolation
├── data/                  # Dataset directory (Software.csv, Video_Games.csv)
└── outputs/
    ├── hybrid/            # Unified experiment outputs (results.json)
    └── memory_reviewed/   # Memory-CF tuning and ablation outputs
```

---

## Datasets

We use the 5-core rating datasets from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UC San Diego):

| Category | Ratings | Users | Items | Role |
| :--- | ---: | ---: | ---: | :--- |
| **Software** | 1,276,840 | 146,396 | 17,591 | Primary domain (model development & hyperparameter tuning) |
| **Video Games** | 814,586 | 94,762 | 25,612 | Target domain (in-domain generalization & cross-domain target) |
| **Shared Users** | — | 4,026 | 0 | Bridge users(70%), hidden validation(10%), hidden test(20%) |

Place `Software.csv` and `Video_Games.csv` inside the `data/` directory before running experiments. Each CSV requires `user_id`, `parent_asin`, `rating`, and `timestamp` columns.

---

## Requirements & Installation

The codebase requires **Python 3.11+** and relies only on standard scientific computing libraries:

```text
numpy==2.5.4
pandas==3.0.1
scipy==1.18.1
matplotlib==3.11.2
```

Install dependencies inside a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
```

---

## Usage

### 1. Run Unified Experiments (`main.py`)

To run the full pipeline (in-domain Software, in-domain Video Games, and Cross-domain evaluation) with the fast BPR configuration (`k=64`, 40 epochs):

```bash
python main.py --bpr-mode fast
```

To run with the full BPR configuration (`k=512`, 188/184/16 epochs):

```bash
python main.py --bpr-mode full
```

Optional arguments:
- `--data-dir`: Path to directory containing `Software.csv` and `Video_Games.csv` (default: `data/`).
- `--output-dir`: Directory where `results.json` will be saved (default: `outputs/hybrid/`).
- `--eval-users`: Maximum number of sampled evaluation users per in-domain dataset (default: `10000`).
- `--max-epochs`: Maximum training epochs for MSE matrix factorization (default: `50`).

### 2. Run Unit Tests

```bash
python -m unittest tests/test_evaluation.py
```

---

## Summary of Results (`--bpr-mode full`)

Results are saved to `outputs/hybrid/results.json`. Metrics below are reported over all evaluated test users (`10,000` users each for Software and Video Games; `805` held-out shared users for Cross-domain).

### Rating Prediction (RMSE — lower is better)

| Model | Software | Video Games | Cross-Domain (SW → VG) |
| :--- | :---: | :---: | :---: |
| **Global mean** | 1.3794 | 1.2526 | 1.1181 |
| **Memory kNN (rating)** | 1.1986 | 1.1260 | **1.0493** |
| **Bias only (MSE)** | 1.1782 | 1.1096 | 1.0557 |
| **MF (MSE, SplitRegMF)** | 1.1779 | 1.1094 | 1.0566 |
| **Hybrid (rating)** | **1.1751** ($\alpha=0.8$) | **1.1090** ($\alpha=0.8$) | **1.0493** ($\alpha=0.0$) |

### Top-10 Recommendation (F1@10 / Recall@10 — higher is better)

| Model | Software F1@10 (Recall@10) | Video Games F1@10 (Recall@10) | Cross-Domain F1@10 (Recall@10) |
| :--- | :---: | :---: | :---: |
| **Popularity** | 0.0131 (5.52%) | 0.0056 (2.29%) | 0.0224 (2.82%) |
| **BPR ($b_i$ only)** | 0.0121 (5.11%) | 0.0053 (2.17%) | 0.0219 (2.77%) |
| **MF (BPR, $k=512$)** | 0.0262 (10.83%) | 0.0128 (5.38%) | 0.0211 (2.72%) |
| **Memory kNN (vote)** | 0.0300 (12.36%) | **0.0167 (7.08%)** | 0.0214 (2.71%) |
| **Hybrid (ranking)** | **0.0311 (12.74%)** ($\alpha=0.7$) | 0.0161 (6.78%) ($\alpha=0.7$) | **0.0237 (3.07%)** ($\alpha=0.6$) |

---

## References

- Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley, *"Bridging Language and Items for Retrieval and Recommendation,"* arXiv:2403.03952, 2024.
- S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, *"BPR: Bayesian Personalized Ranking from Implicit Feedback,"* UAI, 2009.
