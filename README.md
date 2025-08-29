# Product Demand Prescriptive Analytics

Optimize inventory decisions using prescriptive analytics (LP / IP / MILP) on the **Product Demand Forecasting** dataset.

> **Deliverables**: Jupyter Notebook + short report + this README.

---

## 📦 Dataset

**Source**: Kaggle — *Product Demand Forecasting* by felixzhao
URL: [https://www.kaggle.com/felixzhao/productdemandforecasting](https://www.kaggle.com/felixzhao/productdemandforecasting)

Files used:

* `Historical Product Demand.csv`

> ⚠️ You need a Kaggle account + API token to download via CLI. Alternatively, download manually and place files under `data/`.

---

## 🧭 Project Goals

1. **Forecast demand** for products over time (baseline provided; model-agnostic placeholder).
2. **Prescriptive optimization** of inventory (order quantities & timing) via **Linear**, **Integer**, and **Mixed-Integer** Programming.
3. **Evaluation** using **cost, revenue, profit** and operational KPIs.
4. **Sensitivity analysis** for **demand, lead time, holding cost**.

---

## 🗂️ Repository Structure

```
.
├─ data/                         # Put Kaggle CSVs here (not tracked)
├─ notebooks/
│  └─ Product_Demand_Optimization_Notebook.ipynb
├─ results/
│  ├─ figures/                   # Plots exported by the notebook
│  └─ Inventory_Optimization_Results.csv
├─ README.md
└─ requirements.txt (optional)
```

---

## ⚙️ Environment Setup

**Python**: 3.9+ recommended

Install deps:

```bash
pip install pandas numpy matplotlib pulp scikit-learn statsmodels prophet kaggle
```

> If `prophet` fails to install on Windows, you can skip it or use conda:

```bash
conda install -c conda-forge prophet
```

---

## ⬇️ Get the Data

**Option A — Kaggle CLI (recommended)**

```bash
pip install kaggle
# place kaggle.json in ~/.kaggle/kaggle.json with 600 permissions
kaggle datasets download -d felixzhao/productdemandforecasting -p data --unzip
```

**Option B — Manual**

1. Download the dataset zip from the Kaggle page.
2. Extract `Historical Product Demand.csv` into `data/`.

---

## 📓 Run the Notebook

Open and execute:

* `notebooks/Product_Demand_Optimization_Notebook.ipynb`

It contains:

1. **EDA**: Parse dates, clean `Order_Demand` (convert strings/parentheses to numeric).
2. **Baseline forecast**: simple moving average / naive (swap in ARIMA/ETS/Prophet as desired).
3. **Optimization**: Multi-period planning MILP with fixed order cost (K), variable unit cost (c), holding cost (h), optional backorder penalty (p), and lead time (L).
4. **Evaluation**: Cost breakdown + profit.
5. **Sensitivity**: demand / lead-time / holding-cost sweeps with plots.

> If you prefer to run all from root: `jupyter notebook` → open the file → run all.

---

## 🧮 Optimization Models (outline)

**Sets**: periods `t = 1..T`

**Parameters**: demand `D_t`, unit purchase cost `c`, holding `h`, fixed order cost `K`, backorder penalty `p` (optional), lead time `L`.

**Decision vars**:

* `Q_t ≥ 0` — order quantity in period t
* `I_t` — end-of-period on-hand inventory (≥ 0 if no backorders)
* `B_t ≥ 0` — backorders (optional)
* `y_t ∈ {0,1}` — order binary (incurs K if `Q_t > 0`)

**Inventory balance** (with lead time L):

```
I_t - B_t = I_{t-1} - B_{t-1} + Q_{t-L} - D_t
```

**Cost objective (MILP)**:

```
min  Σ_t ( c·Q_t + K·y_t + h·I_t + p·B_t )
subject to  Q_t ≤ M·y_t,  I_t ≥ 0,  B_t ≥ 0,  y_t ∈ {0,1}
```

Set `p=0` and drop `B_t` for no-backorders case. For pure LP, drop fixed cost `K` and `y_t`.

---

## 📈 Metrics & KPIs

* **Ordering Cost**: Σ K·\[Q\_t>0]
* **Purchasing Cost**: Σ c·Q\_t
* **Holding Cost**: Σ h·I\_t
* **Shortage/Backorder Cost**: Σ p·B\_t (if enabled)
* **Total Cost**: sum of above
* **Revenue** (optional, if selling price known): Σ price·sales\_t
* **Profit**: Revenue − Total Cost
* **Service Level**: 1 − (Σ shortages / Σ demand)
* **Capacity Utilization** (if capacity modeled): Σ production / capacity

The notebook prints a monthly table and an overall summary, plus saved CSV in `results/`.

---

## 🔎 Sensitivity Analysis

The notebook varies:

* **Demand**: ±20%
* **Holding Cost (h)**: ±20%
* **Lead Time (L)**: ±20% (modeled via effective holding/backorder impact)

Outputs: line charts for **Profit / Revenue / Total Cost** vs parameter change, plus optional heatmaps.

---

## ▶️ Quick Start (minimal code)

```python
# Load
import pandas as pd
from pathlib import Path
f = Path('data')/"Historical Product Demand.csv"
df = pd.read_csv(f)

# Clean
df['Date'] = pd.to_datetime(df['Date'])
df['Order_Demand'] = (
    df['Order_Demand'].astype(str)
      .str.replace('[(),]','', regex=True)
      .str.replace('−','-', regex=False) # fix minus sign
)
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce').fillna(0)

# Aggregate for one product
pid = df['Product_Code'].iloc[0]
series = (df[df['Product_Code']==pid]
          .groupby('Date')['Order_Demand'].sum()
          .sort_index())

# Baseline forecast (30 periods)
import numpy as np
T = 30
avg = series.rolling(7, min_periods=1).mean().iloc[-1]
D = np.full(T, avg)

# MILP (PuLP)
import pulp
c, h, K, L = 10.0, 0.5, 50.0, 0
prob = pulp.LpProblem('inv', pulp.LpMinimize)
Q = pulp.LpVariable.dicts('Q', range(T), lowBound=0)
y = pulp.LpVariable.dicts('y', range(T), lowBound=0, upBound=1, cat='Binary')
I = pulp.LpVariable.dicts('I', range(T), lowBound=0)
prob += pulp.lpSum(c*Q[t] +
```
