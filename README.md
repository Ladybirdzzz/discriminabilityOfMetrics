# discriminabilityOfMetrics
This repository contains the code and datasets used in paper **"Quantifying discriminability of evaluation metrics in link prediction for real networks"**(https://arxiv.org/pdf/2409.20078).

---

## Environment Requirements
Python 3.9.13  
numpy==1.22.4  
pandas==2.2.3  
openpyxl== 3.1.2  
scikit-learn==1.3.0  
torch==1.12.1+cu113  
torch-geometric==2.3.1  
networkx==2.6.3  
karateclub==1.3.3  
hmeasure==0.1.6  

---

## Project Structure
├── modules/                   # GNN models  
├── networks/  
│   ├── Benchmark/             # Raw dataset 1  
│   ├── edges/                # Raw dataset 2  
│   ├── statistics/             # Processed dataset  
│   └── networks_list.xlsx # Dataset List  
├── discriminability/  
│   ├── single/            # Discriminability of metrics in single network  
│   ├── category/           # Discriminability of metrics in domain networks  
│   ├── discriminability_single.py    # Calculate discriminability of metrics in single network  
│   └── discriminability_domain.py   # Calculate discriminability of metrics in domain networks  
├── results/               # Evaluation metrics scores  
├── data_random.py  # Discriminability of metrics in single network
├── link_prediction_algorithms.py # Algorithms used for link prediction  
├── metrics.py # Metrics used to evaluate link prediction  
├── run.py # Link prediction and metrics evaluating  
└── README.md 

---

## Workflow

### 1. Data processing
python data_random.py  # Generates edges of networks with different *q*

### 2. Link prediction and metrics evaluating
python run.py  # Generates evaluation metrics scores

### 3. Discriminability calculating
python discriminability/discriminability_single.py    # Calculate discriminability of metrics in single network  
python discriminability/discriminability_category.py   # Calculate discriminability of metrics in domain's networks  

---

## Parameters Settings

### `link_prediction_algorithms.py`
```python
# Set hyperparameters of models, for example:
DeepWalk(walk_number=10,walk_length=40,dimensions=32,window_size=10,min_count=0)  
MyGCN(128,32,16)
```

### `run.py`
```python
# Set experiment parameters
RUN = 100
test_p = 0.1

# Specify the algorithm to be used
algorithms = [...]
```

---

## Output Files
- `networks/statistics/A/`:  
  - `A_edge_index.npy`:  Processed network A
  - `A_shuffle_{}.npy`:  Processed and shuffled network A dataset  
  Dataformat of each row: [x, y]
- `results/A/B/details.txt`:  Evaluation metrics scores of algorithm B on network A  
Dataformat of each row: [run, p, Precision, AUC-Precision, AUPR, AUC, AUC-mROC, NDCG, MCC, H-measure]
- `discriminability/`:  
  - `single/A_discriminability.xlsx`:  Discriminability of metrics in network A  
  - `domain/C_discriminability.xlsx`:  Discriminability of metrics in networks of domain C  
  Dataformat of each sheet: **columns**-Value of p, in [0.01, 0.02, ... ,0.1]; **rows**-Metrics, the order matches details.txt
