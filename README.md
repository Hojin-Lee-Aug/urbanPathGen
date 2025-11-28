# 🚁 urbanPathGen

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the official Python implementation for the paper:

> **Drag-Aware Route Planning for Unmanned Aerial Vehicles in Dynamic Urban Environments** 
---

## 📖 Abstract
Unmanned Aerial Vehicles (UAVs) play a pivotal role in modern society, yet their flight time is limited by battery constraints. This study proposes a **drag-based path prediction method** using a **Graph Convolutional Neural Network (GCNN)**. 

We aim to enhance realism by:
1. Accurately predicting wind flow around complex urban buildings.
2. Utilizing the resulting **aerodynamic drag** as a key cost function for path planning.
3. Demonstrating significant energy savings compared to conventional distance-based algorithms.

---

## 🚀 Key Features

- **🌪️ GCNN-based Wind Prediction**: Utilizes Graph Neural Networks to predict complex wind fields around urban geometries (trained with LB simulation data).
- **🔋 Energy-Efficient Pathfinding**: Implements a modified **Dijkstra algorithm** that considers aerodynamic drag ($F_d$) instead of just Euclidean distance.
- **📊 Quantitative Analysis**: Calculates **Work Consumptions** and **Work Savings (%)** relative to a no-wind baseline.

---
## 🛠️ Usage & Reproducibility

This code is specifically configured for the **$Re=5000$** flow regime, corresponding to the experimental setup described in **Table III** of the paper.

### 0. GCNN
`inference.py` is a lightweight, inference-only script. It bypasses the training process and loads the pre-trained checkpoint to generate flow field predictions. 
Upon execution, the predicted flow field file (e.g., PV_pred_914.Netcdf) will be generated in the `gcnn/results` directory.
Action Required: Move (or download) this PV file into the `data` directory to use it for the path planning algorithm.

**Note:** PV_samplefile.Netcdf(`gcnn/data_set/samples`) serves as a template container to hold the PV data for $Re=5000$, as described in the paper.

### 1. Data Setup
Place the downloaded NetCDF files into the `data/` directory. Ensure the filenames match exactly:
* `data/Info.Netcdf` (Geometry info, this file exists already in data)
* `data/PV.Netcdf` (Velocity field for Re=5000)

**Note:** If you wish to use different filenames or paths, you can modify the `info_file` and `pv_file` variables inside `main.py`.
  
### 2. Configuration (Table III Scenarios)
You can modify the **Start Point (SP)**, **Destination (DE)**, and **Waypoints (WPs)** in `main.py` to reproduce the scenarios from **Table III** or to run your own experiments.

Open `main.py` and edit the following section:

```python
# ========================================
# Configuration (Match with Table III)
# ========================================
START_POS = np.array([0.4, -1.6])   # SP
GOAL_POS = np.array([2.5, 0.5])     # DE

STOPOVER_POS_1 = np.array([-1, 2.5])    # WP 1
STOPOVER_POS_2 = np.array([-0.5, -2.5]) # WP 2
STOPOVER_POS_3 = np.array([2.5, -2.5])  # WP 3
```
---

## 🚀 Results
You can extract various values of **distance** and **work savings (%)** for $Re=5000$. Also, we can generate a figure showing the building domain, SP, DE, WPs, and the paths for Opt 1 and Opt 2, which allows for path comparison. This is a visualization of Scenario 1, Case 1 introduced in the paper.

<p align="center">
<img width="800" height="100" alt="Image" src="https://github.com/user-attachments/assets/513fbd2f-bb4e-4038-b520-a78c0d74fb9f" />
<img width="600" height="600" alt="Image" src="https://github.com/user-attachments/assets/0156737e-fe24-4696-844c-38c3e411eef1" />
  
  <br>
  <em>
    Fig 1. The computed paths for Scenario 1, Case 1.
  </em>
</p>

---

## 📂 Directory Structure

├── data/                         
│   ├── Info.Netcdf                # Grid and coordinate information file
│   └── README.txt            
├── gcnn/                         
│   ├── data_set/                  
│   │   ├── adjLists/              
│   │   │   └── adjLst_914.hdf5    # Adjacency list
│   │   ├── coords/                
│   │   │   └── coord_914.hdf5     # Coordinates
│   │   └── samples/               
│   │       └── PV_samplefile.Netcdf # Template container for PV data
│   └── results/                   
│       └── README.txt           
├── src/                           
│   ├── __init__.py               
│   ├── environment.py             # Environment setup and boundary definitions
│   ├── pathfinding.py             # Core pathfinding algorithms (e.g., Dijkstra)
│   ├── physics.py                 # Physics calculations
│   └── visualization.py           # Plotting and visualization tools
├── .gitignore                   
├── LICENSE                  
└── README.md                    
