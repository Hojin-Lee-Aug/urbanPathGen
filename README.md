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
- **📊 Quantitative Analysis**: Calculates **Normalized Energy Consumption** and **Work Savings (%)** relative to a no-wind baseline.

---
## 🛠️ Usage & Reproducibility

This code is specifically configured for the **$Re=5000$** flow regime, corresponding to the experimental setup described in **Table III** of the paper.

### 1. Data Setup
Place the downloaded NetCDF files into the `data/` directory. Ensure the filenames match exactly:
* `data/Info.Netcdf` (Geometry info)
* `data/PV.Netcdf` (Velocity field for Re=5000)

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
---

## 📂 Directory Structure

```bash
├── data/               # Place NetCDF files here
├── src/                # Source codes
│   ├── __init__.py
│   ├── physics.py      # Physical constants & Drag coefficient ($C_d$) interpolation
│   ├── environment.py  # Data loading & Building detection
│   ├── pathfinding.py  # Modified Dijkstra algorithm (Opt 1 & Opt 2)
│   └── visualization.py # Plotting tools
├── main.py             # Main execution script
└── requirements.txt    # Python dependencies
