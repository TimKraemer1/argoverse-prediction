# Argoverse Self-Driving Trajectory Prediction

This project was completed as part of the CSE 251B curriculum — a quarter-long deep learning research course at UC San Diego (Spring 2025). The project was structured as a Kaggle competition, which you can find here:

[kaggle link](https://www.kaggle.com/competitions/cse-251-b-2025) 📲

The models that were researched and presented were collaborated on in conjunction with **[Angus Yick](https://www.linkedin.com/in/angus-yick/) (UCSD)** and **[Mathew Raju](https://www.linkedin.com/in/mathew-raju-6b4517171/) (UCSD)**.

# Project Overview

As part of the CSE 251B curriculum at UC San Diego, students participate in a 
real-world research project focused on advanced deep learning topics. This quarter’s challenge involved building 
robust and innovative models for predicting the future trajectory of self-driving vehicles.

Using the **Argoverse 2 dataset**, we developed models that predict the ego vehicle's future motion based on surrounding agents 
and scene context. Each training scene includes up to 50 dynamic agents (vehicles, pedestrians, etc.), and the task is to 
predict the ego vehicle's future trajectory, step-by-step, over a 6-second horizon.

# Dataset
The **Argoverse 2** dataset is a high-dimensional, multi-agent dataset specifically design for vehicle trajectory prediction. It consists of:

- 10000 training scenes and 2100 test scenes
- Each scene consists of 50 time steps of observed data for 50 agents
- Each agent contains a featureset including:
    - $(x, y)$ position
    - $(v_x, v_y)$ velocity
    - $\theta$ bearing
    - agent type
        - vehicle,
        - pedestrian
        - motorcyclist
        - cyclist
        - bus
        - static
        - background
        - construction
        - riderless bike
        - unknown

## How to access

# Model design

# File Structure
```
.
├── __pycache__
├── models/
│   ├── __pycache__
│   ├── CNNModelClass.py
│   ├── LSTMModelClass.py
│   ├── LinearRegressionModelClass.py
│   └── MLPModelClass.py
├── .DS_store
├── .gitignore
├── DatasetClass.py
├── LICENSE
├── README.md
├── data-modeling.ipynb
└── models.ipynb
```

# How to run

## Requirements
