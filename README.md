# PM2.5 Prediction Framework: Fuzzy Clustering and Dynamic Ensemble Neural Networks

This repository contains the implementation and data for the research paper **"In-Vehicle PM2.5 Prediction during Commuting Periods Based on Fuzzy Clustering and Dynamically Weighted Neural Networks."** The framework is designed to predict in-vehicle PM2.5 concentrations using advanced ensemble learning techniques that integrate General Regression Neural Networks (GRNN), Convolutional Neural Networks (CNN), and Attention Regressors.

---

## Overview

Urban traffic growth has intensified air pollution, raising public health concerns. This study introduces a novel hybrid ensemble learning framework that combines fuzzy clustering with dynamically weighted neural networks to improve the accuracy and reliability of PM2.5 predictions. The framework effectively captures spatial-temporal variations, enabling actionable insights for urban air quality management and transportation planning.

Key components include:
- **Fuzzy C-Means Clustering**: Partitions the feature space into subspaces to enable dynamic model weighting.
- **Dynamic Ensemble Learning**: Integrates GRNN, CNN, and Attention Regressor models with adaptive weights based on subspace-specific errors.
- **Multi-Dimensional Data**: Utilizes meteorological parameters, traffic conditions, and in-vehicle sensor readings.

---

## Directory Structure

### Root Directory
Contains six main Python scripts for data preparation, analysis, modeling, and integration:

1. **`1.Table Association.py`**: Associates raw data tables to prepare for preprocessing.
2. **`2.Data preprocessing.py`**: Cleans, normalizes, and formats data for modeling.
3. **`3.Data analysis and visualisation.py`**: Performs exploratory data analysis and visualizes key insights.
4. **`4.Machine learning modelling.py`**: Implements individual machine learning models, including GRNN, CNN, and Attention Regressors.
5. **`5.Integration model.py`**: Combines individual models into a static ensemble framework.
6. **`6.Dynamic committee integration model.py`**: Implements the dynamic ensemble learning framework using fuzzy clustering.

### `Comparative_experiment/` Directory
Contains scripts for benchmarking the proposed dynamic ensemble model against baseline methods:

- **`AttentionRegressor.py`**: Implements the Attention Regressor model.
- **`CNN.py`**: Implements the Convolutional Neural Network model.
- **`DBN.py`**: Implements the Deep Belief Network model.
- **`GRNN.py`**: Implements the General Regression Neural Network model.
- **`LSTM.py`**: Implements the Long Short-Term Memory network.
- **`RNN.py`**: Implements the Recurrent Neural Network model.
- **`utils.py`**: Contains utility functions for model evaluation and metrics.

---

## Features

- **Fuzzy Clustering**: Dynamically adjusts model contributions to improve prediction accuracy.
- **Hybrid Ensemble Learning**: Combines multiple neural networks to leverage their complementary strengths.
- **Air Quality Insights**: Provides actionable recommendations for mitigating PM2.5 exposure during commutes.

---

## Author

This work was conducted by **Xiangyu Li**, a Ph.D. student in the Department of Civil and Environmental Engineering at Northwestern University. For inquiries or collaborations, contact:

- **Email**: xiangyuli2027@u.northwestern.edu
