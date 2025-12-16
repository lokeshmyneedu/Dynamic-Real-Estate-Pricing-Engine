# Dynamic Real Estate Pricing Engine

## Project Overview
An end-to-end Machine Learning pipeline designed to predict optimal rental pricing for real estate properties. This project utilizes an Enterprise-grade architecture, emphasizing modularity, reproducibility, and advanced feature engineering techniques (e.g., Amenity Parsing, Log-Scaling).

## Key Features
* **Custom Feature Engineering:** Implemented `AmenityScoreEngine` to parse unstructured text data into numerical density scores.
* **Hyperparameter Tuning:** Automated GridSearch pipeline to optimize Random Forest Regressors.
* **Enterprise Architecture:** Separation of concerns between Data, Features, Pipeline, and Training modules.
* **Error Analysis:** Heteroscedasticity checks using residual plotting.

## Project Structure
```text
├── data/
│   ├── raw/            # Place listings.csv here
├── notebooks/          # EDA and prototyping
├── src/                # Source code
│   ├── config.py       # Configuration constants
│   ├── features.py     # Custom Scikit-Learn Transformers
│   ├── pipeline.py     # Pipeline Factory
│   └── train.py        # Training & Evaluation Script
├── models/             # Serialized models (.pkl)
└── requirements.txt