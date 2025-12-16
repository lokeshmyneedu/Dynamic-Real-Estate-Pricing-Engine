# Dynamic Real Estate Pricing Engine 🏠

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green) ![FastAPI](https://img.shields.io/badge/FastAPI-Production-teal) ![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

## 📌 Project Overview
An end-to-end **Machine Learning Solution** designed to predict optimal rental pricing for real estate properties. Unlike standard notebook experiments, this project implements a production-ready **Microservice Architecture**.

It features a **"Model Zoo"** strategy that dynamically benchmarks multiple algorithms (XGBoost, Random Forest, SVR, Regularized Linear Models) to automatically select and deploy the highest-performing champion model.

## 🚀 Key Features
* **🏆 Champion/Challenger Training Loop:** Automated orchestration script (`train.py`) that trains 5+ different model architectures, performs Hyperparameter Tuning via `GridSearchCV`, and promotes the model with the lowest Mean Absolute Error (MAE) to production.
* **🏭 Factory Pattern Pipelines:** Decoupled model definitions from training logic using a `PipelineFactory`. This allows seamless swapping of estimators (e.g., swapping Ridge for XGBoost) without changing preprocessing code.
* **🛠 Custom Feature Engineering:**
    * `AmenityScoreEngine`: A custom Scikit-Learn transformer that parses unstructured text data (e.g., `"{TV,Wifi,Pool}"`) into numerical density scores.
    * **Log-Transformations:** Automatic skew correction for financial data (Price/Income).
* **⚡ High-Performance API:** A **FastAPI** microservice serving predictions with **Pydantic** strict type validation and automatic Swagger UI documentation.
* **🐳 Containerization:** Fully Dockerized application optimized for cloud deployment with multi-stage builds.
* **🍎 M4 Chip Optimization:** Parallel processing tuned (`n_jobs=-2`) to leverage Apple Silicon's Unified Memory Architecture without UI freezing.

## 📂 Enterprise Project Structure
```text
dynamic_pricing_project/
├── data/
│   ├── raw/                  # Original CSV files (e.g., Inside Airbnb)
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Central Config (Hyperparams, Paths, Features)
│   ├── features.py           # Custom Scikit-Learn Transformers
│   ├── pipeline.py           # Pipeline Factory (Architecture Definition)
│   ├── train.py              # Orchestrator: Training, Tuning & Evaluation
│   ├── predict.py            # Inference Engine (Model Wrapper)
│   └── app.py                # FastAPI Microservice Entrypoint
├── models/                   # Serialized Artifacts (.pkl)
├── notebooks/                # EDA & Prototyping
├── Dockerfile                # Production Container Definition
├── requirements.txt          # Dependency pinning
└── README.md                 # Documentation

🛠 Tech Stack
Core: Python 3.9+

Machine Learning: Scikit-Learn, XGBoost, Pandas, NumPy

API Framework: FastAPI, Uvicorn, Pydantic

DevOps: Docker, Joblib (Serialization)

⚡ Quick Start
1. Prerequisite: Data Setup
Download the listings.csv file (e.g., from Inside Airbnb) and place it in: data/raw/listings.csv

2. Environment Setup
Bash

# Create Virtual Environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install Dependencies
pip install -r requirements.txt
3. Train the Model Zoo
Run the orchestration script to race XGBoost vs. Random Forest vs. Linear Models.

Bash

python src/train.py
Output: The script will log the MAE of each model and save the winner (e.g., XGBoost) to models/pricing_model_v1.pkl.

4. Run the API Server
Start the microservice locally.

Bash

uvicorn src.app:app --reload
Access the Interactive API Docs at: http://127.0.0.1:8000/docs

🐳 Docker Deployment
To run this application in a production-like container environment:

Bash

# 1. Build the Image
docker build -t pricing-engine:v1 .

# 2. Run the Container
docker run -p 8000:8000 pricing-engine:v1
The API is now accessible at http://localhost:8000 from any machine.

📊 Model Performance Results
Champion Model: XGBoost Regressor

R² Score: 0.93 (Explains 93% of price variance)

Test Set MAE: ~192.59 (Normalized)

Key Insight: XGBoost outperformed linear baselines (Ridge/Lasso) by over 60% due to its ability to capture non-linear relationships in location and amenity data.

👨‍💻 Author
[Lokesh Myneedu] Solution Architect & Machine Learning Engineer