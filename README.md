---
title: Explainable AI Credit Risk Predictor
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Explainable AI (XAI) Credit Risk Prediction Web App

This project is a complete, end-to-end data science application that predicts the risk of a loan applicant defaulting on a loan. It features a full-stack web interface built with Flask and provides not only a risk score but also a human-readable explanation of the model's decision-making process using SHAP (SHapley Additive exPlanations).

**Live Demo:** [Link to your live Hugging Face Space will go here once deployed]

## Key Features

- **End-to-End ML Pipeline:** The entire workflow from data loading, preprocessing, feature engineering, and model training is captured in a reproducible Jupyter Notebook.
- **High-Performance Modeling:** Utilizes a tuned **XGBoost** model, a state-of-the-art algorithm for tabular data.
- **Explainable AI (XAI):** Integrates the **SHAP** library to provide transparent, feature-level explanations for every prediction, building trust and interpretability.
- **Interactive Web Application:** A user-friendly frontend built with **Flask** and HTML/CSS/JavaScript allows for real-time predictions.
- **Containerized for Deployment:** The entire application is containerized using **Docker**, ensuring portability and consistent execution in any environment.

## Model Interpretation

The model's decisions are explained using SHAP. The summary plot below shows the most important features that influence the model's predictions globally.

![SHAP Summary Plot](SHAP_Summary_Plot.png)

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Credit-Risk.git](https://github.com/your-username/Credit-Risk.git)
    cd Credit-Risk
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Run the Flask application:**
    ```bash
    flask run
    ```
    
4.  Open your browser and navigate to `http://127.0.0.1:5000`.