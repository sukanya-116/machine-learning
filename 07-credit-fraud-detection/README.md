# Credit Card Fraud Detection using XGBoost 📘

A complete ML pipeline with training, evaluation, feature scaling, hyperparameter tuning, and Dockerized model deployment.

## Project Overview 🚀

This project builds an end-to-end **Credit Card Fraud Detection system** leveraging the popular **Kaggle credit-card dataset**.
The workflow includes:

* Exploratory Data Analysis (EDA)

* Train/Validation/Test splits with stratification

* Feature scaling using StandardScaler

* Class imbalance handling using XGBoost’s scale_pos_weight

* Hyperparameter tuning (depth × learning rate grid)

* Final training on full dataset

* Model evaluation on unseen test data

* Exporting the trained model

* Deploying prediction API via **FastAPI** inside a **Docker container**

The goal is to demonstrate a production-ready machine learning pipeline suitable for fintech fraud detection systems.

## Model Architecture 🧠

The model uses **XGBoost**, a gradient boosting framework optimized for tabular data. Key design choices:

* Algorithm: binary:logistic

* Evaluation Metric: AUC

* Tuned hyperparameters:

    *   eta (learning rate)

    *   max_depth

    *   scale_pos_weight → handles heavy class imbalance

* Final model trained on combined Train + Validation dataset

## Dataset 📊 

* Source: Kaggle Credit Card Fraud Dataset

* Rows: 284807

* Features: PCA components V1–V28 + Time + Amount

* Target:

  * 0 → Non-fraud

  * 1 → Fraud

## Model Training Pipeline 🔧 
1. **Data Preprocessing**

    * Remove duplicates

    * Train/Validation/Test Split

    * Feature Scaling using StandardScaler

    * Oversampling is not required (XGBoost handles imbalance well)

2. **Hyperparameter Tuning**

    Grid search over:

        max_depth ∈ {2, 3, 6}
        eta       ∈ {0.01, 0.1, 0.5}


    Selects best combination based on validation AUC.

3. **Full Model Training**

    * Recompute scale_pos_weight on full training set

    * Train with tuned parameters

## Evaluation Results 🧪

    Final Test AUC: 0.96179
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56651
           1       0.95      0.76      0.84        95


The model achieves high recall & precision for detecting fraudulent transactions.

## Saving the Model 💾

The trained XGBoost model is saved as:

model_final.save_model("credit_fraud_detect_model.json")

Scaler is saved as

joblib.dump(scaler, "scaler.pkl")


This file is included in the Docker image for deployment.

## Deployment with Docker 🌐 
1. **Build the image**

        docker build -t credit-card-fraud-detector .

2. **Run the container**

        docker run -p 8000:8000 credit-card-fraud-detector


The FastAPI service is now available at:

        http://localhost:8000/predict

## API Usage 🔌

📌 Example request:

You can send a POST request to the FastAPI server using a Python script - predict-test.py

📌 Example Output

When the API is running (uvicorn main:app --reload --port 8000), running the test script prints something like:

    Status Code: 200
    Response JSON: {
        "probability": 0.9999877214431763,
        "risk": true
    }

➡️ risk: true means the model flagged the transaction as potential fraud.

## Technologies Used 🛠

* Python 3.12

* XGBoost

* Pandas, NumPy, Scikit-Learn

* FastAPI

* Docker

* Matplotlib & Seaborn

## Future Improvements 📌

* Add SHAP explainability dashboard

* Add MLflow for model versioning

* Auto-refresh Docker build via CI/CD

* Move deployment to AWS/GCP/Azure

* Add real-time streaming predictions via Kafka

## Contributing 🤝 

Pull requests and feature suggestions are welcome!