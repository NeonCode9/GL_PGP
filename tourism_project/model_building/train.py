# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from huggingface_hub import HfApi

def train_and_register_model():
    print("🚀 Ingesting partitions for model building...")
    train_df = pd.read_csv("tourism_project/data/train.csv")
    test_df = pd.read_csv("tourism_project/data/test.csv")
    
    X_train = train_df.drop(columns=['ProdTaken'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']
    
    # Dynamically extract lists of continuous vs categorical features
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Setup automated data preprocessing transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), num_cols),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Connect to local MLflow tracking interface
    mlflow.set_experiment("Tourism_Package_Prediction")
    
    with mlflow.start_run(run_name="XGBoost_Ensemble_Optimization"):
        hyper_params = {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "random_state": 42
        }
        # Log our explicit parameters to the MLflow tracking dashboard
        mlflow.log_params(hyper_params)
        
        # Build an atomic, reusable Pipeline containing transformations and our model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(**hyper_params, eval_metric='logloss'))
        ])
        
        # Train our pipeline
        pipeline.fit(X_train, y_train)
        
        # Evaluate model metrics on unseen holdout data
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Log evaluation statistics straight to MLflow
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("F1_Score", f1)
        mlflow.log_metric("AUC_ROC", auc)
        print(f"Logged Metrics: Accuracy={acc:.4f}, AUC-ROC={auc:.4f}, F1-Score={f1:.4f}")
        
        # Save out serialization weights file locally
        local_model_path = "tourism_project/model_building/best_model.pkl"
        joblib.dump(pipeline, local_model_path)
        mlflow.log_artifact(local_model_path)
        
        # Push file to Hugging Face Model Hub
        hf_token = os.getenv("HF_TOKEN")
        hf_user = os.getenv("HF_USER")
        if hf_token and hf_user:
            api = HfApi()
            model_repo = f"{hf_user}/tourism-package-model"
            api.create_repo(repo_id=model_repo, token=hf_token, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=local_model_path, path_in_repo="best_model.pkl", repo_id=model_repo, repo_type="model", token=hf_token)
            print("Best model successfully registered on Hugging Face Model Hub.")

if __name__ == '__main__':
    train_and_register_model()
