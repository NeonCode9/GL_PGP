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

def train_and_register_model():
    print("🚀 Ingesting partitions for model building...")
    train_path = "tourism_project/data/train.csv"
    test_path = "tourism_project/data/test.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['ProdTaken'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']
    
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), num_cols),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    mlflow.set_experiment("Tourism_Package_Prediction")
    
    with mlflow.start_run(run_name="Optimized_XGBoost_Ensemble"):
        hyper_params = {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "random_state": 42
        }
        mlflow.log_params(hyper_params)
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(**hyper_params, eval_metric='logloss'))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("F1_Score", f1)
        mlflow.log_metric("AUC_ROC", auc)
        print(f"🎯 Model Performance Logs: Accuracy={acc:.4f}, AUC-ROC={auc:.4f}, F1-Score={f1:.4f}")
        
        local_model_path = "tourism_project/model_building/best_model.pkl"
        joblib.dump(pipeline, local_model_path)
        mlflow.log_artifact(local_model_path)
        print("✔️ Model binary pipeline saved locally to runner environment workspace.")

if __name__ == '__main__':
    train_and_register_model()
