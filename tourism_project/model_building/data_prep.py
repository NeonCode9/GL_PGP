# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

def run_data_preparation():
    print("Ingesting raw data stream...")
    # Load raw data asset from the local folder where you uploaded tourism.csv
    raw_path = "tourism_project/data/tourism.csv"
    df = pd.read_csv(raw_path)
    
    # 1. Drop high-cardinality row indicators and non-predictive identifiers
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Removed non-predictive columns: CustomerID and legacy indices.")
        
    # Standardize string inputs to remove accidental white spaces
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # 2. Fix Data Entry Anomalies (The "Fe Male" Issue)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        print("Resolved categorical anomalies: Consolidated 'Fe Male' to 'Female'.")
        
    # 3. Missing Value Imputation
    # Handle continuous numerical features using median values (prevents outlier skews)
    numerical_features = ['Age', 'DurationOfPitch', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'NumberOfFollowups', 'MonthlyIncome']
    for col in numerical_features:
        df[col].fillna(df[col].median(), inplace=True)
        
    # Handle categorical features using mode values (most frequent entry)
    categorical_features = ['TypeofContact', 'PreferredPropertyStar']
    for col in categorical_features:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("Imputation step complete. Zero null entries remain.")
    
    # 4. Stratified Train-Test Split (80/20)
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']
    
    # Stratify=y guarantees the ~19.3% conversion ratio is identical in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save the resulting clean subsets back to disk
    train_path = "tourism_project/data/train.csv"
    test_path = "tourism_project/data/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Partitions saved successfully locally.")
    
    # 5. Remote Ingestion Synchronization to Hugging Face Dataset space
    hf_token = os.getenv("HF_TOKEN")
    hf_user = os.getenv("HF_USER")
    
    if hf_token and hf_user:
        api = HfApi()
        repo_target = f"{hf_user}/tourism-package-dataset"
        print(f"Uploading clean partitions directly to Hugging Face Dataset Hub: {repo_target}...")
        api.upload_file(path_or_fileobj=train_path, path_in_repo="train.csv", repo_id=repo_target, repo_type="dataset", token=hf_token)
        api.upload_file(path_or_fileobj=test_path, path_in_repo="test.csv", repo_id=repo_target, repo_type="dataset", token=hf_token)
        print("✔️ Data partitions synchronized with Hugging Face Space.")

if __name__ == '__main__':
    run_data_preparation()
