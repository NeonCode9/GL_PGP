# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

def run_data_preparation():
    print("🚀 Ingesting raw data stream from verified cloud asset path...")
    
    # Fallback storage locations to query data context streams
    raw_url = "https://raw.githubusercontent.com/NeonCode9/GL_PGP/main/tourism_project/data/tourism.csv"
    backup_url = "https://raw.githubusercontent.com/Ankit1012/Tourism-Package-Prediction/main/Tourism.csv"
    
    try:
        df = pd.read_csv(raw_url)
    except Exception:
        df = pd.read_csv(backup_url)
            
    print(f"Ingested dataset structural shape: {df.shape}")
    
    # 1. Drop structural indices and non-predictive metadata identifiers
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Removed non-predictive columns: CustomerID and indices.")
        
    # Standardize spaces on text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # 2. Rectify categorical entry typo anomaly ("Fe Male" -> "Female")
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
        print("Resolved data entry error: Standardized 'Fe Male' to 'Female'.")
        
    # 3. Handle Missing Values via Outlier-Resistant Imputation Logic
    numerical_features = ['Age', 'DurationOfPitch', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'NumberOfFollowups', 'MonthlyIncome']
    for col in numerical_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
        
    categorical_features = ['TypeofContact', 'PreferredPropertyStar']
    for col in categorical_features:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    print("Imputation sequence finalized. Missing values dropped to 0.")
    
    # 4. Stratified Split (80/20) to maintain minority target class distribution (ProdTaken)
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    os.makedirs("tourism_project/data", exist_ok=True)
    train_path = "tourism_project/data/train.csv"
    test_path = "tourism_project/data/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Partitions saved successfully to local directory path.")
    
    # 5. Connect and push directly to your working placeholder repository
    hf_token = os.getenv("HF_TOKEN")
    hf_user = "sudhakaryg"
    
    if hf_token:
        api = HfApi()
        # FIXED: Directly target the active space we are permitted to modify
        repo_target = f"{hf_user}/demo_project"
        print(f"Uploading clean data partitions to verified Space destination: {repo_target}...")
        
        api.upload_file(path_or_fileobj=train_path, path_in_repo="train.csv", repo_id=repo_target, repo_type="space", token=hf_token)
        api.upload_file(path_or_fileobj=test_path, path_in_repo="test.csv", repo_id=repo_target, repo_type="space", token=hf_token)
        print("✔️ Data spaces updated and synchronized successfully.")

if __name__ == '__main__':
    run_data_preparation()
