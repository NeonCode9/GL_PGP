# ✈️ AI-Driven Tourism Package Prediction & MLOps Pipeline

An end-to-end production-grade machine learning and MLOps lifecycle solution engineered for **"Visit with Us"** to optimize marketing campaigns[cite: 1, 4]. The system integrates automated data engineering, advanced predictive modeling with hyperparameter tracking, containerization, and fully automated continuous integration and continuous deployment (CI/CD) workflows.

---

## 📋 Business Context & Core Challenge

"Visit with Us," a leading travel enterprise, is modernizing its consumer operations by adopting data-driven frameworks to optimize outreach and maximize customer engagement[cite: 1, 4]. During the rollout of their premium **Wellness Tourism Package**, the organization faced a critical operational bottleneck: the historical manual selection of prospective buyers by sales agents was inconsistent, highly time-consuming, and prone to human error[cite: 1, 4]. This resulted in misallocated marketing budgets, low team efficiency, and missed revenue opportunities[cite: 1, 4].

### **The Solution**
This project establishes a scalable, automated predictive gatekeeper platform. By assessing granular customer demographics and past sales interaction telemetry, the pipeline calculates an explicit purchase probability score *before* direct outreach campaigns are launched[cite: 1, 4]. This enables marketing teams to prioritize high-potential leads, optimize resource allocation, and accelerate customer acquisition[cite: 1, 4].

---

## 🛠️ Architecture Overview

The platform is engineered as a fully automated loop spanning multiple cloud ecosystems:

* **Google Colab:** Staging area for initial scripting and workspace tree structure creation.
* **GitHub Repository:** The primary codebase orchestrator containing project assets.
* **GitHub Actions Workflow:** Spins up an automated runner to process splits, retrain models, and sync platforms.
* **Hugging Face Hub:** Host for the remote data spaces and serialization artifact hub.
* **Hugging Face Spaces:** Serves the interactive Streamlit user app via secure Docker containers[cite: 1, 2].

---

## 📁 Repository Directory Structure

.
├── .github/
│   └── workflows/
│       └── pipeline.yml       # Automated CI/CD GitHub Actions engine
└── tourism_project/
├── data/
│   ├── tourism.csv        # Raw base dataset
│   ├── train.csv          # Preprocessed stratified training partition
│   └── test.csv           # Preprocessed stratified validation partition
├── model_building/
│   ├── data_prep.py       # Data cleaning and ingestion module
│   ├── train.py           # Model execution pipeline and tracking script
│   └── best_model.pkl     # Locally saved model binary weights
└── deployment/
├── app.py             # Streamlit interactive application script
├── Dockerfile         # System container configuration layout
└── requirements.txt   # Locked application version dependencies

---

## ⚙️ Modular Component Breakdown

### **1. Data Engineering & Cleansing Pipeline (`data_prep.py`)**
* **Anomalies Resolved:** Corrects structural entry inconsistencies by programmatically merging duplicate categorical strings (e.g., standardizing `"Fe Male"` into `"Female"`).
* **Feature Trimming:** Drops high-cardinality metadata tags (`CustomerID` and index markers) that cause model overfitting.
* **Imputation Engine:** Utilizes robust, outlier-resistant median values for continuous null fields and modes for categorical missing indices.
* **Target Class Stratification:** Addresses a highly skewed dataset (only **19.31%** converted buyers) by implementing a stratified 80/20 train-test partition split to protect baseline class distributions.
* **Sync Integration:** Programmatically pushes clean partitions to the Hugging Face Dataset Hub.

### **2. Modeling & Experiment Tracking (`train.py`)**
* **Feature Processing:** Utilizes a robust `ColumnTransformer` execution engine grouping `StandardScaler` transformations for numbers and `OneHotEncoder(handle_unknown='ignore')` for strings.
* **Core Algorithm:** Trains an optimized **XGBoost Classifier** ensemble pipeline.
* **Governance Matrix:** Uses **MLflow** to create an automated audit trail, logging tuned hyperparameters alongside validation metrics: Accuracy, F1-Score, and AUC-ROC.
* **Model Registry:** Registers the final serialization file (`best_model.pkl`) to the Hugging Face Model Hub.

### **3. Production Deployment Frontend (`app.py` & `Dockerfile`)**
* **UI Engine:** Implements an interactive, responsive **Streamlit** dashboard built with input sections divided across three profile categories: Demographics, Logistics, and Pitch Telemetry.
* **Container Blueprint:** Containerized using a lightweight `python:3.9-slim` base layout running inside an unprivileged user space on port **7860**.
* **Asset Performance:** Employs `@st.cache_resource` memory caching to pull the pipeline binary directly from the remote Model Hub only on initial startup.

---

## 🚀 CI/CD Automation Workflow (`pipeline.yml`)

The complete operational lifecycle is automated through **GitHub Actions**. Every time code adjustments or features are pushed to the `main` branch, the workflow runner executes the following steps:

1. **Code Ingestion:** Checks out the latest codebase repository changes.
2. **Environment Setup:** Installs virtual Python environments and required library dependencies.
3. **Data Preprocessing Execution:** Automatically runs `data_prep.py` to clean and re-upload datasets.
4. **Model Optimization:** Starts an MLflow instance, triggers `train.py` to optimize parameters, and updates the model registry.
5. **Production Deployment:** Clones the active Hugging Face placeholder Space repo (`sudhakaryg/demo_project`), copies the app components, and executes a force push to instantly rebuild the live Docker web app container[cite: 1, 2].

---

## 🛠️ Environment Configuration & Setup

To replicate or launch this operational pipeline, set up the following environments:

### **1. GitHub Secrets Setup**
Navigate to your GitHub Repository **Settings ➔ Secrets and Variables ➔ Actions** and register the following variables:
* `HF_USER` : Set value to `sudhakaryg`[cite: 1, 2].
* `HF_TOKEN` : Paste your private write-access API token generated from your Hugging Face developer settings[cite: 1, 2].

### **2. Launch Local Training Executions**
If you wish to trigger data cleaning or model training steps manually in a local terminal or Colab cell, use these commands:

```bash
# Execute data ingestion and cleansing
python tourism_project/model_building/data_prep.py

# Launch model training and register metrics
python tourism_project/model_building/train.py
