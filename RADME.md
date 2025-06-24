squid-project/
│
├── app/
│   ├── main.py              # FastAPI REST API for prediction endpoints
│   └── streamlit_app.py     # Streamlit interface for interactive user interface
│
├── data/
│   ├── raw/                 # Raw image and CSV data
│   └── processed/           # Cleaned and preprocessed data
│
├── models/                  # Directory for saved/trained model files
│
├── src/
│   ├── data_preprocessing.py  # Data loading, cleaning, feature engineering
│   ├── train.py               # Model training scripts (classification/regression)
│   ├── evaluate.py            # Model evaluation and performance metrics
│   └── predict.py             # Inference pipeline using trained models
│
├── Dockerfile                # Containerization for FastAPI and dependencies
├── docker-compose.yml        # Multi-container setup (if applicable)
├── requirements.txt          # Python dependencies
├── dvc.yaml                  # DVC pipeline file for versioning data and models
└── README.md                 # Project overview and documentation


🦑 Squid Classification & Weight Estimation
This repository hosts the machine learning pipeline developed for ThisFish Inc. to classify squid species and estimate weights using both image and tabular data. It includes advanced preprocessing, EDA, ML model training, explainability, and evaluation components.

📌 Project Objectives
Classify squid species using image and structured data.

Estimate total weight (g) of squid specimens.

Utilize MobileNetV2 for deep image feature extraction.

Integrate SHAP, Grad-CAM, LIME for explainability and transparency.

Generate actionable insights for fisheries and inventory systems.

🧰 Tech Stack
Category	Tools & Libraries
Language	Python 3.7+
ML Libraries	scikit-learn, xgboost, tensorflow, keras, imblearn
EDA	pandas, seaborn, matplotlib, ydata-profiling
Explainability	shap, lime, Grad-CAM
Deployment	Jupyter Notebooks, JSON config files

📂 Directory Structure
graphql
Copy
Edit
classification/
├── config/
│   └── config.json         # File paths and column mappings
├── data/
│   └── squid_data.csv      # CSVs with image URLs & features
├── images/
│   └── *.jpg               # Local images for squid specimens
├── models/
│   └── *.pkl               # Trained model files
├── notebooks/
│   └── analysis.ipynb      # Full EDA, modeling, and visualization
├── outputs/
│   └── *.png, *.html       # Visualizations, EDA reports
├── README.md
📈 Pipeline Overview
🔹 Task 1: Data Processing & EDA
Load and inspect CSVs

Download and verify images

Parse Defect column → engineered features (is_skinless, num_defects)

Handle missing values and outliers

Encode categorical features (e.g. Color, Species)

Visualizations: histograms, boxplots, PCA, correlation heatmap

Automated EDA with ydata-profiling

🔹 Task 2: Modeling
Extract image features with MobileNetV2

Merge image features with tabular data

Train & evaluate:

Classification: Random Forest, XGBoost, MLP, SVC, Naive Bayes

Regression: XGBoost, SVR, Linear, MLP

Handle imbalance with SMOTE

🔹 Explainability
Global/Local: SHAP, LIME

Image attention: Grad-CAM

✅ Results Summary
Task	Model	Metric	Score
Classification	XGBoost (Illex)	F1-Score	0.91+
Regression	XGBoost Regressor	R²	0.89+
Explainability	SHAP	Top Features	Total_Length, Image_Embedding

📊 See advanced_eda_analysis.html for interactive visualizations.

⚙️ Configuration Example
json
Copy
Edit
{
  "input_csv": "data/squid_data.csv",
  "image_dir": "images/",
  "model_output": "models/",
  "target_column": "Species",
  "defect_column": "Defect"
}
🔍 Resources
📘 Full Technical Report: TessaAyv- Team Member Docs.docx

📊 EDA Report: advanced_eda_analysis.html

🔗 GitHub Repo: thisfishinc/datascience-squid-tallyvision

🧠 Future Work
Real-time prediction interface

YOLO-based detection integration

AutoML and model selection optimization

Deployment to cloud (e.g., AWS Sagemaker)

👤 Author
Tessa Nejla Ayvazoğlu
Data Scientist @ M2M Tech Inc.
Email: tessaayv@gmail.com
