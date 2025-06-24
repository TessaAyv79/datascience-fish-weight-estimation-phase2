import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport

# Output directories
REPORTS_DIR = 'reports'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Visualization settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

def perform_comprehensive_eda(csv_path='/home/tessaayv/datascience-weight-estimation/TheFishProject4_v1/data/squid_dataset.csv'):
    print("--- 🔬 ADVANCED EXPLORATORY DATA ANALYSIS STARTED 🔬 ---")

    # === LOAD DATA ===
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset Loaded: {csv_path} | Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {csv_path}")
        return

    print("\n--- First 5 Rows ---")
    print(df.head())

    # === DATA INFO ===
    print("\n--- Data Info ---")
    df.info()

    # === DESCRIPTIVE STATS ===
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all').T)

    # === MISSING VALUES ===
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print(missing)
    else:
        print("✅ No missing values found.")

    # === DEFECT COLUMN CLEANING ===
    df['Defect_List'] = df['Defect'].apply(lambda s: ast.literal_eval(s) if pd.notna(s) and isinstance(s, str) and s.startswith('[') else [])
    df['num_defects'] = df['Defect_List'].apply(len)
    df['is_skinless'] = df['Defect_List'].apply(lambda x: 'skinless' in x)
    df['is_headless'] = df['Defect_List'].apply(lambda x: 'headless' in x)

    # === REMOVE OUTLIERS ===
    df = df[(df['Total Weight (g)'] > 0) & (df['Total Length (cm)'] > 0)].copy()

    # === FEATURE ENGINEERING ===
    df['weight_per_cm'] = df['Total Weight (g)'] / df['Total Length (cm)']

    # === ENCODING ===
    for col in ['Color', 'Species']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])

    # === VISUAL ANALYSIS ===
    for col in ['Total Weight (g)', 'Total Length (cm)', 'weight_per_cm']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(df[col], kde=True, ax=axes[0], bins=50)
        axes[0].set_title(f'{col} - Histogram')
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'{col} - Boxplot')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'{col}_distribution.png'))
        plt.close()

    for col in ['Color', 'Species']:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'{col} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'{col}_counts.png'))
        plt.close()

    # Scatter plot: Length vs Weight
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Total Length (cm)', y='Total Weight (g)', hue='Species')
    plt.title('Length vs Weight by Species')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'length_vs_weight.png'))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_matrix.png'))
    plt.close()

    # PCA for 2D visualization
    pca_cols = ['Total Length (cm)', 'Total Weight (g)', 'weight_per_cm']
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[pca_cols])
    df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Species')
    plt.title('PCA Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pca_visualization.png'))
    plt.close()

    # === PROFILE REPORT ===
    profile = ProfileReport(df, title="Squid Dataset Profiling", explorative=True)
    profile.to_file(os.path.join(REPORTS_DIR, "eda_profile_report.html"))

    # === SAVE CLEANED DATA ===
    cleaned_csv_path = '/home/tessaayv/datascience-weight-estimation/TheFishProject4_v1/data/cleaned_squid_dataset.csv'
    df.to_csv(cleaned_csv_path, index=False)
    print(f"✅ Cleaned data saved to: {cleaned_csv_path}")
    print("--- ✅ Analysis and Feature Engineering Completed ---")

if __name__ == '__main__':
    perform_comprehensive_eda()
