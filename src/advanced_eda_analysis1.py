import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats
from ydata_profiling import ProfileReport

# Output directories
BASE_DIR = '/home/tessaayv/datascience-weight-estimation/TheFishProject4_v1'
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)


def perform_advanced_eda(csv_path=os.path.join(DATA_DIR, 'squid_dataset.csv')):
    print("--- 🔬 ADVANCED EXPLORATORY DATA ANALYSIS STARTED ---")

    # Load dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset Loaded: {csv_path} | Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset not found at {csv_path}")
        return

    # Basic info
    print(df.head())
    print(df.info())
    print(df.describe(include='all').T)

    # Missing value visualization
    msno.matrix(df)
    plt.savefig(os.path.join(FIGURES_DIR, 'missing_matrix.png'))
    plt.close()

    # Parse defect list
    df['Defect_List'] = df['Defect'].apply(lambda s: ast.literal_eval(s) if pd.notna(s) and isinstance(s, str) and s.startswith('[') else [])
    df['num_defects'] = df['Defect_List'].apply(len)
    df['is_skinless'] = df['Defect_List'].apply(lambda x: 'skinless' in x)
    df['is_headless'] = df['Defect_List'].apply(lambda x: 'headless' in x)

    # Remove invalid rows
    df = df[(df['Total Weight (g)'] > 0) & (df['Total Length (cm)'] > 0)].copy()

    # Feature engineering
    df['weight_per_cm'] = df['Total Weight (g)'] / df['Total Length (cm)']

    # Encoding
    for col in ['Color', 'Species']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])

    # Outlier detection
    for col in ['Total Weight (g)', 'Total Length (cm)', 'weight_per_cm']:
        df[f'{col}_zscore'] = np.abs(stats.zscore(df[col]))
        df[f'{col}_is_outlier'] = df[f'{col}_zscore'] > 3

    # Histograms and boxplots
    for col in ['Total Weight (g)', 'Total Length (cm)', 'weight_per_cm']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'{col} - Histogram')
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'{col} - Boxplot')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'{col}_distribution.png'))
        plt.close()

    # Categorical balance
    for col in ['Color', 'Species']:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'{col} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'{col}_counts.png'))
        plt.close()

    # Length vs Weight
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

    # PCA visualization
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

    # Profiling report
    profile = ProfileReport(df, title="Squid Dataset Profiling", explorative=True)
    profile.to_file(os.path.join(REPORTS_DIR, "eda_profile_report.html"))

    # Save cleaned dataset
    cleaned_path = os.path.join(DATA_DIR, 'cleaned_squid_dataset.csv')
    df.to_csv(cleaned_path, index=False)
    print(f"✅ Cleaned dataset saved: {cleaned_path}")
    print("--- ✅ ADVANCED EDA COMPLETED ---")


if __name__ == '__main__':
    perform_advanced_eda()