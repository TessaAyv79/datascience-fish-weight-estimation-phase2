import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast # To parse string representations of lists
from io import StringIO # To capture df.info() output

# --- Helper Functions ---

def load_and_preprocess_data(csv_path):
    """
    Loads the dataset and performs basic initial preprocessing.
    Updated to be used within Streamlit.
    """
    try:
        df = pd.read_csv(csv_path)
        st.success(f"✅ Dataset Successfully Loaded from: `{csv_path}`")
        st.write(f"Dataset Dimensions: **{df.shape[0]}** rows, **{df.shape[1]}** columns")
        return df
    except FileNotFoundError:
        st.error(f"❌ ERROR: File `{csv_path}` not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading data: {e}")
        return None

def display_eda_sections(df):
    """
    Displays different sections of the data analysis in Streamlit tabs.
    """
    if df is None:
        return

    st.header("🔬 Comprehensive Exploratory Data Analysis (EDA) Report 🔬")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview & Structure",
        "🕵️‍♀️ Missing Data & Cleaning",
        "📈 Numerical Distributions",
        " categoric Distributions", # This was "categorik Dağılımlar"
        "🔗 Relational Analyses"
    ])

    with tab1:
        st.subheader("Dataset First Look")
        st.write("First 5 Rows of the Dataset:")
        st.dataframe(df.head())

        st.subheader("Data Types and Memory Usage")
        # Capture df.info() output to display as text
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with tab2:
        st.subheader("Missing Data Report")
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percent})
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False)
        
        if not missing_df.empty:
            st.warning("❗️ ATTENTION: Missing data found in the following columns:")
            st.dataframe(missing_df)
        else:
            st.success("✅ All columns are complete, no missing data found.")

        st.subheader("Data Cleaning and Transformation")
        # Safely parse the 'Defect' column
        # It's safer to create a copy for processing
        df_processed = df.copy()
        df_processed['Defect_List'] = df_processed['Defect'].apply(lambda s: ast.literal_eval(s) if pd.notna(s) and isinstance(s, str) and s.startswith('[') else [])
        
        st.write(f"Number of rows where Weight <= 0: **{(df_processed['Total Weight (g)'] <= 0).sum()}**")
        st.write(f"Number of rows where Length <= 0: **{(df_processed['Total Length (cm)'] <= 0).sum()}**")
        
        # Create a cleaned DataFrame
        df_cleaned = df_processed[
            (df_processed['Total Weight (g)'] > 0) & 
            (df_processed['Total Length (cm)'] > 0)
        ].copy() # Ensure it's a copy for consistency
        
        st.success(f"Meaningless values cleaned. Remaining rows: **{len(df_cleaned)}**")
        st.write("First 5 Rows of the Cleaned Dataset:")
        st.dataframe(df_cleaned.head())
        
        st.session_state['cleaned_df'] = df_cleaned # Save the cleaned DataFrame to session_state

    with tab3:
        st.subheader("Numerical Variable Distributions and Outliers")
        cleaned_df = st.session_state.get('cleaned_df', df) # Use cleaned DF if available, otherwise original
        
        for col in ['Total Weight (g)', 'Total Length (cm)']:
            if col in cleaned_df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                sns.histplot(cleaned_df[col], kde=True, ax=axes[0], bins=50)
                axes[0].set_title(f'{col} Distribution (Histogram)')
                sns.boxplot(x=cleaned_df[col], ax=axes[1])
                axes[1].set_title(f'{col} Distribution (Box Plot - Outliers)')
                plt.tight_layout()
                st.pyplot(fig) # Display the plot in Streamlit
                plt.close(fig) # Free up memory
            else:
                st.warning(f"Column '{col}' not found in the cleaned dataset.")

    with tab4:
        st.subheader("Categorical Variable Distributions")
        cleaned_df = st.session_state.get('cleaned_df', df) # Use cleaned DF if available, otherwise original
        
        for col in ['Color', 'Species']:
            if col in cleaned_df.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(y=cleaned_df[col], order=cleaned_df[col].value_counts().index, palette='viridis')
                plt.title(f'{col} Column Value Counts')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                st.pyplot(plt) # Display the plot in Streamlit
                plt.close() # Free up memory
            else:
                st.warning(f"Column '{col}' not found in the cleaned dataset.")

    with tab5:
        st.subheader("Relational Analyses")
        cleaned_df = st.session_state.get('cleaned_df', df) # Use cleaned DF if available, otherwise original

        st.markdown("#### Relationship Between Squid Length and Weight (by Species)")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=cleaned_df, x='Total Length (cm)', y='Total Weight (g)', hue='Species', alpha=0.7)
        plt.title('Relationship Between Squid Length and Weight (by Species)')
        st.pyplot(plt)
        plt.close()

        st.markdown("#### Correlation Heatmap Among Numerical Variables")
        plt.figure(figsize=(8, 6))
        numeric_cols = cleaned_df.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Correlation Among Numerical Variables')
            st.pyplot(plt)
            plt.close()
        else:
            st.info("No numerical columns found for correlation analysis.")

# --- Main Streamlit Application ---

st.set_page_config(layout="wide", page_title="Squid Weight Project EDA")
st.title("🦑 Squid Weight Estimation - Exploratory Data Analysis (EDA)")

# Input for the CSV file path
csv_file_path = st.text_input("Path to the CSV file for analysis:", "data/squid_dataset.csv")

# Button to start the analysis
if st.button("Start EDA"):
    with st.spinner("Preparing data analysis and visualizations..."):
        df = load_and_preprocess_data(csv_file_path)
        if df is not None:
            # Also process and store the initial cleaned DataFrame for consistent tab usage
            df_processed_for_cleaning = df.copy()
            df_processed_for_cleaning['Defect_List'] = df_processed_for_cleaning['Defect'].apply(lambda s: ast.literal_eval(s) if pd.notna(s) and isinstance(s, str) and s.startswith('[') else [])
            df_cleaned_initial = df_processed_for_cleaning[
                (df_processed_for_cleaning['Total Weight (g)'] > 0) & 
                (df_processed_for_cleaning['Total Length (cm)'] > 0)
            ].copy()
            st.session_state['cleaned_df'] = df_cleaned_initial

            display_eda_sections(df)
            st.success("🎉 EDA Successfully Completed!")
            
            # Optionally save the cleaned data (can also be done directly in Streamlit)
            cleaned_csv_path = 'data/cleaned_squid_dataset.csv'
            st.session_state['cleaned_df'].to_csv(cleaned_csv_path, index=False)
            st.success(f"Cleaned dataset saved to: `{cleaned_csv_path}`")
else:
    st.info("Click the 'Start EDA' button above to begin the analysis.")