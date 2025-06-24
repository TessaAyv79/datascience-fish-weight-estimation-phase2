import streamlit as st
import sys
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# Add the project root to the Python path
# This allows finding the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the predict function
from src.predict import predict

# --- Streamlit Interface ---

st.set_page_config(layout="wide", page_title="Squid Weight Project")
st.title("🦑 Advanced Squid Weight Estimation Dashboard")

# --- Mode Selection Sidebar ---
st.sidebar.title("Operation Mode")
analysis_mode = st.sidebar.radio(
    "Select an analysis method:",
    ("Single Image Prediction", "Batch Analysis with Folder")
)

# ==============================================================================
# MODE 1: SINGLE IMAGE UPLOAD
# ==============================================================================
if analysis_mode == "Single Image Prediction":
    st.header("Quick Prediction via Manual Image Upload", divider='rainbow')

    uploaded_files = st.file_uploader(
        "Upload one or more images to estimate their weight...",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Create columns to display results neatly
        cols = st.columns(3)
        col_idx = 0
        
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary location
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Make a prediction with the model
            with st.spinner(f"Processing {uploaded_file.name}..."):
                prediction = predict(temp_path)
            
            # Display the result in the next available column
            with cols[col_idx % 3]:
                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
                st.metric(label="Estimated Weight", value=f"{prediction:.2f} g")
                st.write("---")
            
            col_idx += 1
            os.remove(temp_path)

# ==============================================================================
# MODE 2: BATCH ANALYSIS VIA FOLDER PATH
# ==============================================================================
elif analysis_mode == "Batch Analysis with Folder":
    st.header("Batch Prediction and Performance Evaluation with Test Set", divider='rainbow')
    st.info("This mode analyzes the model's overall performance using an image folder and a CSV file on the server. You can optionally select a specific data range for analysis.")

    # Get folder and file paths from the user
    image_folder_path = st.text_input("1. Path to the Test Images Folder:", "data/processed")
    truth_csv_path = st.text_input("2. Path to the CSV File with Ground Truth Data:", "data/cleaned_squid_dataset.csv")

    # --- Data Range Selection ---
    st.subheader("Data Range Selection")
    batch_analysis_option = st.radio(
        "Select the dataset range for analysis:",
        ("Full Dataset", "First 100", "Between 100 and 2000", "Between 800 and 10000", "Custom Row Range"),
        key="batch_option"
    )

    start_idx = 0
    end_idx = None # None means up to the end of the data

    if batch_analysis_option == "First 100":
        end_idx = 100
    elif batch_analysis_option == "Between 100 and 2000":
        start_idx = 100
        end_idx = 2000
    elif batch_analysis_option == "Between 800 and 10000":
        start_idx = 800
        end_idx = 10000
    elif batch_analysis_option == "Custom Row Range":
        st.markdown("**_Please enter the custom range below:_**")
        min_possible = 0
        max_possible = 999999 # Temporarily large value, will be updated if CSV is loaded

        # Update max_possible if CSV file exists and can be read
        if os.path.exists(truth_csv_path):
            try:
                temp_truth_df = pd.read_csv(truth_csv_path)
                max_possible = len(temp_truth_df) - 1 # Last index
            except Exception as e:
                st.warning(f"Error reading CSV file, maximum row count could not be determined: {e}")
                max_possible = 999999


        custom_start = st.number_input("Starting Row Number (0-indexed):", 
                                       min_value=min_possible, 
                                       max_value=max_possible, 
                                       value=min_possible, 
                                       key="custom_start")
        custom_end = st.number_input("Ending Row Number (exclusive):", 
                                     min_value=custom_start + 1, 
                                     max_value=max_possible + 1, # Can be one more than max_possible for slice end
                                     value=min(custom_start + 100, max_possible + 1), # Default to 100 rows or max possible
                                     key="custom_end")
        start_idx = custom_start
        end_idx = custom_end
        
        if start_idx >= end_idx:
            st.error("Starting row must be less than the ending row.")
            st.stop()


    if st.button("Start Analysis"):
        # Check if necessary files and folders exist
        if not os.path.isdir(image_folder_path) or not os.listdir(image_folder_path):
            st.error(f"Processed images folder '{image_folder_path}' is empty or not found. Please ensure the path is correct and you have run `dvc repro`.")
            st.stop()
        if not os.path.exists(truth_csv_path):
            st.error(f"Ground truth CSV file '{truth_csv_path}' not found. Please check the file path.")
            st.stop()

        with st.spinner("Processing the test set, this may take a moment..."):
            
            truth_df = pd.read_csv(truth_csv_path)
            if 'Image' not in truth_df.columns or 'Total Weight (g)' not in truth_df.columns or 'Total Length (cm)' not in truth_df.columns:
                st.error("The CSV file must contain 'Image', 'Total Weight (g)', and 'Total Length (cm)' columns.")
                st.stop()
            
            # Filter truth_df based on the selected range
            if batch_analysis_option != "Full Dataset":
                # ensure end_idx doesn't exceed DataFrame length
                effective_end_idx = end_idx if end_idx is not None and end_idx <= len(truth_df) else len(truth_df)
                truth_df = truth_df.iloc[start_idx:effective_end_idx]
                if truth_df.empty:
                    st.warning(f"No data found in the specified range ({start_idx}-{effective_end_idx}).")
                    st.stop()
                st.info(f"Analysis will be performed for rows **{start_idx} to {effective_end_idx-1}** of the CSV file.")


            truth_df['ImageName'] = truth_df['Image'].apply(lambda url: str(url).split('/')[-1])
            
            test_images = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            prediction_results = []
            for img_name in test_images:
                img_path = os.path.join(image_folder_path, img_name)
                # Check only images present in the filtered truth_df
                actual_data = truth_df[truth_df['ImageName'] == img_name]
                
                if not actual_data.empty:
                    actual_weight = actual_data['Total Weight (g)'].iloc[0]
                    actual_length = actual_data['Total Length (cm)'].iloc[0]
                    predicted_weight = predict(img_path)
                    
                    prediction_results.append({
                        "Filename": img_name,
                        "Actual Length (cm)": actual_length,
                        "Actual Weight (g)": actual_weight,
                        "Predicted Weight (g)": predicted_weight
                    })

            if not prediction_results:
                st.warning("No corresponding labels found in the CSV file for the images in the folder, or no matches in the selected range. Please ensure filenames match and your range is correct.")
                st.stop()
            
            results_df = pd.DataFrame(prediction_results)
            results_df["Error (g)"] = results_df["Predicted Weight (g)"] - results_df["Actual Weight (g)"]
            results_df["Absolute Error (g)"] = abs(results_df["Error (g)"])

        st.success(f"Predictions for **{len(results_df)}** images completed!")
        
        # --- Performance Metrics ---
        st.subheader("Model Performance Summary")
        mae = results_df["Absolute Error (g)"].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} g", help="Indicates how much the model's predictions deviate from the actual values on average. Lower is better.")
        col2.metric("Max Error", f"{results_df['Absolute Error (g)'].max():.2f} g")
        col3.metric("Min Error", f"{results_df['Absolute Error (g)'].min():.2f} g")

        # --- VISUALIZATION TABS ---
        st.subheader("Visual Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Length-Weight Trend", "📈 Error Distribution", "📉 Residuals Plot", "📋 Detailed Results"])

        with tab1:
            st.markdown("#### Trend of Actual Data vs. Model Predictions")
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.set_style("whitegrid")
            sns.regplot(data=results_df, x="Actual Length (cm)", y="Actual Weight (g)", ax=ax, scatter_kws={'alpha':0.4, 's':40}, line_kws={'color':'royalblue', 'ls':'--'}, label="Actual Data Trend")
            sns.regplot(data=results_df, x="Actual Length (cm)", y="Predicted Weight (g)", ax=ax, scatter_kws={'alpha':0.4, 'marker':'x'}, line_kws={'color':'red'}, label="Model Prediction Trend")
            ax.set_title("Actual vs. Predicted Weight by Length", fontsize=16)
            ax.set_xlabel("Actual Length (cm)", fontsize=12)
            ax.set_ylabel("Weight (g)", fontsize=12)
            ax.legend()
            st.pyplot(fig)
            st.info("This plot shows how well the model's learned relationship (red line) matches the natural relationship in the data (blue dashed line). The closer the lines, the better the model's generalization.")

        with tab2:
            st.markdown("#### Distribution of Prediction Errors")
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.histplot(data=results_df, x="Error (g)", kde=True, ax=ax, bins=30)
            ax.axvline(x=0, color='red', linestyle='--', label='Zero Error (Ideal)')
            ax.set_title("Histogram of Prediction Errors", fontsize=16)
            ax.set_xlabel("Error (g) [Predicted - Actual]", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.legend()
            st.pyplot(fig)
            st.info("Ideally, the errors of a good model should be symmetrically distributed around the red line (zero). A skewed distribution indicates the model may have a bias towards over or under-prediction.")

        with tab3:
            st.markdown("#### Residuals Plot")
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.residplot(data=results_df, x="Predicted Weight (g)", y="Actual Weight (g)", ax=ax, scatter_kws={'alpha':0.5})
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_title("Residuals vs. Predicted Values", fontsize=16)
            ax.set_xlabel("Predicted Weight (g)", fontsize=12)
            ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
            st.pyplot(fig)
            st.info("The points in this plot should be randomly scattered around the horizontal red line. Any clear pattern or shape suggests the model's errors are not random, which could indicate areas for improvement.")

        with tab4:
            st.markdown("#### Detailed Prediction Results Table")
            st.dataframe(results_df[['Filename', 'Actual Length (cm)', 'Actual Weight (g)', 'Predicted Weight (g)', 'Absolute Error (g)']].style.format("{:.2f}", subset=['Actual Length (cm)', 'Actual Weight (g)', 'Predicted Weight (g)', 'Absolute Error (g)']))
