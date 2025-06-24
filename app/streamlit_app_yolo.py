import streamlit as st
import sys
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc,
    balanced_accuracy_score, matthews_corrcoef
)
from ultralytics import YOLO # Make sure this is imported
import torch # Make sure this is imported
import yaml # Make sure this is imported
from pathlib import Path # Make sure this is imported
import argparse # Need for Namespace (for calculate_metrics)

# Add the project root to the Python path
# This allows finding the 'src' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the predict function
from src.predict import predict

# --- HELPER FUNCTIONS FOR COMPREHENSIVE EVALUATION MODE ---
# These functions are moved outside the Streamlit analysis_mode blocks
# so they are defined only once when the script runs.

def load_data_yaml(data_yaml_path: str) -> dict:
    """Load data.yaml configuration."""
    # Check if it's a directory (assuming data.yaml is inside)
    if os.path.isdir(data_yaml_path):
        data_yaml_path = os.path.join(data_yaml_path, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
        
    with open(data_yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_predictions_and_labels(model_path, data_config, split='test'):
    """Get model predictions and true labels for evaluation."""
    model = YOLO(model_path)
    model.eval() # Ensure model is in evaluation mode
    
    split_dir = Path(data_config['path']) / split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory '{split_dir}' not found. Please check data_path and split name.")
    
    class_names_dict = data_config['names']
    class_names_list = []
    for i in range(len(class_names_dict)):
        if str(i) in class_names_dict:
            class_names_list.append(class_names_dict[str(i)])
        elif i in class_names_dict:
            class_names_list.append(class_names_dict[i])
        else:
            class_names_list.append(f"Class {i}")
            
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_filenames = []
    
    # Run validation on the dataset first to get the model warmed up and capture speed metrics
    val_results = model.val(data=data_config['path'], split=split, verbose=False) # Capture val results

    with torch.no_grad():
        # Iterate through actual class folders to get true labels and image paths
        for class_idx_str, class_name in class_names_dict.items():
            class_idx = int(class_idx_str) # Ensure class_idx is integer for comparison
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                results = model(str(img_path), verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'probs') and result.probs is not None:
                        probs = result.probs.data.cpu().numpy()
                        pred_class = np.argmax(probs)
                        
                        all_predictions.append(pred_class)
                        all_probabilities.append(probs)
                        all_true_labels.append(class_idx) # Use the integer class_idx
                        all_filenames.append(img_path.name)
                        
    true_classes_names = [class_names_list[idx] for idx in all_true_labels]
    pred_classes_names = [class_names_list[idx] for idx in all_predictions]
    
    confidences = [prob[pred] for prob, pred in zip(all_probabilities, all_predictions)]
    
    # Pass val_results back to calculate_metrics for speed data
    return true_classes_names, pred_classes_names, confidences, class_names_list, all_filenames, all_probabilities, val_results

def calculate_metrics(true_classes, pred_classes, confidences, class_names, filenames, data_config, args_namespace, all_probabilities_raw, val_results):
    """Calculate comprehensive evaluation metrics."""
    # Ensure correct handling of zero_division in classification_report
    report = classification_report(true_classes, pred_classes, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_classes, pred_classes, labels=class_names)
    
    roc_curves = {}
    pr_curves = {}
    roc_auc = None
    pr_auc = None
    
    if len(class_names) == 2:
        # Check if the positive class (class_names[1]) is present in true_classes
        if class_names[1] in true_classes:
            binary_true = np.array([1 if c == class_names[1] else 0 for c in true_classes])
            
            # Extract probabilities for the positive class (index 1)
            # Ensure all_probabilities_raw has the correct shape (num_samples x num_classes)
            if all_probabilities_raw and all(len(p) == len(class_names) for p in all_probabilities_raw):
                positive_class_probs = np.array([p[class_names.index(class_names[1])] for p in all_probabilities_raw])
                
                fpr, tpr, _ = roc_curve(binary_true, positive_class_probs)
                roc_curves = {'fpr': fpr, 'tpr': tpr}
                roc_auc = auc(fpr, tpr)
                
                precision, recall, _ = precision_recall_curve(binary_true, positive_class_probs)
                pr_curves = {'precision': precision, 'recall': recall}
                pr_auc = auc(recall, precision)
            else:
                st.warning("Could not compute ROC/PR curves: Probability data for binary classification is not in expected format.")
        else:
            st.warning(f"Could not compute ROC/PR curves: Positive class '{class_names[1]}' not found in true labels for binary classification.")
    
    # Use speed values from YOLO's validation output
    # Ensure val_results.speed exists and contains keys
    if hasattr(val_results, 'speed') and isinstance(val_results.speed, dict):
        preprocess_times = [val_results.speed.get('preprocess', 0.0)]
        inference_times = [val_results.speed.get('inference', 0.0)]
        postprocess_times = [val_results.speed.get('postprocess', 0.0)]
    else:
        # Fallback to default/placeholder if speed metrics are not available
        preprocess_times = [0.0]
        inference_times = [0.0]
        postprocess_times = [0.0]
    
    balanced_acc = balanced_accuracy_score(true_classes, pred_classes)
    mcc = matthews_corrcoef(true_classes, pred_classes)
    
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(true_classes, pred_classes)) if true != pred]
    misclassified_rate = len(misclassified_indices) / len(true_classes) if len(true_classes) > 0 else 0
    
    metrics = {
        'accuracy': report.get('accuracy', 0.0),
        'balanced_accuracy': balanced_acc,
        'matthews_correlation_coefficient': mcc,
        'macro_avg_f1': report.get('macro avg', {}).get('f1-score', 0.0),
        'weighted_avg_f1': report.get('weighted avg', {}).get('f1-score', 0.0),
        'speed_preprocess_ms': np.mean(preprocess_times),
        'speed_inference_ms': np.mean(inference_times),
        'speed_postprocess_ms': np.mean(postprocess_times),
        'inference_fps': 1000 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0,
        'misclassification_rate': misclassified_rate,
        'misclassifications_count': len(misclassified_indices),
        'confusion_matrix': cm
    }
    
    for class_name in class_names:
        if class_name in report and isinstance(report[class_name], dict):
            metrics[f'{class_name}_precision'] = report[class_name].get('precision', 0.0)
            metrics[f'{class_name}_recall'] = report[class_name].get('recall', 0.0)
            metrics[f'{class_name}_f1'] = report[class_name].get('f1-score', 0.0)
            metrics[f'{class_name}_support'] = report[class_name].get('support', 0)
        
    return metrics, roc_curves, pr_curves, roc_auc, pr_auc, misclassified_indices

# --- Streamlit Interface ---

st.set_page_config(layout="wide", page_title="Squid Weight Project")
st.title("🦑 Advanced Squid Weight Estimation Dashboard")

# --- Mode Selection Sidebar ---
st.sidebar.title("Operation Mode")
analysis_mode = st.sidebar.radio(
    "Select an analysis method:",
    ("Single Image Prediction", "Batch Analysis with Folder", "Comprehensive Model Evaluation")
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

# ==============================================================================
# MODE 3: COMPREHENSIVE MODEL EVALUATION
# ==============================================================================
elif analysis_mode == "Comprehensive Model Evaluation":
    st.header("Comprehensive Model Evaluation", divider='green')
    st.info("This mode provides a detailed performance analysis of your classification model.")

    # Input fields for model and data paths
    model_path = st.text_input("Path to trained model (e.g., outputs/best_model/weights/best.pt):", "outputs/best_model/weights/best.pt")
    data_path = st.text_input("Path to data directory (containing data.yaml):", "/home/tim/2_squid/3_classification_skin_status/1_train_test_split/outputs")
    split_to_evaluate = st.radio("Dataset split to evaluate:", ('test', 'val', 'train'))
    
    # Create a temporary output directory for plots if not already created
    temp_eval_output_dir = "temp_evaluation_results"
    os.makedirs(temp_eval_output_dir, exist_ok=True)

    if st.button("Start Comprehensive Evaluation"):
        # Validate paths
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: `{model_path}`")
            st.stop()
        if not os.path.isdir(data_path) and not os.path.exists(os.path.join(data_path, 'data.yaml')):
            st.error(f"❌ Data directory or data.yaml not found at: `{data_path}`")
            st.stop()

        with st.spinner("Running comprehensive model evaluation... This may take a moment."):
            try:
                # Load data config
                data_config = load_data_yaml(data_path)
                
                # Get predictions and labels
                true_classes, pred_classes, confidences, class_names, filenames, all_probabilities_raw, val_results = get_predictions_and_labels(
                    model_path, data_config, split_to_evaluate
                )
                
                st.write(f"Processed **{len(true_classes)}** images from the **{split_to_evaluate}** dataset.")

                # Calculate metrics
                # Pass args_namespace (dummy for argparse), and all_probabilities_raw, val_results
                metrics, roc_curves, pr_curves, roc_auc, pr_auc, misclassified_indices = calculate_metrics(
                    true_classes, pred_classes, confidences, class_names, filenames, data_config, argparse.Namespace(model_path=model_path, data_path=data_path, split=split_to_evaluate), all_probabilities_raw, val_results
                )
                
                st.success("✅ Evaluation complete! Displaying results.")

                # --- Display Metrics ---
                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                col2.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")
                col3.metric("MCC", f"{metrics['matthews_correlation_coefficient']:.4f}")

                col4, col5, col6 = st.columns(3)
                col4.metric("Macro F1-Score", f"{metrics['macro_avg_f1']:.4f}")
                col5.metric("Weighted F1-Score", f"{metrics['weighted_avg_f1']:.4f}")
                
                # ROC AUC and PR AUC display
                if roc_auc is not None:
                    col6.metric("ROC AUC", f"{roc_auc:.4f}")
                else:
                    col6.info("ROC AUC is for binary classification only.")

                if pr_auc is not None:
                    # Place PR AUC below ROC AUC if they share the same column
                    # Or use a separate column if you want them side by side
                    st.columns(3)[2].metric("PR AUC", f"{pr_auc:.4f}")
                else:
                    st.columns(3)[2].info("PR AUC is for binary classification only.")


                st.markdown("---")
                st.subheader("Speed Metrics (per image in ms)")
                col_speed1, col_speed2, col_speed3, col_speed4 = st.columns(4)
                col_speed1.metric("Preprocess", f"{metrics['speed_preprocess_ms']:.2f}")
                col_speed2.metric("Inference", f"{metrics['speed_inference_ms']:.2f}")
                col_speed3.metric("Postprocess", f"{metrics['speed_postprocess_ms']:.2f}")
                col_speed4.metric("Inference FPS", f"{metrics['inference_fps']:.2f}")


                # --- Tabs for detailed results and plots ---
                st.subheader("Detailed Analysis")
                tab_metrics, tab_conf_matrix, tab_roc_pr, tab_misclass = st.tabs([
                    "Class Metrics", "Confusion Matrix", "ROC & PR Curves", "Misclassifications"
                ])

                with tab_metrics:
                    st.write("### Per-Class Metrics")
                    class_metrics_data = {}
                    for class_name in class_names:
                        if f'{class_name}_f1' in metrics:
                            class_metrics_data[class_name] = {
                                'Precision': metrics[f'{class_name}_precision'],
                                'Recall': metrics[f'{class_name}_recall'],
                                'F1-Score': metrics[f'{class_name}_f1'],
                                'Support': int(metrics[f'{class_name}_support'])
                            }
                    if class_metrics_data:
                        st.dataframe(pd.DataFrame.from_dict(class_metrics_data, orient='index')) # Transpose to make classes as rows
                    else:
                        st.warning("No per-class metrics available.")

                with tab_conf_matrix:
                    st.write("### Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('True')
                    ax_cm.set_title('Confusion Matrix')
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                with tab_roc_pr:
                    if len(class_names) == 2 and roc_curves and pr_curves and roc_auc is not None and pr_auc is not None:
                        st.write("### ROC Curve")
                        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
                        ax_roc.plot(roc_curves['fpr'], roc_curves['tpr'], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax_roc.set_xlim([0.0, 1.0])
                        ax_roc.set_ylim([0.0, 1.05])
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax_roc.legend(loc='lower right')
                        ax_roc.grid(True)
                        st.pyplot(fig_roc)
                        plt.close(fig_roc)

                        st.write("### Precision-Recall Curve")
                        fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
                        ax_pr.plot(pr_curves['recall'], pr_curves['precision'], color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
                        ax_pr.axhline(y=0.5, color='red', linestyle='--', label='No Skill')
                        ax_pr.set_xlim([0.0, 1.0])
                        ax_pr.set_ylim([0.0, 1.05])
                        ax_pr.set_xlabel('Recall')
                        ax_pr.set_ylabel('Precision')
                        ax_pr.set_title('Precision-Recall Curve')
                        ax_pr.legend(loc='lower left')
                        ax_pr.grid(True)
                        st.pyplot(fig_pr)
                        plt.close(fig_pr)
                    else:
                        st.info("ROC and PR Curves are applicable for binary classification only. Please ensure a binary classification setup and sufficient data.")

                with tab_misclass:
                    st.write("### Misclassified Samples")
                    if len(misclassified_indices) > 0:
                        misclassified_data = []
                        for idx in misclassified_indices:
                            misclassified_data.append({
                                'filename': filenames[idx],
                                'true_class': true_classes[idx],
                                'predicted_class': pred_classes[idx],
                                'prediction_confidence': f"{confidences[idx]:.4f}", # Format confidence
                            })
                        misclassified_df = pd.DataFrame(misclassified_data)
                        st.write(f"Total Misclassifications: **{len(misclassified_indices)}**")
                        st.dataframe(misclassified_df)
                    else:
                        st.info("No misclassifications found for the selected dataset split.")

            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")