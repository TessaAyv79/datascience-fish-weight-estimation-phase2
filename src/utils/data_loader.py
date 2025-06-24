import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path, resize=(224, 224)):
    images = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, resize)
                images.append(img)
                filenames.append(filename)
    
    print(f"{len(images)} images loaded from {folder_path}")
    return images, filenames

def display_sample_images(images, filenames=None, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i+1)
        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        title = filenames[i] if filenames else f"Image {i+1}"
        plt.title(title)
        plt.axis("off")
    plt.show()

def load_labels(label_csv_path):
    return pd.read_csv(label_csv_path)

def merge_images_with_labels(filenames, labels_df, image_column="filename"):
    df = pd.DataFrame({"filename": filenames})
    merged_df = pd.merge(df, labels_df, on=image_column, how="left")
    return merged_df
