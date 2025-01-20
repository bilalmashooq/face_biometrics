import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")  # To suppress warnings for cleaner output

# 1. Define the dataset directory
DATASET_DIR = 'M&FDataset'  # Update this path based on your setup

# 2. Function to load images and labels with a limit of 50 images per class
def load_dataset(dataset_dir, num_images_per_class=50):
    image_paths = []
    labels = []
    
    # Male Faces
    male_dir = os.path.join(dataset_dir, 'Male Faces')
    male_images = [img for img in os.listdir(male_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(male_images)
    selected_male_images = male_images[:num_images_per_class]
    
    for img_name in selected_male_images:
        img_path = os.path.join(male_dir, img_name)
        image_paths.append(img_path)
        labels.append('Male')
    
    # Female Faces
    female_dir = os.path.join(dataset_dir, 'Female Faces')
    female_images = [img for img in os.listdir(female_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    np.random.shuffle(female_images)
    selected_female_images = female_images[:num_images_per_class]
    
    for img_name in selected_female_images:
        img_path = os.path.join(female_dir, img_name)
        image_paths.append(img_path)
        labels.append('Female')
    
    # Create a DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    
    return df

# 3. Load the dataset with 50 images per class
df = load_dataset(DATASET_DIR, num_images_per_class=50)
print(f'Total samples: {len(df)}')  # Should print 100

# 4. Split the data
# With 100 samples, split into Training (60%), Validation (20%), Testing (20%)
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f'Training samples: {len(train_df)}')   # Should be 60
print(f'Validation samples: {len(val_df)}')   # Should be 20
print(f'Testing samples: {len(test_df)}')     # Should be 20

# 5. Initialize Pipelines
print("\nLoading Models...")
# Model 1: rizvandwiki/gender-classification
pipe_model1 = pipeline("image-classification", model="rizvandwiki/gender-classification")

# Model 2: mrm8488/mobilevit-small-finetuned-agegender
pipe_model2 = pipeline("image-classification", model="rizvandwiki/gender-classification-2")

print("Models Loaded Successfully.")

# 6. Function to load and preprocess images
def load_and_preprocess_image(image_path):
    """
    Loads an image from the given path and preprocesses it for the model.
    
    Parameters:
    - image_path (str): Path to the image file.
    
    Returns:
    - image (PIL.Image): Preprocessed image or None if loading fails.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')
        return None

# 7. Prepare Test Images
print("\nLoading Test Images...")
test_images = []
for img_path in tqdm(test_df['image_path'], desc="Loading Test Images"):
    img = load_and_preprocess_image(img_path)
    if img is not None:
        test_images.append(img)

print(f'Total test images loaded: {len(test_images)}')  # Should be 20

# 8. Define a Function to Parse Gender Labels
def parse_gender_label(label):
    """
    Parses the gender label from the model's prediction.
    
    Parameters:
    - label (str): The gender label predicted by the model.
    
    Returns:
    - gender (str): 'Male' or 'Female'. Returns 'Unknown' for unrecognized labels.
    """
    if label.lower() == 'male':
        return 'Male'
    elif label.lower() == 'female':
        return 'Female'
    else:
        return 'Unknown'

# 9. Make Predictions with Both Models
def make_gender_predictions(pipeline, images):
    """
    Makes gender predictions on a list of images using the provided pipeline.
    
    Parameters:
    - pipeline: Hugging Face pipeline for image classification.
    - images (list): List of PIL.Image objects.
    
    Returns:
    - predictions (list): List of predicted genders ('Male', 'Female', 'Unknown').
    """
    predictions = []
    for img in tqdm(images, desc="Predicting"):
        preds = pipeline(img)
        label = preds[0]['label']
        gender = parse_gender_label(label)
        predictions.append(gender)
    return predictions

# Model 1 Predictions
print("\nPredicting with Model 1 (rizvandwiki/gender-classification)...")
predictions_model1 = make_gender_predictions(pipe_model1, test_images)

# Model 2 Predictions
print("\nPredicting with Model 2 (mrm8488/mobilevit-small-finetuned-agegender)...")
predictions_model2 = make_gender_predictions(pipe_model2, test_images)

# 10. Add Predictions to DataFrame
test_df = test_df.copy()  # To avoid SettingWithCopyWarning
test_df['pred_model1'] = predictions_model1
test_df['pred_model2'] = predictions_model2

# 11. Data Cleaning: Remove 'Unknown' Predictions
evaluation_df = test_df[(test_df['pred_model1'] != 'Unknown') & (test_df['pred_model2'] != 'Unknown')].reset_index(drop=True)
print(f'\nEvaluation DataFrame shape: {evaluation_df.shape}')  # Should be <=20

if evaluation_df.empty:
    print("\nError: No valid predictions found. Please check the label parsing logic and model outputs.")
    exit()

# 12. True Labels
true_labels = evaluation_df['label']

# 13. Predictions
pred_labels_model1 = evaluation_df['pred_model1']
pred_labels_model2 = evaluation_df['pred_model2']

# 14. Calculate Metrics for Model 1
accuracy1 = accuracy_score(true_labels, pred_labels_model1)
precision1 = precision_score(true_labels, pred_labels_model1, pos_label='Male')
recall1 = recall_score(true_labels, pred_labels_model1, pos_label='Male')
f1_1 = f1_score(true_labels, pred_labels_model1, pos_label='Male')

print("\n----- Model 1 (rizvandwiki/gender-classification) -----")
print(f"Accuracy: {accuracy1:.2f}")
print(f"Precision (Male): {precision1:.2f}")
print(f"Recall (Male): {recall1:.2f}")
print(f"F1-Score (Male): {f1_1:.2f}")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels_model1))

# 15. Calculate Metrics for Model 2
accuracy2 = accuracy_score(true_labels, pred_labels_model2)
precision2 = precision_score(true_labels, pred_labels_model2, pos_label='Male')
recall2 = recall_score(true_labels, pred_labels_model2, pos_label='Male')
f1_2 = f1_score(true_labels, pred_labels_model2, pos_label='Male')

print("\n----- Model 2 (mrm8488/mobilevit-small-finetuned-agegender) -----")
print(f"Accuracy: {accuracy2:.2f}")
print(f"Precision (Male): {precision2:.2f}")
print(f"Recall (Male): {recall2:.2f}")
print(f"F1-Score (Male): {f1_2:.2f}")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels_model2))

# 16. Generate Confusion Matrices
def plot_confusion_matrix_cm(cm, model_name):
    """
    Plots the confusion matrix.
    
    Parameters:
    - cm (array): Confusion matrix.
    - model_name (str): Name of the model.
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Male', 'Female'], yticklabels=['Male', 'Female'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Confusion Matrix for Model 1
cm1 = confusion_matrix(true_labels, pred_labels_model1, labels=['Male', 'Female'])
plot_confusion_matrix_cm(cm1, 'Model 1 (rizvandwiki/gender-classification)')

# Confusion Matrix for Model 2
cm2 = confusion_matrix(true_labels, pred_labels_model2, labels=['Male', 'Female'])
plot_confusion_matrix_cm(cm2, 'Model 2 (mrm8488/mobilevit-small-finetuned-agegender)')
