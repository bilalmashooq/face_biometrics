import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from transformers import pipeline
import re
import warnings
warnings.filterwarnings("ignore")  # To suppress warnings for cleaner output

# 1. Define the dataset directory
DATASET_DIR = 'UTKFace'  # Update this path based on your setup

# 2. Function to parse age from filename
def parse_age(filename):
    try:
        age = int(filename.split('_')[0])
        return age
    except:
        return None

# 3. Create a DataFrame with image paths and age labels
data = []
for img_name in os.listdir(DATASET_DIR):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        age = parse_age(img_name)
        if age is not None:
            img_path = os.path.join(DATASET_DIR, img_name)
            data.append({'image_path': img_path, 'age': age})

df = pd.DataFrame(data)
print(f'Total samples: {len(df)}')

# 4. Split the data into training, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1765, random_state=42)  # 0.1765 * 0.85 H 0.15

print(f'Training samples: {len(train_df)}')
print(f'Validation samples: {len(val_df)}')
print(f'Testing samples: {len(test_df)}')

# 5. Load Both Models Using Pipelines
print("\nLoading Models...")
pipe_model1 = pipeline("image-classification", model="dima806/facial_age_image_detection")
pipe_model2 = pipeline("image-classification", model="nateraw/vit-age-classifier")
print("Models Loaded Successfully.")

# 6. Prepare the Test Data for Prediction
# Number of images to use for prediction
NUM_IMAGES = 50  # Adjust based on your computational resources

# Select a subset from the test set
test_subset_df = test_df.sample(n=NUM_IMAGES, random_state=42).reset_index(drop=True)

# Load images
test_images = []
for img_path in tqdm(test_subset_df['image_path'], desc="Loading Test Images"):
    try:
        img = Image.open(img_path).convert('RGB')
        test_images.append(img)
    except Exception as e:
        print(f'Error loading image {img_path}: {e}')
        test_images.append(None)  # Placeholder for failed images

# Remove any failed images
valid_indices = [i for i, img in enumerate(test_images) if img is not None]
test_subset_df = test_subset_df.iloc[valid_indices].reset_index(drop=True)
test_images = [img for img in test_images if img is not None]

print(f'Test subset after removing invalid images: {len(test_images)}')

# 7. Define a Function to Parse Age Labels
def parse_age_label(label):
    """
    Parses the age label from the model's prediction.
    
    Parameters:
    - label (str): The age label predicted by the model.
    
    Returns:
    - age (float): The numerical age value.
    """
    # Check if the label is a single number
    if re.match(r'^\d+$', label):
        return float(label)
    # Check if the label is a range like '25-30'
    elif re.match(r'^\d+-\d+$', label):
        ages = label.split('-')
        lower = float(ages[0])
        upper = float(ages[1])
        return (lower + upper) / 2
    else:
        print(f"Unrecognized label format: {label}")
        return np.nan  # Return NaN for unrecognized formats

# 8. Make Predictions with Both Models
# Model 1 Predictions
print("\nPredicting with dima806/facial_age_image_detection...")
predictions_model1 = []
for img in tqdm(test_images, desc="Model 1 Prediction"):
    preds = pipe_model1(img)
    age_label = preds[0]['label']
    age = parse_age_label(age_label)
    predictions_model1.append(age)

# Model 2 Predictions
print("\nPredicting with nateraw/vit-age-classifier...")
predictions_model2 = []
for img in tqdm(test_images, desc="Model 2 Prediction"):
    preds = pipe_model2(img)
    age_label = preds[0]['label']
    age = parse_age_label(age_label)
    predictions_model2.append(age)

# 9. Add Predictions to DataFrame
test_subset_df['pred_model1'] = predictions_model1
test_subset_df['pred_model2'] = predictions_model2

# 10. Data Cleaning: Remove NaN predictions
evaluation_df = test_subset_df.dropna(subset=['pred_model1', 'pred_model2']).reset_index(drop=True)
print(f'Evaluation DataFrame shape: {evaluation_df.shape}')

if evaluation_df.empty:
    print("\nError: No valid predictions found. Please check the label parsing logic and model outputs.")
    exit()

# 11. True ages
true_ages = evaluation_df['age'].values

# 12. Predictions
pred_model1 = evaluation_df['pred_model1'].values
pred_model2 = evaluation_df['pred_model2'].values

# 13. Calculate Mean Absolute Error (MAE)
mae_model1 = mean_absolute_error(true_ages, pred_model1)
mae_model2 = mean_absolute_error(true_ages, pred_model2)

print(f'\nModel 1 (dima806/facial_age_image_detection) MAE: {mae_model1:.2f} years')
print(f'Model 2 (nateraw/vit-age-classifier) MAE: {mae_model2:.2f} years')

# 14. Plot Cumulative Score Curves
import matplotlib.pyplot as plt

def plot_cumulative_score(true, pred, model_name, max_error=10):
    """
    Plots the cumulative percentage of predictions within a specified error range.
    
    Parameters:
    - true (array): True age labels.
    - pred (array): Predicted age labels.
    - model_name (str): Name of the model.
    - max_error (int): Maximum error to consider.
    """
    errors = np.abs(true - pred)
    cumulative = [np.mean(errors <= i) * 100 for i in range(max_error + 1)]
    
    plt.figure(figsize=(8,6))
    plt.plot(range(max_error + 1), cumulative, marker='o', label=model_name)
    plt.title('Cumulative Score Curve')
    plt.xlabel('Maximum Error (years)')
    plt.ylabel('Cumulative Percentage (%)')
    plt.grid(True)
    plt.legend()
    plt.xticks(range(0, max_error + 1, 1))
    plt.yticks(range(0, 101, 10))
    plt.show()

# Plot for Model 1
plot_cumulative_score(true_ages, pred_model1, 'Model 1 (dima806/facial_age_image_detection)')

# Plot for Model 2
plot_cumulative_score(true_ages, pred_model2, 'Model 2 (nateraw/vit-age-classifier)')

# 15. Distribution of Errors
errors_model1 = np.abs(true_ages - pred_model1)
errors_model2 = np.abs(true_ages - pred_model2)

plt.figure(figsize=(12,6))
sns.histplot(errors_model1, color='blue', label='Model 1', kde=True, stat="density", linewidth=0)
sns.histplot(errors_model2, color='green', label='Model 2', kde=True, stat="density", linewidth=0, alpha=0.7)
plt.title('Distribution of Age Prediction Errors')
plt.xlabel('Absolute Error (years)')
plt.ylabel('Density')
plt.legend()
plt.show()

# 16. Optional: Detailed Classification Report (Binning Ages)
# Since age estimation is a regression task, classification metrics can be less informative.
# However, for analysis, we can bin ages into categories.

def age_binning(age):
    if age < 20:
        return 'Teen'
    elif 20 <= age < 40:
        return 'Young Adult'
    elif 40 <= age < 60:
        return 'Middle-aged'
    else:
        return 'Senior'

# Bin true and predicted ages
binned_true = [age_binning(age) for age in true_ages]
binned_pred_cnn = [age_binning(pred) for pred in pred_model1]
binned_pred_resnet = [age_binning(pred) for pred in pred_model2]

# Classification Report for Model 1
print("\nModel 1 (dima806/facial_age_image_detection) Classification Report (Binned Ages):")
print(classification_report(binned_true, binned_pred_cnn, target_names=['Teen', 'Young Adult', 'Middle-aged', 'Senior']))

# Classification Report for Model 2
print("\nModel 2 (nateraw/vit-age-classifier) Classification Report (Binned Ages):")
print(classification_report(binned_true, binned_pred_resnet, target_names=['Teen', 'Young Adult', 'Middle-aged', 'Senior']))
