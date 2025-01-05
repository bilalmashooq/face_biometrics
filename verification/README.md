
# Face Verification System

## Overview

This project implements a **Face Verification System** combining deep learning models with a user-friendly **PyQt5-based GUI**. The system supports:

1. **Registration** of individuals with unique IDs and images.
2. **Face Verification** using models such as **Facenet**, **ArcFace**, and **VGG-Face** for 1-to-1 verification.
3. **Performance Evaluation** using metrics like ROC curves, EER, confusion matrices, and classification metrics.

The repository includes Python scripts for embedding extraction, dataset management, pair generation, distance computation, and performance evaluation.

---

## Features

- **GUI Interface:**
  - Intuitive tabs for registration and verification.
  - Upload, save, and manage face images.
  - Verify faces by comparing query images with reference images.
  
- **Deep Learning Models:**
  - Use state-of-the-art face recognition models like **Facenet**, **ArcFace**, and **VGG-Face** via the **DeepFace** library.

- **Evaluation Utilities:**
  - Compute distances between embeddings.
  - Plot ROC curves and evaluate Equal Error Rate (EER).
  - Generate confusion matrices and performance metrics.

---

## Files Description

1. **`main_verification.py`:**
   - Main script for launching the PyQt5 GUI for face registration and verification.
   - Supports switching between registration and verification tabs.
   - Includes functionality to save images to uniquely named folders and verify faces.

2. **`evalution.py`:**
   - Core script for managing datasets, generating pairs, computing distances, and evaluating model performance.
   - Includes utilities for:
     - Assigning unique IDs to dataset folders.
     - Generating pairs for both single and multiple models.
     - Calculating metrics like FAR, FRR, EER, and plotting evaluation curves.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-verification-system.git
   cd face-verification-system
   ```

2. **Install dependencies:**
   Ensure Python 3.7+ is installed. Then, install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the dataset:**
   - Place images in a folder structured as:
     ```
     dataset/
       person1/
         image1.jpg
         image2.jpg
       person2/
         image1.jpg
         image2.jpg
     ```
   - Update the dataset path in `main_verification.py` and `evalution.py`.

4. **Run the GUI:**
   ```bash
   python main_verification.py
   ```

---

## Usage

### 1. **Register Individuals**
- Switch to the **Registration Tab**.
- Upload an image and assign a unique ID and name. The system will create a folder for the individual.

### 2. **Verify Faces**
- Switch to the **Verification Tab**.
- Upload a query image and enter the ID of the individual to verify.
- The system will compare the query image with the reference image(s) using the selected model.

### 3. **Evaluate Models**
- Use the functions in `evalution.py` to:
  - Generate pairs and distances for evaluation.
  - Compute metrics like ROC, FAR/FRR, and EER.
  - Compare models using their respective outputs.

---

## Examples

### Generate Results for a Single Model:
```python
from evalution import generate_results_single_model

generate_results_single_model(
    dataset_dir="path/to/dataset",
    results_csv="results_model1.csv",
    model_name="Facenet",
    max_same_pairs=5,
    max_diff_pairs=5
)
```

### Evaluate a Model's Performance:
```python
from evalution import evaluate_single_csv

evaluate_single_csv("results_model1.csv", model_name="Facenet")
```

---

## Requirements

- Python 3.7+
- PyQt5
- DeepFace
- OpenCV
- Matplotlib
- NumPy
- Scikit-learn
- Seaborn



---

## Screenshots

### Registration Tab:
- !![img_2.png](img_2.png)

### Verification Tab:
- ![img_1.png](img_1.png)

---

## License

This project is licensed under the MIT License.

---

## Contact

- **Author:** Muhammad Bilal


---

If you have any additional content or screenshots you'd like to add, let me know!