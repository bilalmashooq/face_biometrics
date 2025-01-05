
# Face Identification System

## Overview

This project implements a **Face Identification System** using deep learning models integrated with a user-friendly **PyQt5-based GUI**. The system supports face registration and identification, allowing multiple registered faces to be recognized in real-time.

---

## Features

1. **GUI Interface:**
   - Intuitive tabs for **Registration** and **Recognition**.
   - User-friendly interface for uploading, registering, and identifying faces.

2. **Deep Learning Models:**
   - Utilizes state-of-the-art models such as **Facenet** and **VGG-Face** via the **DeepFace** library.

3. **Pre-Built Embeddings:**
   - Caches face embeddings for faster recognition during subsequent runs.

4. **Multiple Face Handling:**
   - Handles multiple faces in the query image using **DeepFace.find**.

5. **Performance and Scalability:**
   - Accelerates recognition with prebuilt embeddings stored in `.deepface` folders.

---

## Files Description

1. **`main_identification.py`:**
   - Main script for the GUI application, providing face registration and identification functionality.
   - Automates embedding generation for datasets to speed up recognition.

2. **`evalution1/2.py`:**
   - Python files are used to evaluate model performance, analyze embeddings, and visualize results.


### Step 3: Set Up Dataset
Organize your dataset as follows:
```
dataset/
  person1/
    image1.jpg
    image2.jpg
  person2/
    image1.jpg
    image2.jpg
```
Update the `dataset_path` in `main_identification.py` to point to your dataset.

---

## Usage

### 1. **Register Individuals**
- Switch to the **Registration Tab**.
- Upload an image, assign a name, and save the registration.
- Images are stored in the dataset folder under individual subfolders.

### 2. **Identify Individuals**
- Switch to the **Recognition Tab**.
- Upload a query image to find the closest matches from the dataset.
- Results include the top match and similarity scores.

### 3. **Evaluate Performance**
- Use the `evalution_identification.ipynb` notebook to:
  - Analyze embeddings.
  - Visualize distances and similarity metrics.
  - Evaluate model performance.



## License

This project is licensed under the MIT License.
