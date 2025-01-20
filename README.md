# Face Biometrics Applications

This repository contains multiple applications focused on **face biometrics**, including **age estimation, pose estimation, emotion recognition, gender classification, and face verification/identification systems.** Each application leverages deep learning models and provides a user-friendly **PyQt5-based GUI** for real-time processing.

---

## Projects Overview

### 1. [Face Age Estimation](face-age-estimation/)
   - **Description:** Predicts a person's age from facial images using deep learning.
   - **Features:** 
     - Supports image uploads and real-time webcam processing.
     - Saves results with predicted ages.
   - **Models Used:** 
     - `dima806/facial_age_image_detection`
     - `nateraw/vit-age-classifier`
  

---

### 2. [Face Pose Estimation](face-pose-estimation/)
   - **Description:** Estimates head pose angles (Yaw, Pitch, Roll) using facial landmarks.
   - **Features:**
     - Image and video processing with real-time webcam support.
     - Visual feedback with pose axes drawn on the face.
   - **Models Used:**  
     - 7-landmark model (`model.pkl`)  
     - 468-landmark model (`SVR_model.sav`)  


---

### 3. [Facial Emotion Recognition](emotion-recognition/)
   - **Description:** Recognizes facial emotions in images and video streams.
   - **Features:**
     - Detects emotions such as happy, sad, angry, surprise, etc.
     - Provides real-time results and result saving.
   - **Models Used:**  
     - `trpakov/vit-face-expression`  
     - `motheecreator/vit-Facial-Expression-Recognition`  
 
---

### 4. [Gender Classification](gender-classification/)
   - **Description:** Classifies gender based on facial features using machine learning.
   - **Features:**
     - Supports dataset preprocessing and model evaluation.
     - Performance metrics such as accuracy, precision, and recall.
   - **Models Used:**  
     - `rizvandwiki/gender-classification`
     - `mrm8488/mobilevit-small-finetuned-agegender`
  
---

### 5. [Face Identification System](face-identification/)
   - **Description:** Identifies registered individuals using facial recognition.
   - **Features:**
     - Register faces and identify from uploaded images.
     - Performance evaluation tools and GUI-based interface.
   - **Models Used:**  
     - `Facenet`
     - `VGG-Face`
     - `ArcFace`

---

### 6. [Face Verification System](face-verification/)
   - **Description:** Verifies if two faces belong to the same person (1-to-1 matching).
   - **Features:**
     - Secure verification with pre-built embeddings.
     - Evaluation using FAR, FRR, and ROC curves.
   - **Models Used:**  
     - `Facenet`
     - `ArcFace`
     - `VGG-Face`


---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/face_biometrics.git
   cd face_biometrics
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up datasets:**  
   Download the required datasets (e.g., UKTface, FER2013) and place them in respective subdirectories.



## Requirements

Ensure you have the following installed:

- Python 3.8+
- PyQt5
- OpenCV
- Transformers (Hugging Face)
- DeepFace
- Pandas, NumPy, Matplotlib, Seaborn

---

## License

This project is licensed under the MIT License.
