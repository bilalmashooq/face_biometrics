# Facial Emotion Recognition Application

This project provides a facial emotion recognition system using deep learning models and the **FER2013** dataset. The application supports image uploads and real-time webcam-based predictions with a user-friendly GUI built using PyQt5.

## Features

- **Emotion Recognition:** Predicts facial emotions using two pre-trained models.
- **Image & Webcam Support:** Upload images or use the webcam for real-time detection.
- **Save Results:** Save analyzed images with predicted emotions.
- **Dataset Used:** FER2013 dataset for training and evaluation.

## Installation

1. Clone the repository:

2. Download the FER2013 dataset and place it in the `dataset/` folder.

[Dataset](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

3. Install dependencies

4. Run the application:

   ```bash
   python emotion_recognition_app.py
   ```

## Usage

- **Upload Image:** Click "Upload Image" to analyze an image.
- **Webcam Mode:** Click "Start Webcam" for real-time emotion detection.
- **Save Results:** Click "Save Results" to store the analyzed image.
- **Exit:** Click "Exit" to close the application.

## Models Used
Models are loaded from the Hugging Face model hub.

- **Model 1:** `trpakov/vit-face-expression` (Vision Transformer-based).
- **Model 2:** `motheecreator/vit-Facial-Expression-Recognition` (CNN-based).

## License

Licensed under the MIT License.