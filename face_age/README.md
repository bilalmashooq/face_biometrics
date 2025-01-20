
# Face Age Estimation Application

This project provides a facial age estimation system using deep learning models and the **UKTface** dataset. The application supports image uploads and real-time webcam-based predictions with an intuitive GUI built using PyQt5.

## Features

- **Age Estimation:** Predicts age from facial images using two pre-trained models.
- **Image & Webcam Support:** Upload images or use the webcam for real-time estimation.
- **Save Results:** Save annotated images with predicted ages.
- **Dataset Used:** UKTface dataset for training and evaluation.

## Installation



### Steps

1. Clone the repository:

2. Download the **UKTface** dataset and place it in the `dataset/` folder.

3. Run the application:

   ```bash
   python face_age_app.py
   ```

## Usage

1. **Upload Image:** Click "Upload Image" to select an image.
2. **Webcam Mode:** Click "Start Webcam" for real-time age estimation.
3. **Save Results:** Click "Save Results" to store the analyzed image.
4. **Exit:** Close the app by clicking "Exit".

## Dataset

The **UKTface** dataset contains facial images with age labels and can be downloaded from the official site.


## Model Details

The application uses two pre-trained models from Hugging face for age estimation:

1. **Model 1 (dima806/facial_age_image_detection):** Vision Transformer-based model.
2. **Model 2 (nateraw/vit-age-classifier):** CNN-based model for robust age classification.



## License

Licensed under the MIT License.
