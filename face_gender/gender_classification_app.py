import sys
import os
import cv2
import io
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont

# Update the path to your Haar Cascade if needed
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"


class WebcamThread(QThread):
    """
    QThread class to handle webcam capture in the background,
    perform face detection, and run gender classification on each face.
    """
    change_pixmap_signal = pyqtSignal(QImage)
    face_summary_signal = pyqtSignal(str)

    def __init__(self, pipe_model1, pipe_model2):
        super().__init__()
        self.pipe_model1 = pipe_model1
        self.pipe_model2 = pipe_model2
        self._run_flag = True

        # Initialize the Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    def run(self):
        # Capture from webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(None, "Webcam Error", "Cannot access the webcam.")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                # Convert the frame to RGB (PIL format) for annotation
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                draw = ImageDraw.Draw(pil_image)

                face_summaries = []
                for idx, (x, y, w, h) in enumerate(faces, start=1):
                    # Crop the face region
                    face_crop = pil_image.crop((x, y, x + w, y + h))

                    # Model 1 prediction
                    try:
                        preds1 = self.pipe_model1(face_crop)
                        label1 = preds1[0]['label']
                        score1 = preds1[0]['score']
                        gender1 = self.parse_gender_label(label1)
                    except Exception:
                        gender1, score1 = "Error", 0.0

                    # Model 2 prediction
                    try:
                        preds2 = self.pipe_model2(face_crop)
                        label2 = preds2[0]['label']
                        score2 = preds2[0]['score']
                        gender2 = self.parse_gender_label(label2)
                    except Exception:
                        gender2, score2 = "Error", 0.0

                    # Draw bounding box
                    draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
                    # Put text above the bounding box
                    text_info = f"Face {idx}: M1={gender1}, M2={gender2}"
                    draw.text((x, y - 25), text_info, fill="red", font=ImageFont.truetype("arial.ttf", 16))

                    summary = (f"Face {idx}: "
                               f"Model1={gender1}({score1:.2f}), "
                               f"Model2={gender2}({score2:.2f})")
                    face_summaries.append(summary)

                # Prepare summary for the entire frame
                summary_text = "\n".join(face_summaries) if face_summaries else "No faces detected."
                self.face_summary_signal.emit(summary_text)

                # Convert the PIL image with annotations back to a NumPy array
                annotated_frame = np.array(pil_image)

                # Convert annotated_frame to QImage for display in QLabel
                h, w, ch = annotated_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Scale the final QImage to keep consistent display size
                qt_img_scaled = qt_img.scaled(800, 600, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(qt_img_scaled)

        # Release the webcam
        cap.release()

    def stop(self):
        """Stops the webcam thread."""
        self._run_flag = False
        self.wait()

    @staticmethod
    def parse_gender_label(label):
        """
        Convert the raw pipeline label (e.g., 'male' or 'female') to a standardized label.
        """
        if label.lower() == 'male':
            return 'Male'
        elif label.lower() == 'female':
            return 'Female'
        else:
            return 'Unknown'


class GenderClassifierGUI(QWidget):
    """
    Main GUI class for the Gender Classification application.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gender Classification")
        self.setGeometry(100, 100, 1200, 800)

        # Apply an overall stylesheet for a more modern look
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f4f8;
                color: #333333;
                font-family: Arial, sans-serif;
            }
            QLabel#titleLabel {
                color: #2c3e50;
                font-size: 28px;
                font-weight: bold;
            }
            QLabel#imageLabel {
                border: 2px dashed #ccc;
                background-color: #ffffff;
                color: #777777;
                font-size: 16px;
            }
            QLabel#resultLabel {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #007bff;
                color: #fff;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                min-height: 50px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
        """)

        # Initialize models
        self.init_models()

        # Initialize UI components
        self.init_ui()

        # Variables
        self.image = None
        self.webcam_thread = None

    def init_models(self):
        """
        Load the two gender classification models using Hugging Face pipelines.
        """
        print("Loading Models...")
        try:
            # Model 1
            self.pipe_model1 = pipeline("image-classification", model="rizvandwiki/gender-classification")
            print("Model 1 loaded successfully.")
        except Exception as e:
            print(f"Error loading Model 1: {e}")
            self.pipe_model1 = None
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load Model 1:\n{e}")

        try:
            # Model 2
            self.pipe_model2 = pipeline("image-classification", model="rizvandwiki/gender-classification-2")
            print("Model 2 loaded successfully.")
        except Exception as e:
            print(f"Error loading Model 2: {e}")
            self.pipe_model2 = None
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load Model 2:\n{e}")

    def init_ui(self):
        """
        Set up the GUI layout and all interactive widgets.
        """
        # Title label at the top
        self.title_label = QLabel("Gender Classification", objectName="titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Image display in the center-right
        self.image_label = QLabel("No image selected.", objectName="imageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(400, 300)

        # Results label at bottom (wrapped in a QFrame for a 'card' look)
        self.result_label = QLabel("Model Predictions", objectName="resultLabel")
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.result_label.setMinimumHeight(100)

        # Buttons
        self.btn_load = self.create_button("Load Image", self.load_image)
        self.btn_start_webcam = self.create_button("Start Webcam", self.start_webcam)
        self.btn_stop_webcam = self.create_button("Stop Webcam", self.stop_webcam, enabled=False)
        self.btn_clear = self.create_button("Clear Canvas", self.clear_display)
        self.btn_exit = self.create_button("Exit", self.close_app)

        # Set maximum width for buttons to prevent stretching
        button_max_width = 200
        for btn in [self.btn_load, self.btn_start_webcam, self.btn_stop_webcam, self.btn_clear, self.btn_exit]:
            btn.setMaximumWidth(button_max_width)

        # Left vertical layout for the buttons
        left_button_layout = QVBoxLayout()
        left_button_layout.setSpacing(20)
        left_button_layout.addWidget(self.btn_load)
        left_button_layout.addWidget(self.btn_start_webcam)
        left_button_layout.addWidget(self.btn_stop_webcam)
        left_button_layout.addWidget(self.btn_clear)
        left_button_layout.addWidget(self.btn_exit)
        left_button_layout.addStretch()

        # Main layout: title on top, then a row with left buttons and right image area
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(30)
        main_layout.addWidget(self.title_label)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(30)
        row_layout.addLayout(left_button_layout)

        # Right side layout: image + results
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        right_layout.addWidget(self.image_label)

        # Frame for the result label to give a bit more spacing
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.NoFrame)
        result_frame_layout = QVBoxLayout()
        result_frame_layout.setContentsMargins(0, 0, 0, 0)
        result_frame_layout.addWidget(self.result_label)
        result_frame.setLayout(result_frame_layout)

        right_layout.addWidget(result_frame)
        row_layout.addLayout(right_layout)

        main_layout.addLayout(row_layout)
        self.setLayout(main_layout)

    def create_button(self, text, slot, enabled=True):
        """
        Helper method to create a styled QPushButton.
        """
        button = QPushButton(text)
        button.setEnabled(enabled)
        button.clicked.connect(slot)
        return button

    def close_app(self):
        """Exit the application."""
        self.close()

    def load_image(self):
        """
        Select an image from disk, detect faces, run classification on each face,
        and display results.
        """
        self.clear_display()  # Clear canvas before loading a new image

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )
        if not file_name:
            return

        # Check if webcam is running
        if self.webcam_thread and self.webcam_thread.isRunning():
            QMessageBox.warning(self, "Webcam Running", "Please stop the webcam before loading a new image.")
            return

        # Load the image with PIL
        try:
            self.image = Image.open(file_name).convert('RGB')
        except Exception as e:
            QMessageBox.critical(self, "Image Loading Error", f"Failed to load image:\n{e}")
            return

        annotated_pil, summary_text = self.process_image_for_faces(self.image)
        self.result_label.setText(summary_text)

        # Convert annotated PIL image to QPixmap
        img_byte_arr = io.BytesIO()
        annotated_pil.save(img_byte_arr, format='PNG')
        qimage = QImage.fromData(img_byte_arr.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def process_image_for_faces(self, pil_image):
        """
        Detect faces using Haar Cascades, run both models on each face,
        and draw bounding boxes + text. Returns:
        1) The annotated PIL image
        2) A summary text of face predictions
        """
        # Convert PIL to NumPy (RGB)
        cv_image = np.array(pil_image)
        # Convert from RGB to BGR for OpenCV face detection
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # Convert to grayscale for detection
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Haar Cascade
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

        annotated_pil = pil_image.copy()
        draw = ImageDraw.Draw(annotated_pil)
        face_summaries = []

        if len(faces) == 0:
            return annotated_pil, "No faces detected."

        for idx, (x, y, w, h) in enumerate(faces, start=1):
            face_crop = annotated_pil.crop((x, y, x + w, y + h))

            # Model 1
            if self.pipe_model1:
                try:
                    preds1 = self.pipe_model1(face_crop)
                    label1 = preds1[0]['label']
                    score1 = preds1[0]['score']
                    gender1 = self.parse_gender_label(label1)
                except Exception:
                    gender1, score1 = "Error", 0.0
            else:
                gender1, score1 = "NotLoaded", 0.0

            # Model 2
            if self.pipe_model2:
                try:
                    preds2 = self.pipe_model2(face_crop)
                    label2 = preds2[0]['label']
                    score2 = preds2[0]['score']
                    gender2 = self.parse_gender_label(label2)
                except Exception:
                    gender2, score2 = "Error", 0.0
            else:
                gender2, score2 = "NotLoaded", 0.0

            # Draw bounding box
            draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
            # Overlay text above face bounding box
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            text_info = f"Face {idx}: M1={gender1}, M2={gender2}"
            draw.text((x, y - 25), text_info, fill="red", font=font)

            summary = (f"Face {idx}: "
                       f"Model1={gender1}({score1:.2f}), "
                       f"Model2={gender2}({score2:.2f})")
            face_summaries.append(summary)

        summary_text = "\n".join(face_summaries)
        return annotated_pil, summary_text

    def start_webcam(self):
        """
        Start the webcam thread for real-time face detection and gender classification.
        """
        self.clear_display()  # Clear canvas before starting webcam

        if self.pipe_model1 is None or self.pipe_model2 is None:
            QMessageBox.critical(self, "Model Not Loaded",
                                 "Cannot start webcam because a model failed to load.")
            return

        if self.webcam_thread and self.webcam_thread.isRunning():
            QMessageBox.warning(self, "Webcam Running", "Webcam is already running.")
            return

        self.webcam_thread = WebcamThread(self.pipe_model1, self.pipe_model2)
        self.webcam_thread.change_pixmap_signal.connect(self.update_image)
        self.webcam_thread.face_summary_signal.connect(self.update_result_label)
        self.webcam_thread.start()

        self.btn_start_webcam.setEnabled(False)
        self.btn_stop_webcam.setEnabled(True)
        self.btn_load.setEnabled(False)

    def stop_webcam(self):
        """
        Stop the webcam thread and reset the display.
        """
        if self.webcam_thread is not None:
            self.webcam_thread.stop()
            self.webcam_thread = None

        self.clear_display()
        self.btn_start_webcam.setEnabled(True)
        self.btn_stop_webcam.setEnabled(False)
        self.btn_load.setEnabled(True)

    def update_image(self, qimage):
        """Update the QLabel (image_label) with the new frame from webcam."""
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def update_result_label(self, summary):
        """Update the result_label with the face summary text from webcam."""
        self.result_label.setText(summary)

    def clear_display(self):
        """Clear or reset the image label and result label."""
        self.image_label.clear()
        self.image_label.setText("No image selected.")
        self.result_label.setText("Model Predictions")

    @staticmethod
    def parse_gender_label(label):
        """
        Convert the raw label from the pipeline to 'Male', 'Female', or 'Unknown'.
        """
        if label.lower() == 'male':
            return 'Male'
        elif label.lower() == 'female':
            return 'Female'
        else:
            return 'Unknown'

    def closeEvent(self, event):
        """
        Ensure the webcam thread is properly closed when quitting.
        """
        self.stop_webcam()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = GenderClassifierGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
