import sys
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from transformers import pipeline


class AgeEstimationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Age Estimation")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            background-color: #edf2f7;  /* Light grayish-blue background */
            color: #333333;            /* Dark gray text color */
        """)

        # Initialize Models
        self.init_models()

        # Setup UI
        self.setup_ui()

        # Initialize Webcam Thread
        self.thread = None

    def init_models(self):
        """Initializes the age estimation models using Hugging Face pipelines."""
        print("Loading Age Estimation Models...")
        try:
            self.pipe_age1 = pipeline("image-classification", model="dima806/facial_age_image_detection")
            self.pipe_age2 = pipeline("image-classification", model="nateraw/vit-age-classifier")
            print("Models Loaded Successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"Error loading models:\n{e}")
            sys.exit(1)

    def setup_ui(self):
        """Sets up the user interface components."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)

        # Application Title
        title = QLabel("Face Age Estimation App")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 30px; 
            font-weight: 600; 
            color: #2b2d42; 
            margin-top: 20px;
        """)
        main_layout.addWidget(title)

        # Subtitle for University
        subtitle = QLabel("Université Paris-Est Créteil (UPEC)\nDepartment of Computer Vision and Biometrics")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 18px; 
            color: #444; 
            margin-bottom: 20px;
        """)
        main_layout.addWidget(subtitle)

        # Content Layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # Button Layout (Left side)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)

        # Upload Image Button
        self.upload_btn = self.create_button("Upload Image", "#0078D7", self.upload_image)
        button_layout.addWidget(self.upload_btn)

        # Webcam Button
        self.webcam_btn = self.create_button("Start Webcam", "#28a745", self.toggle_webcam)
        self.webcam_btn.setCheckable(True)
        button_layout.addWidget(self.webcam_btn)

        # Save Results Button
        self.save_btn = self.create_button("Save Results", "#ffc107", self.save_results)
        self.save_btn.setEnabled(False)  # Disabled until an image or webcam is active
        button_layout.addWidget(self.save_btn)

        # Clear Canvas Button (Optional: to allow user to reset the display)
        self.clear_btn = self.create_button("Clear Canvas", "#6c757d", self.clear_canvas)
        button_layout.addWidget(self.clear_btn)

        # Exit Application Button
        self.exit_btn = self.create_button("Exit", "#dc3545", self.close_application)
        button_layout.addWidget(self.exit_btn)

        button_layout.addStretch()
        content_layout.addLayout(button_layout)

        # Display Frame (Right side)
        self.display_frame = QFrame(self)
        self.display_frame.setFrameShape(QFrame.NoFrame)
        # A layout for the display_frame
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(10, 10, 10, 10)
        frame_layout.setSpacing(15)

        # Image Display Label
        self.image_label = QLabel("Your Image Will Appear Here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(700, 500)
        self.image_label.setStyleSheet("""
            border: 2px dashed #aaa; 
            background-color: #ffffff; 
            color: #888; 
            font-size: 16px;
        """)
        frame_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Results Container Frame
        self.results_container = QFrame()
        self.results_container.setFrameShape(QFrame.StyledPanel)
        self.results_container.setStyleSheet("""
            background-color: #f7f9fc; 
            border: 1px solid #ccc; 
            border-radius: 8px;
        """)
        self.results_container.setFixedSize(700, 150)

        # Layout for the results
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(20, 20, 20, 20)
        results_layout.setSpacing(10)

        # Results Title
        results_title = QLabel("Results")
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #2b2d42;")
        results_layout.addWidget(results_title)

        # Model 1 Results
        self.prediction_model1_label = QLabel("Model 1 - Age: N/A")
        self.prediction_model1_label.setAlignment(Qt.AlignCenter)
        self.prediction_model1_label.setStyleSheet("font-size: 16px; color: #333;")
        results_layout.addWidget(self.prediction_model1_label)

        # Model 2 Results
        self.prediction_model2_label = QLabel("Model 2 - Age: N/A")
        self.prediction_model2_label.setAlignment(Qt.AlignCenter)
        self.prediction_model2_label.setStyleSheet("font-size: 16px; color: #333;")
        results_layout.addWidget(self.prediction_model2_label)

        self.results_container.setLayout(results_layout)

        frame_layout.addWidget(self.results_container, alignment=Qt.AlignCenter)
        self.display_frame.setLayout(frame_layout)
        content_layout.addWidget(self.display_frame, alignment=Qt.AlignCenter)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        # Track current image path and predictions
        self.current_image_path = None
        self.current_predictions = {
            'Model 1': {'Age': np.nan},
            'Model 2': {'Age': np.nan}
        }

    def create_button(self, text, color, func):
        """Helper method to create styled buttons."""
        button = QPushButton(text)
        button.setFixedSize(150, 50)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: #fff;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:disabled {{
                background-color: #a0a0a0;
            }}
            QPushButton:hover {{
                filter: brightness(110%);
            }}
        """)
        button.clicked.connect(func)
        return button

    def clear_canvas(self):
        """Clears the image display and resets predictions."""
        # Clear the image label
        self.image_label.clear()
        self.image_label.setText("Your Image Will Appear Here")
        self.image_label.setStyleSheet("""
            border: 2px dashed #aaa;
            background-color: #ffffff;
            color: #888; 
            font-size: 16px;
        """)

        # Reset predictions
        self.prediction_model1_label.setText("Model 1 - Age: N/A")
        self.prediction_model2_label.setText("Model 2 - Age: N/A")

        # If a webcam thread is running, stop it
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.webcam_btn.setChecked(False)
            self.webcam_btn.setText("Start Webcam")

        # Reset current data
        self.current_image_path = None
        self.current_predictions = {
            'Model 1': {'Age': np.nan},
            'Model 2': {'Age': np.nan}
        }
        self.save_btn.setEnabled(False)

    def upload_image(self):
        """Handles image upload functionality with enhanced error handling."""
        self.clear_canvas()  # Clear canvas before loading a new image
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
            options=options
        )
        if file_name:
            try:
                self.display_image(file_name)
                self.save_btn.setEnabled(True)

                # Estimate ages
                age1 = self.estimate_age_model1(file_name)
                age2 = self.estimate_age_model2(file_name)

                self.prediction_model1_label.setText(f"Model 1 - Age: {age1:.2f} years")
                self.prediction_model2_label.setText(f"Model 2 - Age: {age2:.2f} years")

                self.current_image_path = file_name
                self.current_predictions = {
                    'Model 1': {'Age': age1},
                    'Model 2': {'Age': age2}
                }
            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"An error occurred while processing the image:\n{e}")

    def display_image(self, img_path):
        """Displays the selected image in the GUI."""
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def estimate_age_model1(self, img_path):
        """Estimates age using Model 1."""
        try:
            img = Image.open(img_path).convert('RGB')
            age_preds = self.pipe_age1(img)
            age_label = age_preds[0]['label']
            age = self.parse_label(age_label)
            return age
        except Exception as e:
            print(f"Error in Model 1 age estimation: {e}")
            QMessageBox.warning(self, "Estimation Warning",
                                f"Model 1 failed to estimate age.\n{e}")
            return np.nan

    def estimate_age_model2(self, img_path):
        """Estimates age using Model 2."""
        try:
            img = Image.open(img_path).convert('RGB')
            age_preds = self.pipe_age2(img)
            age_label = age_preds[0]['label']
            age = self.parse_label(age_label)
            return age
        except Exception as e:
            print(f"Error in Model 2 age estimation: {e}")
            QMessageBox.warning(self, "Estimation Warning",
                                f"Model 2 failed to estimate age.\n{e}")
            return np.nan

    def parse_label(self, label):
        """Parses the label from the model's prediction to extract numerical values."""
        # Single number
        if re.match(r'^\d+$', label):
            return float(label)
        # Range like '25-30'
        elif re.match(r'^\d+-\d+$', label):
            lower, upper = map(float, label.split('-'))
            return (lower + upper) / 2
        # More than a certain age
        elif label.lower().startswith("more than"):
            age = float(re.findall(r'\d+', label)[0])
            return age + 5
        # Less than a certain age
        elif label.lower().startswith("less than"):
            age = float(re.findall(r'\d+', label)[0])
            # Just approximate a bit lower
            return max(age - 5, 0)
        else:
            print(f"Unrecognized label format: {label}")
            return np.nan

    def toggle_webcam(self):
        """Starts or stops the webcam based on the button state."""
        if self.webcam_btn.isChecked():
            self.clear_canvas()  # Clear canvas before starting webcam
            self.webcam_btn.setText("Stop Webcam")
            self.upload_btn.setEnabled(False)
            self.save_btn.setEnabled(False)

            self.thread = WebcamThread(self.pipe_age1, self.pipe_age2)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.age_estimated_signal.connect(self.update_predictions)
            self.thread.start()
        else:
            self.webcam_btn.setText("Start Webcam")
            self.upload_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            if self.thread:
                self.thread.stop()

    def update_image(self, cv_img):
        """Updates the image_label with a new OpenCV image."""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def update_predictions(self, age1, age2):
        """Updates the prediction labels with the estimated ages."""
        self.prediction_model1_label.setText(f"Model 1 - Age: {age1:.2f} years")
        self.prediction_model2_label.setText(f"Model 2 - Age: {age2:.2f} years")

        self.current_predictions = {
            'Model 1': {'Age': age1},
            'Model 2': {'Age': age2}
        }

    def convert_cv_qt(self, cv_img):
        """Converts an OpenCV image to QPixmap for display."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        return QPixmap.fromImage(p)

    def save_results(self):
        """Saves the current image with overlaid age estimations."""
        try:
            if self.current_image_path:
                # Open the image from disk
                img = Image.open(self.current_image_path).convert("RGBA")
            elif self.thread and self.thread.current_frame is not None:
                # Capture from webcam
                img = Image.fromarray(cv2.cvtColor(self.thread.current_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            else:
                QMessageBox.warning(self, "Save Warning", "No image available to save.")
                return

            draw = ImageDraw.Draw(img)

            # Define font
            try:
                font = ImageFont.truetype("arial.ttf", size=22)
            except IOError:
                font = ImageFont.load_default()

            # Overlay text with predictions
            text_lines = []
            for model, features in self.current_predictions.items():
                text_lines.append(f"{model}:  Age = {features['Age']:.2f} years")

            overlay_text = "\n".join(text_lines)

            # Create a semi-transparent overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            text_size = overlay_draw.multiline_textsize(overlay_text, font=font)
            padding = 20
            # Bottom-left corner position
            text_x = 10
            text_y = img.height - text_size[1] - padding

            # Draw rectangle for background
            rect_pos = [
                (text_x - 10, text_y - 10),
                (text_x + text_size[0] + 20, text_y + text_size[1] + 10)
            ]
            overlay_draw.rectangle(rect_pos, fill=(0, 0, 0, 128))

            # Draw text
            overlay_draw.multiline_text(
                (text_x, text_y), overlay_text,
                font=font, fill=(255, 255, 255, 255)
            )

            # Merge overlay
            img = Image.alpha_composite(img, overlay)

            # Save file
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)",
                options=QFileDialog.Options()
            )
            if save_path:
                img.convert("RGB").save(save_path)
                QMessageBox.information(self, "Image Saved",
                                        f"Image saved successfully at:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error",
                                 f"An error occurred while saving the image:\n{e}")

    def close_application(self):
        """Gracefully closes the application."""
        reply = QMessageBox.question(
            self,
            'Exit Application',
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.thread and self.thread.isRunning():
                self.thread.stop()
            QApplication.instance().quit()


class WebcamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    age_estimated_signal = pyqtSignal(float, float)

    def __init__(self, model_age1, model_age2):
        super().__init__()
        self._run_flag = True
        self.model_age1 = model_age1
        self.model_age2 = model_age2
        self.current_frame = None

    def run(self):
        """Runs the webcam thread to capture frames and estimate age in real-time."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(None, "Webcam Error", "Could not access the webcam.")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.change_pixmap_signal.emit(frame)

                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    # Model 1
                    age_preds1 = self.model_age1(pil_img)
                    age_label1 = age_preds1[0]['label']
                    age1 = self.parse_label(age_label1)

                    # Model 2
                    age_preds2 = self.model_age2(pil_img)
                    age_label2 = age_preds2[0]['label']
                    age2 = self.parse_label(age_label2)

                    self.age_estimated_signal.emit(age1, age2)
                except Exception as e:
                    print(f"Error during real-time age estimation: {e}")
                    self.age_estimated_signal.emit(np.nan, np.nan)

        cap.release()

    def stop(self):
        """Sets the run flag to False and waits for the thread to finish."""
        self._run_flag = False
        self.wait()

    def parse_label(self, label):
        """Parses the label from the model's prediction to extract numerical values."""
        if re.match(r'^\d+$', label):
            return float(label)
        elif re.match(r'^\d+-\d+$', label):
            lower, upper = map(float, label.split('-'))
            return (lower + upper) / 2
        elif label.lower().startswith("more than"):
            age = float(re.findall(r'\d+', label)[0])
            return age + 5
        elif label.lower().startswith("less than"):
            age = float(re.findall(r'\d+', label)[0])
            return max(age - 5, 0)
        else:
            print(f"Unrecognized label format: {label}")
            return np.nan


def main():
    """Main function to run the Age Estimation Application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AgeEstimationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
