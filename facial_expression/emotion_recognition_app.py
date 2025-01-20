# emotion_recognition_app.py

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QMessageBox, QProgressBar, QMenuBar, QAction, QGridLayout, QDialog, QTableWidget,
    QTableWidgetItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer
import cv2
from transformers import pipeline
import torch
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("About")
        self.setFixedSize(400, 300)
        layout = QVBoxLayout()

        about_text = """
        <h2>Facial Emotion Recognition Application</h2>
        <p><strong>Author:</strong> Muhammad Bilal</p>
        <p><strong>Supervisor:</strong> Prof. Dr. Amin Nait-Ali</p>
        <p><strong>Master of Biometrics and Intelligent Vision</strong></p>
        <p><strong>Department of Science and Technology, UPEC</strong></p>
        <br>
        <p>This application recognizes facial emotions using two advanced models. It was developed as part of the Master's program at UPEC.</p>
        """

        label = QLabel(about_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        self.setLayout(layout)


class EmotionRecognitionApp(QWidget):
    def __init__(self, model1_name="trpakov/vit-face-expression",
                 model2_name="motheecreator/vit-Facial-Expression-Recognition"):
        super().__init__()
        self.setWindowTitle("Facial Emotion Recognition")
        self.setGeometry(100, 100, 1400, 800)  # Increased width for better layout

        # Initialize models
        self.device = 0 if torch.cuda.is_available() else -1
        try:
            self.pipe_vit_face = pipeline(
                "image-classification",
                model=model1_name,
                device=self.device
            )
            self.pipe_mothee_vit = pipeline(
                "image-classification",
                model=model2_name,
                device=self.device
            )
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error",
                                 f"An error occurred while loading models:\n{str(e)}")
            sys.exit(1)

        # Define emotion labels and corresponding emojis
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_emojis = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😮',
            'neutral': '😐'
        }
        self.emotion_colors = {
            'angry': 'red',
            'disgust': 'darkgreen',
            'fear': 'purple',
            'happy': 'yellow',  # You mentioned yellow was "disgusting", but keeping for reference
            'sad': 'blue',
            'surprise': 'orange',
            'neutral': 'grey'
        }

        # Initialize emotion statistics
        self.emotion_stats = {emotion: 0 for emotion in self.emotion_labels}

        # Initialize UI components
        self.initUI()

    def initUI(self):
        # Create Menu Bar
        menu_bar = QMenuBar(self)
        help_menu = menu_bar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # Create Header Labels
        header_layout = QVBoxLayout()

        # App Name
        app_name_label = QLabel("Facial Emotion Recognition")
        app_name_label.setAlignment(Qt.AlignCenter)
        app_name_label.setFont(QFont('Arial', 28, QFont.Bold))
        header_layout.addWidget(app_name_label)

        # University and Department
        university_label = QLabel("Master of Biometrics and Intelligent Vision\n"
                                   "Department of Science and Technology, UPEC")
        university_label.setAlignment(Qt.AlignCenter)
        university_label.setFont(QFont('Arial', 14, QFont.StyleItalic))
        header_layout.addWidget(university_label)

        # Create buttons
        self.upload_btn = QPushButton("Upload Image")
        upload_icon_path = os.path.join('assets', 'upload_icon.png')
        if os.path.exists(upload_icon_path):
            self.upload_btn.setIcon(QIcon(upload_icon_path))
        self.upload_btn.setFixedSize(150, 50)
        self.upload_btn.clicked.connect(self.upload_image)

        self.webcam_btn = QPushButton("Start Webcam")
        webcam_icon_path = os.path.join('assets', 'webcam_icon.png')
        if os.path.exists(webcam_icon_path):
            self.webcam_btn.setIcon(QIcon(webcam_icon_path))
        self.webcam_btn.setFixedSize(150, 50)
        self.webcam_btn.clicked.connect(self.start_webcam)

        # Theme Toggle Button as a small icon in the upper right corner
        self.theme_btn = QPushButton()
        theme_icon_path = os.path.join('assets', 'theme_icon_light.png')  # Ensure you have theme_icon_light.png
        if os.path.exists(theme_icon_path):
            self.theme_btn.setIcon(QIcon(theme_icon_path))
        else:
            self.theme_btn.setText("🌙")  # Default to moon emoji if icon not found
        self.theme_btn.setFixedSize(40, 40)
        self.theme_btn.clicked.connect(self.toggle_theme)

        # Create labels to display images and results
        # Add a QLabel above the image to display the predictions
        self.result_label = QLabel("Predictions will appear here.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont('Arial', 16))
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
        """)

        self.image_label = QLabel("No Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(700, 700)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #000;
                border-radius: 10px;
                background-color: #fff;
            }
        """)

        # Initialize the results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['Emotion', 'Count', 'Percentage'])
        self.results_table.setRowCount(len(self.emotion_labels))
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #f0f0f0;
                font-size: 14px;
            }
            QHeaderView::section {
                background-color: #d0d0d0;
                font-weight: bold;
            }
        """)

        for i, emotion in enumerate(self.emotion_labels):
            # Combine emotion name with its emoji
            emotion_with_emoji = f"{emotion.capitalize()} {self.emotion_emojis.get(emotion, '')}"
            emotion_item = QTableWidgetItem(emotion_with_emoji)
            emotion_item.setTextAlignment(Qt.AlignCenter)
            count_item = QTableWidgetItem("0")
            count_item.setTextAlignment(Qt.AlignCenter)
            percentage_item = QTableWidgetItem("0%")
            percentage_item.setTextAlignment(Qt.AlignCenter)

            self.results_table.setItem(i, 0, emotion_item)
            self.results_table.setItem(i, 1, count_item)
            self.results_table.setItem(i, 2, percentage_item)

        # Initialize the matplotlib FigureCanvas
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Emotion Distribution")
        self.ax.set_xlabel("Emotions")
        self.ax.set_ylabel("Count")
        self.bar_plot = None
        self.update_graph()

        # Progress Bar (Hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Initialize timer for webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None  # Initialize webcam capture

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.webcam_btn)
        button_layout.addStretch()  # Push buttons to the left

        # Image and Result Layout
        image_result_layout = QHBoxLayout()
        image_and_label_layout = QVBoxLayout()
        image_and_label_layout.addWidget(self.result_label)  # Add the result label above the image
        image_and_label_layout.addWidget(self.image_label, stretch=3)
        image_result_layout.addLayout(image_and_label_layout, stretch=3)

        # Right side layout for results and graph
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<h3>Results:</h3>"))
        right_layout.addWidget(self.results_table)
        right_layout.addWidget(QLabel("<h3>Emotion Statistics:</h3>"))
        right_layout.addWidget(self.canvas)
        right_layout.addStretch()
        image_result_layout.addLayout(right_layout, stretch=1)

        # Combine all layouts
        main_layout = QVBoxLayout()
        main_layout.setMenuBar(menu_bar)
        main_layout.addLayout(header_layout)
        main_layout.addLayout(image_result_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)

        # Create a top layout for theme toggle
        top_layout = QHBoxLayout()
        top_layout.addLayout(main_layout)
        top_layout.addWidget(self.theme_btn, alignment=Qt.AlignTop | Qt.AlignRight)

        self.setLayout(top_layout)

        # Load default theme (light)
        self.load_theme('light')

        # Initialize theme state
        self.dark_mode = False

    def load_theme(self, theme_name):
        """
        Loads a QSS stylesheet based on the theme name.
        """
        theme_file = os.path.join('themes', f"{theme_name}.qss")
        if os.path.exists(theme_file):
            with open(theme_file, 'r') as f:
                self.setStyleSheet(f.read())
        else:
            QMessageBox.warning(self, "Theme File Missing",
                                f"The theme file '{theme_file}' does not exist.")

    def show_about_dialog(self):
        dialog = AboutDialog()
        dialog.exec_()

    def toggle_theme(self):
        if self.dark_mode:
            # Switch to Light Theme
            self.load_theme('light')
            # Update theme icon to moon
            theme_icon_path = os.path.join('assets', 'theme_icon_light.png')  # Ensure you have theme_icon_light.png
            if os.path.exists(theme_icon_path):
                self.theme_btn.setIcon(QIcon(theme_icon_path))
                self.theme_btn.setText("")
            else:
                self.theme_btn.setText("🌙")  # Fallback emoji
        else:
            # Switch to Dark Theme
            self.load_theme('dark')
            # Update theme icon to sun
            theme_icon_path = os.path.join('assets', 'theme_icon_dark.png')  # Ensure you have theme_icon_dark.png
            if os.path.exists(theme_icon_path):
                self.theme_btn.setIcon(QIcon(theme_icon_path))
                self.theme_btn.setText("")
            else:
                self.theme_btn.setText("☀️")  # Fallback emoji

        self.dark_mode = not self.dark_mode

    def upload_image(self):
        # Open file dialog
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)", options=options
        )
        if file_name:
            try:
                # Display the image
                pixmap = QPixmap(file_name)
                if pixmap.isNull():
                    QMessageBox.warning(self, "Invalid Image", "The selected file is not a valid image.")
                    return
                pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)

                # Open and preprocess the image
                image = Image.open(file_name).convert('RGB')
                image_resized = image.resize((224, 224))

                # Show progress bar
                self.progress_bar.setVisible(True)

                # Make predictions
                preds_vit_face = self.pipe_vit_face(image_resized)
                preds_mothee_vit = self.pipe_mothee_vit(image_resized)

                # Hide progress bar
                self.progress_bar.setVisible(False)

                # Extract labels
                label_vit_face = preds_vit_face[0]['label'].lower()
                label_mothee_vit = preds_mothee_vit[0]['label'].lower()

                # Display predictions in the result_label
                self.result_label.setText(
                    f"Model 1 Prediction: {label_vit_face.capitalize()} {self.emotion_emojis.get(label_vit_face, '')}\n"
                    f"Model 2 Prediction: {label_mothee_vit.capitalize()} {self.emotion_emojis.get(label_mothee_vit, '')}"
                )

                # Update emotion statistics based on a single consolidated prediction
                # For simplicity, we'll prioritize Model 1's prediction
                # Alternatively, you can implement a consensus or averaging mechanism
                primary_label = label_vit_face  # Using Model 1's prediction
                if primary_label in self.emotion_stats:
                    self.emotion_stats[primary_label] += 1

                # Update results table
                self.update_results_table()

                # Update graph
                self.update_graph()

            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"An error occurred while processing the image:\n{str(e)}")

    def start_webcam(self):
        if self.timer.isActive():
            self.timer.stop()
            self.webcam_btn.setText("Start Webcam")
            if self.cap:
                self.cap.release()
            self.image_label.setText("No Image")
            self.result_label.setText("Predictions will appear here.")
            self.results_table.setRowCount(len(self.emotion_labels))  # Reset table
            for i, emotion in enumerate(self.emotion_labels):
                self.results_table.setItem(i, 0, QTableWidgetItem(f"{emotion.capitalize()} {self.emotion_emojis.get(emotion, '')}"))
                self.results_table.setItem(i, 1, QTableWidgetItem("0"))
                self.results_table.setItem(i, 2, QTableWidgetItem("0%"))
            self.emotion_stats = {emotion: 0 for emotion in self.emotion_labels}  # Reset stats
            self.update_graph()
            return
        else:
            # Start the webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Webcam Error", "Cannot access webcam.")
                return

            # Inform the user how to exit
            QMessageBox.information(self, "Webcam",
                                    "Webcam started. Click the 'Start Webcam' button again to stop.")

            # Start timer to capture frames
            self.timer.start(2000)  # Capture frame every 2 seconds
            self.webcam_btn.setText("Stop Webcam")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image_resized = pil_image.resize((224, 224))

            try:
                # Make predictions
                preds_vit_face = self.pipe_vit_face(pil_image_resized)
                preds_mothee_vit = self.pipe_mothee_vit(pil_image_resized)

                label_vit_face = preds_vit_face[0]['label'].lower()
                label_mothee_vit = preds_mothee_vit[0]['label'].lower()
            except Exception as e:
                label_vit_face = "error"
                label_mothee_vit = "error"

            # Display predictions in the result_label
            self.result_label.setText(
                f"Model 1 Prediction: {label_vit_face.capitalize()} {self.emotion_emojis.get(label_vit_face, '')}\n"
                f"Model 2 Prediction: {label_mothee_vit.capitalize()} {self.emotion_emojis.get(label_mothee_vit, '')}"
            )

            # Update emotion statistics based on a single consolidated prediction
            # For simplicity, we'll prioritize Model 1's prediction
            primary_label = label_vit_face  # Using Model 1's prediction
            if primary_label in self.emotion_stats:
                self.emotion_stats[primary_label] += 1

            # Update results table
            self.update_results_table()

            # Update graph
            self.update_graph()

            # Display predictions on the frame without emojis
            cv2.putText(frame, f"Model 1: {label_vit_face.capitalize()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Model 2: {label_mothee_vit.capitalize()}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert frame to QImage and display in QLabel
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.image_label.width(),
                                                     self.image_label.height(),
                                                     Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def update_emotion_stats(self, emotion):
        emotion = emotion.lower()
        if emotion in self.emotion_stats:
            self.emotion_stats[emotion] += 1

    def update_results_table(self):
        total = sum(self.emotion_stats.values())
        for i, emotion in enumerate(self.emotion_labels):
            count = self.emotion_stats[emotion]
            percentage = f"{(count / total * 100):.1f}%" if total > 0 else "0%"
            self.results_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.results_table.setItem(i, 2, QTableWidgetItem(percentage))

    def update_graph(self):
        self.ax.clear()
        emotions = list(self.emotion_stats.keys())
        counts = list(self.emotion_stats.values())

        # Append emojis to emotions for the graph
        emotions_with_emojis = [f"{emotion.capitalize()} {self.emotion_emojis.get(emotion, '')}" for emotion in emotions]

        bars = self.ax.bar(emotions_with_emojis, counts, color='skyblue')
        self.ax.set_title("Emotion Distribution")
        self.ax.set_xlabel("Emotions")
        self.ax.set_ylabel("Count")
        self.ax.set_ylim(0, max(counts) + 1 if counts else 1)

        # Add counts on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}',
                        ha='center', va='bottom')

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    window = EmotionRecognitionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
