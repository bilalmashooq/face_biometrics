import sys
import os
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QComboBox, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error
)

# -------------- Pose Estimation Logic --------------

# Paths to models (Update these paths as per your environment)
MODEL_A_PATH = r"C:\Users\muham\PycharmProjects\pythonProject\face_pose\models\model.pkl"      # 7-landmark model
MODEL_B_PATH = r"C:\Users\muham\PycharmProjects\pythonProject\face_pose\models\SVR_model.sav"          # 468-landmark model

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh_static = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

# For Model A (7 Key Landmarks):
LANDMARKS_A = {
    'NOSE': 1,
    'FOREHEAD': 10,
    'LEFT_EYE': 33,
    'MOUTH_LEFT': 61,
    'CHIN': 199,
    'RIGHT_EYE': 263,
    'MOUTH_RIGHT': 291
}

# Columns for Model A
COLS_A = []
for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
    for dim in ('x', 'y'):
        COLS_A.append(pos + dim)

# Load Models
def load_models():
    try:
        with open(MODEL_A_PATH, 'rb') as fA:
            modelA = pickle.load(fA)
        print("Model A loaded successfully.")
    except Exception as e:
        print(f"Error loading Model A: {e}")
        modelA = None

    try:
        modelB = pickle.load(open(MODEL_B_PATH, 'rb'))
        print("Model B loaded successfully.")
    except Exception as e:
        print(f"Error loading Model B: {e}")
        modelB = None

    return modelA, modelB

modelA, modelB = load_models()

# Feature Extraction for Model A
def extract_features_modelA(img_rgb, face_mesh_processor):
    result = face_mesh_processor.process(img_rgb)
    if not result.multi_face_landmarks:
        return None

    face_landmarks = result.multi_face_landmarks[0]
    # Collect 14 points (x,y) from 7 key landmarks
    features = []
    for idx_name in LANDMARKS_A.values():
        lm = face_landmarks.landmark[idx_name]
        features.append(lm.x)
        features.append(lm.y)

    # Convert to DataFrame for normalization
    df = pd.DataFrame([features], columns=COLS_A)
    df_normalized = normalize_modelA(df)
    return df_normalized.values  # shape (1,14)

def normalize_modelA(poses_df):
    df = poses_df.copy()
    for dim in ['x', 'y']:
        # Center around nose
        for feat in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim,
                     'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            df[feat] = poses_df[feat] - poses_df['nose_'+dim]

        # Scale
        diff = df['mouth_right_'+dim] - df['left_eye_'+dim]
        for feat in ['forehead_'+dim, 'nose_'+dim, 'mouth_left_'+dim,
                     'mouth_right_'+dim, 'left_eye_'+dim, 'chin_'+dim, 'right_eye_'+dim]:
            df[feat] = df[feat] / diff
    return df

# Feature Extraction for Model B
def extract_features_modelB(img_rgb, face_mesh_processor, width, height):
    result = face_mesh_processor.process(img_rgb)
    if not result.multi_face_landmarks:
        return None

    face = result.multi_face_landmarks[0]
    x_val = np.array([lm.x * width for lm in face.landmark])
    y_val = np.array([lm.y * height for lm in face.landmark])

    # Center around the nose (landmark #1)
    x_center = x_val[1]
    y_center = y_val[1]
    x_val -= x_center
    y_val -= y_center

    # Normalize by max absolute val
    x_max = np.max(np.abs(x_val)) if np.max(np.abs(x_val)) != 0 else 1
    y_max = np.max(np.abs(y_val)) if np.max(np.abs(y_val)) != 0 else 1
    x_val /= x_max
    y_val /= y_max

    return np.concatenate([x_val, y_val]).reshape(1, -1)  # shape (1, 936)

# Prediction Function
def predict_pose(image, selected_model):
    img_bgr = image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_bgr.shape

    # Process image with Mediapipe
    result = face_mesh_static.process(img_rgb)

    if not result.multi_face_landmarks:
        return None, None, None, "No face detected.", None

    face_landmarks = result.multi_face_landmarks[0]  # Extract the first face's landmarks

    if selected_model == 'Model A' and modelA is not None:
        featA = extract_features_modelA(img_rgb, face_mesh_static)
        if featA is not None:
            try:
                pitchA, yawA, rollA = modelA.predict(featA)[0]
                return pitchA, yawA, rollA, None, face_landmarks
            except Exception as e:
                print(f"Model A Prediction Error: {e}")
                return None, None, None, "Model A Prediction Error.", None
        else:
            return None, None, None, "No face detected for Model A.", None

    elif selected_model == 'Model B' and modelB is not None:
        featB = extract_features_modelB(img_rgb, face_mesh_static, w, h)
        if featB is not None:
            try:
                pitchB, yawB, rollB = modelB.predict(featB)[0]
                return pitchB, yawB, rollB, None, face_landmarks
            except Exception as e:
                print(f"Model B Prediction Error: {e}")
                return None, None, None, "Model B Prediction Error.", None
        else:
            return None, None, None, "No face detected for Model B.", None
    else:
        return None, None, None, "Selected model not loaded.", None

# Function to draw axes on the image
def draw_axes(image, yaw, pitch, roll, face_landmarks, image_width, image_height):
    """
    Draws 3D axes on the image based on the yaw, pitch, and roll angles.
    """
    # Define axis length
    axis_length = 50

    # Calculate rotation matrices
    yaw_rad = yaw
    pitch_rad = pitch
    roll_rad = roll

    # Rotation matrices around the x, y, z axes
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    R_y = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])

    R_z = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Define the axes in 3D space
    axes = np.float32([
        [axis_length, 0, 0],  # X axis (Red)
        [0, axis_length, 0],  # Y axis (Green)
        [0, 0, axis_length]   # Z axis (Blue)
    ])

    # Get nose position from landmarks
    nose_landmark = face_landmarks.landmark[1]  # Nose is landmark #1
    nose_x = int(nose_landmark.x * image_width)
    nose_y = int(nose_landmark.y * image_height)

    # Project the 3D axes onto the 2D image
    projected_axes = []
    for axis in axes:
        rotated_axis = R @ axis
        end_point = (int(nose_x + rotated_axis[0]),
                     int(nose_y - rotated_axis[1]))  # y-axis is inverted in image coordinates
        projected_axes.append(end_point)

    # Draw the axes on the image
    cv2.line(image, (nose_x, nose_y), projected_axes[0], (0, 0, 255), 2)  # X axis in Red
    cv2.line(image, (nose_x, nose_y), projected_axes[1], (0, 255, 0), 2)  # Y axis in Green
    cv2.line(image, (nose_x, nose_y), projected_axes[2], (255, 0, 0), 2)  # Z axis in Blue

    return image

# Function to interpret angles
def interpret_angles(yaw, pitch, roll):
    """
    Provides textual feedback based on the yaw, pitch, and roll angles.
    """
    directions = []

    # Yaw interpretation
    if yaw < -0.3:
        directions.append("Looking Left")
    elif yaw > 0.3:
        directions.append("Looking Right")

    # Pitch interpretation
    if pitch < -0.3:
        directions.append("Looking Upward")
    elif pitch > 0.3:
        directions.append("Looking Downward")

    # Roll interpretation
    if roll < -0.3:
        directions.append("Head Tilted Left")
    elif roll > 0.3:
        directions.append("Head Tilted Right")

    if not directions:
        directions.append("Looking Forward")

    return ', '.join(directions)

# -------------- GUI Components --------------

class PoseEstimationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Pose Estimation")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()
        self.selected_model = 'Model A'  # Default selection
        self.webcam_active = False  # To track webcam state

    def init_ui(self):
        # Main Layout
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Face Pose Estimation")
        title_font = QFont('Arial', 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2E86C1;")
        main_layout.addWidget(title_label)

        # Horizontal layout for buttons and display
        h_layout = QHBoxLayout()

        # Left Panel - Buttons
        button_panel = QVBoxLayout()
        button_panel.setAlignment(Qt.AlignTop)
        button_panel.setSpacing(20)

        # Upload Image Button
        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_image_btn.setStyleSheet(self.button_style())
        self.upload_image_btn.clicked.connect(self.upload_image)
        button_panel.addWidget(self.upload_image_btn)

        # Upload Video Button
        self.upload_video_btn = QPushButton("Upload Video")
        self.upload_video_btn.setStyleSheet(self.button_style())
        self.upload_video_btn.clicked.connect(self.upload_video)
        button_panel.addWidget(self.upload_video_btn)

        # Webcam Start Button
        self.webcam_start_btn = QPushButton("Start Webcam")
        self.webcam_start_btn.setStyleSheet(self.button_style())
        self.webcam_start_btn.clicked.connect(self.start_webcam)
        button_panel.addWidget(self.webcam_start_btn)

        # Webcam Stop Button
        self.webcam_stop_btn = QPushButton("Stop Webcam")
        self.webcam_stop_btn.setStyleSheet(self.button_style())
        self.webcam_stop_btn.clicked.connect(self.stop_webcam)
        self.webcam_stop_btn.setEnabled(False)  # Initially disabled
        button_panel.addWidget(self.webcam_stop_btn)

        # Model Selection ComboBox
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Model A', 'Model B'])
        self.model_combo.setStyleSheet(self.combo_style())
        self.model_combo.currentTextChanged.connect(self.model_selection_changed)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        button_panel.addWidget(model_group)

        # Exit Button
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setStyleSheet(self.button_style())
        self.exit_btn.clicked.connect(self.close_application)
        button_panel.addWidget(self.exit_btn)

        # Spacer
        button_panel.addStretch()

        # Add button panel to left layout
        h_layout.addLayout(button_panel, 1)

        # Right Panel - Display and Scores
        display_panel = QVBoxLayout()

        # Image/Video Display
        self.image_label = QLabel("Image/Video will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color : #D5DBDB; border: 2px solid #2E86C1; }")
        self.image_label.setFixedSize(800, 600)
        display_panel.addWidget(self.image_label)

        # Scores Display
        scores_group = QGroupBox("Pose Scores")
        scores_layout = QGridLayout()
        scores_group.setLayout(scores_layout)
        scores_group.setStyleSheet("QGroupBox { font-weight: bold; color: #1B4F72; }")

        # Yaw
        yaw_label = QLabel("Yaw:")
        self.yaw_value = QLabel("N/A")
        self.yaw_value.setStyleSheet("color: #CB4335; font-size: 16px;")
        scores_layout.addWidget(yaw_label, 0, 0)
        scores_layout.addWidget(self.yaw_value, 0, 1)

        # Pitch
        pitch_label = QLabel("Pitch:")
        self.pitch_value = QLabel("N/A")
        self.pitch_value.setStyleSheet("color: #28B463; font-size: 16px;")
        scores_layout.addWidget(pitch_label, 1, 0)
        scores_layout.addWidget(self.pitch_value, 1, 1)

        # Roll
        roll_label = QLabel("Roll:")
        self.roll_value = QLabel("N/A")
        self.roll_value.setStyleSheet("color: #1F618D; font-size: 16px;")
        scores_layout.addWidget(roll_label, 2, 0)
        scores_layout.addWidget(self.roll_value, 2, 1)

        # Direction Feedback
        direction_label = QLabel("Direction:")
        self.direction_value = QLabel("N/A")
        self.direction_value.setStyleSheet("color: #7D3C98; font-size: 16px;")
        scores_layout.addWidget(direction_label, 3, 0)
        scores_layout.addWidget(self.direction_value, 3, 1)

        display_panel.addWidget(scores_group)

        # Add display panel to right layout
        h_layout.addLayout(display_panel, 3)

        # Add horizontal layout to main layout
        main_layout.addLayout(h_layout)

        # Set main layout
        self.setLayout(main_layout)

        # Initialize Threads
        self.webcam_thread = None

    def button_style(self):
        return """
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #95A5A6;
            }
        """

    def combo_style(self):
        return """
            QComboBox {
                background-color: #ECF0F1;
                padding: 5px;
                font-size: 16px;
            }
            QComboBox::drop-down {
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
        """

    def model_selection_changed(self, text):
        self.selected_model = text
        print(f"Selected Model: {self.selected_model}")

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "Error", "Failed to load the image.")
                return
            pitch, yaw, roll, error, face_landmarks = predict_pose(image, self.selected_model)
            if error:
                QMessageBox.warning(self, "Error", error)
                return
            annotated_image = self.annotate_image(image, pitch, yaw, roll, face_landmarks)
            self.display_image(annotated_image)
            self.update_scores(pitch, yaw, roll)

    def upload_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                                   "Videos (*.mp4 *.avi *.mov *.mkv)", options=options)
        if file_path:
            self.video_path = file_path
            self.video_thread = VideoThread(file_path, self.selected_model, modelA, modelB)
            self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
            self.video_thread.update_scores_signal.connect(self.update_scores)
            self.video_thread.start()
            QMessageBox.information(self, "Info", "Video processing started.")

    def start_webcam(self):
        if not self.webcam_active:
            self.webcam_thread = WebcamThread(self.selected_model, modelA, modelB)
            self.webcam_thread.change_pixmap_signal.connect(self.update_video_frame)
            self.webcam_thread.update_scores_signal.connect(self.update_scores)
            self.webcam_thread.start()
            self.webcam_active = True
            self.webcam_start_btn.setEnabled(False)
            self.webcam_stop_btn.setEnabled(True)
            print("Webcam started.")

    def stop_webcam(self):
        if self.webcam_active:
            self.webcam_thread.stop()
            self.webcam_active = False
            self.webcam_start_btn.setEnabled(True)
            self.webcam_stop_btn.setEnabled(False)
            print("Webcam stopped.")

    def close_application(self):
        reply = QMessageBox.question(self, 'Exit',
                                     "Are you sure you want to exit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stop_webcam()  # Ensure webcam is stopped
            QApplication.instance().quit()

    def annotate_image(self, image, pitch, yaw, roll, face_landmarks):
        # Draw axes
        annotated_image = image.copy()
        if face_landmarks:
            annotated_image = draw_axes(annotated_image, yaw, pitch, roll, face_landmarks, image.shape[1], image.shape[0])

        # Add pose information text
        text = f"Yaw: {yaw:.2f} rad\nPitch: {pitch:.2f} rad\nRoll: {roll:.2f} rad"
        cv2.putText(annotated_image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_image

    def display_image(self, image):
        # Resize image to fit the label while maintaining aspect ratio
        height, width, channels = image.shape
        aspect_ratio = width / height
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        if aspect_ratio > 1:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert image to QPixmap
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def update_scores(self, pitch, yaw, roll):
        if pitch is not None and yaw is not None and roll is not None:
            self.pitch_value.setText(f"{pitch:.2f} rad")
            self.yaw_value.setText(f"{yaw:.2f} rad")
            self.roll_value.setText(f"{roll:.2f} rad")
            direction = interpret_angles(yaw, pitch, roll)
            self.direction_value.setText(direction)
        else:
            self.pitch_value.setText("N/A")
            self.yaw_value.setText("N/A")
            self.roll_value.setText("N/A")
            self.direction_value.setText("N/A")

    def update_video_frame(self, cv_img, pitch, yaw, roll):
        if cv_img is not None:
            # Annotate image
            annotated_image = annotate_frame_with_pose(cv_img, pitch, yaw, roll, None)  # Face landmarks already drawn in thread
            self.display_image(annotated_image)
            self.update_scores(pitch, yaw, roll)

    def closeEvent(self, event):
        # Handle thread termination on closing the app
        self.stop_webcam()
        try:
            if self.video_thread.isRunning():
                self.video_thread.stop()
        except:
            pass
        event.accept()

# -------------- Video Processing Threads --------------

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, float, float, float)
    update_scores_signal = pyqtSignal(float, float, float)

    def __init__(self, video_path, selected_model, modelA, modelB):
        super().__init__()
        self._run_flag = True
        self.video_path = video_path
        self.selected_model = selected_model
        self.modelA = modelA
        self.modelB = modelB

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                pitch, yaw, roll, error, face_landmarks = predict_pose(frame, self.selected_model)
                if error:
                    # Optionally, display error on the frame
                    cv2.putText(frame, error, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.change_pixmap_signal.emit(frame, 0, 0, 0)
                else:
                    # Annotate frame
                    annotated_frame = annotate_frame_with_pose(frame, pitch, yaw, roll, face_landmarks)
                    self.change_pixmap_signal.emit(annotated_frame, pitch, yaw, roll)
                    self.update_scores_signal.emit(pitch, yaw, roll)
            else:
                break
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class WebcamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, float, float, float)
    update_scores_signal = pyqtSignal(float, float, float)

    def __init__(self, selected_model, modelA, modelB):
        super().__init__()
        self._run_flag = True
        self.selected_model = selected_model
        self.modelA = modelA
        self.modelB = modelB

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                pitch, yaw, roll, error, face_landmarks = predict_pose(frame, self.selected_model)
                if error:
                    # Optionally, display error on the frame
                    cv2.putText(frame, error, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.change_pixmap_signal.emit(frame, 0, 0, 0)
                else:
                    # Annotate frame
                    annotated_frame = annotate_frame_with_pose(frame, pitch, yaw, roll, face_landmarks)
                    self.change_pixmap_signal.emit(annotated_frame, pitch, yaw, roll)
                    self.update_scores_signal.emit(pitch, yaw, roll)
            else:
                break
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# Helper function to annotate frame
def annotate_frame_with_pose(frame, pitch, yaw, roll, face_landmarks):
    annotated_image = frame.copy()
    if face_landmarks:
        annotated_image = draw_axes(annotated_image, yaw, pitch, roll, face_landmarks, frame.shape[1], frame.shape[0])
    # Add pose information text
    text = f"Yaw: {yaw:.2f} rad\nPitch: {pitch:.2f} rad\nRoll: {roll:.2f} rad"
    y0, dy = 30, 30
    for i, line in enumerate(text.split('\n')):
        cv2.putText(annotated_image, line, (10, y0 + i*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return annotated_image

# -------------- Main Execution --------------

def main():
    app = QApplication(sys.argv)
    window = PoseEstimationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
