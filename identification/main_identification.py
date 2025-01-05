# -*- coding: utf-8 -*-
"""
Author: Muhammad Bilal
Supervisor: Prof. Dr. Amin Nait-Ali
Master of Biometrics and Intelligent Vision
Department of Science and Technology, UPEC
University Paris Est Creteil
Date created: 21/12/2024
Last Date modified: 25/12/2024
Description: Main code for face verification using deep learning with a PyQt5 GUI.

This code:
1) Builds face embeddings for your dataset at startup ("before click on any button").
2) Provides a Registration tab (Upload, Register, View Registered) with the original pushButton_6, pushButton_7, pushButton_8.
3) Provides a Recognition tab with pushButton_3, pushButton_4, pushButton_9.
4) Speeds up recognition using DeepFace.find (cached embeddings).
5) Shows how to handle multiple faces (list of DataFrames) from DeepFace.find.
"""

import os
import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout,
    QInputDialog, QLineEdit
)
from PyQt5.QtCore import QDir

# DeepFace
from deepface import DeepFace

# Import your generated UI from the .ui file
# Make sure identification_GUI.py matches your .ui file exactly.
from identification_GUI import Ui_MainWindow


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ------------------- #
        #  Basic GUI Setups   #
        # ------------------- #
        self.image1 = None              # Used for "Recognition" tab
        self.image1_path = None         # Path to the loaded Recognition image
        self.registration_image = None  # Used for "Registration" tab
        self.fig = None
        self.canvas = None

        # Path to your dataset folder
        self.dataset_path = r"C:\Users\muham\PycharmProjects\pythonProject\Identification\lfw_subset"
        # Adjust this path to point to your dataset

        # Connect theme comboBox
        self.comboBox.currentIndexChanged.connect(self.ThemeChanged)
        self.ThemeChanged(0)               # Set default theme
        self.tabWidget.setCurrentIndex(0)  # Start on the "Main" tab

        # ---------------------------------------------------------------------
        # 1) MAIN TAB (index=0) Buttons
        # ---------------------------------------------------------------------
        self.pushButton.clicked.connect(self.registeration_tab)  # "Registeration"
        self.pushButton_2.clicked.connect(self.recognition_tab)  # "Recognition"

        # ---------------------------------------------------------------------
        # 2) RECOGNITION TAB (index=2) Buttons
        # ---------------------------------------------------------------------
        self.pushButton_3.clicked.connect(self.load_image)  # "Upload Image" (Recognition)
        self.pushButton_4.clicked.connect(lambda: self.recognize_face_find(model_name='VGG-Face'))
        self.pushButton_9.clicked.connect(lambda: self.recognize_face_find(model_name='Facenet'))

        self.pushButton_5.clicked.connect(self.reset_all_displays)

        # ---------------------------------------------------------------------
        # 3) REGISTRATION TAB (index=1) Buttons (original names: 6, 7, 8)
        # ---------------------------------------------------------------------
        self.pushButton_6.clicked.connect(self.registration_load_image)   # "Upload"
        self.pushButton_7.clicked.connect(self.registration_save_image)   # "Registeration"
        self.pushButton_8.clicked.connect(self.registration_view_dataset) # "View Registered"

        # ---------------------------------------------------------------------
        # 4) BUILD EMBEDDINGS AT STARTUP
        # ---------------------------------------------------------------------
        # This step will generate a .deepface folder in each subdirectory
        # of self.dataset_path. Future searches will be much faster.
        try:
            # Build embeddings for VGG-Face model
            self.build_embeddings(model_name='VGG-Face')
            # Build embeddings for Facenet model (optional, if you plan to use it)
            self.build_embeddings(model_name='Facenet')
        except Exception as e:
            print(f"[Startup] Could not build embeddings: {e}")

    # -------------------------------------------------------------------------
    #                            PRE-BUILD EMBEDDINGS
    # -------------------------------------------------------------------------
    def build_embeddings(self, model_name='VGG-Face'):
        """
        Build or update the face embeddings cache for the entire dataset
        by calling DeepFace.find with a dummy query. This will scan the dataset
        and store embeddings in .deepface folders.

        If your dataset is large, this can take time initially, but speeds up
        subsequent recognition calls significantly.
        """
        # Create a dummy image (black square 224x224) as a temporary file
        dummy_path = "temp_dummy.jpg"
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(dummy_path, dummy_img)

        print(f"[Build Embeddings] Model: {model_name} => Building dataset embeddings ...")

        # This call triggers embedding creation for your entire dataset
        # (If they aren't already cached in .deepface)
        DeepFace.find(
            img_path=dummy_path,
            db_path=self.dataset_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='skip'
        )

        print(f"[Build Embeddings] Completed for model: {model_name}\n")

        # Clean up dummy file (optional)
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

    # -------------------------------------------------------------------------
    #                               TAB SWITCHING
    # -------------------------------------------------------------------------
    def registeration_tab(self):
        """Switch to the registration tab (index=1)."""
        self.tabWidget.setCurrentIndex(1)

    def recognition_tab(self):
        """Switch to the recognition tab (index=2)."""
        self.tabWidget.setCurrentIndex(2)

    # -------------------------------------------------------------------------
    #                                THEME HANDLING
    # -------------------------------------------------------------------------
    def ThemeChanged(self, index):
        """Apply the selected theme."""
        self.SetTheme(index)

    def SetTheme(self, index):
        """Set the application theme based on comboBox."""
        theme_options = ['Default', 'Dark', 'Light']
        selected_theme = theme_options[index]
        file = None

        if selected_theme == 'Dark':
            file = 'QtStylesheet/dark.qss'
        elif selected_theme == 'Light':
            file = 'QtStylesheet/light.qss'

        if file is not None and os.path.exists(file):
            with open(file, 'r') as f:
                self.setStyleSheet(f.read())
        else:
            self.setStyleSheet('')

    # -------------------------------------------------------------------------
    #                             REGISTRATION TAB
    # -------------------------------------------------------------------------
    def registration_load_image(self):
        """
        Load an image for registration (pushButton_6),
        then display it in groupBox_2.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Select Image File', QDir.currentPath(),
            'Image Files (*.png *.jpg *.jpeg *.bmp)'
        )
        if not filename:
            return

        try:
            img = cv2.imread(filename)
            if img is None:
                raise Exception("Could not load the image.")
            # Convert to RGB for matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.registration_image = img
            self.update_canvas_in_groupbox(
                groupbox=self.groupBox_2,
                rgb_image=self.registration_image,
                title="Registration Image"
            )

            QMessageBox.information(
                self, 'Image Loaded',
                f'Image loaded for registration:\n{filename}'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error loading image file: {e}")

    def registration_save_image(self):
        """
        'Register' the loaded image (pushButton_7) by copying it
        into the dataset folder under a user-specified subfolder (person name).
        """
        if self.registration_image is None:
            QMessageBox.warning(self, 'Error', "No image loaded for registration.")
            return

        # Ask the user for a person's name
        person_name, ok_pressed = QInputDialog.getText(
            self, "Registration", "Enter person's name:", QLineEdit.Normal, ""
        )
        if not ok_pressed or not person_name.strip():
            QMessageBox.warning(self, "Registration", "Name cannot be empty.")
            return
        person_name = person_name.strip()

        # Create (if needed) the folder for this person
        person_folder = os.path.join(self.dataset_path, person_name)
        os.makedirs(person_folder, exist_ok=True)

        # Build a filename, e.g. "registered_image.jpg" or a timestamp
        new_filename = os.path.join(person_folder, "registered_image.jpg")

        # Convert back to BGR for saving
        img_bgr = cv2.cvtColor(self.registration_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_filename, img_bgr)

        QMessageBox.information(
            self, "Registration",
            f"Image has been saved in:\n{person_folder}\n\n"
            "Embeddings will be built/updated next time you run or do a recognition."
        )

    def registration_view_dataset(self):
        """
        Show a list of person folders, let user select one,
        then display the first image found in that person's folder.
        """
        # 1) Collect all persons (subfolders) in dataset_path
        persons = []
        for name in sorted(os.listdir(self.dataset_path)):
            full_path = os.path.join(self.dataset_path, name)
            if os.path.isdir(full_path):
                persons.append(name)

        if not persons:
            QMessageBox.warning(self, "View Registered", "No registered persons found.")
            return

        # 2) Ask user to pick one person from the list
        person, ok = QInputDialog.getItem(
            self, "Registered Persons", "Select Person:", persons, 0, False
        )
        if not ok:
            return  # user canceled

        # 3) Show the first image from that person's folder
        person_folder = os.path.join(self.dataset_path, person)
        found_image_path = None
        for f in os.listdir(person_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                found_image_path = os.path.join(person_folder, f)
                break

        if not found_image_path:
            QMessageBox.warning(self, "View Registered", f"No images found for {person}.")
            return

        try:
            img = mpimg.imread(found_image_path)  # Usually returns an RGB array
            self.update_canvas_in_groupbox(
                groupbox=self.groupBox_2,
                rgb_image=img,
                title=f"Person: {person}"
            )
            QMessageBox.information(
                self, "View Registered",
                f"Showing: {found_image_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")

    # -------------------------------------------------------------------------
    #                          RECOGNITION TAB (index=2)
    # -------------------------------------------------------------------------
    def load_image(self):
        """
        Load an image for the Recognition tab (pushButton_3) and display it
        in groupBox.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Select Image File', QDir.currentPath(),
            'Image Files (*.png *.jpg *.bmp)'
        )
        if not filename:
            return

        try:
            self.image1_path = filename  # Keep track of its path for DeepFace.find
            self.image1 = cv2.imread(filename)
            if self.image1 is None:
                raise Exception("Could not load the image.")
            self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)

            self.update_canvas_in_groupbox(
                groupbox=self.groupBox,
                rgb_image=self.image1,
                title="Loaded Image"
            )

            QMessageBox.information(
                self, 'File Loaded',
                f'File Path: {filename}\nFile loaded successfully.'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error loading image file: {e}")

    def recognize_face_find(self, model_name='VGG-Face'):
        """
        Speed up recognition using DeepFace.find, which uses cached embeddings
        in .deepface subfolders. We handle the possibility of multiple faces
        (list of DataFrames).
        """
        if not hasattr(self, 'image1_path') or not self.image1_path:
            QMessageBox.warning(self, 'Error', "Please load an image first.")
            return

        try:
            # 1) Use DeepFace.find to retrieve a DataFrame or a list of DataFrames
            result = DeepFace.find(
                img_path=self.image1_path,
                db_path=self.dataset_path,
                model_name=model_name,
                enforce_detection=False
            )
            # If multiple faces are detected in the query, result is a list of DataFrames
            # If single face, result is a single DataFrame

            # 2) Convert result -> first DataFrame if it's a list
            if isinstance(result, list):
                if len(result) == 0:
                    QMessageBox.warning(self, "No Face", "No faces were detected in the query.")
                    return
                df = result[0]  # Take the first face
            else:
                df = result  # It's already a DataFrame

            # Check if empty
            if df.empty:
                QMessageBox.warning(self, 'No Match', "No match found in the dataset.")
                return

            # 3) The top row (index=0) is the best match (lowest distance)
            best_match_path = df.iloc[0]['identity']
            distance = df.iloc[0]['distance']
            similarity_score = 1.0 - distance

            # 4) Extract person's name from path
            # E.g.: best_match_path = ".../lfw_subset/Maria_Sharapova/img_001.jpg"
            person_name = os.path.basename(os.path.dirname(best_match_path))

            msg = (
                f"Closest match found:\n"
                f"Person: {person_name}\n"
                f"File: {best_match_path}\n"
                f"Model: {model_name}\n"
                f"Distance: {distance:.3f}\n"
                f"Similarity: {similarity_score:.3f}"
            )
            QMessageBox.information(self, "Recognition Result", msg)

            # 5) Load the best match image
            matched_img = mpimg.imread(best_match_path)

            # 6) Show side-by-side
            self.update_canvas_side_by_side(
                left_image=self.image1,
                right_image=matched_img,
                left_title="Query Image",
                right_title=f"{person_name}\nSim={similarity_score:.3f}"
            )

        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Could not process recognition: {e}")

    # -------------------------------------------------------------------------
    #                            CANVAS HELPERS
    # -------------------------------------------------------------------------
    def update_canvas_in_groupbox(self, groupbox, rgb_image, title=''):
        """
        Utility function to display a single image in a specified QGroupBox.
        """
        if groupbox.layout() is None:
            groupbox.setLayout(QVBoxLayout())

        # Clear any existing widgets
        self.clear_layout(groupbox.layout())

        if rgb_image is None:
            # If no image, just clear the canvas
            return

        fig = plt.Figure(figsize=(4, 3))
        canvas = FigureCanvas(fig)
        groupbox.layout().addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.imshow(rgb_image)
        ax.set_title(title)
        ax.axis('off')
        canvas.draw()

    def update_canvas_side_by_side(self, left_image, right_image,
                                   left_title='Image1',
                                   right_title='Image2'):
        """
        Utility function to show a side-by-side comparison in self.groupBox.
        (Used for recognition result display.)
        """
        if self.groupBox.layout() is None:
            self.groupBox.setLayout(QVBoxLayout())

        # Clear old widgets
        self.clear_layout(self.groupBox.layout())

        fig = plt.Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)
        self.groupBox.layout().addWidget(canvas)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(left_image)
        ax1.set_title(left_title)
        ax1.axis('off')

        ax2.imshow(right_image)
        ax2.set_title(right_title)
        ax2.axis('off')

        canvas.draw()

    def reset_all_displays(self):
        """
        Clears the groupBox (Recognition) and groupBox_2 (Registration) layouts,
        removing any displayed Matplotlib figures or images.
        Also resets self.image1 and self.registration_image to None.
        """
        # Clear the Recognition tab's groupBox layout
        if self.groupBox.layout() is not None:
            self.clear_layout(self.groupBox.layout())

        # Clear the Registration tab's groupBox_2 layout
        if self.groupBox_2.layout() is not None:
            self.clear_layout(self.groupBox_2.layout())

        # Reset stored images
        self.image1 = None
        self.image1_path = None
        self.registration_image = None

        # If you use self.canvas or self.fig, you can reset them too:
        self.canvas = None
        self.fig = None

        # Optional: show a small message
        QMessageBox.information(self, "Reset", "All displays have been cleared.")

    def clear_layout(self, layout):
        """Remove all widgets from the given layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # -------------------------------------------------------------------------
    #                              MAIN EXECUTION
    # -------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec_())
