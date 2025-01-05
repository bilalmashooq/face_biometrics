# -*- coding: utf-8 -*-
"""
Author: Muhammad Bilal
Supervisor: Prof. Dr. Amin Nait-Ali
Master of Biometrics and Intelligent Vision
Department of Science and Technology, UPEC
University Paris Est Creteil
Date created: 21/12/2024
Last Date modified: 21/12/2024
Description:
  Main code for face verification (1-to-1) using a PyQt5 GUI.
  We re-use the verification UI, with Registration + Verification tabs.

  In this version, we assign a unique ID to each newly registered person,
  storing images in a folder named "<ID>_<personName>".
  During verification, we only ask for the person's unique ID and find
  that folder to load a reference image.
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

# DeepFace for face verification
from deepface import DeepFace

# Import the generated UI class from the .ui file (verification.ui → verification_GUI.py)
from verification_GUI import Ui_MainWindow


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # --------------------- #
        #  Basic GUI Variables  #
        # --------------------- #
        self.registration_image = None  # For the "Registration" tab
        self.verification_image = None  # For the "Verification" tab
        self.verification_image_path = None

        self.fig = None
        self.canvas = None

        # Path to store the dataset (subfolders named: "<ID>_<personName>")
        self.dataset_path = r"C:\Users\muham\PycharmProjects\pythonProject\Verification\lfw_subset"
        # Adjust to your actual dataset path

        # --------------------- #
        #     Connect Buttons   #
        # --------------------- #
        # Main Tab (index=0)
        self.pushButton.clicked.connect(self.go_to_registration_tab)   # "Registeration"
        self.pushButton_2.clicked.connect(self.go_to_verification_tab) # "Verification"

        # Registration Tab (index=1)
        self.pushButton_6.clicked.connect(self.registration_load_image)   # "Upload"
        self.pushButton_7.clicked.connect(self.registration_save_image)   # "Registeration"
        self.pushButton_8.clicked.connect(self.registration_view_dataset) # "View Registered"

        # Verification Tab (index=2)
        self.pushButton_3.clicked.connect(self.verification_load_image)   # "Upload Image"
        # Two model buttons:
        self.pushButton_9.clicked.connect(lambda: self.verify_face_1to1(model_name='VGG-Face'))  # "Model 1"
        self.pushButton_4.clicked.connect(lambda: self.verify_face_1to1(model_name='Facenet'))   # "Model 2"

        # Reset button (pushButton_5)
        self.pushButton_5.clicked.connect(self.reset_displays)

        # Theme comboBox
        self.comboBox.currentIndexChanged.connect(self.theme_changed)
        self.theme_changed(0)  # Set default theme

        # Start on the "Main" tab
        self.tabWidget.setCurrentIndex(0)

    # -------------------------------------------------------------------------
    #                               TAB SWITCHING
    # -------------------------------------------------------------------------
    def go_to_registration_tab(self):
        """Switch to the registration tab (index=1)."""
        self.tabWidget.setCurrentIndex(1)

    def go_to_verification_tab(self):
        """Switch to the verification tab (index=2)."""
        self.tabWidget.setCurrentIndex(2)

    # -------------------------------------------------------------------------
    #                                THEME HANDLING
    # -------------------------------------------------------------------------
    def theme_changed(self, index):
        """Apply the selected theme from comboBox."""
        self.set_theme(index)

    def set_theme(self, index):
        """Set the application theme based on comboBox index."""
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
            # Default / no style
            self.setStyleSheet('')

    # -------------------------------------------------------------------------
    #                             REGISTRATION TAB
    # -------------------------------------------------------------------------
    def registration_load_image(self):
        """
        Load an image for registration, display it in groupBox_2.
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.registration_image = img
            self.update_canvas_in_groupbox(
                groupbox=self.groupBox_2,
                rgb_image=self.registration_image,
                title="Registration Image"
            )

            QMessageBox.information(
                self,
                'Image Loaded',
                f'Image loaded for registration:\n{filename}'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error loading image file: {e}")

    def registration_save_image(self):
        """
        Prompt for a person's name + unique ID, create a folder "<ID>_<Name>" in the dataset,
        and save the registration_image there.
        """
        if self.registration_image is None:
            QMessageBox.warning(self, 'Error', "No image loaded for registration.")
            return

        # Ask for person's name
        person_name, ok1 = QInputDialog.getText(
            self, "Registration", "Enter person's name:", QLineEdit.Normal, ""
        )
        if not ok1 or not person_name.strip():
            QMessageBox.warning(self, "Registration", "Name cannot be empty.")
            return
        person_name = person_name.strip()

        # Ask for unique ID (you can generate automatically or let user type it)
        unique_id, ok2 = QInputDialog.getText(
            self, "Registration", "Enter unique ID:", QLineEdit.Normal, ""
        )
        if not ok2 or not unique_id.strip():
            QMessageBox.warning(self, "Registration", "ID cannot be empty.")
            return
        unique_id = unique_id.strip()

        # Create folder named "<ID>_<personName>"
        folder_name = f"{unique_id}_{person_name}"
        person_folder = os.path.join(self.dataset_path, folder_name)
        os.makedirs(person_folder, exist_ok=True)

        # Save the image
        new_filename = os.path.join(person_folder, "registered_image.jpg")
        img_bgr = cv2.cvtColor(self.registration_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_filename, img_bgr)

        QMessageBox.information(
            self, "Registration",
            f"Image has been saved in folder:\n{person_folder}"
        )

    def registration_view_dataset(self):
        """
        List all folders (which should be "<ID>_<Name>") in the dataset,
        let user pick one, display the first image from that folder in groupBox_2.
        """
        folder_list = []
        for entry in sorted(os.listdir(self.dataset_path)):
            full_path = os.path.join(self.dataset_path, entry)
            if os.path.isdir(full_path):
                folder_list.append(entry)

        if not folder_list:
            QMessageBox.warning(self, "View Registered", "No registered folders found.")
            return

        chosen, ok = QInputDialog.getItem(
            self, "Registered Folders", "Select folder:", folder_list, 0, False
        )
        if not ok:
            return

        folder_path = os.path.join(self.dataset_path, chosen)
        found_image_path = None
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                found_image_path = os.path.join(folder_path, f)
                break

        if not found_image_path:
            QMessageBox.warning(self, "View Registered", f"No images found in {chosen}.")
            return

        try:
            img = mpimg.imread(found_image_path)
            self.update_canvas_in_groupbox(
                groupbox=self.groupBox_2,
                rgb_image=img,
                title=f"Folder: {chosen}"
            )
            QMessageBox.information(
                self, "View Registered",
                f"Showing: {found_image_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Could not load image: {e}")

    # -------------------------------------------------------------------------
    #                             VERIFICATION TAB
    # -------------------------------------------------------------------------
    def verification_load_image(self):
        """
        Load a query image for 1-to-1 verification, display in groupBox.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Select Image File', QDir.currentPath(),
            'Image Files (*.png *.jpg *.jpeg *.bmp)'
        )
        if not filename:
            return

        try:
            self.verification_image_path = filename
            img = cv2.imread(filename)
            if img is None:
                raise Exception("Could not load the image.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.verification_image = img
            self.update_canvas_in_groupbox(
                groupbox=self.groupBox,
                rgb_image=self.verification_image,
                title="Image to Verify"
            )

            QMessageBox.information(
                self, 'Image Loaded',
                f'Image loaded for verification:\n{filename}'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error loading image file: {e}")

    def verify_face_1to1(self, model_name='VGG-Face'):
        """
        1-to-1 face verification using a unique ID. We prompt the user for an ID,
        locate a folder named "<ID>_...", pick a reference image, then do verify.
        """
        if self.verification_image is None:
            QMessageBox.warning(self, 'Error', "Please upload an image first.")
            return

        # Ask user for the person's ID
        input_id, ok = QInputDialog.getText(
            self, "Verification", "Enter the unique ID:", QLineEdit.Normal, ""
        )
        if not ok or not input_id.strip():
            QMessageBox.warning(self, "Verification", "ID cannot be empty.")
            return
        input_id = input_id.strip()

        # Find a folder that starts with "<ID>_"
        target_folder = None
        all_folders = os.listdir(self.dataset_path)
        for fold in all_folders:
            if fold.startswith(input_id + "_") and \
               os.path.isdir(os.path.join(self.dataset_path, fold)):
                target_folder = os.path.join(self.dataset_path, fold)
                break

        if not target_folder:
            QMessageBox.warning(self, "Verification",
                                f"No folder found starting with ID '{input_id}'.")
            return

        # Pick the first reference image
        ref_image_path = None
        for ff in os.listdir(target_folder):
            if ff.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                ref_image_path = os.path.join(target_folder, ff)
                break

        if not ref_image_path:
            QMessageBox.warning(self, "Verification",
                                f"No reference images found in {target_folder}.")
            return

        # Now do the 1-to-1 verify with DeepFace
        try:
            result = DeepFace.verify(
                img1_path=self.verification_image_path,
                img2_path=ref_image_path,
                model_name=model_name,
                enforce_detection=False
            )
            verified = result.get('verified', False)
            distance = result.get('distance', None)

            folder_name = os.path.basename(target_folder)  # e.g. "1001_John"
            if distance is not None:
                msg = (f"ID: {input_id}\n"
                       f"Folder: {folder_name}\n"
                       f"Reference: {ref_image_path}\n"
                       f"Model: {model_name}\n"
                       f"Distance: {distance:.3f}\n"
                       f"VERIFIED: {verified}")
            else:
                msg = (f"ID: {input_id}\n"
                       f"Folder: {folder_name}\n"
                       f"Reference: {ref_image_path}\n"
                       f"Model: {model_name}\n"
                       f"VERIFIED: {verified}")

            QMessageBox.information(self, "Verification", msg)

            # Show side-by-side
            ref_img = mpimg.imread(ref_image_path)
            result_text = "MATCH" if verified else "NO MATCH"
            self.update_canvas_side_by_side(
                left_image=self.verification_image,
                right_image=ref_img,
                left_title="Query Image",
                right_title=f"{folder_name}\n{result_text}"
            )
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Verification error: {e}")

    # -------------------------------------------------------------------------
    #                                RESET METHOD
    # -------------------------------------------------------------------------
    def reset_displays(self):
        """
        Clears the displays in both the Registration tab (groupBox_2)
        and Verification tab (groupBox).
        Resets the images to None.
        """
        # Clear groupBox_2 (Registration)
        if self.groupBox_2.layout() is not None:
            self.clear_layout(self.groupBox_2.layout())

        # Clear groupBox (Verification)
        if self.groupBox.layout() is not None:
            self.clear_layout(self.groupBox.layout())

        # Reset images
        self.registration_image = None
        self.verification_image = None
        self.verification_image_path = None

        # Reset figure/canvas references
        self.fig = None
        self.canvas = None

        QMessageBox.information(self, "Reset", "All displays have been cleared.")

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
                                   left_title='Image1', right_title='Image2'):
        """
        Utility function to show a side-by-side comparison in self.groupBox
        (the Verification tab).
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
