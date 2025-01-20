# Title: Gender Classification Using Machine Learning

This project revolves around building a gender classification model using machine learning. It involves loading images, training gender classifying models, making predictions, and evaluating model performance.

## Prerequisites:

Make sure the following Python libraries are installed:
- numpy, pandas: for data manipulation
- PIL: for image processing
- matplotlib, seaborn: for data visualization
- tqdm: for displaying progress bars
- sklearn: for machine learning tasks like data splitting and model evaluation
- transformers: for loading pre-trained models

The current open file in this project is a Jupyter notebook named "face_gender_evaluation.ipynb".

## Process:

1. **Import the required libraries:** All necessary Python libraries as mentioned in the prerequisites are imported. A random seed is set for reproducibility and warnings are suppressed for cleaner output.

2. **Define the dataset directory:** Ensure that the dataset directory exists. Update the path based on your setup.

3. **Prepare the dataset:** Randomly select 50 images per class (male and female). If the directories for male and female images are not found, an error is raised. The images are loaded along with their labels into a pandas DataFrame.

4. **Display sample images:** To observe the loaded dataset's contents, display some sample images. Here, it is set to display 5 images per class.

5. **Split the Data:** Split the data into training, validation, and testing sets in a 60%, 20% and 20% ratio respectively.

6. **Load Models:** Initialize the machine learning models. Here, two models are used - 'rizvandwiki/gender-classification', 'mrm8488/mobilevit-small-finetuned-agegender'.

7. **Prepare Test Images:** Load the test images and display a sample of the test images.

8. **Make Predictions:** Provide the loaded test images to both models and make gender predictions.

8. **Data Cleaning:** Remove entries with 'Unknown' predictions.

10. **Model Evaluation:** Calculate metrics like Accuracy, Precision, Recall, F1 Score, and provide the Classification report for each model. Also, plot the confusion matrix for a visual understanding of the model performance.

11. **Visualize the results:** Display the test images with their actual labels and predicted labels. 

The project involves the aforementioned steps, performed using the provided Python code. Each section of the code is separated into different cells in the Jupyter notebook for easier understanding and readability. 

**Note:** To ensure the best output, adjust the path to your dataset and the number of images per class according to your requirements and system capabilities. 

**Disclaimer:** The code provided should be used as a basis and might need adjustments based on the specific requirements and conditions.

This project provides a good understanding of how pre-trained models can be leveraged for classification tasks. It is a good starting point for anyone interested in gender classification using machine learning techniques.
