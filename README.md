# ML-Project

# BREAST CANCER PREDICTION MODEL

# OVERVIEW

This project implements a machine learning model to predict whether a breast tumor is benign or malignant based on various diagnostic features. The model is trained using the Breast Cancer Wisconsin Dataset and evaluated for its performance. The Random Forest Classifier is utilized for classification, and the results include a classification report, confusion matrix, and feature importance visualization.

# DATASET

The dataset used in this project is BreastCancer.csv. It contains diagnostic measurements of breast tumors. The key columns in the dataset are:

**id:** Unique identifier for each observation (dropped in preprocessing).

**diagnosis:** Target variable indicating tumor type (B for Benign and M for Malignant).

**Feature Columns:** 30 numeric features derived from image analysis, including:

radius_mean, texture_mean, perimeter_mean, etc.

radius_se, texture_se, perimeter_se, etc.

radius_worst, texture_worst, perimeter_worst, etc.

# PREREQUISITES

To run the code, you need the following libraries installed:

pandas

numpy

sklearn

matplotlib

seaborn

pickle

Install them using:

pip install pandas numpy scikit-learn matplotlib seaborn

# STEPS IN THE CODE

Import Libraries: Necessary libraries are imported for data processing, model training, evaluation, visualization, and serialization.

**Load and Explore Data:**
The dataset is loaded using pandas.read_csv.
Basic information, missing values, and sample rows are displayed.

**Data Preprocessing:**
Unnecessary columns (id and Unnamed: 32) are dropped.
Missing values in numeric columns are imputed using the mean strategy.
The target variable (diagnosis) is mapped to binary values (0 for Benign, 1 for Malignant).

**Visualization:**
A correlation heatmap is generated to analyze feature relationships.

**Model Training:**
The dataset is split into training and testing sets using train_test_split.
A Random Forest Classifier is trained on the training data.

**Evaluation:**
Predictions are made on the test set.
Model performance is evaluated using a classification report and confusion matrix.

**Feature Importance:**
The importance of each feature in prediction is visualized using a bar plot.

**Model Serialization:**
The trained model is saved as a pickle file (breast_cancer_model.pkl) for later use.

# HOW TO RUN

Save the dataset as BreastCancer.csv in the working directory.

Run the Python script.

Outputs include:

Dataset information and sample rows.

Correlation heatmap.

Classification report and confusion matrix.

Feature importance visualization.

Serialized model file (breast_cancer_model.pkl).

# EXPECTED OUTPUTS

**Correlation Heatmap:** Displays relationships between features.

**Classification Report:** Shows precision, recall, F1-score, and accuracy.

**Confusion Matrix:** Provides true positives, true negatives, false positives, and false negatives.

**Feature Importance Plot:** Highlights which features are most important for the predictions.

# USAGE OF PICKLE FILE

The trained model is saved as a pickle file for reuse:
import pickle

**Load the saved model**

with open('breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

**Use the model for prediction**

prediction = model.predict(new_data)
print("Prediction:", prediction)

