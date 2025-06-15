# password-scoring-ML

Password Strength Scoring: Machine Learning Approach
This repository contains a Google Colab notebook (Password_Scoring.ipynb) that implements a machine learning solution for classifying the strength of passwords. The project explores various feature engineering techniques, feature selection methods, and machine learning models (Random Forest, SVM, Logistic Regression, and CNN) to accurately predict whether a given password is weak, medium, or strong.

Project Overview
The goal of this project is to develop a robust system for password strength classification. This is crucial for enhancing cybersecurity by providing real-time feedback on password quality and identifying potential vulnerabilities. The notebook covers:

1. Data Loading and Preprocessing: Handling a large dataset of passwords and their pre-assigned strengths.

2. Feature Engineering: Creating insightful features from raw passwords such as length, character types (digits, uppercase, lowercase, special characters), character variety, entropy, and pattern detection.

3. Feature Selection: Utilizing various methods like Mutual Information, Correlation Matrix, SelectKBest (ANOVA F-value), and Recursive Feature Elimination (RFE) to identify the most impactful features for classification.

4. Model Training: Implementing and training different machine learning models, including:

- Random Forest Classifier

- Support Vector Machine (SVM)

- Logistic Regression

- Convolutional Neural Network (CNN)

5. Model Evaluation: Assessing model performance using standard metrics like accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and Precision-Recall curves.

6. Hyperparameter Tuning: Employing techniques like GridSearchCV and RandomizedSearchCV (with SMOTE for imbalance handling) to optimize model parameters.

Dataset
The dataset used for this project is sourced from Kaggle:
Password Strength Classifier Dataset by Bhavik B.
You can download it using the opendatasets library directly within the Colab notebook.

The dataset typically contains passwords and a strength column, which is usually categorized (e.g., 0 for weak, 1 for medium, 2 for strong).

Installation and Setup
This project is designed to run on Google Colab, so most dependencies are pre-installed or can be installed with simple pip commands within the notebook.

To get started:

1. Open the Colab Notebook: Navigate to Password_Scoring.ipynb in this repository and open it in Google Colab.

2. Install Dependencies: The first cells in the notebook will install necessary libraries such as opendatasets and tensorflow.

!pip install opendatasets
!pip install tensorflow
!pip install imblearn # For SMOTE

3. Kaggle Credentials: To download the dataset, you will need to provide your Kaggle username and API key when prompted by opendatasets.download(). Follow the instructions provided by http://bit.ly/kaggle-creds to obtain your key.

Feature Engineering
Several features are engineered from the raw password strings to help the models learn patterns indicative of strength:

length: Total length of the password.

num_digits: Number of digits in the password.

num_upper: Number of uppercase letters.

num_lower: Number of lowercase letters.

num_special: Number of special characters.

char_variety_score: A score indicating the diversity of character types present (e.g., presence of lowercase, uppercase, digits, special characters).

entropy: Calculated using Shannon entropy, based on the character set size and password length.

pattern_detect: A binary feature indicating the presence of simple patterns like sequences (e.g., "abc", "123") or repeated characters (e.g., "aaa").

dictionary_word: A binary feature indicating if the password contains common dictionary words (using common-passwords.txt).

Additionally, TfidfVectorizer is used to create character-level n-gram features from the passwords.

Feature Selection
Various methods were employed to identify the most relevant features and reduce dimensionality:

Mutual Information: Measures the dependency between features and the target variable.

Correlation Matrix: Visualizes the linear relationships between numerical features.

SelectKBest (ANOVA F-value): Selects features based on the highest scores from ANOVA F-value statistical test.

Recursive Feature Elimination (RFE): Recursively removes features and builds a model on the remaining features, with the goal of selecting the optimal subset.

The selected features for model training often include: length, num_upper, char_variety_score, entropy, and pattern_detect.

Models Trained
The project evaluates the performance of the following classification models:

1. Random Forest Classifier: An ensemble learning method that builds multiple decision trees and merges their predictions to improve accuracy and control overfitting.

2. Support Vector Machine (SVM): A powerful model that finds an optimal hyperplane to separate data points into classes.

3. Logistic Regression: A linear model used for binary or multiclass classification.

4. Convolutional Neural Network (CNN): A deep learning model specifically designed for sequence data (like text, treated as character sequences here) to capture local patterns.

Evaluation Metrics
The models are evaluated using a comprehensive set of metrics to provide a complete picture of their performance:

Accuracy: The proportion of correctly classified instances.

Precision: The proportion of true positive predictions among all positive predictions.

Recall (Sensitivity): The proportion of true positive predictions among all actual positive instances.

F1-score: The harmonic mean of precision and recall, providing a balance between the two.

Confusion Matrix: A table that visualizes the performance of a classification model, showing true positives, true negatives, false positives, and false negatives.

ROC Curve (Receiver Operating Characteristic): Plots the true positive rate against the false positive rate at various threshold settings.

Precision-Recall Curve: Plots precision against recall at various threshold settings, particularly useful for imbalanced datasets.

Results
Based on the provided evaluation, the models performed as follows:

Classifier

Test Accuracy

Random Forest

0.9912

SVM

0.9902

Logistic Regression

0.9022

CNN

0.9395

Random Forest and SVM models demonstrated superior performance in terms of test accuracy compared to Logistic Regression and CNN for this specific dataset and feature set. Detailed classification reports and confusion matrices for each model are available in the notebook.

How to Run the Notebook
1. Clone the repository (or download the Password_Scoring.ipynb file).

2. Open in Google Colab.

3. Run all cells sequentially (Runtime -> Run all).

4. Provide Kaggle credentials when prompted for dataset download.

5. Observe the output, including data insights, feature engineering results, model training progress, and evaluation metrics.

Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Any contributions, suggestions, or bug reports are welcome!

License
This project is open-source and available under the MIT License.
