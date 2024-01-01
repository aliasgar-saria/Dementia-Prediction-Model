# %% [markdown]
# 
#     This notebook covers the analysis of a dementia dataset, including data preprocessing, exploratory data analysis (EDA), 
#     machine learning model application with various algorithms used, and implemented Deep learning neural networks using Tensorflow and Keras framework.
# 
#     

# %%
import pandas as pd

# Load the dataset to take a preliminary look at its structure
file_path = 'dementia_dataset.csv'
dementia_data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset
dementia_data.head()


# %% [markdown]
# 
#     ## Exploratory Data Analysis (EDA)
#     Performing an initial analysis to understand the structure and characteristics of the data.
#     

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initial data analysis and visualization

# Checking for missing values
missing_values = dementia_data.isnull().sum()

# Distribution of the target variable 'Group'
group_distribution = dementia_data['Group'].value_counts()

# Age distribution
age_distribution = dementia_data['Age'].describe()

# Gender distribution
gender_distribution = dementia_data['M/F'].value_counts()

# Visualizations
plt.figure(figsize=(20, 5))

# Plot for missing values
plt.subplot(1, 3, 1)
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.xticks(rotation=90)
plt.title("Missing Values in Each Column")

# Plot for group distribution
plt.subplot(1, 3, 2)
sns.barplot(x=group_distribution.index, y=group_distribution.values)
plt.title("Distribution of Dementia Groups")

# Plot for gender distribution
plt.subplot(1, 3, 3)
sns.barplot(x=gender_distribution.index, y=gender_distribution.values)
plt.title("Gender Distribution")

plt.tight_layout()
plt.show()

missing_values, group_distribution, age_distribution, gender_distribution


# %%
# Plotting appropriate graphs for the dementia dataset project

# 1. Histogram for Age Distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(dementia_data['Age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# 2. Box Plot for MMSE (Mini-Mental State Examination) scores
plt.subplot(1, 3, 2)
sns.boxplot(x='Group', y='MMSE', data=dementia_data)
plt.title('MMSE Score by Dementia Group')
plt.xlabel('Group')
plt.ylabel('MMSE Score')

# 3. Heatmap for Correlations between features
plt.subplot(1, 3, 3)
corr_matrix = dementia_data[['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()


# %% [markdown]
# 
#     ## Data Cleaning and Feature Engineering
#     Preparing the data for modeling by handling missing values, feature scaling, and engineering.
#     

# %%
# Data Cleaning

# Imputing missing values for SES with its mode
ses_mode = dementia_data['SES'].mode()[0]
dementia_data['SES'].fillna(ses_mode, inplace=True)

# Imputing missing values for MMSE with its median
mmse_median = dementia_data['MMSE'].median()
dementia_data['MMSE'].fillna(mmse_median, inplace=True)

# Feature Engineering
# For demonstration, let's create a new feature 'AgeGroup' to categorize subjects into age groups
bins = [60, 70, 80, 90, 100]
labels = ['60-69', '70-79', '80-89', '90-99']
dementia_data['AgeGroup'] = pd.cut(dementia_data['Age'], bins=bins, labels=labels, right=False)

# Feature Scaling
# Scaling continuous features like 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', and 'ASF'
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = dementia_data[['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
scaled_features = scaler.fit_transform(scaled_features)

# Replacing original columns with scaled values
dementia_data[['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']] = scaled_features

# Checking the dataset after cleaning and feature engineering
dementia_data.head()


# %% [markdown]
# 
#     ## Applying Machine Learning Algorithms
#     Implementing and evaluating different machine learning models such as Logistic Regression, Random Forest, and SVM.
#     

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Preparing the data for modeling
# Encoding the categorical variables
dementia_data_encoded = pd.get_dummies(dementia_data, columns=['Group', 'M/F', 'Hand', 'AgeGroup'], drop_first=True)

# Separating the features and the target variable
X = dementia_data_encoded.drop(['Subject ID', 'MRI ID', 'Group_Demented', 'Group_Nondemented'], axis=1)
y = dementia_data_encoded['Group_Nondemented']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying various machine learning algorithms
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Support Vector Machine Classifier
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

# Evaluating the models
log_reg_report = classification_report(y_test, y_pred_log_reg)
rf_report = classification_report(y_test, y_pred_rf)
svm_report = classification_report(y_test, y_pred_svm)

log_reg_report, rf_report, svm_report


# %% [markdown]
# 
#     ## Deep Learning Model Template (TensorFlow)
#     This section is a template for a deep learning model using TensorFlow. Since TensorFlow is not available in this environment, 
#     this section is commented out. You can run this on your local machine with TensorFlow installed.
#     

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Balancing the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning for Random Forest Classifier
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                              param_grid=rf_params, 
                              cv=5, 
                              verbose=2, 
                              n_jobs=-1)
grid_search_rf.fit(X_train_smote, y_train_smote)
best_rf_clf = grid_search_rf.best_estimator_

# Predictions using the best Random Forest model
y_pred_best_rf = best_rf_clf.predict(X_test)

# Metrics calculation for the best Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_best_rf)
precision_rf = precision_score(y_test, y_pred_best_rf)
recall_rf = recall_score(y_test, y_pred_best_rf)
f1_rf = f1_score(y_test, y_pred_best_rf)

# Metrics for original models
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)
f1_log_reg = f1_score(y_test, y_pred_log_reg)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Plotting the performance metrics for all models
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
log_reg_metrics = [accuracy_log_reg, precision_log_reg, recall_log_reg, f1_log_reg]
rf_metrics = [accuracy_rf, precision_rf, recall_rf, f1_rf]
svm_metrics = [accuracy_svm, precision_svm, recall_svm, f1_svm]

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
plt.bar(x - 0.2, log_reg_metrics, width=0.2, label='Logistic Regression')
plt.bar(x, rf_metrics, width=0.2, label='Random Forest')
plt.bar(x + 0.2, svm_metrics, width=0.2, label='SVM')
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Performance Metrics of Different Models")
plt.legend()
plt.show()

best_rf_clf, (accuracy_rf, precision_rf, recall_rf, f1_rf)


# %%
# Hyperparameter tuning for Random Forest Classifier without SMOTE
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                              param_grid=rf_params, 
                              cv=5, 
                              verbose=2, 
                              n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_clf = grid_search_rf.best_estimator_

# Predictions using the best Random Forest model
y_pred_best_rf = best_rf_clf.predict(X_test)

# Metrics calculation for the best Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_best_rf)
precision_rf = precision_score(y_test, y_pred_best_rf)
recall_rf = recall_score(y_test, y_pred_best_rf)
f1_rf = f1_score(y_test, y_pred_best_rf)

# Plotting the performance metrics for all models
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
plt.bar(x - 0.2, log_reg_metrics, width=0.2, label='Logistic Regression')
plt.bar(x, rf_metrics, width=0.2, label='Random Forest (Optimized)')
plt.bar(x + 0.2, svm_metrics, width=0.2, label='SVM')
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Performance Metrics of Different Models")
plt.legend()
plt.show()

best_rf_clf, (accuracy_rf, precision_rf, recall_rf, f1_rf)


# %%
# Recalculating the metrics for Logistic Regression and SVM models

# Metrics for Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)
f1_log_reg = f1_score(y_test, y_pred_log_reg)
log_reg_metrics = [accuracy_log_reg, precision_log_reg, recall_log_reg, f1_log_reg]

# Metrics for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
svm_metrics = [accuracy_svm, precision_svm, recall_svm, f1_svm]

# Plotting the performance metrics for the original Logistic Regression and SVM models
plt.figure(figsize=(10, 6))
x_labels = np.arange(len(metrics_labels))
plt.bar(x_labels - 0.1, log_reg_metrics, width=0.2, label='Logistic Regression')
plt.bar(x_labels + 0.1, svm_metrics, width=0.2, label='SVM')
plt.xticks(x_labels, metrics_labels)
plt.ylabel("Score")
plt.title("Performance Metrics of Logistic Regression vs SVM")
plt.legend()
plt.show()


# %%
# Plotting a line graph for the performance metrics of Logistic Regression and SVM models

plt.figure(figsize=(10, 6))

# Plotting lines for Logistic Regression and SVM
plt.plot(metrics_labels, log_reg_metrics, marker='o', label='Logistic Regression')
plt.plot(metrics_labels, svm_metrics, marker='o', label='SVM')

# Adding labels and title
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Line Graph of Performance Metrics for Logistic Regression vs SVM')
plt.legend()

plt.show()


# %%
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Calculating additional metrics and plotting graphs for Logistic Regression and SVM

# Confusion Matrices
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# ROC Curves and AUC
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_log_reg)
roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plotting
plt.figure(figsize=(15, 5))

# Confusion Matrix for Logistic Regression
plt.subplot(1, 3, 1)
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Confusion Matrix for SVM
plt.subplot(1, 3, 2)
sns.heatmap(conf_matrix_svm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# ROC Curve
plt.subplot(1, 3, 3)
plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

roc_auc_log_reg, roc_auc_svm


# %%
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# Calculating additional metrics for Logistic Regression and SVM models

# ROC AUC Score
roc_auc_score_log_reg = roc_auc_score(y_test, y_pred_log_reg)
roc_auc_score_svm = roc_auc_score(y_test, y_pred_svm)

# Precision-Recall Curve and Average Precision Score
precision_log_reg, recall_log_reg, _ = precision_recall_curve(y_test, y_pred_log_reg)
ap_score_log_reg = average_precision_score(y_test, y_pred_log_reg)

precision_svm, recall_svm, _ = precision_recall_curve(y_test, y_pred_svm)
ap_score_svm = average_precision_score(y_test, y_pred_svm)

# Plotting Precision-Recall Curves
plt.figure(figsize=(10, 5))

plt.plot(recall_log_reg, precision_log_reg, label=f'Logistic Regression (AP = {ap_score_log_reg:.2f})')
plt.plot(recall_svm, precision_svm, label=f'SVM (AP = {ap_score_svm:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.show()

# Displaying ROC AUC scores and Average Precision scores
roc_auc_score_log_reg, roc_auc_score_svm, ap_score_log_reg, ap_score_svm


# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Convert the data to float32 if not already
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# %%

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Model summary
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2)

# Evaluating the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# %%
# model.save('modeldementia.h5')

# %%
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()



