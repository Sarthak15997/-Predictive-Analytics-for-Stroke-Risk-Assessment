# -*- coding: utf-8 -*-
"""
##Attribute Information

1) id: unique identifier.

2) gender: "Male", "Female" or "Other".

3) age: age of the patient.

4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension.

5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease.

6) ever_married: "No" or "Yes".

7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed".

8) Residence_type: "Rural" or "Urban".

9) avg_glucose_level: average glucose level in blood.

10) bmi: body mass index.

11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*.

12) stroke: 1 if the patient had a stroke or 0 if not.

##Stroke Prediction with 9 Algorithms

(1)Logistic Regression

(2)KNN

(3)SVM

(4)Naive Bayes

(5)Random Forest

(6)Decision Tree

(7)Using xgboostClassifier of tree class to use Decision Tree Algorithm

(8)Using SGDClassifier of tree class to use Decision Tree Algorithm

(9)Using AdaBoostClassifier of tree class to use Decision Tree Algorithm

##Importing the necessary libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

df=pd.read_csv("/content/healthcare-dataset-stroke-data.csv")
df.head()

"""Dropping the id column"""

# Drop id column
df.drop(columns=['id'],inplace=True)

df.info()

df.describe()

df.isna()

df.isna().sum()

df.isnull().sum()

#Imputing the missing values with the mean
df=df.fillna(np.mean(df['bmi']))
df.info()

df.isnull().sum()

# Convert Marrital Status, Residence and Gender into 0's and 1's
df['gender']=df['gender'].apply(lambda x : 1 if x=='Male' else 0) 
df["Residence_type"] = df["Residence_type"].apply(lambda x: 1 if x=="Urban" else 0)
df["ever_married"] = df["ever_married"].apply(lambda x: 1 if x=="Yes" else 0)

"""This code is converting the values in three columns of a dataframe ('gender', 'Residence_type', and 'ever_married') into binary form, where 1 indicates a certain value and 0 indicates another.

Specifically, it is converting the values in the 'gender' column where 'Male' is converted to 1 and 'Female' to 0. The 'Residence_type' column values where 'Urban' is converted to 1 and 'Rural' to 0. The 'ever_married' column values where 'Yes' is converted to 1 and 'No' to 0.

This conversion is often done to prepare categorical variables for machine learning algorithms that can only handle numerical data.
"""

# Removing the observations that have smoking_status type unknown. 
df=df[df['smoking_status']!='Unknown']

df

# used One Hot encoding smoking_status, work_type
data_dummies = df[['smoking_status','work_type']]
data_dummies=pd.get_dummies(data_dummies)
df.drop(columns=['smoking_status','work_type'],inplace=True)

"""This code performs one-hot encoding on the 'smoking_status' and 'work_type' columns in the 'df' DataFrame. It creates new columns for each unique value in the original columns, and sets the value to 1 if that value is present in the row and 0 otherwise. The new columns are added to a new DataFrame called 'data_dummies'.

Finally, the original columns ('smoking_status' and 'work_type') are dropped from the original DataFrame ('df') using the 'drop' method with the 'columns' parameter set to a list of the column names to be dropped. The 'inplace' parameter is set to 'True' to modify the 'df' DataFrame in place.
"""

data_dummies

df

y=df['stroke']
df.drop(columns=['stroke'],inplace=True)
x=df.merge(data_dummies,left_index=True, right_index=True,how='left')

"""The code is separating the target variable stroke from the features in the DataFrame df. It then removes the stroke column from the df DataFrame and saves it as the target variable y.

The code then performs one-hot encoding on the smoking_status and work_type columns in the df DataFrame and saves the result in a new DataFrame data_dummies.

Finally, the code merges the one-hot encoded data_dummies DataFrame with the original df DataFrame on their indexes, dropping any rows that are not present in both DataFrames. The resulting DataFrame x contains all the original features with the one-hot encoded columns added to it.
"""

x

"""##Since the data is imabalanced 

imbalanced-learn is a Python package used for handling imbalanced datasets, which are datasets with unequal distribution of classes. It provides various techniques for resampling the data to balance the classes, which can help improve the performance of machine learning models trained on such datasets. By installing/upgrading the imbalanced-learn package, the user can access its functions and classes for building more effective models on imbalanced datasets.
"""

!pip install -U imbalanced-learn

"""
performing random oversampling of the minority class in the dataset to address class imbalance."""

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(random_state=0)
x, y = oversample.fit_resample(x, y)

"""This code is performing oversampling using RandomOverSampler from the imbalanced-learn library. The purpose of oversampling is to balance the class distribution in the target variable 'y' by creating more samples of the minority class (i.e., the class with fewer observations) to match the majority class.

The RandomOverSampler randomly selects samples from the minority class and creates new samples until both classes have an equal number of observations. The resulting oversampled dataset has more observations and a balanced distribution of the target variable, which can improve the performance of machine learning models, particularly when the original dataset is imbalanced.

In the code, 'x' is the feature matrix and 'y' is the target variable. The oversample.fit_resample() method applies the oversampling technique to 'x' and 'y' and returns the oversampled feature matrix 'x' and target variable 'y'.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.20,random_state=0)

"""This code is performing a train-test split on the data, where x represents the feature matrix and y represents the target variable.

The train_test_split function from the sklearn.model_selection module is used to randomly split the dataset into two parts - a training set and a testing set. The test_size parameter is set to 0.20, which means that 20% of the data will be used for testing and the remaining 80% will be used for training the model.

The random_state parameter is set to 0, which ensures that the same split is obtained every time the code is run, allowing for reproducibility. The function returns four arrays: X_train, which contains the training features, X_test, which contains the testing features, Y_train, which contains the training target variable, and Y_test, which contains the testing target variable.
"""

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""This code is performing feature scaling using standardization on the training and testing sets. Standardization is a common preprocessing technique used in machine learning to transform the data to have a mean of 0 and standard deviation of 1. The StandardScaler() function from sklearn.preprocessing is used to perform this transformation.

The fit_transform() method of the sc object is called on X_train to fit the scaler on the training data and transform it. The transform() method is called on X_test to transform the testing data using the same scaler that was fit on the training data. This ensures that the same scaling is applied to both the training and testing data, preventing data leakage and ensuring that the model is trained and evaluated on data that has been processed in the same way.
"""

X_train

# count the number of missing values in each column
missing_values = df.isnull().sum()

# print the result
print(missing_values)

"""##(1)Logistic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# create a Logistic Regression model
lr = LogisticRegression(random_state=0)

# fit the model on the training data
lr.fit(X_train, Y_train)

# predict the class labels for the test set
Y_pred = lr.predict(X_test)

# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)


# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n", classification_report(Y_test, Y_pred))

"""This code is training a Logistic Regression model on the training data and predicting the class labels for the test set. It then calculates the confusion matrix, accuracy, true positives, true negatives, false positives, and false negatives for the model's predictions. It then prints these values along with a classification report, which includes precision, recall, and F1-score for each class as well as the overall accuracy, macro average, and weighted average.

##(2)KNN

##K = 5
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Create an instance of the classifier with n_neighbors=5
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data
knn.fit(X_train, Y_train)

# Predict the test data
Y_pred = knn.predict(X_test)

# Evaluate the model on the test data
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)
# print("Accuracy: {:.2f}%".format(accuracy * 100))


# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion Matrix:\n", cm)



# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))

print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n", classification_report(Y_test, Y_pred))

"""This code trains a K-nearest neighbor (KNN) classifier on some training data X_train and Y_train. The trained model is then used to predict the labels of some test data X_test, and the accuracy of the model is evaluated using the accuracy_score function from the sklearn.metrics module.

The confusion matrix is printed using the confusion_matrix function from the sklearn.metrics module, and then the true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) are calculated. These metrics give a more detailed understanding of how well the classifier is performing.

Finally, the classification_report function from the sklearn.metrics module is used to print a report of precision, recall, and f1-score for each class, as well as the overall accuracy, precision, recall, and f1-score of the classifier.

##K = 3
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Create an instance of the classifier with n_neighbors=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
knn.fit(X_train, Y_train)

# Predict the test data
Y_pred = knn.predict(X_test)

# Evaluate the model on the test data
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)
# print("Accuracy: {:.2f}%".format(accuracy * 100))


# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion Matrix:\n", cm)



# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n", classification_report(Y_test, Y_pred))

"""## K =10"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Create an instance of the classifier with n_neighbors=10
knn = KNeighborsClassifier(n_neighbors=10)

# Fit the model to the training data
knn.fit(X_train, Y_train)

# Predict the test data
Y_pred = knn.predict(X_test)

# Evaluate the model on the test data
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)
# print("Accuracy: {:.2f}%".format(accuracy * 100))


# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion Matrix:\n", cm)



# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n", classification_report(Y_test, Y_pred))

"""##(3)SVM"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Define the SVM model with linear kernel
svm = SVC(kernel='linear')

# Train the model on the training data
svm.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = svm.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(Y_test, Y_pred)



# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)


# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n", classification_report(Y_test, Y_pred))

"""This code trains a Support Vector Machine (SVM) model with a linear kernel on some training data X_train and Y_train. Then, it makes predictions on some test data X_test and calculates the accuracy score and confusion matrix to evaluate the performance of the model. The classification report is also printed which shows precision, recall, and f1-score for each class in the target variable.

##(4)Naive Bayes
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# create Gaussian Naive Bayes model object
nb = GaussianNB()

# train the model using training set
nb.fit(X_train, Y_train)

# predict using the trained model
Y_pred = nb.predict(X_test)

# evaluate the performance of the model
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)

# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in %: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n", classification_report(Y_test, Y_pred))

"""This code trains a Gaussian Naive Bayes classification model on a training set, uses it to predict labels for a test set, and then evaluates its performance using metrics such as accuracy, confusion matrix, and classification report. The Gaussian Naive Bayes algorithm is a probabilistic model that assumes features are independent and that their distribution follows a Gaussian or normal distribution. The model is trained on a set of labeled data and uses Bayes' theorem to calculate the probability of each class given a set of features.

The accuracy of the model is then calculated by comparing the predicted labels to the true labels in the test set. The confusion matrix is a table that shows the number of true positives, true negatives, false positives, and false negatives, which can be used to calculate metrics such as precision, recall, and F1 score. The classification report is a summary of these metrics for each class, as well as the overall accuracy of the model.

##(5)Random Forest

n_estimators: The number of trees in the forest (default=100). In this case, 1000 trees will be used in the forest.

random_state: Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node. This ensures that the results are reproducible.
max_leaf_nodes: The maximum number of leaf nodes in each 


decision tree. This limits the depth of the tree, thereby avoiding overfitting.

min_samples_split: The minimum number of samples required to split an internal node. If a node has fewer samples than this number, it will not be split. This parameter also helps to prevent overfitting by setting a threshold on the minimum number of samples required to split a node.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Instantiate the model
rf = RandomForestClassifier(n_estimators=1000, random_state=1,max_leaf_nodes=20, min_samples_split=15)

# Train the model
rf.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = rf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)



# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)



# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Model[0] Testing Accuracy = {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n", classification_report(Y_test, Y_pred))

"""This code trains a Random Forest Classifier model on some training data (X_train and Y_train) and evaluates its performance on a test set (X_test and Y_test). The Random Forest Classifier is instantiated with some hyperparameters, such as the number of trees (n_estimators), the maximum number of leaf nodes per tree (max_leaf_nodes), and the minimum number of samples required to split a node (min_samples_split).

After training the model, it makes predictions on the test set and evaluates its performance using the accuracy score and a confusion matrix. The code also calculates the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) to further evaluate the performance of the model. Finally, it prints the results including the accuracy, confusion matrix, and a classification report.

##(6)Decision Tree
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Create a decision tree classifier object
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', min_samples_split=5,splitter='random', random_state=1)

# Fit the model on training data
dt.fit(X_train, Y_train)

# Predict on test data
y_pred = dt.predict(X_test)

# Calculate accuracy score
acc = accuracy_score(Y_test, Y_pred)




# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)



# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", acc)
print("Accuracy in percentage: {:.2f}%".format(acc * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))

print("classification report:\n", classification_report(Y_test, Y_pred))

"""This code performs binary classification using a decision tree classifier. It creates a decision tree classifier object with specified hyperparameters, fits the model on training data, predicts on the test data, and calculates the accuracy score of the model.

The confusion matrix is printed along with the number of true positives, true negatives, false positives, and false negatives. The classification report is also printed which includes precision, recall, F1-score, and support for each class.

##(7)Using xgboostClassifier of tree class to use Decision Tree Algorithm
"""

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Create XGBClassifier object and set hyperparameters
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, 
                        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)

# Fit the model on training data
model.fit(X_train, Y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(Y_test, Y_pred)




# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)


# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
#print("Accuracy:", accuracy)
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n",classification_report(Y_test, Y_pred))

"""This code uses the XGBoost library to create a gradient boosting classifier. The hyperparameters of the model are set and the model is fit on the training data. After training, the model makes predictions on the test data, and the accuracy score is calculated using the accuracy_score function from sklearn.metrics.

The confusion matrix is also printed, which shows the number of true positives, true negatives, false positives, and false negatives. Finally, the classification report is printed using the classification_report function from sklearn.metrics, which shows various classification metrics such as precision, recall, and F1-score for each class.

##(8)Using  SGDClassifier of tree class to use Decision Tree Algorithm
"""

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# Create an SGDClassifier object
clf = SGDClassifier()

# Fit the classifier to the training data
clf.fit(X_train, Y_train)

# Predict the labels of the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = clf.score(X_test, Y_test)

# Print the accuracy in percentage
# Evaluate the performance of the classifier
# print("Accuracy:", clf.score(X_test, Y_test))
# print("Accuracy:", round(accuracy * 100, 2), "%")


# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion Matrix:\n", cm)

# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n",classification_report(Y_test, Y_pred))

"""This code trains a Stochastic Gradient Descent (SGD) classifier using the SGDClassifier class from the sklearn.linear_model module. The classifier is trained on a training dataset, and then used to make predictions on a test dataset. The accuracy of the classifier is calculated using the accuracy_score function from the sklearn.metrics module, and the confusion matrix is generated using the confusion_matrix function from the same module. The true positives, true negatives, false positives, and false negatives are then calculated using the values in the confusion matrix. Finally, a classification report is generated using the classification_report function from the sklearn.metrics module, which shows the precision, recall, and f1-score for each class, as well as the support (the number of samples in each class).

##(9)Using  AdaBoostClassifier of tree class to use Decision Tree Algorithm
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# initialize the decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=1)

# initialize the adaboost classifier
ada_clf = AdaBoostClassifier(n_estimators=2000, random_state = 0)

# fit the model on training data
ada_clf.fit(X_train, Y_train)

# make predictions on test data
Y_pred = ada_clf.predict(X_test)

# evaluate model performance
accuracy = accuracy_score(Y_test, Y_pred)
# print("Accuracy:", accuracy)
# print("Accuracy:", round(accuracy * 100, 2), "%")


# print the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion Matrix:\n", cm)

# calculate accuracy, TP, TN, FP, FN
accuracy = accuracy_score(Y_test, Y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# print the results
print("Accuracy:", accuracy)
print("Accuracy in percentage: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", cm)
print("True Positives(TP) = {}".format(TP))
print("True Negatives(TN) = {}".format(TN))
print("False Positives(FP) = {}".format(FP))
print("False Negatives(FN) = {}".format(FN))
print("classification report:\n",classification_report(Y_test, Y_pred))

"""This code trains an AdaBoost classifier on a decision tree classifier with a max depth of 1. The trained model is then used to make predictions on test data, and the accuracy, confusion matrix, and classification report are printed. Specifically, the code imports necessary modules and initializes the decision tree classifier and AdaBoost classifier. Then, the AdaBoost classifier is fit to the training data and used to make predictions on test data. The accuracy of the classifier is calculated, and the confusion matrix, as well as true positives, true negatives, false positives, and false negatives are printed. Finally, a classification report is printed, providing precision, recall, and F1 score for each class."""