# Import necessary libraries
import pandas as pd  # Import Pandas for data manipulation
import seaborn as sns  # Import Seaborn for data visualization
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from sklearn.model_selection import train_test_split, GridSearchCV  # Import scikit-learn modules for model selection and evaluation
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix  # Import metrics for model evaluation
import joblib  # Import joblib for saving models

# Load your dataset 
df = pd.read_csv('Project.csv')

# Visualize the 3D data
# create three separate Pandas Series (x, y, z) from the DataFrame 'df', each containing the data from columns 'X', 'Y', and 'Z', respectively
x = df['X']
y = df['Y']
z = df['Z']

#set up a Matplotlib 3D plot. It creates a figure, adds a 3D subplot, and plots the data from 'x', 'y', and 'z' on the plot. 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='z vs. (x, y)', color='blue', marker='o', linestyle='-')

# add labels to the X, Y, and Z axes of the 3D plot, set a title, add a legend, enable the grid, and display the 3D plot.
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Line Plot of Z vs. (X, Y)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the correlation matrix
correlation_matrix1 = df[['X', 'Y', 'Z', 'Step']].corr()

# Create a heatmap for the correlation matrix using Seaborn
# They set the figure size, annotations, color map, formatting, and then display the heatmap.
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Define your features (X) and target variable (y)
X = df[['X', 'Y', 'Z']]
y = df['Step']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models, creates instances of three different classifiers
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()
svm_model = SVC()

# Define hyperparameter grids for tuning the Random Forest Classifier
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt']  
}

# Define hyperparameter grids for tuning the Decision Tree Classifier
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Define hyperparameter grids for tuning the Support Vector Classifier
svm_param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Create GridSearchCV objects for each of the 3 models
# 'cv=5' specifies 5-fold cross-validation
rf_grid_search = GridSearchCV(rf_model, param_grid=rf_param_grid, cv=5)
dt_grid_search = GridSearchCV(dt_model, param_grid=dt_param_grid, cv=5)
svm_grid_search = GridSearchCV(svm_model, param_grid=svm_param_grid, cv=5)

# Fit the models to the training data using grid search
rf_grid_search.fit(X_train, y_train)
dt_grid_search.fit(X_train, y_train)
svm_grid_search.fit(X_train, y_train)

# Decide and retrieve the best hyperparameters
best_rf_params = rf_grid_search.best_params_
best_dt_params = dt_grid_search.best_params_
best_svm_params = svm_grid_search.best_params_

# create new instances of each model with the best hyperparameters found during the grid search.
best_rf_model = RandomForestClassifier(**best_rf_params)
best_dt_model = DecisionTreeClassifier(**best_dt_params)
best_svm_model = SVC(**best_svm_params)

# fit the best models to the training data using the best hyperparameters, ensuring that the models are now optimized for performance
best_rf_model.fit(X_train, y_train)
best_dt_model.fit(X_train, y_train)
best_svm_model.fit(X_train, y_train)

# make predictions on the test set using the best models for the 3 models
rf_pred = best_rf_model.predict(X_test)
dt_pred = best_dt_model.predict(X_test)
svm_pred = best_svm_model.predict(X_test)

# calculate various performance metrics (F1 score, precision, and accuracy) for each model on the test set using the 'micro' average, which considers all classes equally. The results are stored in variables for further analysis.
rf_f1 = f1_score(y_test, rf_pred, average='micro')
rf_precision = precision_score(y_test, rf_pred, average='micro')
rf_accuracy = accuracy_score(y_test, rf_pred)

dt_f1 = f1_score(y_test, dt_pred, average='micro')
dt_precision = precision_score(y_test, dt_pred, average='micro')
dt_accuracy = accuracy_score(y_test, dt_pred)

svm_f1 = f1_score(y_test, svm_pred, average='micro')
svm_precision = precision_score(y_test, svm_pred, average='micro')
svm_accuracy = accuracy_score(y_test, svm_pred)

# create a dictionary called model_scores to store performance metrics and models for different machine learning models. The keys are model names, and the values are tuples containing F1 score, precision, accuracy, and the corresponding best model.
model_scores = {
    "Random Forest": (rf_f1, rf_precision, rf_accuracy, best_rf_model),
    "Decision Tree": (dt_f1, dt_precision, dt_accuracy, best_dt_model),
    "Support Vector Machine": (svm_f1, svm_precision, svm_accuracy, best_svm_model),
}

# Determine the best model based on the highest F1 score

best_model = max(model_scores, key=lambda k: model_scores[k][0])  # Select based on F1 score

# Print performance metrics for all 3 models
print(f"Random Forest - F1 Score: {rf_f1}, Precision: {rf_precision}, Accuracy: {rf_accuracy}")
print(f"Decision Tree - F1 Score: {dt_f1}, Precision: {dt_precision}, Accuracy: {dt_accuracy}")
print(f"Support Vector Machine - F1 Score: {svm_f1}, Precision: {svm_precision}, Accuracy: {svm_accuracy}")

#prints out the best performace model 
print(f"The best model is {best_model}")

# Save the best model in a joblib format
best_model_filename = "best_model.joblib"
joblib.dump(model_scores[best_model][3], best_model_filename)

print(f"The best model ({best_model}) has been saved as {best_model_filename}")

# Create a confusion matrix for the best model
# making predictions on the test data and comparing them with the actual labels
y_pred = model_scores[best_model][3].predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix as a heatmap
# helps you understand the model's performance in terms of true positives, false positives, true negatives, and false negatives.
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for {best_model}')
plt.show()

# Data points for prediction
data_points = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

# Make predictions using the best model
predictions = model_scores[best_model][3].predict(data_points)

# Print the predicted maintenance steps for each data point
for i, data_point in enumerate(data_points):
    print(f"Data Point {i+1}: Features = {data_point}, Predicted Step = {predictions[i]}")
