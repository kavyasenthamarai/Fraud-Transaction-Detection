# Fraud-Transaction-Detection

## Data Loading:
The credit card transactions dataset is loaded into a Pandas DataFrame (df) from a CSV file.
## Handling Missing Values:
Any missing values in the dataset are dropped.
## Data Preparation:
Features (X) and the target variable (y) are separated.
The dataset is split into training and testing sets using the train_test_split function.
## Data Standardization:
The features are standardized using StandardScaler.
## Handling Class Imbalance:
Synthetic Minority Over-sampling Technique (SMOTE) is applied to address class imbalance in the training set.
## Model Building:
A Random Forest Classifier is created with a default configuration.
## Hyperparameter Tuning:
GridSearchCV is used for hyperparameter tuning. The grid search is performed over the number of estimators and maximum depth of the trees in the Random Forest.
## Model Training:
The Random Forest model is trained with the best hyperparameters obtained from the grid search.

# code:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Load your credit card transactions dataset
# Assuming the dataset has columns 'V1', 'V2', ..., 'V28' for features and 'Class' for the target variable

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('/content/creditcard.csv')

# Handle missing values if any
df = df.dropna()

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_resampled, y_resampled)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
best_rf_classifier = grid_search.best_estimator_
best_rf_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = best_rf_classifier.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
misclassified_indices = np.where(y_test.values.flatten() != y_pred.flatten())[0]
misclassified_transactions = X_test.iloc[misclassified_indices]
print("Misclassified Transactions:")
print(misclassified_transactions)
```
## Output:

![image](https://github.com/kavyasenthamarai/Fraud-Transaction-Detection/assets/118668727/b05359b9-ea41-411f-8613-06a736d89eba)


