import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Logistic Regression
"""
    - Logistic regression is similar to linear regression but unlike linear regression, which predicts a continuous outcome, logistic 
      regression predicts discrete outcomes (0 or 1, True or False)
    - Instead of fitting a straight line to the data, we aim to fit an s-shaped logistic function
    - The logistic function outputs a value between 0 and 1, we choose a threshold value (usually 0.5) to decide which category the 
      prediction falls into
    - The probability is constrained to be between 0 and 1 but if we transform the y-axis using the log_odds = log(p/1-p) then the y-axis 
      can take any value, thus allowing us to use all the same theory from standard linear models
    - Maximum likelihood is used instead of least squares to fit the logistic function to the data
        - Since the log_odds of 0 or 1 is -inf or +inf, we cannot calculate residuals and therefore we can't use least squares
        - Plot a logistic function and compute the likelihood of observing each sample given the logistic function
        - Multiply each of these together (assume independence) to get the overall likelihood of observing the data
        - Move the logistic function and recalculate the likelihood given the new logistic function
        - Choose the logistic function which maximises the overall likelihood
    
"""

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Drop rows with missing values (Logistic Regression can't handle these)
df.dropna(inplace=True)

# Encode the 'Sex' and 'Embarked' columns
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Define the target variable and feature set
y = df['Survived']
X = df.drop(columns=['Survived'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Initialize the Logistic Regression model
logistic_regression = LogisticRegression(
    penalty='l2',  # Regularization term
    dual=False,
    tol=1e-4,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver="lbfgs",
    max_iter=100,
    multi_class="deprecated",
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
)

# Train the model
logistic_regression.fit(X_train, y_train)

# Make predictions
y_pred = logistic_regression.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
