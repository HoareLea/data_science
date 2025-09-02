import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# Support Vector Machines
"""
    - The main idea behind support vector machines is to try to fit a shape to the data which divides the data into groups
        - 1 dimensional data -> fit a point
        - 2 dimensional data -> fit a line
        - 3 dimensional data -> fit a plane
        - n dimensional data -> fit an n-1 dimensional shape
    -
"""

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Drop rows with missing values (SCV can't handle these)
df.dropna(inplace=True)

# Encode the 'Sex' and 'Embarked' column
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Define the target variable and feature set
y = df['Survived']
X = df.drop(columns=['Survived'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Initialize the Random Forest Classifier
svm = SVC(
    C=1.0,
    kernel="rbf",
    degree=3,
    gamma="scale",
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape="ovr",
    break_ties=False,
    random_state=None,
)

# Train the model
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))