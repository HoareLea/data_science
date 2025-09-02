import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Random Forest
"""
    - The decision tree algorithm is generally quite a poor performing model and is often prone to over fitting
    - The random forest model seeks to address the short-comings of using a single decision tree by building many decision trees and
      aggregating the predictions


    1. Create a bootstrapped data set
        - Sample the data with replacement to build a 'bootstrapped' data set of the same size as the original data where each entry could 
          appear multiple times and not all data points are necessarily included in the bootstrapped data
    2. Build a decision tree using the bootstrapped data and a subset of variables
        - Choose a subset of the variables and use these to build a decision tree based on the bootstrapped data
    3. Repeat to build many trees (i.e. a forest)
    4. Evaluate performance using the random forest
        - Use every tree in the forest to make a prediction and aggregate the results to get an overall prediction
        - For a discrete response you could use the mode
        - For a continuous response you could use the mean or median

    - Ideally we would be able to train each decision tree on a completely new data set, however in practice this isn't usually possible
      (e.g. due to the impracticalities of obtaining more data); bootstrapping is the next best option, giving us quasi-new data to train
      each tree with
    - The process of building a bootstrapped data set and aggregating the predictions across multiple models is called bagging
    - By spreading out the predictions across multiple trees we diversify the prediction and generally improve its accuracy versus using a
      single tree
    
"""

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Encode the 'Sex' and 'Embarked column
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Define the target variable and feature set
y = df['Survived']
X = df.drop(columns=['Survived'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Initialize the Random Forest Classifier
random_forest = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None,
)

# Train the model
random_forest.fit(X_train, y_train)

# Make predictions
y_pred = random_forest.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
