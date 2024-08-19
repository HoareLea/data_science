import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Decision Tree
"""
    - The decision tree algorithm can be used for regression (i.e. predicting continuous values) or classification (i.e. predicting discrete
    values)
    - A decision tree is made up of nodes and branches where each node represents a statement about a variable and the branches correspond
      to whether that statement is true or false about the given data
    - At each node, the data is split into two mutually exclusive groups
    - The top level node is called the root, the nodes below that are decision nodes and the final nodes are leaf nodes
    - The leaf nodes simply correspond to data rather than a statement
    - The main idea is that each node splits the data into smaller partitions of similar data
    - Predictions can then be made by following the decision tree and answering the questions about the given data you want to make a
    prediction for
    - When predicting a continuous value, some kind of summary metric of the data in the leaf node is used such as the mean


    1. Choose a root node variable and criteria
        - Take each variable in the data set and use it to partition the data
        - If the variable is binary simply use a True/False condition (e.g. is male)
        - For discrete variables, one-hot encode them and then treat them as binary
        - If the variable is continuous, order it by size and test each pair-wise average
        - Evaluate the split using a chosen metric (e.g. Gini impurity, entropy or chi-squared for classification and mean square error for
          regression
        - Choose the best split according to the evaluation metric and choose the variable with the best overall score according to the 
          evaluation metric
    2. Recursively repeating the splitting
        - Move down into each decision node and repeat step 1, treating the decision node as a root node
    3. Stop when certain criteria are met
        - Maximum Tree Depth: Prevents the tree from growing too large and over fitting
        - Minimum Number of Samples per Split: Ensures that each split has enough samples to be statistically meaningful
        - Minimum Impurity Decrease: Splitting stops if the decrease in impurity is below a threshold
    4. Prune the tree (optional)
        - Instead of applying stopping criteria instead allow the model to grow to its full depth with no constraints (this will result in
          a highly complex and likely overfit model
        - Evaluate the tree on validation data using a performance metric (e.g. accuracy for classification or mean square error for 
          regression)
        - Loop over each pair of leaf nodes and replace the corresponding decision node with a leaf node formed by combining its two leaf
          nodes
        - Re-evaluate the performance of each of these resultant trees using the same performance metric as before
        - Compute the cost for each tree by using the formula performance_metric + penalty * number_of_leaves
        - Choose the tree with the lowest cost
        - Repeat the process of combining leaves and computing the cost until no further improvements are made
        - This should simplify the tree and thus help to prevent over fitting
        - Note the penalty should be chosen based on how much you want to penalise the model for over fitting
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

# Initialize the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=72,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
)

# Train the model
decision_tree.fit(X_train, y_train)

# Make predictions
y_pred = decision_tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
