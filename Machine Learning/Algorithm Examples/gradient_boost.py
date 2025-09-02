import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

# Gradient Boost
"""
    - Similarly to random forest, gradient boost works by building multiple decision trees
    - However, unlike random forest, gradient boost does not build the decision trees independently of one another, each tree tries to
      predict the errors of the previous tree

    1. Generate an initial prediction
        - For a continuous response variable use the mean
        - For a binary response use the log odds = log( mean / (1 - mean) ) treating true as 1 and false as 0
    2. Calculate the residual for each sample using the initial prediction as the predicted value for each
        - residual = actual - predicted
    3. Build a new tree which predicts the residuals
        - Rather than predicting the response, this tree should try to predict the residuals from the previous round
        - We can apply constraints here to limit the number of variables, layers, leaves etc
    4. Combine the trees using a learning rate to make further predictions
        - Take the initial prediction and add the result of the next tree scaled by a learning rate (typically 0.1)
        - The learning rate is designed to limit the influence of the new tree and this prevent over fitting
    5. Repeat to build more trees
        - Each time a new tree is created, we scale its prediction by the learning rate and add it to the result from all other trees
        - We stop after a given number of trees have been created or the residuals don't improve by a given threshold


    - When the response is binary we cannot simply take the output value from a leaf and combine it with the initial prediction because the
      initial predicition is in terms of the log odds but the tree outputs a probability
    - To address this, we need to transform the probability output by a tree using:
        sum(residuals) / sum(previous_probabilities * (1 - previous_probabilities))
    - When we generate predictions they are in terms of the log odds so to extract the probability of being in the positive class use:
        e^log_odds / (1 + e^log_ods)
    - We can then use these predicted probabilities to calculate residuals (using 1 for positive and 0 for negative actual values)

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

# Initialize the Gradient Boosting Classifier (use HistGradientBoostingClassifier as it can deal with NaNs)
gradient_boost = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_features=1.0,
    max_bins=255,
    categorical_features="warn",
    monotonic_cst=None,
    interaction_cst=None,
    warm_start=False,
    early_stopping="auto",
    scoring="loss",
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-7,
    verbose=0,
    random_state=None,
    class_weight=None,
)

# Train the model
gradient_boost.fit(X_train, y_train)

# Make predictions
y_pred = gradient_boost.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

