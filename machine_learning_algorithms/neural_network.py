import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Neural Network
"""
    - Consists of layers nodes and edges connecting the nodes
        - Input layer has node corresponding to each variable
        - Hidden layers contain many nodes and allow the model to capture more complex shapes within the data
        - Output layer has a node for each possible outcome (e.g. 2 for binary classification)
    - The nodes in the hidden layers contain activation functions (e.g. softplus, relu or sigmoid) and the values passed in are x-axis
      coordinates for these functions, the output is the corresponding y-axis coordinate according to the function
    - Values are passed along the network
        - When they pass along a branch there is a linear equation ax + b where a are called weights and b are called biases
        - This value is then passed into the activation function at the node
        - Ultimately the values are combined to create a prediction
    - The number of layers and the nodes in each layer is found by trial and error, generally more layers/nodes are needed to model more
      complex relationships but can lead to over fitting
    - Effectively the weights and bias stretch, flip and combine the activation functions to create more complex shapes to fit to the data

    Gradient Decent
    - Gradient decent is used when we want to find the optimal value for a weight or bias in a neural network
    - The optimal value is the value that minimises the sum of squared residuals
    - We could use trial and error by plugging in lots of values for the weight/bias and recalculating the sum of squared residuals but this
          would be slow
    - We use the chain rule to take the derivative of the equation for the sum of squared residuals with respect to the weight/bias
    - We are looking for the value of the weight/bias such that the derivative of the curve of the sum of squared residuals is 0
    - We plug the initial value for the weight/bias into this equation and this gives us the slop of the curve of sum of squared
      residuals against the weight/bias at the value of the weight/bias
    - We multiple this slop value by the learning rate (typically 0.1) and to compute the step size
    - We subtract the step size from current guess of the value and this becomes our new estimate of the weight/bias
    - We repeat the process, again using our new estimate to make predictions and then plugging these into the equation for the derivative
      of the curve of the sum of squared residuals to get its slop at our estimate of the weight/bias
    - We can then take another step (usually smaller) towards the minimum point of the curve
    - We stop when the step size gets below a certain threshold


    Back Propagation
    1. Initialise the weights and biases with starting values
        - Typically we set biases to 0 and weights are drawn from a standard random normal distribution
    2. Use gradient decent to optimise the value of each parameter
    3, Repeat until all parameters have been optimised

    Argmax & softmax
    - When we have multi-class classification we will have one output node for each class
    - However the output values will not necessarily be between 0 and 1 (i.e. they don't map to probabilities)
    - So we need a way to make the values more interpretable
    - Argmax set the largest value to 1 and all others to 0 but cannot be used for optimising the weights and biases since its derivative
      is 0
    - Softmax is an increasing function (so it preserves the order of the original output values) which has a non-zero derivative
    - Softmax = e^a / (e^a + e^b + ... e^n) where a,...,n are the values from each of the output nodes and a is the current output node
    - Softmax also ensures all values are between 0 and 1
    - The output values are not probabilities since they depend on the randomly selected initial weights and biases; indeed different
      initial weights and biases could be just as good at classifying the data but could lead to different output values
    - Softmax can be used for back propagation (unlike argmax) since its derivative is not always 0
    - Often softmax is used for training (since it can be used in backpropagation) but argmax is used for testing since it can be more
      easily interpreted

    Cross Entropy
    - For multiclass classification we use cross entropy as the cost function instead of SSR
    - Cross entropy = sum(-log(predicted probability for observed class))
    - We could use SSR but its derivative between 0 and 1 is small compared to cross entropy
    - So cross entropy helps us to take larger steps during gradient devent

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

# Build the neural network model
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Input layer
    Dense(16, activation='relu'),  # Hidden layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))