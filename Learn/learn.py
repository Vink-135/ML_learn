# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures

# # Placeholder data and models (replace with your actual data and models)
# X_train = np.random.uniform(-1, 1, (50, 1))
# y_train = np.sin(X_train * np.pi) + np.random.normal(0, 0.1, X_train.shape)
# X_test = np.random.uniform(-1, 1, (20, 1))
# y_test = np.sin(X_test * np.pi) + np.random.normal(0, 0.1, X_test.shape)

# # Fit linear model
# linear_model = LinearRegression().fit(X_train, y_train)
# # Fit polynomial model
# poly = PolynomialFeatures(degree=15)
# X_train_poly = poly.fit_transform(X_train)
# poly_model = LinearRegression().fit(X_train_poly, y_train)

# # Make smooth X range for prediction curve
# X_smooth = np.linspace(-1, 1, 200).reshape(-1, 1)
# X_smooth_poly = poly.transform(X_smooth)

# y_smooth_linear = linear_model.predict(X_smooth)
# y_smooth_poly = poly_model.predict(X_smooth_poly)

# plt.figure(figsize=(10, 5))
# plt.scatter(X_train, y_train, label='Training Data', color='blue')
# plt.scatter(X_test, y_test, label='Test Data', color='orange')
# plt.plot(X_smooth, y_smooth_linear, color='green', label='Linear Regression')
# plt.plot(X_smooth, y_smooth_poly, color='red', label='Polynomial Regression (Degree=15)')
# plt.title("Overfitting Example")
# plt.legend()
# plt.grid(True)
# plt.show()




# # Plot decision boundary
# plt.figure(figsize=(8,6))
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr', alpha=0.8)
# plt.title("Logistic Regression - Binary Classification")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.grid(True)
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix

# # 1️⃣ Generate binary classification data
# X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, random_state=42)

# # 2️⃣ Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 3️⃣ Fit Logistic Regression
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 4️⃣ Predict and evaluate
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # 5️⃣ Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()






# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report

# # Load dataset (3-class classification)
# data = load_iris()
# X = data.data
# y = data.target

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = GaussianNB()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=data.target_names))




# for multinomial



# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # Sample data
# texts = ["free money win", "hello friend", "project deadline", "win cash", "team meeting"]
# labels = [1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

# # Convert text to features
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(texts)

# # Train Naive Bayes
# model = MultinomialNB()
# model.fit(X, labels)

# # Predict new message
# new_text = vectorizer.transform(["free cash"])
# print("Spam probability:", model.predict(new_text)[0])


# Decision tree Regression



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error


# # Generate some non-linear data
# X = np.sort(np.random.rand(100, 1) * 10, axis=0)
# y = np.sin(X).ravel() + np.random.randn(100) * 0.1  # sin wave + some noise


# model = DecisionTreeRegressor(max_depth=3, random_state=0)
# model.fit(X, y)


# # Predict on a smooth X range
# X_test = np.linspace(0, 10, 500).reshape(-1, 1)
# y_pred = model.predict(X_test)

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.scatter(X, y, s=20, label="Training Data")
# plt.plot(X_test, y_pred, color="red", linewidth=2, label="Prediction")
# plt.title("Decision Tree Regression")
# plt.xlabel("X")
# plt.ylabel("y")
# plt.legend()
# plt.grid(True)
# plt.show()


# # Evaluate the model
# y_train_pred = model.predict(X)
# mse = mean_squared_error(y, y_train_pred)
# print("MSE on training data:", round(mse, 4))




# gini vs entropy + pre -pruning


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load dataset
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Gini-based tree (default)
# gini_tree = DecisionTreeClassifier(criterion='gini', max_depth=3)
# gini_tree.fit(X_train, y_train)
# print("Gini Accuracy:", accuracy_score(y_test, gini_tree.predict(X_test)))

# # Entropy-based tree
# entropy_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
# entropy_tree.fit(X_train, y_train)
# print("Entropy Accuracy:", accuracy_score(y_test, entropy_tree.predict(X_test)))



# post pruning


# # Full tree first
# full_tree = DecisionTreeClassifier(random_state=0)
# full_tree.fit(X_train, y_train)

# # Get effective alphas and pruning path
# path = full_tree.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas = path.ccp_alphas

# # Prune with different alphas
# for alpha in ccp_alphas:
#     pruned_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
#     pruned_tree.fit(X_train, y_train)
#     print(f"Alpha: {alpha:.5f} → Accuracy: {accuracy_score(y_test, pruned_tree.predict(X_test)):.2f}")


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Generate a classification dataset
# X, y = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=0, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a base tree to get ccp_alphas
# clf = DecisionTreeClassifier(random_state=0)
# path = clf.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas = path.ccp_alphas

# alphas = []
# accuracies = []

# for alpha in ccp_alphas:
#     tree = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
#     tree.fit(X_train, y_train)
#     acc = accuracy_score(y_test, tree.predict(X_test))
#     alphas.append(alpha)
#     accuracies.append(acc)

# plt.plot(alphas, accuracies, marker='o')
# plt.xlabel("ccp_alpha")
# plt.ylabel("Accuracy")
# plt.title("Pruning Path")
# plt.grid(True)
# plt.show()



# //single perceptron


# import numpy as np

# # Inputs (x1, x2)
# x = np.array([6, 7])  # Good comm. and tech skills

# # Weights
# w = np.array([0.6, 0.8])

# # Bias
# b = -8

# # Step function
# def step(z):
#     return 1 if z >= 0 else 0

# # Weighted sum
# z = np.dot(w, x) + b
# output = step(z)

# print("Candidate Score:", z)
# print("Hire (1) or Reject (0):", output)


# we can use the tts method too with importing Perceptron from sklearn.linear_model import perceptron

# code for simple perceptron to check inout and output predictions

# import numpy as np

# # Input layer: 3 features
# X = np.array([1, 2, 3])  # x1, x2, x3

# # Weights and bias for 2 neurons in hidden layer
# W_hidden = np.array([[0.1, 0.2, 0.3],   # Neuron 1 weights
#                      [0.4, 0.5, 0.6]])  # Neuron 2 weights
# b_hidden = np.array([0.1, 0.2])         # Bias for each neuron

# # Activation function (ReLU)
# def relu(z):
#     return np.maximum(0, z)

# # Forward pass to hidden layer
# z_hidden = np.dot(W_hidden, X) + b_hidden
# a_hidden = relu(z_hidden)

# # Output layer weights and bias (single neuron)
# W_output = np.array([0.7, 0.8])  # weights from H1 and H2
# b_output = 0.3

# # Final output
# z_output = np.dot(W_output, a_hidden) + b_output
# output = relu(z_output)

# print("Hidden layer output:", a_hidden)
# print("Final prediction:", output)


#front and back-propogation

import numpy as np

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)  # derivative of sigmoid

# Input dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output labels (XOR task)
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases randomly
np.random.seed(42)
w1 = np.random.rand(2, 4)  # 2 inputs to 4 neurons
b1 = np.zeros((1, 4))
w2 = np.random.rand(4, 1)  # 4 neurons to 1 output
b2 = np.zeros((1, 1))

# Training loop
for epoch in range(10000):

    # FORWARD PASS
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # LOSS (Mean Squared Error)
    loss = np.mean((y - a2) ** 2)

    # BACKWARD PASS
    d_loss_output = (a2 - y) * sigmoid_derivative(a2)

    d_w2 = np.dot(a1.T, d_loss_output)
    d_b2 = np.sum(d_loss_output, axis=0, keepdims=True)

    d_hidden = np.dot(d_loss_output, w2.T) * sigmoid_derivative(a1)

    d_w1 = np.dot(X.T, d_hidden)
    d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

    # UPDATE WEIGHTS (Gradient Descent)
    lr = 0.1
    w2 -= lr * d_w2
    b2 -= lr * d_b2
    w1 -= lr * d_w1
    b1 -= lr * d_b1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")