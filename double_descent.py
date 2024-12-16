# I have to investigate double descent phenomenon in the context of least-squares linear regression.

# I have to demonstrate it by training linear models using gradient descend with the model parameters initialized to 0 and increasing it for different values of the regularization parameter.

# I will use the following steps:

# 1. Generate a dataset with 1000 samples and 1000 features.

# 2. Split the dataset into training and testing sets.

# 3. Train a linear model using gradient descent with the model parameters initialized to 0 and increasing it for different values of the regularization parameter.

# 4. Evaluate the model on the testing set.

# 5. Repeat steps 3 and 4 for different values of the regularization parameter.

# 6. Plot the test error as a function of the regularization parameter.

# 7. Explain the double descent phenomenon.

# Let's start by importing the required libraries:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Step 1: Generate a dataset with 1000 samples and 1000 features.
# np.random.seed(42)
n_samples = 50
n_features = 500
noise = 0.2
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)
# y = np.dot(X, np.random.randn(n_features)) + noise * np.random.randn(n_samples)

# Step 2: Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 3: Train a linear model using gradient descent with the model parameters initialized to 0 and increasing it for different values of the regularization parameter.
# alphas = np.logspace(2, 10, 5)
alphas = np.logspace(-4, 6, 5)
train_errors = []
test_errors = []

for alpha in alphas:
    for d in range(1, n_features + 1):
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train[:, :d], y_train)
        y_pred = model.predict(X_train[:, :d])
        train_error = mean_squared_error(y_train, y_pred)
        y_pred = model.predict(X_test[:, :d])
        test_error = mean_squared_error(y_test, y_pred)
        train_errors.append(train_error)
        test_errors.append(test_error)

# Plot the double descent curve for each regularization parameter.
fig, ax = plt.subplots(1, 5, figsize=(15, 5))
for i, alpha in enumerate(alphas):
    print(i, alpha)
    ax[i].plot(range(1, n_features + 1), test_errors[i*n_features:i*n_features + n_features], label=f'Test')
    ax[i].plot(range(1, n_features + 1), train_errors[i*n_features:i*n_features + n_features], label=f'Train')
    ax[i].set_xlabel('Number of features')
    ax[i].set_ylabel(f'Train and Test errors with alpha={alpha}')
    # ax[i].set_ylim(0, 2000)
    ax[i].legend()

plt.tight_layout()
plt.show()

