import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Linear_Regression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)

        for _ in range(self.n_iterations):
            Y_prediction = self.predict(X)
            dw = (1 / n_samples) * np.dot(X.T, (Y_prediction - Y))
            db = (1 / n_samples) * np.sum(Y_prediction - Y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

if __name__ == "__main__":
    train = pd.read_csv("train_processed.csv")

    print("Original Train Shape:", train.shape)
    print(train.head())

    X = train.drop("count", axis=1)
    Y = train["count"]

    print("Features Shape:", X.shape)
    print("Target Shape:", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test:", Y_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Linear_Regression(
        learning_rate=0.01,
        n_iterations=1000
    )

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    mae = np.mean(np.abs(Y_test - predictions))
    print("\nMAE:", mae)

    mse = np.mean((Y_test - predictions) ** 2)
    print("MSE:", mse)

    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    r2 = 1 - (
        np.sum((Y_test - predictions) ** 2)
        /
        np.sum((Y_test - Y_test.mean()) ** 2)
    )
    print("R²:", r2)

    print("Sample Predictions:")
    print(predictions[:10])

    print("\nActual Values:")
    print(Y_test.values[:10])

    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, predictions, alpha=0.5, color='dodgerblue', label='Model Predictions')

    min_val = np.min(Y_test)
    max_val = np.max(Y_test)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

    plt.title('Actual vs. Predicted Bike Rental Demand')
    plt.xlabel('Actual Demand (Y_test)')
    plt.ylabel('Predicted Demand (predictions)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('actual_vs_predicted_final.png', bbox_inches='tight')
    plt.show()