import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from solar_energy_forecast.preprocess import get_data

"""import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler"""

def main():
    df = get_data(verbose=True)

    # Set your target column
    target_col = 'relative_power'  # replace this with your actual column

    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop NaNs if any
    X = X.dropna()
    y = y.loc[X.index]

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)  # you can tune n_estimators
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # 1:1 line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Random Forest: Predicting {target_col}')
    plt.tight_layout()
    plt.savefig('output/Tree.png')
    plt.clf()


    # Print metrics
    print(f'R² Score: {r2_score(y_test, y_pred):.3f}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
    print(f'mean_absolute_error: {mean_absolute_error(y_test, y_pred):.3f}')

    """# Set your target column
    target_col = 'relative_power'  # replace this with your actual column

    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop NaNs if any
    X = X.dropna()
    y = y.loc[X.index]

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Random Forest model
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # 1:1 line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Random Hist: Predicting {target_col}')
    plt.tight_layout()
    plt.savefig('output/Hist.png')

    # Print metrics
    print(f'R² Score: {r2_score(y_test, y_pred):.3f}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')
    print(f'mean_absolute_error: {mean_absolute_error(y_test, y_pred):.3f}')



    # Optional: Log-transform the target
    log_transform = True
    if log_transform:
        y = np.log1p(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define a deeper NN model with dropout
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Add early stopping
    early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Predict
    y_pred = model.predict(X_test).flatten()

    # Optional: Reverse log-transform
    if log_transform:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Neural Network: Predicting {target_col}')
    plt.tight_layout()
    plt.savefig('output/NN.png')
    plt.clf()

    # Plot training/validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/loss_curve.png')
    plt.clf()

    # Print metrics
    print(f'R² Score: {r2_score(y_test, y_pred):.3f}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')"""

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    main()
