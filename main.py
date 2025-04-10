import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from solar_energy_forecast.preprocess import get_data


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

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    main()
