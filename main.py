import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
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

    # Initialize and train Ridge model
    ridge = Ridge(alpha=1.0)  # you can tune alpha
    ridge.fit(X_train, y_train)

    # Predict
    y_pred = ridge.predict(X_test)

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # 1:1 line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Ridge Regression: Predicting {target_col}')
    plt.tight_layout()
    plt.savefig('output/all.png')

    # Print metrics
    print(f'R² Score: {r2_score(y_test, y_pred):.3f}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}')

    """df.to_csv("output/complete_data.csv")

    df["hour"] = df.index.hour
    sns.lineplot(df, y="relative_power", x="hour", errorbar="sd")
    plt.savefig('output/relative_power_by_hour.png')
    plt.clf()

    df["day_of_year"] = df.index.day_of_year
    df["date"] = df.index.date
    df["week"] = df.index.day_of_year // 7
    sns.lineplot(
        df.groupby("date").agg(avg_relative_power=("relative_power", "mean"), week=("week", "first")),
        y="avg_relative_power", x="week", errorbar="sd", estimator='mean')
    plt.savefig('output/avg_relative_power_by_week.png')
    plt.clf()

    X = df.select_dtypes(include='number').drop(columns="relative_power")
    y = df["relative_power"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred = np.where(y_pred > 0, y_pred, 0)

    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    normlized_rmse = np.mean(
        mean_squared_error(y_test, y_pred, multioutput='raw_values') /
        np.max(np.stack([y_pred, y_test.to_numpy()]), axis=0)
    )
    {"RMSE": rmse, "MAE": mae, "nRMSE": float(normlized_rmse)}

    df_params = X_train.mean().rename("mean_value").to_frame()
    df_params["coef"] = clf.coef_
    df_params["mean_weight"] = df_params["coef"] * df_params["mean_value"]
    df_params.sort_values("mean_weight", ascending=False, inplace=True)

    df_params[df_params["mean_weight"] > 0]["mean_weight"].plot(kind="bar")
    plt.savefig('output/test.png')


    x = df['relative_power']
    y = df['target_column']

    plt.scatter(x, y)
    plt.xlabel('some_column')
    plt.ylabel('target_column')
    plt.title('Scatter plot of X vs Y')
    plt.show()


    target_col = 'relative_power'  # Replace with your actual target column name

    for col in df.columns:
        if col == target_col:
            continue  # Skip plotting the target column against itself

        plt.figure(figsize=(6, 4))
        plt.scatter(df[col], df[target_col])
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.title(f'{col} vs {target_col}')
        plt.tight_layout()
        plt.savefig(f'output/{col}.png')
        plt.clf()"""

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    main()
