import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(input_file_path, model_save_path, results_save_path, metrics_save_path, graph_save_path):
    try:
        data = pd.read_csv(input_file_path)
        print(data.info())
        print(data.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    categorical_columns = ['Crate']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    try:
        encoded_columns = encoder.fit_transform(data[categorical_columns])
        encoded_column_names = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names, index=data.index)
        data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)
    except Exception as e:
        print(f"Error in one-hot encoding: {e}")
        return

    X = data.drop(['GrossWeight', "_id"], axis=1)
    y = data["GrossWeight"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVR(kernel='linear', C=8, epsilon=0.09)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error in training model: {e}")
        return

    try:
        joblib.dump(model, model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")
        return

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)

    metrics = {
        "Root Mean Squared Error (RMSE)": rmse,
        "Mean Absolute Error (MAE)": mae,
        "R^2 Score": r2,
        "Explained Variance Score (EVS)": evs,
        "Median Absolute Error (MedAE)": medae
    }

    try:
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        metrics_df.to_csv(metrics_save_path, index=False)
        print(f"Metrics saved to {metrics_save_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    result_df = X_test.copy()
    result_df['ActualGrossWeight'] = y_test
    result_df['PredictedGrossWeight'] = y_pred

    try:
        result_df.to_csv(results_save_path, index=False)
        print(f"Predicted results saved to {results_save_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual GrossWeight')
    plt.ylabel('Predicted GrossWeight')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='green')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')

    plt.tight_layout()
    plt.savefig(f"{graph_save_path}/graphs.png")
    print(f"Graphs saved to {graph_save_path}/graphs.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.6, color='purple')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Residuals vs Actual GrossWeight')
    plt.xlabel('Actual GrossWeight')
    plt.ylabel('Residuals')
    plt.savefig(f"{graph_save_path}/residuals_vs_actual.png")
    print(f"Residuals vs Actual plot saved to {graph_save_path}/residuals_vs_actual.png")

    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(f"{graph_save_path}/correlation_heatmap.png")
    print(f"Correlation heatmap saved to {graph_save_path}/correlation_heatmap.png")

input_file_path = r"D:\Inzpire-Solutions\gross-weight-workspace\merged_data_original_transformed.csv"
model_save_path = r"D:\Inzpire-Solutions\gross-weight-workspace\svr_model.pkl"
results_save_path = r"D:\Inzpire-Solutions\gross-weight-workspace\predicted_results.csv"
metrics_save_path = r"D:\Inzpire-Solutions\gross-weight-workspace\metrics.csv"
graph_save_path = r"D:\Inzpire-Solutions\gross-weight-workspace\graphs"

train_and_evaluate(input_file_path, model_save_path, results_save_path, metrics_save_path, graph_save_path)
