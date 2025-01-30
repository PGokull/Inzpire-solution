import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

def train_and_predict(train_file, test_file, output_file):
    try:
        # Load and prepare training data
        data = pd.read_csv(train_file)
        X = data.drop(['GrossWeight', "_id_x", "Crate"], axis=1)
        y = data["GrossWeight"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = SVR(kernel='linear', C=8, epsilon=0.09)
        model.fit(X_train, y_train)
        
        # Load and prepare test data
        test_data = pd.read_csv(test_file)
        X_new = test_data[X.columns]
        
        # Make predictions
        predictions = model.predict(X_new)
        
        # Save predictions to an Excel file
        results_df = pd.DataFrame({'Predicted GrossWeight': predictions})
        results_df.to_excel(output_file, index=False)
        
        print(f"Predictions saved successfully to {output_file}")
        return predictions
    except Exception as e:
        print(f"Error in training or inference: {e}")
        return None
