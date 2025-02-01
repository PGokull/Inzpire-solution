from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

app = Flask(__name__)

def train_and_predict(train_file, test_file, output_file):
    try:
        data = pd.read_csv(train_file)
        X = data.drop(['GrossWeight', "_id_x", "Crate"], axis=1)
        y = data["GrossWeight"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SVR(kernel='linear', C=8, epsilon=0.09)
        model.fit(X_train, y_train)
        
        test_data = pd.read_csv(test_file)
        
        missing_cols = set(X.columns) - set(test_data.columns)
        for col in missing_cols:
            test_data[col] = 0
        
        test_data = test_data[X.columns]
        
        predictions = model.predict(test_data)
        
        results_df = pd.DataFrame({'Predicted GrossWeight': predictions})
        results_df.to_csv(output_file, index=False)
        
        print(f"Predictions saved successfully to {output_file}")
        return predictions
    except Exception as e:
        print(f"Error in training or inference: {e}")
        return None

def transform_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    required_columns = ['_id', 'Crate', 'Content', 'NetWeight', 'GrossWeight']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The dataset must contain the '{col}' column.")
    
    df['GrossWeight'] = df['GrossWeight'].fillna(0)
    df = df[df['NetWeight'] < df['GrossWeight']]
    
    weight_map = df[['_id', 'Crate', 'NetWeight', 'GrossWeight']].drop_duplicates(subset=['_id'])
    content_counts = df.groupby(['_id','Crate', 'Content']).size().reset_index(name='ContentCount')
    pivot_df = content_counts.pivot(index=['_id','Crate'], columns='Content', values='ContentCount').fillna(0)
    pivot_df = pd.merge(pivot_df, weight_map, on=['_id','Crate'], how='left')
    pivot_df.columns.name = None
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    pivot_df.to_csv(output_file, index=False)

def merge_crate_weight(input_file1, input_file2, output_file):
    data1 = pd.read_csv(input_file1)
    data2 = pd.read_csv(input_file2)
    combined_data = pd.merge(data1, data2, on='Crate', how='left')
    combined_data.dropna(inplace=True)
    combined_data.drop(columns=['_id','Crate'], axis=1, inplace=True)
    combined_data.to_csv(output_file, index=False)
    print("Done")

@app.route('/weightcheck', methods=['POST'])
def process():
    try:
        data = request.json
        input_path = data.get('input_path')
        crate_path = r"D:\Inzpire-Solutions\GrossWeightPrediction\crate.csv"
        jan_path = r"D:\Inzpire-Solutions\GrossWeightPrediction\outliers_free_jan_data_transformed.csv"
        output1 = "output.csv"
        output2 = "output2.csv"
        result = "result.csv"
        
        transform_data(input_path, output1)
        merge_crate_weight(output1, crate_path, output2)
        train_and_predict(jan_path, output2, result)
        
        return jsonify({"message": "Processing complete", "result_file": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
