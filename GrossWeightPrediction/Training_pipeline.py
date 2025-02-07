import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error , mean_squared_error

class DataTransformation:
    def __init__(self):
        self.base_folder_jan_data = r'D:\Inzpire-Solutions\GrossWeightPrediction\dataset\raw_data'
        self.crate_weights_data = r'D:\Inzpire-Solutions\GrossWeightPrediction\dataset\crate_data\crate.csv'
        self.merged_data = r'D:\Inzpire-Solutions\GrossWeightPrediction\dataset\merged_data\merged.csv'
        self.transformed_data = r'D:\Inzpire-Solutions\GrossWeightPrediction\dataset\transformed_data\transformed_data.csv'
        self.crate_added_data = r'D:\Inzpire-Solutions\GrossWeightPrediction\dataset\crate_added_data\crate_added_dataset.csv'
        self.dataset = r'D:\Inzpire-Solutions\GrossWeightPrediction\dataset\dataset.csv'

    def merge_dataset(self):
        try:
            new_dataframe = pd.DataFrame()
            for file in os.listdir(self.base_folder_jan_data):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.base_folder_jan_data,file)
                    current_dataframe = pd.read_csv(file_path)
                    new_dataframe = pd.concat([new_dataframe,current_dataframe],ignore_index=True)
            new_dataframe.to_csv(self.merged_data,index=False)
            print('Completed Merging of Dataset')
        except Exception as e:
            print("Error Occured", e)

    def transform_data(self):
        try:
            merged_dataset = pd.read_csv(self.merged_data)
            merged_dataset['GrossWeight'] = merged_dataset['GrossWeight'].fillna(0)
            weight_map = merged_dataset[['_id', 'Crate', 'NetWeight', 'GrossWeight']].drop_duplicates(subset=['_id'])
            content_counts = merged_dataset.groupby(['_id','Crate', 'Content']).size().reset_index(name='ContentCount')
            pivot_df = content_counts.pivot(index=['_id','Crate'], columns='Content', values='ContentCount').fillna(0)
            pivot_df = pd.merge(pivot_df, weight_map, on=['_id','Crate'], how='left')
            pivot_df.columns.name = None
            pivot_df.columns = [str(col) for col in pivot_df.columns]
            pivot_df.to_csv(self.transformed_data,index=False)
            print("Completed Transforming Data")
        except Exception as e:
            print("Error Occured", e)

    def add_crate_data(self):
        try:
            transformed_dataset = pd.read_csv(self.transformed_data)
            crate_data = pd.read_csv(self.crate_weights_data)
            crate_added_dataset = pd.merge(transformed_dataset,crate_data,on='Crate',how='left')
            crate_added_dataset.to_csv(self.crate_added_data,index=False)
            print("Completed Adding Crate Data")
        except Exception as e:
            print("Error Occured", e)
    
    def preprocess_data(self):
        try:
            final_dataset = pd.read_csv(self.crate_added_data)
            final_dataset = final_dataset[final_dataset['GrossWeight'] > 0]
            final_dataset = final_dataset.dropna(subset=['CrateWeight'])
            final_dataset = final_dataset[final_dataset['NetWeight'] < final_dataset['GrossWeight']]
            final_dataset.to_csv(self.dataset,index=False)
            print("Completed Data preprocessing successfully") 
        except Exception as e:
            print("Error Occurred:", e)

class TrainingModel:
    def __init__(self):
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f'D:\\Inzpire-Solutions\\GrossWeightPrediction\\model\\Trained_model-{date}.pkl'
        self.model = path
        self.dataset = r'D:\Inzpire-Solutions\GrossWeightPrediction\dataset\dataset.csv'
        
    def training(self):
        try:
            dataset = pd.read_csv(self.dataset)
            x = dataset.drop(['GrossWeight', "_id", "Crate"], axis=1)
            y = dataset["GrossWeight"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            model = SVR(kernel='linear', C=8, epsilon=0.09)
            model.fit(x_train, y_train)
            joblib.dump(model, self.model)
            print("Completed Training successfully")
            y_pred = model.predict(x_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print(f"Root Mean Squared Error (RMSE): {rmse}")
            print(f"Mean Absolute Error (MAE): {mae}")
        except Exception as e:
            print("Error Occurred:", e)

class GrossWeightPredictionPipeline:
    def __init__(self):
        self.data_transformer = DataTransformation()
        self.model_trainer = TrainingModel()

    def run_pipeline(self):
        print("Starting Data Transformation...")
        self.data_transformer.merge_dataset()
        self.data_transformer.transform_data()
        self.data_transformer.add_crate_data()
        self.data_transformer.preprocess_data()
        
        print("\nStarting Model Training...")
        self.model_trainer.training()
        print("\nPipeline Execution Completed Successfully!")

if __name__ == "__main__":
    pipeline = GrossWeightPredictionPipeline()
    pipeline.run_pipeline()
