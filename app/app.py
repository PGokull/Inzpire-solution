import os
import cv2
from flask import Flask, request, jsonify
from datetime import datetime
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd
import numpy as np
import shutil


model = xgb.Booster()
model.load_model(r'D:\Inzpire-Solutions\app\static\weights\estimation-weights\xgboost\xgboost_gross_weight_model.json')

app = Flask(__name__)

BASE_DIR = "Output-Images"
BRANCHES = [
    "ambattur", "kolathur", "velachery", "tambaram", "adyar", "porur",
    "mogappair", "annanagar", "t.nagar", "guindy", "chromepet", 
    "nungambakkam", "pallavaram", "thiruvanmiyur", "perambur"
]

def transformCSV(csv_path, output_csv):
    df = pd.read_csv(csv_path)
    df['GrossWeight'] = df['GrossWeight'].fillna(0)
    df = df[df['NetWeight'] < df['GrossWeight']]
    weight_map = df[['_id', 'Crate', 'NetWeight', 'GrossWeight']].drop_duplicates(subset=['_id'])
    content_counts = df.groupby(['_id', 'Content']).size().reset_index(name='ContentCount')
    pivot_df = content_counts.pivot(index='_id', columns='Content', values='ContentCount').fillna(0)
    pivot_df = pd.merge(pivot_df, weight_map, on='_id', how='left')
    pivot_df['CrateWeight'] = pivot_df['GrossWeight'] - pivot_df['NetWeight']
    columns = [col for col in pivot_df.columns if col != 'GrossWeight'] + ['GrossWeight']
    pivot_df = pivot_df[columns]
    pivot_df.columns.name = None
    pivot_df.columns = [str(col) for col in pivot_df.columns]
    pivot_df.to_csv(output_csv, index=False)
    return output_csv

def create_directories():
    try:
        for branch in BRANCHES:
            branch_path = os.path.join(BASE_DIR, branch)
            os.makedirs(branch_path, exist_ok=True)
    except Exception as e:
        raise Exception(f"Error creating directories: {str(e)}")

create_directories()

def generate_unique_id(branch_name):
    try:
        date_str = datetime.now().strftime("%d%m%Y")
        branch_code = branch_name[:3].upper()
        date_folder_path = os.path.join(BASE_DIR, branch_name, datetime.now().strftime("%Y/%m/%d"))
        os.makedirs(date_folder_path, exist_ok=True)
        existing_files = [f for f in os.listdir(date_folder_path) if f.startswith("IMG")]
        file_count = len(existing_files) + 1
        return f"IMG-{branch_code}-{file_count:03d}-{date_str}"
    except Exception as e:
        raise Exception(f"Error generating unique ID: {str(e)}")


@app.route('/Detect_glass_sanitation', methods=['POST'])
def Detect_glass_sanitation():
    try:
        data = request.json
        input_image_path = data.get("image_path")
        branch_name = data.get("branch_name")

        if branch_name not in BRANCHES:
            return jsonify({"error": f"Invalid branch name. Valid options are: {BRANCHES}", "status": False}), 400
        
        if not os.path.exists(input_image_path):
            return jsonify({"error": "Input image does not exist.", "status": False}), 400
        
        detection_model = YOLO(r"D:\Glass-Sanitation\app\static\weights\detection-weights\best-v2.pt")
        classification_model = YOLO(r"D:\Glass-Sanitation\app\static\weights\classification-weights\best-v1.pt")

        unique_id = generate_unique_id(branch_name)
        
        base_output_path = os.path.join(BASE_DIR, branch_name, datetime.now().strftime("%Y/%m/%d"), unique_id)
        detected_images_dir = os.path.join(base_output_path, "detected_images")
        cropped_images_dir = os.path.join(base_output_path, "cropped_images")
        clean_images_dir = os.path.join(base_output_path, "clean_glass_images")
        dirty_images_dir = os.path.join(base_output_path, "dirty_glass_images")
        
        os.makedirs(detected_images_dir, exist_ok=True)
        os.makedirs(cropped_images_dir, exist_ok=True)
        os.makedirs(clean_images_dir, exist_ok=True)
        os.makedirs(dirty_images_dir, exist_ok=True)
        
        image = cv2.imread(input_image_path)
        detection_results = detection_model(image)
        
        annotated_image = image.copy()
        glasses_detected = 0
        clean_count = 0
        dirty_count = 0
        dirty_glasses = []
        
        for i, box in enumerate(detection_results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            glasses_detected += 1

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, "Glass", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cropped_image_path = os.path.join(cropped_images_dir, f"{unique_id}-CRP-{i + 1}.jpg")
            cv2.imwrite(cropped_image_path, crop)
            
            classification_results = classification_model(crop)
            predicted_class = classification_results[0].probs.top1
            confidence_score = classification_results[0].probs.top1conf.item()
            class_name = "Clean" if predicted_class == 0 else "Dirty"
            
            cv2.putText(crop, f"{class_name} ({confidence_score:.2f})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if predicted_class == 0:
                clean_count += 1
                clean_image_path = os.path.join(clean_images_dir, f"{unique_id}-CLN-{i + 1}.jpg")
                cv2.imwrite(clean_image_path, crop)
            elif predicted_class == 1:
                dirty_count += 1
                dirty_image_path = os.path.join(dirty_images_dir, f"{unique_id}-DIR-{i + 1}.jpg")
                cv2.imwrite(dirty_image_path, crop)
                dirty_glasses.append(dirty_image_path)

        detected_image_path = os.path.join(detected_images_dir, f"{unique_id}-DET.jpg")
        cv2.imwrite(detected_image_path, annotated_image)
        
        return jsonify({
            "input_image": input_image_path,
            "detected_image": detected_image_path,
            "dirty_images": dirty_glasses,
            "dirty_images_count": dirty_count,
            "clean_images_count": clean_count,
            "glasses_detected": glasses_detected,
            "branch": branch_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "status": True
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": False}), 500
    
@app.route('/Detect_Rangoli', methods=['POST'])
def Detect_Rangoli():
    try:
        data = request.json
        input_image_path = data.get("image_path")
        branch_name = data.get("branch_name")

        if branch_name not in BRANCHES:
            return jsonify({"error": f"Invalid branch name. Valid options are: {BRANCHES}", "status": False}), 400

        if not os.path.exists(input_image_path):
            return jsonify({"error": "Input image does not exist.", "status": False}), 400

        detection_model = YOLO(r"D:\Glass-Sanitation\app\static\weights\Rangoli-Detection-weights\best-v1.pt")

        unique_id = generate_unique_id(branch_name)

        base_output_path = os.path.join(BASE_DIR, branch_name, datetime.now().strftime("%Y/%m/%d"), unique_id)
        detected_images_dir = os.path.join(base_output_path, "detected_images")
        rangoli_present_dir = os.path.join(base_output_path, "rangoli_present")
        no_rangoli_dir = os.path.join(base_output_path, "no_rangoli")

        os.makedirs(detected_images_dir, exist_ok=True)
        os.makedirs(rangoli_present_dir, exist_ok=True)
        os.makedirs(no_rangoli_dir, exist_ok=True)

        image = cv2.imread(input_image_path)
        detection_results = detection_model(image)

        annotated_image = image.copy()
        rangoli_detected = False
        rangoli_count = 0

        for i, box in enumerate(detection_results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, "Rangoli", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            rangoli_detected = True
            rangoli_count += 1

        detected_image_path = os.path.join(detected_images_dir, f"{unique_id}-RANGOLI.jpg")
        cv2.imwrite(detected_image_path, annotated_image)

        if rangoli_detected:
            output_image_path = os.path.join(rangoli_present_dir, f"{unique_id}-RANGOLI.jpg")
            cv2.imwrite(output_image_path, annotated_image)
        else:
            output_image_path = os.path.join(no_rangoli_dir, f"{unique_id}-NORANGOLI.jpg")
            cv2.imwrite(output_image_path, image)

        return jsonify({
            "input_image": input_image_path,
            "detected_image": detected_image_path,
            "rangoli_detected": rangoli_detected,
            "rangoli_count": rangoli_count,
            "branch": branch_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "status": True
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": False}), 500
    
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        data = request.json
        branch_name = data.get('branch_name')
        file_path = data.get('file_path')

        if branch_name not in BRANCHES:
            return jsonify({"error": f"Invalid branch name. Valid options are: {BRANCHES}", "status": False}), 400

        if not os.path.exists(file_path):
            return jsonify({"error": "The provided file path does not exist.", "status": False}), 400

        unique_id = generate_unique_id(branch_name)
        base_output_path = os.path.join(BASE_DIR, branch_name, datetime.now().strftime("%Y/%m/%d"), unique_id)
        os.makedirs(base_output_path, exist_ok=True)

        original_file_path = os.path.join(base_output_path, 'original_input.csv')
        shutil.copy(file_path, original_file_path)

        transformed_file_path = transformCSV(original_file_path, os.path.join(base_output_path, 'transformed_data.csv'))

        data = pd.read_csv(transformed_file_path)

        features = ['AH', 'AMP', 'ANJR', 'BI', 'BM', 'BMP', 'CBI', 'CGMP', 'COCBB', 'DAB', 'GA', 'GJ', 'GKB', 
                    'GL', 'GM', 'GMK', 'GMP', 'KAB', 'KAJBT', 'KM', 'KMR', 'KMT', 'MAD', 'MB', 'MCR', 'MDM', 'MH200',
                    'MK', 'ML', 'MLD', 'MPRL', 'MRP', 'MS', 'MUT', 'NB', 'NHB', 'NL', 'NM', 'NMP', 'NPK', 'NTB',
                    'OM', 'ORB', 'PM', 'PR', 'PS', 'PT', 'RS', 'SCB', 'SES', 'SID', 'SOPP', 'SPK', 'SRS', 'STR',
                    'STS', 'STT', 'TPM', 'CCR', 'KRBT', 'MNJ', 'RPA', 'npc', 'GD250', 'GD500', 'PLT450', 'PLT900',
                    'SUB', 'SVR250', 'SVR500', 'APGB', 'ASW1', 'ASW250', 'ASW500', 'SBX400', 'AYSL', 'GD1', 'GFT250',
                    'KJU500', 'SKB', 'SSCS', 'SVR1', 'CrateWeight', 'NetWeight']

        missing_features = [col for col in features if col not in data.columns]
        if missing_features:
            return jsonify({"error": f"Missing features in the input CSV: {missing_features}", "status": False}), 400

        label_encoder = LabelEncoder()
        for col in features:
            if data[col].dtype == 'object':
                data[col] = label_encoder.fit_transform(data[col])

        X = data[features]
        dmatrix = xgb.DMatrix(X)

        predictions = model.predict(dmatrix)
        data['PredictedGrossWeight'] = predictions
        #std_dev = float(np.std(predictions)) 
        std_dev = 0.5
        print(std_dev)
        data['GrossWeightValid'] = np.abs(data['GrossWeight'] - predictions) <= std_dev

        results_path = os.path.join(base_output_path, 'Predicted_Results.csv')
        data.to_csv(results_path, index=False)

        return jsonify({
            "status": True,
            "transformed_file_path": results_path,
            "standard_deviation": std_dev
        }), 200

    except Exception as e:
        return jsonify({"error": str(e), "status": False}), 500

@app.route('/New_test_endpoint',methods=['POST','GET'])
def New_test_endpoint():
    try:
        print("Hello world")
    except Exception as e:
        return jsonify({"error": str(e), "status": False}), 500

if __name__ == "__main__":
    app.run(debug=True)
