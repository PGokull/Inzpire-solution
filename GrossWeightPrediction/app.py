from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = r"D:\Inzpire-Solutions\GrossWeightPrediction\Models\svr_model.pkl"
model = joblib.load(MODEL_PATH)
PRODUCTS = ["AH", "AMP", "BM200", "BMCO", "CGMP", "GA", "GJ", "GL", "GLCO", "GM", "GM200", "GMK200", "GMP", "GMPC",
            "KAB200", "KM", "KMR", "KMT", "KS", "MB", "MDM", "MDM200", "MH 200", "MK", "MK200", "ML", "MS", "MS200",
            "MSC", "MUT", "NB", "NHB", "NL", "NLC", "NM", "NM200", "NMC", "NMP", "NPC200", "NPK", "NTB", "OM200",
            "OMC", "PM200", "PR", "PS", "PS200", "PSCO", "PT", "SES200", "SPK", "SRS200", "STS", "TPM200", "TPMC"]

def transform_data(crate_data):
    transformed_data = {product: 0 for product in PRODUCTS}
    for product in crate_data["contents"]["products"]:
        if product in transformed_data:
            transformed_data[product] += 1
    transformed_data["NetWeight"] = crate_data["net_weight"]
    transformed_data["CrateWeight"] = crate_data["crate_weight"]

    return pd.DataFrame([transformed_data])

@app.route('/predict', methods=['POST'])
def predict_gross_weight():
    try:
        crate_data = request.get_json()
        required_keys = ["contents", "net_weight", "crate_weight", "gross_weight"]
        for key in required_keys:
            if key not in crate_data:
                return jsonify({"error": f"Missing key: {key}"}), 400
        input_df = transform_data(crate_data)
        predicted_gross_weight = model.predict(input_df)[0]
        actual_gross_weight = crate_data["gross_weight"]
        difference = round(abs(predicted_gross_weight - actual_gross_weight), 3)
        response = {
            "predicted_gross_weight": round(predicted_gross_weight, 3),
            "actual_gross_weight": actual_gross_weight,
            "difference": difference
        }
        if difference > 0.5:
            response["message"] = "Difference is large."
        else:
            response["message"] = "All products fit."
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
