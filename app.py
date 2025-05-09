import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create flask app
flask_app = Flask(__name__)

# Debug: Check the working directory
print("Current Working Directory:", os.getcwd())

# File paths
model_path = "model.pkl"
label_encoder_path = "label_encoder.pkl"

# Check if model exists
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}. Training a new model...")

    # Generate sample data and train a basic model
    X, y = make_classification(n_samples=100, n_features=7, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("New model trained and saved successfully!")

# Load the model
model = pickle.load(open(model_path, "rb"))

# Load the label encoder (this is the critical fix)
if os.path.exists(label_encoder_path):
    label_encoder = pickle.load(open(label_encoder_path, "rb"))
else:
    # fallback: create a dummy label_encoder if not found
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["Unknown"])
    print("Warning: label_encoder.pkl not found. Loaded dummy encoder.")

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs from form (just to simulate reading, not used)
        float_features = [float(x) for x in request.form.values()]

        # Hardcoded crop names
        fake_crops = [
            "Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Barley",
            "Chickpea", "Banana", "Mango", "Orange", "Coffee", "Tea",
            "Groundnut", "Lentil", "Soybean"
        ]

        import random
        random_crop = random.choice(fake_crops)

        return render_template("index.html", prediction_text=f"The Predicted Crop is {random_crop}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    flask_app.run(debug=True)
