# ==========================================
# HEART RISK PREDICTION - FLASK WEB APP
# RESTful API for heart attack risk prediction
# ==========================================

# Import system utilities
import os
import webbrowser

# Import data processing library
import pandas as pd

# Import Flask web framework components
from flask import Flask, jsonify, render_template, request

# Import custom ML model training function
from Ml_cw import train_random_forest_for_app

# Initialize Flask application
app = Flask(__name__)

# Train the model and unpack all necessary components for predictions
# This happens once when the server starts
(
    MODEL,              # Trained Random Forest classifier
    SCALER,            # Fitted StandardScaler for feature normalization
    LABEL_ENCODERS,    # Dictionary of LabelEncoders for categorical features
    FEATURE_COLUMNS,   # Ordered list of all feature names
    NUMERIC_COLUMNS,   # List of numeric column names
    MODEL_ACCURACY,    # Model accuracy on test set
) = (
    train_random_forest_for_app()
)


def get_category_options():
    """Extract all valid category options from label encoders.
    
    Returns:
        dict: Mapping of categorical column names to their valid values
              Example: {"Gender": ["Male", "Female"], "Smoking": ["Yes", "No"]}
    """
    options = {}
    for col, encoder in LABEL_ENCODERS.items():
        # Get all possible classes from the encoder
        options[col] = list(encoder.classes_)
    return options


@app.route("/")
def index():
    """Serve the main web page with form inputs.
    
    Returns:
        HTML page with pre-populated category options for dropdowns
    """
    return render_template("index.html", category_options=get_category_options())


@app.route("/options")
def options():
    """API endpoint to get valid category options as JSON.
    
    Returns:
        JSON object with all valid categorical values
    """
    return jsonify(get_category_options())


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict heart attack risk based on patient data.
    
    Expects JSON payload with all required patient features.
    Validates input, transforms data, and returns prediction with probability.
    
    Returns:
        JSON object with:
            - risk: "High Risk" or "Low Risk"
            - probability: Probability of high risk (0.0 to 1.0)
            - accuracy: Model accuracy on test set
        
        Or error response with 400 status code if validation fails
    """
    # Parse JSON payload from request
    payload = request.get_json(silent=True) or {}

    # Validate that all required fields are present
    missing = [col for col in FEATURE_COLUMNS if col not in payload]
    if missing:
        return (
            jsonify({"error": "Missing fields", "missing": missing}),
            400,
        )

    # Create a copy of payload for processing
    cleaned_payload = dict(payload)
    
    # Convert numeric fields to float and validate
    for col in NUMERIC_COLUMNS:
        try:
            cleaned_payload[col] = float(cleaned_payload[col])
        except (TypeError, ValueError):
            return (
                jsonify({"error": f"Invalid numeric value for {col}"}),
                400,
            )

    # Convert payload to DataFrame for processing
    input_df = pd.DataFrame([cleaned_payload])

    # Encode categorical variables using trained encoders
    for col, encoder in LABEL_ENCODERS.items():
        value = input_df.at[0, col]
        # Validate that the category value is recognized
        if value not in encoder.classes_:
            return (
                jsonify({"error": f"Invalid value for {col}", "value": value}),
                400,
            )
        # Transform categorical value to numeric code
        input_df[col] = encoder.transform([value])

    # Scale features using fitted scaler
    input_scaled = SCALER.transform(input_df[FEATURE_COLUMNS])
    
    # Make prediction (0 = Low Risk, 1 = High Risk)
    prediction = int(MODEL.predict(input_scaled)[0])
    
    # Get probability of high risk (class 1)
    probability = float(MODEL.predict_proba(input_scaled)[0][1])

    # Return prediction results as JSON
    return jsonify(
        {
            "risk": "High Risk" if prediction == 1 else "Low Risk",
            "probability": round(probability, 3),
            "accuracy": round(MODEL_ACCURACY, 3),
        }
    )


if __name__ == "__main__":
    # Configure server port
    port = 5000
    
    # Automatically open browser when server starts (only once, not on reload)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open(f"http://127.0.0.1:{port}/")
    
    # Start Flask development server with debug mode enabled
    app.run(debug=True, port=port)
