# Import system utilities
import os

# Import data visualization libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import machine learning models and utilities
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define paths for dataset location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "heart_attack_risk_dataset_1200.csv")


def load_dataset():
    """Load the heart attack risk dataset from CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset containing patient health data and risk labels
    """
    return pd.read_csv(DATASET_PATH)


def handle_missing_values(df, verbose=False):
    """Handle missing values in the dataset using appropriate imputation strategies.
    
    Numeric columns are filled with median values.
    Categorical columns are filled with mode (most frequent value).
    
    Args:
        df (pd.DataFrame): Dataset that may contain missing values
        verbose (bool): If True, print missing value statistics before and after
        
    Returns:
        pd.DataFrame: Dataset with missing values imputed
    """
    if verbose:
        print("\nMissing Values Before Handling:")
        print(df.isnull().sum())
    
    # Fill numeric columns with median
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode (most frequent value)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    if verbose:
        print("\nMissing Values After Handling:")
        print(df.isnull().sum())
    
    return df


def encode_categoricals(df):
    """Convert categorical text columns to numerical values using Label Encoding.
    
    Args:
        df (pd.DataFrame): The dataset with categorical columns
        
    Returns:
        tuple: (encoded_df, label_encoders_dict)
            - encoded_df: DataFrame with categorical columns converted to numbers
            - label_encoders_dict: Dictionary mapping column names to their LabelEncoder objects
    """
    label_encoders = {}
    # Iterate through all columns with object (text) data type
    for col in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        # Transform categorical values to numerical codes
        df[col] = encoder.fit_transform(df[col])
        # Store encoder for later use in predictions
        label_encoders[col] = encoder
    return df, label_encoders


def train_random_forest_for_app():
    """Train a Random Forest model for use in the Flask web application.
    
    This function prepares the data, trains the model, and returns all necessary
    components for making predictions on new patient data.
    
    Returns:
        tuple: (model, scaler, label_encoders, feature_columns, numeric_columns, accuracy)
            - model: Trained RandomForestClassifier
            - scaler: Fitted StandardScaler for normalizing input features
            - label_encoders: Dictionary of LabelEncoders for categorical features
            - feature_columns: List of feature column names
            - numeric_columns: List of numeric column names
            - accuracy: Model accuracy on test set
    """
    # Load the dataset
    df = load_dataset()
    
    # Handle any missing values in the dataset
    df = handle_missing_values(df, verbose=False)

    # Identify categorical and numeric columns
    categorical_columns = list(df.select_dtypes(include="object").columns)
    numeric_columns = [
        col
        for col in df.columns
        if col not in categorical_columns and col != "Heart Attack Risk"
    ]

    # Encode categorical variables to numerical values
    df, label_encoders = encode_categoricals(df)

    # Separate features (X) from target variable (y)
    X = df.drop("Heart Attack Risk", axis=1)
    y = df["Heart Attack Risk"]

    # Split data into training (80%) and testing (20%) sets
    # stratify=y ensures balanced class distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardize features by removing mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest classifier with 200 decision trees
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Evaluate model performance on test set
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Return all components needed for making predictions
    feature_columns = list(X.columns)
    return rf, scaler, label_encoders, feature_columns, numeric_columns, accuracy


def predict_heart_risk(input_data, model, scaler, label_encoders, feature_columns):
    """Predict heart attack risk for a single patient.
    
    Args:
        input_data (dict): Patient data with feature names as keys
        model: Trained machine learning model
        scaler: Fitted StandardScaler for feature normalization
        label_encoders (dict): Dictionary of LabelEncoders for categorical features
        feature_columns (list): Ordered list of feature column names
        
    Returns:
        str: "High Risk" or "Low Risk" prediction
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features using the same encoders from training
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])

    # Scale features using the same scaler from training
    input_scaled = scaler.transform(input_df[feature_columns])
    
    # Make prediction (0 = Low Risk, 1 = High Risk)
    prediction = model.predict(input_scaled)[0]

    return "High Risk" if prediction == 1 else "Low Risk"


def main():
    """Main function to train models and generate visualizations.
    
    This function:
    1. Loads and preprocesses the data
    2. Trains Logistic Regression and Linear Regression models
    3. Evaluates model performance
    4. Generates visualizations for analysis
    5. Demonstrates prediction on a sample patient
    """
    # Load the dataset
    df = load_dataset()

    # Display basic dataset information
    print("Dataset Shape:", df.shape)
    print(df.head())
    
    # Handle missing values with verbose output
    df = handle_missing_values(df, verbose=True)

    # Encode categorical variables
    df, label_encoders = encode_categoricals(df)

    # Visualize feature correlations using a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Prepare features and target variable
    X = df.drop("Heart Attack Risk", axis=1)
    y = df["Heart Attack Risk"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardize features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    # MODEL 1: LOGISTIC REGRESSION
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    y_pred_log = log_model.predict(X_test_scaled)

    print("\n===== Logistic Regression Results =====")
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

    cm_log = confusion_matrix(y_test, y_pred_log)
    plt.figure()
    sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature importance (coefficients)
    coefficients = pd.DataFrame(
        {
            "Feature": X.columns,
            "Coefficient": log_model.coef_[0],
        }
    ).sort_values(by="Coefficient", ascending=False)

    plt.figure()
    sns.barplot(data=coefficients, x="Coefficient", y="Feature")
    plt.title("Feature Importance - Logistic Regression")
    plt.tight_layout()
    plt.show()

    # MODEL 2: LINEAR REGRESSION
    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train)

    y_pred_lin = lin_model.predict(X_test_scaled)

    print("\n===== Linear Regression Results =====")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lin))
    print("R2 Score:", r2_score(y_test, y_pred_lin))

    plt.figure()
    plt.scatter(y_test, y_pred_lin)
    plt.xlabel("Actual Risk")
    plt.ylabel("Predicted Risk")
    plt.title("Linear Regression Predictions")
    plt.show()

    # SAMPLE PREDICTION
    # Create a sample patient profile for testing prediction
    sample = {
        "Age": 60,
        "Gender": "Male",
        "Cholesterol": 250,
        "Blood Pressure": 150,
        "Smoking": "Yes",
        "Obesity": "Yes",
        "Diabetes": "No",
        "Previous Heart Problems": "Yes",
        "Medication Use": "Yes",
        "Physical Activity Level": 1,
    }

    # Demonstrate prediction on sample patient using Logistic Regression model
    print(
        "\nSample Prediction (Logistic Regression):",
        predict_heart_risk(
            sample,
            log_model,
            scaler,
            label_encoders,
            list(X.columns),
        ),
    )

    # Prepare sample for linear regression prediction
    sample_df = pd.DataFrame([sample])
    for col in label_encoders:
        sample_df[col] = label_encoders[col].transform([sample_df.at[0, col]])
    sample_scaled = scaler.transform(sample_df[list(X.columns)])
    linear_pred = lin_model.predict(sample_scaled)[0]
    print("Sample Prediction (Linear Regression):", round(float(linear_pred), 3))

    print("\nTraining Complete.")


# Execute main function when script is run directly
if __name__ == "__main__":
    main()