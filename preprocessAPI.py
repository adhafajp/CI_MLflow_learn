# preprocessAPI.py
import numpy as np
import pandas as pd
import joblib
import requests
import logging
from pathlib import Path
from typing import Dict, Union, List

# ----------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------
MODEL_DIR = "model"
# MLFLOW_URL = "http://127.0.0.1:5001/invocations"   # sesuaikan dengan endpoint Anda
MLFLOW_URL = "http://127.0.0.1:9000/invocations"  # sesuaikan dengan endpoint Anda
RUNNING_ON = "docker"
# RUNNING_ON = "mlflow"

# Kolom yang di-drop
DROPPED_COLUMNS = [
    "Customer_ID",
    "Month",
    "Occupation",
    "Type_of_Loan",
    "Credit_Utilization_Ratio",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Credit_History_Age",
]

CATEGORICAL_COLUMNS = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

PCA1_COLUMNS = [
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_History_Age",
]

PCA2_COLUMNS = [
    "Monthly_Inhand_Salary",
    "Monthly_Balance",
    "Amount_invested_monthly",
    "Total_EMI_per_month",
]

FINAL_FEATURE_ORDER = [
    "Age",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "pc1_1",
    "pc1_2",
    "pc1_3",
    "pc1_4",
    "pc1_5",
    "pc2_1",
    "pc2_2",
]

# ----------------------------------------------------------------------
# Setup logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Global cache (models & session)
# ----------------------------------------------------------------------
_scalers = {}
_encoders = {}
_pca1 = None
_pca2 = None
_target_encoder = None
_http_session = None


def _load_models():
    """Load all required models (scaler, encoder, PCA, target encoder) once."""
    global _scalers, _encoders, _pca1, _pca2, _target_encoder
    model_path = Path(MODEL_DIR)

    # Scalers
    for col in NUMERICAL_COLUMNS:
        f = model_path / f"scaler_{col}.joblib"
        if not f.exists():
            raise FileNotFoundError(f"Scaler not found: {f}")
        _scalers[col] = joblib.load(f)

    # Encoders for categorical features
    for col in CATEGORICAL_COLUMNS:
        f = model_path / f"encoder_{col}.joblib"
        if not f.exists():
            raise FileNotFoundError(f"Encoder not found: {f}")
        _encoders[col] = joblib.load(f)

    # PCA
    pca1_file = model_path / "pca_1.joblib"
    pca2_file = model_path / "pca_2.joblib"
    if not pca1_file.exists() or not pca2_file.exists():
        raise FileNotFoundError("PCA models missing (pca_1.joblib / pca_2.joblib)")
    _pca1 = joblib.load(pca1_file)
    _pca2 = joblib.load(pca2_file)

    # Target encoder (untuk decode hasil prediksi)
    target_enc_file = model_path / "encoder_target.joblib"
    if not target_enc_file.exists():
        raise FileNotFoundError(f"Target encoder not found: {target_enc_file}")
    _target_encoder = joblib.load(target_enc_file)

    logger.info("All models loaded successfully.")


def _get_session():
    """Reuse HTTP connection for MLflow requests."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
    return _http_session


def preprocess_input(user_data: Dict, model_dir: str = MODEL_DIR) -> np.ndarray:
    """
    Preprocess single user dictionary into feature vector (2D array).
    """
    global _scalers, _encoders, _pca1, _pca2
    if not _scalers:
        _load_models()

    df = pd.DataFrame([user_data])

    # Drop columns
    cols_to_drop = [c for c in DROPPED_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, axis=1, inplace=True)

    # Scale numerical
    for col in NUMERICAL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")
        scaler = _scalers[col]
        df[col] = scaler.transform(df[[col]].values).flatten()

    # Encode categorical
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")
        encoder = _encoders[col]
        try:
            df[col] = encoder.transform(df[col])
        except ValueError as e:
            raise ValueError(f"Unknown category in '{col}': {df[col].iloc[0]}") from e

    # PCA1
    pca1_input = df[PCA1_COLUMNS].values
    pc1 = _pca1.transform(pca1_input)  # shape (1,5)
    df.drop(columns=PCA1_COLUMNS, axis=1, inplace=True)
    for i in range(5):
        df[f"pc1_{i+1}"] = pc1[:, i]

    # PCA2
    pca2_input = df[PCA2_COLUMNS].values
    pc2 = _pca2.transform(pca2_input)  # shape (1,2)
    df.drop(columns=PCA2_COLUMNS, axis=1, inplace=True)
    for i in range(2):
        df[f"pc2_{i+1}"] = pc2[:, i]

    # Order columns
    missing = set(FINAL_FEATURE_ORDER) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns after preprocessing: {missing}")
    df_final = df[FINAL_FEATURE_ORDER]
    return df_final.values.astype(np.float32)


def predict_from_dataframe(preprocessed_df: pd.DataFrame) -> str:
    """
    Send already-preprocessed DataFrame to MLflow model endpoint.

    Args:
        preprocessed_df: DataFrame with exactly the 11 final features (order matters).

    Returns:
        Predicted label: 'Good', 'Standard', or 'Poor'.
    """
    if not _target_encoder:
        _load_models()

    # Convert to JSON split format expected by MLflow
    data_json = preprocessed_df.to_json(orient="split")
    payload = f'{{"dataframe_split": {data_json}}}'

    session = _get_session()
    headers = {"Content-Type": "application/json"}
    try:
        response = session.post(MLFLOW_URL, data=payload, headers=headers, timeout=300)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"MLflow request failed: {e}")
        raise RuntimeError(f"Prediction service error: {e}") from e

    predictions = response.json().get("predictions")
    if predictions is None:
        raise RuntimeError("MLflow response missing 'predictions' field")

    # Decode numeric prediction to label
    label = _target_encoder.inverse_transform(predictions)[0]
    return label


def predict_from_dict(user_data: Dict) -> str:
    """
    End-to-end: raw dict -> preprocessing -> MLflow prediction -> label.
    """
    features = preprocess_input(user_data)  # shape (1,11)
    df_features = pd.DataFrame(features, columns=FINAL_FEATURE_ORDER)
    return predict_from_dataframe(df_features)


# ----------------------------------------------------------------------
# Opsional: kompatibilitas dengan fungsi `prediction` yang Anda punya
# (tetapi lebih efisien karena tidak load encoder berulang)
# ----------------------------------------------------------------------
def prediction(data: pd.DataFrame) -> str:
    """
    Original function signature (kept for backward compatibility).
    Now uses optimized predict_from_dataframe.
    """
    return predict_from_dataframe(data)


# ----------------------------------------------------------------------
# Contoh penggunaan jika dijalankan langsung
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_input = {
        "Customer_ID": "TEST001",
        "Month": "Jan",
        "Age": 30,
        "Occupation": "Engineer",
        "Annual_Income": 60000,
        "Monthly_Inhand_Salary": 5000,
        "Num_Bank_Accounts": 2,
        "Num_Credit_Card": 2,
        "Interest_Rate": 12,
        "Num_of_Loan": 1,
        "Delay_from_due_date": 2,
        "Num_of_Delayed_Payment": 1,
        "Changed_Credit_Limit": 0.2,
        "Num_Credit_Inquiries": 3,
        "Outstanding_Debt": 12000,
        "Total_EMI_per_month": 600,
        "Amount_invested_monthly": 300,
        "Monthly_Balance": 900,
        "Credit_History_Age": 96,
        "Type_of_Loan": "Personal",
        "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "High_spent_Small_value_payments",
        "Credit_Utilization_Ratio": 0.25,
        "Credit_Score": "Good",  # tidak dipakai
    }
    try:
        result = predict_from_dict(sample_input)
        print(f"Prediction result: {result}")
    except Exception as e:
        print(f"Error: {e}")
