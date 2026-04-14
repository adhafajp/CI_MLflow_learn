import pandas as pd
from preprocessAPI import preprocess_input, prediction, FINAL_FEATURE_ORDER

print("=== Memulai Simulasi Pengguna ===")

# 1. Definisikan kolom input (sama seperti di atas)
columns = [
    "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Age", 
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", 
    "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Monthly_Inhand_Salary", 
    "Monthly_Balance", "Amount_invested_monthly", "Total_EMI_per_month", 
    "Credit_History_Age"
]

data = [
    "Good", "No", "Low_spent_Small_value_payments", 23, 3, 4, 3, 4, 3, 7, 
    11.27, 5, 809.98, 1824.80, 186.26, 236.64, 49.50, 216
]

# 2. Buat dictionary
user_dict = dict(zip(columns, data))

# 3. Preprocessing -> menghasilkan numpy array (1, 11)
print("[1/3] Preprocessing data...")
features_array = preprocess_input(user_dict)   # shape (1, 11)

# 4. Konversi ke DataFrame dengan nama kolom yang benar
df_features = pd.DataFrame(features_array, columns=FINAL_FEATURE_ORDER)

# 5. Prediksi ke MLflow
print("[2/3] Mengirim ke model MLflow...")
hasil_prediksi = prediction(df_features)   # fungsi prediction sudah menerima DataFrame

print("[3/3] Selesai.")
print("\n=== Hasil Inferensi ===")
print(f"Prediksi Credit Scoring: {hasil_prediksi}")