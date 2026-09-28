"""
Tugas Machine Learning - Regresi Linear Multivariate
Studi Kasus: Prediksi Harga Smartphone
"""

# =========================================================
# 1. IMPORT LIBRARY
# =========================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================================================
# 2. MEMBUAT DATASET
# =========================================================
data = {
    'Ram': [1, 2, 1, 2, 1],
    'Storage': [1, 1, 2, 2, 3],
    'Kamera': [0, 1, 0, 1, 1],
    'Harga': [3, 6, 4, 5, 6]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)
print()

# =========================================================
# 3. MENENTUKAN X DAN y
# =========================================================
X = df[['Ram', 'Storage', 'Kamera']]
y = df['Harga']

print("Variabel X")
print(X)
print()

print("Variabel y")
print(y)
print()

# =========================================================
# 4. TRAIN-TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)

print(f"Jumlah data train : {len(X_train)}")
print(f"Jumlah data test   : {len(X_test)}")
print()

# =========================================================
# 5. MEMBUAT MODEL
# =========================================================
model = LinearRegression()

# =========================================================
# 6. TRAINING
# =========================================================
model.fit(X_train, y_train)

# =========================================================
# 7. MENDAPATKAN b0, b1, b2, b3
# =========================================================
b0 = model.intercept_
b1, b2, b3 = model.coef_

print("Koefisien Model:")
print(f"b0 (intercept) = {b0:.4f}")
print(f"b1 (Ram)       = {b1:.4f}")
print(f"b2 (Storage)   = {b2:.4f}")
print(f"b3 (Kamera)    = {b3:.4f}")
print()

# =========================================================
# 8. MEMBENTUK PERSAMAAN REGRESI
# =========================================================
print("Persamaan Regresi:")
print(f"Y = {b0:.4f} + ({b1:.4f})*Ram + ({b2:.4f})*Storage + ({b3:.4f})*Kamera")
print()

# =========================================================
# 9. TESTING / PREDIKSI
# =========================================================
y_pred = model.predict(X_test)

# =========================================================
# 10. BANDINGKAN AKTUAL VS PREDIKSI
# =========================================================
hasil = pd.DataFrame({
    'Aktual': y_test.values,
    'Prediksi': y_pred
})
print("Perbandingan Aktual vs Prediksi (data test):")
print(hasil)
print()

# =========================================================
# 11. EVALUASI MODEL (MSE)
# =========================================================
mse = mean_squared_error(y_test, y_pred)

print("Evaluasi Model:")
print(f"MSE : {mse:.4f}")

# Catatan: dengan hanya 5 sample dan 1 data test, nilai R2 bisa
# tidak stabil/tidak representatif. Untuk hasil yang lebih valid,
# idealnya dataset diperbanyak.
