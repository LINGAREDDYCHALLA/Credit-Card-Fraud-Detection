import pandas as pd
import joblib
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras

# 1. Load dataset (auto-download)
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
response = requests.get(dataset_url)
df = pd.read_csv(StringIO(response.text))

# 2. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Scale Amount and Time
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())

# 6. Train Keras model for TF.js
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_res, y_train_res, epochs=10, batch_size=1024, verbose=1, validation_split=0.1)

# 7. Predictions on test set
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# 8. Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 9. Save for web
joblib.dump(scaler, 'scaler.pkl')
scaler_mean = scaler.mean_
scaler_scale = scaler.scale_
np.save('scaler_mean.npy', scaler_mean)
np.save('scaler_scale.npy', scaler_scale)
model.save('credit_fraud_model.h5')
print("\nSaved: credit_fraud_model.h5, scaler.pkl, scaler_mean.npy, scaler_scale.npy")

print("\nNext steps for TF.js:")
print("pip install tensorflowjs")
print("tensorflowjs_converter --input_format=keras credit_fraud_model.h5 fraud_model_tfjs")
print("Copy fraud_model_tfjs/ to web dir.")
