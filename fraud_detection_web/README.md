# Credit Card Fraud Detection Web App

## Setup
1. Train model: `python "../Credit card predection.py.py"` (generates .h5, .pkl, .npy)
2. Convert: `tensorflowjs_converter --input_format=keras ../credit_fraud_model.h5 model/`
3. Load scaler: Convert .npy to JS array in script.js `SCALER_MEAN`, `SCALER_SCALE`
4. Copy `fraud_model_tfjs/` to `model/` folder
5. Open `index.html`

## Features
- 30-feature input form with sliders
- Real-time model inference with TensorFlow.js
- Responsive UI, fraud probability gauge
- Scaled/normalized inputs

## Test
- High Amount (>10000): ~fraud
- Normal values: safe

Model accuracy ~99% on test set (recall for fraud >0.8).
