import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
try:
    import pandas as pd
    import numpy as np
    from flask import Flask, request, jsonify, render_template
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import pickle
    import os
except ImportError as e:
    logger.error(f"Import error: {e}. Please ensure pandas, numpy, scikit-learn, and flask are installed.")
    logger.info("Run: pip install pandas numpy scikit-learn flask")
    sys.exit(1)

logger.info("Starting the application")

def load_and_preprocess_data():
    try:
        if not os.path.exists('weather_dataset.csv'):
            logger.error("weather_dataset.csv not found in the current directory.")
            sys.exit(1)
        
        data = pd.read_csv('weather_dataset.csv')
        logger.info("Data loaded successfully")
        
        required_columns = ['mandal', 'temp_min', 'temp_max']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Dataset missing required columns: {required_columns}")
            sys.exit(1)
        
        data['temp_min'] = data['temp_min'].fillna(data['temp_min'].mean())
        data['temp_max'] = data['temp_max'].fillna(data['temp_max'].mean())
        logger.info("Missing values handled")
        
        label_encoder = LabelEncoder()
        data['mandal_encoded'] = label_encoder.fit_transform(data['mandal'])
        logger.info("Mandal encoded")
        
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        return data, label_encoder
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        sys.exit(1)

def train_models(data):
    try:
        features = data[['mandal_encoded']]
        target_temp_min = data['temp_min']
        target_temp_max = data['temp_max']
        
        X_train, X_test, y_train_temp_min, y_test_temp_min, y_train_temp_max, y_test_temp_max = train_test_split(
            features, target_temp_min, target_temp_max, test_size=0.2, random_state=42
        )
        logger.info("Data split into training and testing sets")
        
        model_temp_min = RandomForestRegressor(random_state=42)
        model_temp_max = RandomForestRegressor(random_state=42)
        
        model_temp_min.fit(X_train, y_train_temp_min)
        model_temp_max.fit(X_train, y_train_temp_max)
        logger.info("Models trained successfully")
        
        # Evaluate models
        pred_temp_min = model_temp_min.predict(X_test)
        pred_temp_max = model_temp_max.predict(X_test)
        
        mae_temp_min = mean_absolute_error(y_test_temp_min, pred_temp_min)
        mae_temp_max = mean_absolute_error(y_test_temp_max, pred_temp_max)
        rmse_temp_min = np.sqrt(mean_squared_error(y_test_temp_min, pred_temp_min))
        rmse_temp_max = np.sqrt(mean_squared_error(y_test_temp_max, pred_temp_max))
        
        logger.info(f"Model Evaluation - Temp Min: MAE = {mae_temp_min:.2f}, RMSE = {rmse_temp_min:.2f}")
        logger.info(f"Model Evaluation - Temp Max: MAE = {mae_temp_max:.2f}, RMSE = {rmse_temp_max:.2f}")
        
        with open('model_temp_min.pkl', 'wb') as f:
            pickle.dump(model_temp_min, f)
        with open('model_temp_max.pkl', 'wb') as f:
            pickle.dump(model_temp_max, f)
        logger.info("Models saved successfully")
        
        return model_temp_min, model_temp_max
    except Exception as e:
        logger.error(f"Error in training models: {e}")
        sys.exit(1)

def load_models():
    try:
        with open('model_temp_min.pkl', 'rb') as f:
            model_temp_min = pickle.load(f)
        with open('model_temp_max.pkl', 'rb') as f:
            model_temp_max = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info("Models and label encoder loaded successfully")
        return model_temp_min, model_temp_max, label_encoder
    except Exception as e:
        logger.error(f"Error loading models or label encoder: {e}")
        sys.exit(1)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        mandal = request.form['mandal']
        logger.info(f"Received mandal: {mandal}")
        
        model_temp_min, model_temp_max, label_encoder = load_models()
        
        try:
            mandal_encoded = label_encoder.transform([mandal])[0]
        except ValueError:
            logger.error(f"Mandal not found: {mandal}")
            return jsonify({'error': 'Mandal not found'}), 400
        
        features = np.array([[mandal_encoded]])
        
        pred_temp_min = model_temp_min.predict(features)[0]
        pred_temp_max = model_temp_max.predict(features)[0]
        avg_temp = (pred_temp_min + pred_temp_max) / 2
        logger.info(f"Predictions - Min Temp: {pred_temp_min}, Max Temp: {pred_temp_max}, Avg Temp: {avg_temp}")
        
        return jsonify({
            'avg_temp': round(avg_temp, 2),
            'temp_min': round(pred_temp_min, 2),
            'temp_max': round(pred_temp_max, 2)
        })
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    logger.info(f"Running script from: {os.getcwd()}")
    
    if not os.path.exists('model_temp_min.pkl') or not os.path.exists('model_temp_max.pkl'):
        logger.info("Training models...")
        data, _ = load_and_preprocess_data()
        train_models(data)
    else:
        logger.info("Using existing models...")
    
    logger.info("Starting Flask server")
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
        sys.exit(1)