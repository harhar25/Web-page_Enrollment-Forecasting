from flask import Flask, render_template, request, jsonify, send_file, url_for, session
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_TYPE'] = 'filesystem'  # Enable server-side session

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Department and course configurations
DEPARTMENTS = {
    'BED': ['BSBA-FM', 'BSBA-MM'],
    'CED': ['BSIT', 'BSCS']
}

YEAR_LEVELS = [1, 2, 3, 4]
YEARS = [str(year) for year in range(2020, 2026)]
SEMESTERS = [1, 2]

# Global variables for models and data
models = {}
scalers = {}
data_store = {}
model_histories = {}

def prepare_data(data, course, year_level, seq_length=4):
    """Prepare data for LSTM model training with missing value handling"""
    # Filter data for specific course and year level
    course_data = data[(data['Course'] == course) & (data['Year_Level'] == year_level)].copy()
    course_data = course_data.sort_values('Year_Semester')
    
    # Get all possible year-semesters in the date range
    start_date = course_data['Year_Semester'].min()
    end_date = course_data['Year_Semester'].max()
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    # Create a complete date range
    all_dates = []
    for year in range(start_year, end_year + 1):
        all_dates.extend([f"{year}-1", f"{year}-2"])
    
    all_dates = [d for d in all_dates if start_date <= d <= end_date]
    
    # Create a complete DataFrame with all dates
    complete_data = pd.DataFrame({
        'Year_Semester': all_dates,
        'Course': course,
        'Year_Level': year_level
    })
    
    # Merge with actual data
    course_data = pd.merge(
        complete_data,
        course_data,
        on=['Year_Semester', 'Course', 'Year_Level'],
        how='left'
    )
    
    # Calculate missing value percentage
    missing_pct = course_data['Enrollees'].isnull().mean()
    
    # If 50% or more values are missing, raise an error
    if missing_pct >= 0.5:
        raise ValueError(f'Too many missing values for {course} year {year_level}. Missing: {missing_pct:.1%}')
    
    # Handle missing values with mean
    if course_data['Enrollees'].isnull().any():
        mean_value = course_data['Enrollees'].mean()
        course_data['Enrollees'] = course_data['Enrollees'].fillna(mean_value)
    
    if len(course_data) < seq_length + 1:
        raise ValueError(f'Not enough data points for {course} year {year_level}. Need at least {seq_length + 1} points.')
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(course_data['Enrollees'].values.reshape(-1, 1))
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length])
    
    if not X or not y:
        raise ValueError(f'Could not create sequences for {course} year {year_level}')
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # For small datasets, use simple train-test split
    if len(X) < 6:
        split = max(1, len(X) // 2)  # Ensure at least 1 validation sample
        X_train, X_val = X[:-split], X[-split:]
        y_train, y_val = y[:-split], y[-split:]
    else:
        # Use TimeSeriesSplit for larger datasets
        tscv = TimeSeriesSplit(n_splits=2)
        train_idx, val_idx = next(tscv.split(X))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
    
    return X_train, X_val, y_train, y_val, scaler

def build_model():
    """Build Random Forest model for time series forecasting"""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

def predict_future(model, scaler, last_sequence, n_future=6):
    """Generate future predictions using the trained model"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        # Reshape sequence for Random Forest
        current_sequence_2d = current_sequence.reshape(1, -1)
        
        # Predict next value
        next_pred = model.predict(current_sequence_2d)
        
        # Inverse transform the prediction
        next_pred_original = scaler.inverse_transform(next_pred.reshape(1, -1))[0][0]
        predictions.append(next_pred_original)
        
        # Update sequence
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    return predictions

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape)
    }

def train_model(data, course, year_level):
    """Train Random Forest model for a specific course and year level"""
    try:
        # Prepare data
        X_train, X_val, y_train, y_val, scaler = prepare_data(data, course, year_level)
        
        # Reshape data for Random Forest (flatten the sequence)
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        
        # Create and train model
        model = build_model()
        model.fit(X_train_2d, y_train)
        
        # Calculate validation score
        val_score = model.score(X_val_2d, y_val)
        
        # Create a simple history object to maintain compatibility
        history = {
            'val_loss': [val_score]
        }
        
        return model, scaler, history
        
    except ValueError as ve:
        print(f'Validation error for {course} year {year_level}: {str(ve)}')
        raise
    except Exception as e:
        print(f'Error training model for {course} year {year_level}: {str(e)}')
        raise

def load_and_train_initial_data():
    """Load initial dataset and train models"""
    try:
        # Load initial dataset
        df = pd.read_csv('sample_data.csv')
        data_store['training_data'] = df
        
        # Train models for each course and year level
        for department, courses in DEPARTMENTS.items():
            for course in courses:
                for year_level in YEAR_LEVELS:
                    model_key = f"{course}_year{year_level}"
                    print(f'Training model for {model_key}...')
                    model, scaler, history = train_model(df, course, year_level)
                    models[model_key] = model
                    scalers[model_key] = scaler
                    model_histories[model_key] = history
                    print(f'Finished training {model_key} model')
        
        return True
    except Exception as e:
        print(f'Error training models: {str(e)}')
        return False

@app.route('/')
def index():
    # Check if models are trained
    if not models:
        success = load_and_train_initial_data()
        if not success:
            return 'Error: Could not train initial models', 500
    
    return render_template('index.html', 
                           departments=DEPARTMENTS,
                           year_levels=YEAR_LEVELS)

@app.route('/get_filters')
def get_filters():
    try:
        filters = {
            'courses': DEPARTMENTS,
            'year_levels': YEAR_LEVELS,
            'years': YEARS,
            'semesters': SEMESTERS
        }
        
        return jsonify(filters)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filters')
def api_filters():
    df = data_store.get('training_data')
    if df is None:
        return jsonify({'error': 'No data available'}), 404
    
    years = sorted(df['Year_Semester'].str[:4].unique())
    semesters = ['1', '2']
    
    return jsonify({
        'years': years,
        'semesters': semesters,
        'departments': DEPARTMENTS,
        'year_levels': YEAR_LEVELS
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        department = data.get('department')
        course = data.get('course')
        year = data.get('year')
        semester = data.get('semester')
        
        if not all([department, course, year, semester]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Load and prepare data
        df = pd.read_csv('sample_data.csv')
        
        # Initialize results dictionary
        results = {
            'historical': {
                'dates': [],
                'values': [],
                'by_year_level': {}
            },
            'predictions': {},
            'metrics': {}
        }
        
        # Process each year level
        for year_level in YEAR_LEVELS:
            # Prepare data for this year level
            X_train, X_val, y_train, y_val, scaler = prepare_data(df, course, year_level)
            
            # Train model if not exists
            model_key = f"{course}_year{year_level}"
            if model_key not in models:
                model, scaler, history = train_model(df, course, year_level)
                models[model_key] = model
                scalers[model_key] = scaler
                model_histories[model_key] = history
            
            # Make prediction
            model = models[model_key]
            scaler = scalers[model_key]
            
            # Prepare input sequence
            input_seq = get_input_sequence(df, course, year_level)
            input_seq = scaler.transform(input_seq.reshape(-1, 1)).reshape(1, -1, 1)
            
            # Generate prediction
            pred = model.predict(input_seq)
            pred = scaler.inverse_transform(pred)[0][0]
            
            # Get historical data
            historical = get_historical_data(df, course, year_level)
            
            # Add to results
            if not results['historical']['dates']:
                results['historical']['dates'] = historical['dates']
            
            results['historical']['by_year_level'][year_level] = historical['values']
            results['predictions'][f"Year {year_level}"] = float(pred)
            
            # Calculate metrics for this year level
            metrics = calculate_metrics(historical['values'][-4:], [pred])
            results['metrics'][f"Year {year_level}"] = metrics
        
        # Calculate total values
        results['historical']['values'] = [sum(results['historical']['by_year_level'][yl][i] 
                                             for yl in YEAR_LEVELS) 
                                         for i in range(len(results['historical']['dates']))]
        
        # Calculate overall metrics
        total_pred = sum(results['predictions'].values())
        total_actual = sum(v[-1] for v in results['historical']['by_year_level'].values())
        results['metrics']['total'] = {
            'mse': ((total_pred - total_actual) ** 2),
            'rmse': abs(total_pred - total_actual),
            'mape': abs((total_pred - total_actual) / total_actual) * 100 if total_actual != 0 else 0
        }

        # Prepare response data
        response_data = {
            'dates': results['historical']['dates'],
            'values': results['historical']['values'],
            'courses': list(DEPARTMENTS[department.upper()]),
            'course_enrollments': [sum(results['historical']['by_year_level'][yl][-1] for yl in YEAR_LEVELS)],
            'actual_dates': results['historical']['dates'][-4:],
            'actual_values': results['historical']['values'][-4:],
            'forecast_dates': [f'{year}-{semester}'],
            'forecast_values': [total_pred],
            'metrics': results['metrics']
        }

        # Store in session for download feature
        dept = department.lower()
        session[f'{dept}_line_chart'] = json.dumps({
            'dates': response_data['dates'],
            'values': response_data['values']
        })

        session[f'{dept}_bar_chart'] = json.dumps({
            'courses': response_data['courses'],
            'values': response_data['course_enrollments']
        })

        session[f'{dept}_comparison_chart'] = json.dumps({
            'actual_dates': results['historical']['dates'][-4:],
            'actual_values': results['historical']['values'][-4:],
            'forecast_dates': [f'{year}-{semester}'],
            'forecast_values': [total_pred]
        })

        # Bar chart data
        session[f'{dept}_bar_chart'] = {
            'courses': list(DEPARTMENTS[department]),
            'values': [sum(results['historical']['by_year_level'][yl][-1] for yl in YEAR_LEVELS)]
        }

        # Comparison chart data
        session[f'{dept}_comparison_chart'] = {
            'actual_dates': results['historical']['dates'][-4:],
            'actual_values': results['historical']['values'][-4:],
            'forecast_dates': [f'{year}-{semester}' for _ in range(4)],
            'forecast_values': [total_pred] * 4
        }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_chart/<department>/<chart_type>')
def download_chart(department, chart_type):
    try:
        # Get chart data from session and deserialize JSON
        chart_data_json = session.get(f'{department}_{chart_type}_chart')
        if not chart_data_json:
            return jsonify({'error': 'No chart data available'}), 404

        # Create a temporary file
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{department}_{chart_type}_chart.png')

        # Convert chart data to PNG
        plt.figure(figsize=(12, 6))
        if chart_type == 'line':
            plt.plot(chart_data_json['dates'], chart_data_json['values'], marker='o')
            plt.title(f'{department} Enrollment Trend')
            plt.xlabel('Year-Semester')
            plt.ylabel('Number of Students')
        elif chart_type == 'bar':
            plt.bar(chart_data_json['courses'], chart_data_json['values'])
            plt.title(f'{department} Course Distribution')
            plt.xlabel('Course')
            plt.ylabel('Number of Students')
        elif chart_type == 'comparison':
            plt.plot(chart_data_json['actual_dates'], chart_data_json['actual_values'], marker='o', label='Actual')
            plt.plot(chart_data_json['forecast_dates'], chart_data_json['forecast_values'], marker='o', linestyle='--', label='Forecast')
            plt.title(f'{department} Actual vs Forecast')
            plt.xlabel('Year-Semester')
            plt.ylabel('Number of Students')
            plt.legend()

        plt.tight_layout()
        plt.savefig(temp_file, dpi=300, bbox_inches='tight', format='png')
        plt.close()

        return send_file(
            temp_file,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'{department}_{chart_type}_chart.png'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary file
        if 'temp_file' in locals() and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400
    
    if not models:
        return jsonify({'error': 'Models not trained. Please restart the server.'}), 500
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read and process the uploaded file
            df = pd.read_csv(file, index_col='Year_Semester')
            
            # Validate columns
            required_columns = ['BSBA-FM', 'BSBA-MM', 'BSIT', 'BSCS']
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': 'CSV must contain columns: ' + ', '.join(required_columns)}), 400
            
            # Store for predictions
            data_store['prediction_data'] = df
            
            # Make predictions using trained models
            results = {}
            for course in required_columns:
                # Get last sequence for prediction
                scaled_data = scalers[course].transform(df[course].values.reshape(-1, 1))
                last_sequence = scaled_data[-4:]
                
                # Make predictions
                future_predictions = predict_future(models[course], scalers[course], last_sequence, n_future=6)
                
                # Calculate metrics on test data
                X_test, y_test = create_sequences(scaled_data, 4)
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                y_pred = models[course].predict(X_test).flatten()
                y_pred = scalers[course].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = df[course].values[-len(y_pred):]
                
                metrics = calculate_metrics(y_true, y_pred)
                
                # Generate future dates
                last_date = pd.to_datetime(df.index[-1])
                future_dates = []
                for i in range(6):
                    if (i % 2) == 0:
                        future_dates.append(f"{last_date.year + (i//2) + 1}-1")
                    else:
                        future_dates.append(f"{last_date.year + (i//2) + 1}-2")
                
                results[course] = {
                    'historical': {
                        'dates': df.index.tolist(),
                        'values': df[course].tolist()
                    },
                    'predictions': {
                        'dates': future_dates,
                        'values': future_predictions
                    },
                    'metrics': metrics,
                    'model_summary': {
                        'architecture': str(models[course].summary()),
                        'training_history': str(model_histories[course].history)
                    }
                }
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
