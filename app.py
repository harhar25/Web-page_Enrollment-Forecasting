from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib
import json
from prophet.serialize import model_from_json
import pickle
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Explicit course-to-model mapping
MODEL_PATHS = {
    "BSCS": os.path.join(app.config['MODEL_FOLDER'], "BSCS_prophet_model.pkl"),
    "BSIT": os.path.join(app.config['MODEL_FOLDER'], "BSIT_prophet_model.pkl"),
    "BSBA-FINANCIAL_MANAGEMENT": os.path.join(app.config['MODEL_FOLDER'], "BSBA-FINANCIAL_MANAGEMENT_prophet_model.pkl"),
    "BSBA-MARKETING_MANAGEMENT": os.path.join(app.config['MODEL_FOLDER'], "BSBA-MARKETING_MANAGEMENT_prophet_model.pkl")
}

# Create folders if they don’t exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Department and course configurations
DEPARTMENTS = {
    'BED': ['BSBA-FINANCIAL_MANAGEMENT', 'BSBA-MARKETING_MANAGEMENT'],
    'CED': ['BSIT', 'BSCS']
}

YEAR_LEVELS = [1, 2, 3, 4]
YEARS = [str(year) for year in range(2020, 2026)]
SEMESTERS = [1, 2]

# Dictionary to hold Prophet models
models = {}

# Load trained Prophet models from .pkl
def load_models():
    try:
        for file in os.listdir(app.config['MODEL_FOLDER']):
            if file.endswith("_prophet_model.pkl"):
                course_name = file.replace("_prophet_model.pkl", "")
                file_path = os.path.join(app.config['MODEL_FOLDER'], file)
                with open(file_path, "rb") as f:
                    models[course_name] = pickle.load(f)
        print(f"✅ Loaded models: {list(models.keys())}")
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")


def convert_enrollment_to_prophet_format():
    """Convert the enrollees_dataset.csv to Prophet-friendly format for each course"""
    try:
        # Read the enrollment data
        enroll_file = os.path.join(app.config['UPLOAD_FOLDER'], "enrollees_dataset.csv")
        if not os.path.exists(enroll_file):
            print("❌ enrollees_dataset.csv not found")
            return
        
        df = pd.read_csv(enroll_file)
        print(f"✅ Read enrollment data with {len(df)} rows")
        
        # Convert School_Year to proper dates
        def year_to_date(row):
            year_start = int(row['School_Year'].split('-')[0])
            if row['Semester'] == '1st':
                return f"{year_start}-06-01"  # Start of first semester
            else:
                return f"{year_start}-12-01"  # Start of second semester
        
        df['ds'] = df.apply(year_to_date, axis=1)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Group by course and create separate files
        for course in df['Course'].unique():
            course_data = df[df['Course'] == course].copy()
            course_data = course_data.sort_values('ds')
            
            # Create Prophet format data (date, total enrollment)
            prophet_data = course_data[['ds', 'total_enrollees']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Save to course-specific file
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{course}_history.csv")
            prophet_data.to_csv(output_file, index=False)
            print(f"✅ Created historical data for {course}: {len(prophet_data)} records")
            
    except Exception as e:
        print(f"❌ Error converting enrollment data: {str(e)}")


# Load models on startup
# Load models
models = {}
for course_name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[course_name] = pickle.load(f)
print(f"✅ Loaded models: {list(models.keys())}")

convert_enrollment_to_prophet_format()

@app.route("/")
def index():
    return render_template("frontface.html")

@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        course = request.form.get("course")
        year = request.form.get("year")
        semester = request.form.get("semester")

        print(f"DEBUG: Received course='{course}', year='{year}', semester='{semester}'")

        # Validate all fields are provided
        if not course or not year or not semester:
            flash("Please complete all fields")
            return redirect(url_for("bed_filter" if "BSBA" in course else "ced_filter"))

        if course not in models:
            flash("Invalid course selected")
            return redirect(url_for("index"))

        # Determine which department the course belongs to
        if "BSBA" in course:
            target_redirect = "bed_filter"
        else:
            target_redirect = "ced_filter"

        if semester not in ["1", "2"]:
            flash("Please select a valid semester")
            return redirect(url_for(target_redirect))
        
        semester_text = "1st" if semester == "1" else "2nd"

        model = models[course]
        print(f"DEBUG: Model loaded successfully for {course}")

        # Load historical data from enrollees_dataset.csv
        enroll_file = os.path.join(app.config['UPLOAD_FOLDER'], "enrollees_dataset.csv")
        actual_data = []
        labels = []
        
        if os.path.exists(enroll_file):
            print(f"DEBUG: Loading enrollment data from {enroll_file}")
            enroll_df = pd.read_csv(enroll_file)
            
            # Filter for the selected course
            course_enroll = enroll_df[enroll_df['Course'] == course].copy()
            
            if not course_enroll.empty:
                # Convert School_Year to dates
                def year_to_date(row):
                    year_start = int(row['School_Year'].split('-')[0])
                    if row['Semester'] == '1st':
                        return f"{year_start}-06-01"
                    else:
                        return f"{year_start}-12-01"
                
                course_enroll['ds'] = course_enroll.apply(year_to_date, axis=1)
                course_enroll['ds'] = pd.to_datetime(course_enroll['ds'])
                course_enroll = course_enroll.sort_values('ds')
                
                actual_data = course_enroll['total_enrollees'].tolist()
                labels = course_enroll['ds'].dt.strftime('%Y-%m').tolist()
                print(f"DEBUG: Found {len(actual_data)} historical data points for {course}")
            else:
                print("DEBUG: No enrollment data found for this course")
        else:
            print("DEBUG: No enrollment data file found")

        # Forecast next 6 semesters
        print("DEBUG: Generating forecast...")
        future = model.make_future_dataframe(periods=6, freq="6ME")
        forecast = model.predict(future)
        
        # Get all forecast data including confidence intervals
        forecast_labels = forecast['ds'].dt.strftime('%Y-%m').tolist()
        forecast_data = forecast['yhat'].round(2).tolist()
        forecast_lower = forecast['yhat_lower'].round(2).tolist()
        forecast_upper = forecast['yhat_upper'].round(2).tolist()

        # Combine historical + forecast for chart
        combined_labels = labels + forecast_labels[len(labels):]
        # Ensure we only add None for the exact difference
        none_count = len(forecast_labels) - len(labels)
        combined_actual = actual_data + [None] * none_count if none_count > 0 else actual_data
        print(f"DEBUG: Combined labels length: {len(combined_labels)}, Combined actual length: {len(combined_actual)}")
                
        # For confidence intervals, we need to align with forecast data only
        conf_lower = [None]*len(labels) + forecast_lower[len(labels):]
        conf_upper = [None]*len(labels) + forecast_upper[len(labels):]

        print(f"DEBUG: Success! Rendering result with {len(combined_labels)} data points")

        return render_template(
            "BED_result.html",
            course=course,
            year=year,
            semester=semester_text,
            labels=combined_labels,
            forecast_data=forecast_data,
            actual_data=combined_actual,
            forecast_lower=conf_lower,    # Add this
            forecast_upper=conf_upper     # Add this
        )

    except Exception as e:
        print(f"ERROR in forecast: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f"Error: {str(e)}")
        if course and "BSBA" in course:
            return redirect(url_for("bed_filter"))
        else:
            return redirect(url_for("ced_filter"))
        
@app.route('/comparefilter')
def comparefilter():
    return render_template('comparefilter.html')

@app.route('/compare_results', methods=['POST'])
def compare_results():
    department = request.form['department']
    course1 = request.form['course1']
    course2 = request.form['course2']
    return f"Comparing {course1} vs {course2} in {department}"

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        course = request.form.get('course')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded CSV and add to historical data
            process_uploaded_dataset(filepath, course)
            
            return jsonify({'success': True, 'message': 'Dataset uploaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_uploaded_dataset(filepath, course):
    """Process uploaded CSV and add to historical data"""
    try:
        # Read the uploaded CSV
        new_data = pd.read_csv(filepath)
        
        # Convert to Prophet format (adjust based on your CSV structure)
        if 'ds' in new_data.columns and 'y' in new_data.columns:
            # Already in Prophet format
            prophet_data = new_data
        else:
            # Convert from your enrollment format
            prophet_data = convert_enrollment_to_prophet_format(new_data, course)
        
        # Append to existing historical data
        hist_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{course}_history.csv")
        if os.path.exists(hist_file):
            existing_data = pd.read_csv(hist_file)
            combined_data = pd.concat([existing_data, prophet_data]).drop_duplicates()
            combined_data.to_csv(hist_file, index=False)
        else:
            prophet_data.to_csv(hist_file, index=False)
            
    except Exception as e:
        print(f"Error processing uploaded dataset: {str(e)}")
        raise

@app.route("/bed_filter")
def bed_filter():
    return render_template("BEDfilter.html")

@app.route("/ced_filter")
def ced_filter():
    return render_template("CEDfilter.html")

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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        department = data.get('department')
        course = data.get('course')
        year = data.get('year')
        semester = data.get('semester')

        # Validate inputs
        if not all([department, course, year, semester]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Year validation
        current_year = datetime.now().year
        target_year = int(year.split("-")[0])
        if target_year > current_year + 3:
            return jsonify({'error': 'The model can only predict 3 years ahead at least for now'}), 400

        # Load model
        if course not in models:
            return jsonify({'error': f"No trained model found for {course}"}), 404
        model = models[course]

        # Map semester → approximate date
        if semester == "1st":
            target_date = pd.to_datetime(f"{target_year}-06-01")
        else:
            target_date = pd.to_datetime(f"{target_year}-12-01")

        # Create future DataFrame for Prophet
        future = pd.DataFrame({"ds": [target_date]})
        forecast = model.predict(future)

        # Extract forecast values
        predicted = round(float(forecast.iloc[0]['yhat']), 2)
        lower = round(float(forecast.iloc[0]['yhat_lower']), 2)
        upper = round(float(forecast.iloc[0]['yhat_upper']), 2)

        # Generate a unique ID for this prediction
        prediction_id = str(uuid.uuid4())
        
        # Store prediction data temporarily in session
        session[prediction_id] = {
            'course': course,
            'department': department,
            'year': year,
            'semester': semester,
            'predicted': predicted,
            'lower': lower,
            'upper': upper,
            'target_date': target_date.strftime('%Y-%m-%d')
        }

        return jsonify({
            "prediction_id": prediction_id,
            "course": course,
            "department": department,
            "year": year,
            "semester": semester,
            "predicted_enrollment": predicted,
            "lower_bound": lower,
            "upper_bound": upper
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/prediction_chart/<prediction_id>')
def prediction_chart(prediction_id):
    try:
        # Retrieve prediction data from session
        prediction_data = session.get(prediction_id)
        if not prediction_data:
            return "Prediction not found", 404
        
        course = prediction_data['course']
        target_date = pd.to_datetime(prediction_data['target_date'])
        predicted = prediction_data['predicted']
        lower = prediction_data['lower']
        upper = prediction_data['upper']
        
        # Load historical data if exists
        enroll_file = os.path.join(app.config['UPLOAD_FOLDER'], "enrollees_dataset.csv")
        if os.path.exists(enroll_file):
            enroll_df = pd.read_csv(enroll_file)
            course_enroll = enroll_df[enroll_df['Course'] == course].copy()
            
            if not course_enroll.empty:
                # Convert School_Year to dates
                def year_to_date(row):
                    year_start = int(row['School_Year'].split('-')[0])
                    if row['Semester'] == '1st':
                        return f"{year_start}-06-01"
                    else:
                        return f"{year_start}-12-01"
                
                course_enroll['ds'] = course_enroll.apply(year_to_date, axis=1)
                course_enroll['ds'] = pd.to_datetime(course_enroll['ds'])
                course_enroll = course_enroll.sort_values('ds')
                
                labels = course_enroll['ds'].dt.strftime('%Y-%m').tolist() + [target_date.strftime('%Y-%m')]
                actual_data = course_enroll['total_enrollees'].tolist() + [None]
            else:
                labels = [target_date.strftime('%Y-%m')]
                actual_data = [None]
        else:
            labels = [target_date.strftime('%Y-%m')]
            actual_data = [None]

        forecast_data = [predicted]

        # Create chart
        plt.figure(figsize=(10, 5))
        plt.plot(labels[:-1], actual_data[:-1], marker='o', label='Historical', color='blue')
        plt.plot(labels[-1:], forecast_data, marker='o', label='Prediction', color='red')
        plt.fill_between(labels[-1:], [lower], [upper], color='pink', alpha=0.4, label='Confidence Interval')
        plt.title(f'{course} Enrollment Prediction')
        plt.xlabel('Year-Month')
        plt.ylabel('Number of Students')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart to memory
        from io import BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        plt.close()
        img_buffer.seek(0)
        
        # Return image
        return send_file(img_buffer, mimetype='image/png', download_name=f'{course}_prediction.png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/select_department", methods=["POST"])
def select_department():
    department = request.form.get("department")
    if department == "BED":
        return redirect(url_for("bed_filter"))
    elif department == "CED":
        return redirect(url_for("ced_filter"))
    else:
        flash("Invalid department selected")
        return redirect(url_for("index"))


@app.route('/download_chart/<department>/<chart_type>')
def download_chart(department, chart_type):
    try:
        chart_data_json = session.get(f'{department}_{chart_type}')
        if not chart_data_json:
            return jsonify({'error': 'No chart data available'}), 404

        chart_data = json.loads(chart_data_json)
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{department}_{chart_type}.png')

        # Plot chart
        plt.figure(figsize=(12, 6))
        if chart_type == 'forecast':
            plt.plot(chart_data['forecast_dates'], chart_data['forecast_values'], marker='o', label='Forecast')
            plt.fill_between(chart_data['forecast_dates'], chart_data['lower_bound'], chart_data['upper_bound'], color='lightblue', alpha=0.4)
            plt.title(f'{department} Forecast')
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
            download_name=f'{department}_{chart_type}.png'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'temp_file' in locals() and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
# Add this new route for visualization charts
@app.route('/visualization/<course>')
def visualization(course):
    try:
        # Load historical data
        enroll_file = os.path.join(app.config['UPLOAD_FOLDER'], "enrollees_dataset.csv")
        actual_data = []
        labels = []
        
        if os.path.exists(enroll_file):
            enroll_df = pd.read_csv(enroll_file)
            course_enroll = enroll_df[enroll_df['Course'] == course].copy()
            
            if not course_enroll.empty:
                # Convert School_Year to dates
                def year_to_date(row):
                    year_start = int(row['School_Year'].split('-')[0])
                    if row['Semester'] == '1st':
                        return f"{year_start}-06-01"
                    else:
                        return f"{year_start}-12-01"
                
                course_enroll['ds'] = course_enroll.apply(year_to_date, axis=1)
                course_enroll['ds'] = pd.to_datetime(course_enroll['ds'])
                course_enroll = course_enroll.sort_values('ds')
                
                actual_data = course_enroll['total_enrollees'].tolist()
                labels = course_enroll['ds'].dt.strftime('%Y-%m').tolist()

        return render_template(
            "visualization.html",
            course=course,
            labels=labels,
            actual_data=actual_data
        )

    except Exception as e:
        print(f"ERROR in visualization: {str(e)}")
        flash(f"Error loading visualization: {str(e)}")
        return redirect(url_for('index'))

# Add this new function to handle visualization data
@app.route('/get_visualization_data/<course>')
def get_visualization_data(course):
    try:
        enroll_file = os.path.join(app.config['UPLOAD_FOLDER'], "enrollees_dataset.csv")
        actual_data = []
        labels = []
        
        if os.path.exists(enroll_file):
            enroll_df = pd.read_csv(enroll_file)
            course_enroll = enroll_df[enroll_df['Course'] == course].copy()
            
            if not course_enroll.empty:
                # Convert School_Year to dates
                def year_to_date(row):
                    year_start = int(row['School_Year'].split('-')[0])
                    if row['Semester'] == '1st':
                        return f"{year_start}-06-01"
                    else:
                        return f"{year_start}-12-01"
                
                course_enroll['ds'] = course_enroll.apply(year_to_date, axis=1)
                course_enroll['ds'] = pd.to_datetime(course_enroll['ds'])
                course_enroll = course_enroll.sort_values('ds')
                
                actual_data = course_enroll['total_enrollees'].tolist()
                labels = course_enroll['ds'].dt.strftime('%Y-%m').tolist()

        return jsonify({
            'labels': labels,
            'actual_data': actual_data,
            'course': course
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
