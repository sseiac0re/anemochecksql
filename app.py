"""
AnemoCheck - Anemia Classification Web Application
-------------------------------------------------
This Flask application provides a web interface for the anemia classification system.
It features user authentication, real-time updates, and a comprehensive admin dashboard.

Date: April 28, 2025
"""

import os
import json
import logging
# SMTP imports removed - now using Brevo API
# import smtplib
import random
import string
from datetime import datetime, timedelta
from timezone_utils import get_philippines_time, format_philippines_time, get_philippines_time_for_db, get_philippines_time_plus_minutes, format_philippines_time_ampm
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms import StringField, PasswordField, FloatField, TextAreaField, BooleanField, SelectField, SubmitField, HiddenField, IntegerField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import io
import base64

import database as db
import simple_chat
from email_service import send_result_email_brevo, send_otp_email_brevo


from anemia_model import AnemiaCBCModel
import joblib
from xgboost_ml_module import xgboost_predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'a-very-secret-key-for-anemocheck'
csrf = CSRFProtect(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize anemia model
anemia_model = AnemiaCBCModel()


# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = user_data['id']
        self.username = user_data['username']
        self.email = user_data['email']
        self.first_name = user_data['first_name']
        self.last_name = user_data['last_name']
        self.gender = user_data['gender']
        self.date_of_birth = user_data['date_of_birth']
        self.medical_id = user_data['medical_id']
        self.is_admin = user_data['is_admin']
        self.created_at = user_data['created_at']
        self.last_login = user_data['last_login']


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    user_data = db.get_user(user_id)
    if user_data:
        return User(user_data)
    return None


# Form classes
class LoginForm(FlaskForm):
    """Login form."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    """Registration form."""
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    first_name = StringField('First Name', validators=[Length(max=64)])
    last_name = StringField('Last Name', validators=[Length(max=64)])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    date_of_birth = StringField('Date of Birth (DD-MM-YYYY)', validators=[Optional()])
    medical_id = StringField('Medical ID (Optional)', validators=[Optional(), Length(max=64)])
    submit = SubmitField('Register')


class ProfileForm(FlaskForm):
    """Form for updating user profile."""
    first_name = StringField('First Name', validators=[Length(max=64)])
    last_name = StringField('Last Name', validators=[Length(max=64)])
    username = StringField('Username', validators=[Length(max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')])
    date_of_birth = StringField('Date of Birth (DD-MM-YYYY)', validators=[Optional()])
    medical_id = StringField('Medical ID', validators=[Optional(), Length(max=64)])
    current_password = PasswordField('Current Password', validators=[Optional()])
    new_password = PasswordField('New Password', validators=[Optional(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password', validators=[EqualTo('new_password')])
    submit = SubmitField('Update Profilaze')


class MedicalDataForm(FlaskForm):
    """Form for updating medical data."""
    height = FloatField('Height (cm)', validators=[Optional(), NumberRange(min=50, max=250)])
    weight = FloatField('Weight (kg)', validators=[Optional(), NumberRange(min=1, max=500)])
    blood_type = SelectField('Blood Type', choices=[
        ('', 'Unknown'),
        ('A+', 'A+'), ('A-', 'A-'),
        ('B+', 'B+'), ('B-', 'B-'),
        ('AB+', 'AB+'), ('AB-', 'AB-'),
        ('O+', 'O+'), ('O-', 'O-')
    ], validators=[Optional()])
    medical_conditions = TextAreaField('Medical Conditions', validators=[Optional(), Length(max=1000)])
    medications = TextAreaField('Current Medications', validators=[Optional(), Length(max=1000)])
    submit = SubmitField('Update Medical Data')


class CBCForm(FlaskForm):
    """Form for CBC data input."""
    hemoglobin = FloatField('Hemoglobin (g/dL)', validators=[
        DataRequired(), 
        NumberRange(min=1, max=25, message='Please enter a valid value between 1 and 25')
    ])
    notes = TextAreaField('Notes', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Detect Anemia')


class AdminUserForm(FlaskForm):
    """Form for admin to edit user data."""
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    first_name = StringField('First Name', validators=[Length(max=64)])
    last_name = StringField('Last Name', validators=[Length(max=64)])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')])
    date_of_birth = StringField('Date of Birth (DD-MM-YYYY)', validators=[Optional(), Length(max=10)])
    medical_id = StringField('Medical ID', validators=[Optional(), Length(max=64)])
    is_admin = BooleanField('Administrator')
    password = PasswordField('New Password (Leave blank to keep unchanged)', validators=[Optional(), Length(min=8)])
    user_id = HiddenField('User ID')
    submit = SubmitField('Update User')


class SystemSettingsForm(FlaskForm):
    """Form for system settings."""
    # General Settings
    site_name = StringField('Site Name', validators=[DataRequired(), Length(max=100)])
    site_description = TextAreaField('Site Description', validators=[Length(max=500)])
    max_users = IntegerField('Maximum Users', validators=[DataRequired(), NumberRange(min=1)])
    session_timeout = IntegerField('Session Timeout (minutes)', validators=[DataRequired(), NumberRange(min=5)])
    
    # ML Model Settings
    model_confidence_threshold = FloatField('Model Confidence Threshold', validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    model_version = StringField('Model Version', validators=[DataRequired(), Length(max=50)])
    enable_auto_retrain = BooleanField('Enable Auto Retrain')
    
    # Email Settings (Brevo API)
    brevo_api_key = PasswordField('Brevo API Key', validators=[Length(max=200)])
    brevo_sender_email = StringField('Sender Email', validators=[Email(), Length(max=100)])
    brevo_sender_name = StringField('Sender Name', validators=[Length(max=100)])
    enable_email_notifications = BooleanField('Enable Email Notifications')
    
    # Security Settings
    password_min_length = IntegerField('Minimum Password Length', validators=[DataRequired(), NumberRange(min=6)])
    max_login_attempts = IntegerField('Maximum Login Attempts', validators=[DataRequired(), NumberRange(min=1)])
    enable_two_factor = BooleanField('Enable Two-Factor Authentication')
    
    submit = SubmitField('Save Settings')


# Routes
@app.route('/')
def index():
    """Home page."""
    if current_user.is_authenticated and current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    
    form = CBCForm()
    return render_template('index.html', form=form)


@app.route('/about')
def about():
    """About page with chart data."""
    # Temporary static dataset
    hemoglobin_values = [13.2, 12.8, 13.5, 14.0, 13.7]
    dates = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]

    return render_template(
        'about.html',
        hemoglobin_values=hemoglobin_values,
        dates=dates
    )



@app.route('/faq')
def faq():
    """FAQ page."""
    return render_template('faq.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    
    # Check if user just completed registration
    if request.args.get('registered') == 'true':
        flash('Registration successful! Please log in with your credentials.', 'success')
    
    form = LoginForm()
    if form.validate_on_submit():
        success, result = db.verify_user(form.username.data, form.password.data)
        if success:
            user = User(result)
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                # Redirect to admin dashboard if user is admin, otherwise dashboard
                if user.is_admin:
                    next_page = url_for('admin_dashboard')
                else:
                    next_page = url_for('dashboard')
            return redirect(next_page)
        else:
            flash(result)
    
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page - Step 1: Collect user data and send OTP."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Generate OTP and store user data temporarily
        otp_code = generate_otp()
        expires_at = get_philippines_time_plus_minutes(OTP_EXPIRY_MINUTES)
        
        # Store OTP verification data
        success = db.store_otp_verification(
            email=form.email.data,
            otp_code=otp_code,
            username=form.username.data,
            password_hash=generate_password_hash(form.password.data),
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            gender=form.gender.data,
            date_of_birth=form.date_of_birth.data,
            medical_id=form.medical_id.data,
            expires_at=expires_at
        )
        
        if success:
            # Send OTP email
            email_sent = send_otp_email_brevo(form.email.data, otp_code, form.username.data)
            if email_sent:
                # Store email and username in session for verification step
                session['pending_email'] = form.email.data
                session['pending_username'] = form.username.data
                flash('Verification code sent to your email. Please check your inbox.')
                return redirect(url_for('verify_registration'))
            else:
                flash('Failed to send verification email. Please try again.')
        else:
            flash('Registration failed. Please try again.')
    
    return render_template('register.html', form=form)


@app.route('/verify-registration', methods=['GET', 'POST'])
@csrf.exempt
def verify_registration():
    """OTP verification page - Step 2: Verify OTP and complete registration."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    # Check if user has pending email
    pending_email = session.get('pending_email')
    if not pending_email:
        flash('No pending registration found. Please register first.', 'error')
        return redirect(url_for('register'))
    
    if request.method == 'POST':
        logger.info("OTP verification form submitted")
        logger.info(f"CSRF token in form: {request.form.get('csrf_token')}")
        logger.info(f"CSRF token in headers: {request.headers.get('X-CSRFToken')}")
        logger.info(f"Form data: {dict(request.form)}")
        
        # Check if this is an AJAX request (by checking if it's a fetch request)
        is_ajax = request.headers.get('Content-Type') == 'application/x-www-form-urlencoded'
        
        otp_code = request.form.get('otp_code', '').strip()
        
        if not otp_code:
            if is_ajax:
                return jsonify({'success': False, 'error': 'Please enter the verification code.'}), 400
            else:
                flash('Please enter the verification code.')
                return render_template('verify_registration.html', email=pending_email)
        
        # Verify OTP code
        logger.info(f"Verifying OTP for email: {pending_email}")
        logger.info(f"OTP code received: {otp_code}")
        logger.info(f"OTP code length: {len(otp_code)}")
        
        user_data = db.verify_otp_code(pending_email, otp_code)
        logger.info(f"OTP verification result: {user_data is not None}")
        
        if user_data:
            # Create the user account
            success, result = db.create_user(
                username=user_data['username'],
                password_hash=user_data['password_hash'],
                email=pending_email,
                first_name=user_data['first_name'],
                last_name=user_data['last_name'],
                gender=user_data['gender'],
                date_of_birth=user_data['date_of_birth'],
                medical_id=user_data['medical_id']
            )
            
            if success:
                # Clean up session and OTP data
                session.pop('pending_email', None)
                session.pop('pending_username', None)
                db.cleanup_expired_otp()
                
                if is_ajax:
                    return jsonify({'success': True, 'message': 'Registration successful! Please log in.'})
                else:
                    flash('Registration successful! Please log in.')
                    return redirect(url_for('login'))
            else:
                if is_ajax:
                    return jsonify({'success': False, 'error': f'Registration failed: {result}'}), 400
                else:
                    flash(f'Registration failed: {result}', 'error')
        else:
            if is_ajax:
                return jsonify({'success': False, 'error': 'Invalid or expired verification code. Please try again.'}), 400
            else:
                flash('Invalid or expired verification code. Please try again.', 'error')
    
    return render_template('verify_registration.html', email=pending_email)


@app.route('/resend-verification-otp', methods=['POST'])
@csrf.exempt
def resend_verification_otp():
    """Resend verification OTP for registration."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'success': False, 'error': 'Email is required'})
        
        # Check if email has pending registration
        pending_email = session.get('pending_email')
        if not pending_email or pending_email != email:
            return jsonify({'success': False, 'error': 'No pending registration found for this email'})
        
        # Generate new OTP
        otp_code = generate_otp()
        
        # Update OTP in database (set expiry to 10 minutes from now)
        expires_at = get_philippines_time_plus_minutes(10)
        otp_updated = db.update_otp_code(email, otp_code, expires_at)
        
        if not otp_updated:
            return jsonify({'success': False, 'error': 'No pending registration found for this email'})
        
        # Send OTP email
        # Get username from session or use email as fallback
        username = session.get('pending_username', email.split('@')[0])
        email_sent = send_otp_email_brevo(email, otp_code, username)
        
        if not email_sent:
            return jsonify({'success': False, 'error': 'Failed to send verification email. Please try again.'})
        
        logger.info(f"Verification OTP resent to {email}")
        return jsonify({'success': True, 'message': 'Verification code has been resent to your email'})
        
    except Exception as e:
        logger.error(f"Error resending verification OTP: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to resend verification code. Please try again.'})


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard."""
    # Get user's recent classification history
    history = db.get_user_classification_history(current_user.id, limit=5)
    
    # Get medical data
    medical_data = db.get_medical_data(current_user.id)
    
    # Create form for hemoglobin input
    form = CBCForm()
    
    # Prepare data for charts
    hemoglobin_values = []
    rbc_values = []
    hct_values = []
    mcv_values = []
    dates = []
    
    for record in history:
        hemoglobin_values.append(record['hgb'])
        rbc_values.append(record['rbc'])
        mcv_values.append(record['mcv'])
        hct_values.append(record['hct'])
        # Convert timestamp to Philippines time for display
        from timezone_utils import parse_philippines_time
        created_at = parse_philippines_time(record['created_at'])
        if created_at:
            dates.append(created_at.strftime('%Y-%m-%d'))
        else:
            dates.append(record['created_at'][:10])  # Fallback to first 10 chars
    
    # Reverse lists to show chronological order
    hemoglobin_values.reverse()
    dates.reverse()
    return render_template(
        'dashboard.html',
        history=history,
        medical_data=medical_data,
        hemoglobin_values=hemoglobin_values,
        rbc_values=rbc_values,
        mcv_values=mcv_values,
        hct_values=hct_values,
        dates=dates,
        form=form
    )


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page."""
    form = ProfileForm()
    
    if request.method == 'GET':
        # Pre-populate form with current user data
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email
        form.gender.data = current_user.gender
        form.date_of_birth.data = current_user.date_of_birth
        form.medical_id.data = current_user.medical_id
    
    if form.validate_on_submit():
        # Check current password if provided
        if form.current_password.data:
            success, _ = db.verify_user(current_user.username, form.current_password.data)
            if not success:
                flash('Current password is incorrect.')
                return redirect(url_for('profile'))
            
            # If new password is provided, update it
            if form.new_password.data:
                success, result = db.update_user(
                    current_user.id,
                    password=form.new_password.data
                )
                if not success:
                    flash(f'Password update failed: {result}')
                    return redirect(url_for('profile'))
        
        # Update user information
        success, result = db.update_user(
            current_user.id,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            gender=form.gender.data,
            date_of_birth=form.date_of_birth.data,
            medical_id=form.medical_id.data
        )
        
        if success:
            flash('Profile updated successfully!')
            return redirect(url_for('profile'))
        else:
            flash(f'Profile update failed: {result}')
    
    return render_template('profile.html', form=form)


@app.route('/medical-data', methods=['GET', 'POST'])
@login_required
def medical_data():
    """User medical data page."""
    form = MedicalDataForm()
    medical_data = db.get_medical_data(current_user.id)
    
    if request.method == 'GET' and medical_data:
        # Pre-populate form with current medical data
        form.height.data = medical_data.get('height')
        form.weight.data = medical_data.get('weight')
        form.blood_type.data = medical_data.get('blood_type')
        form.medical_conditions.data = medical_data.get('medical_conditions')
        form.medications.data = medical_data.get('medications')
    
    if form.validate_on_submit():
        # Update medical data
        success, result = db.update_medical_data(
            current_user.id,
            height=form.height.data,
            weight=form.weight.data,
            blood_type=form.blood_type.data,
            medical_conditions=form.medical_conditions.data,
            medications=form.medications.data
        )
        
        if success:
            flash('Medical data updated successfully!')
            return redirect(url_for('medical_data'))
        else:
            flash(f'Medical data update failed: {result}')
    
    return render_template('medical_data.html', form=form, medical_data=medical_data)


@app.route('/history')
@login_required
def history():
    """User classification history page."""
    # Get page parameter for pagination
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    
    # Get user's classification history with pagination
    history_data = db.get_user_classification_history_paginated(current_user.id, page=page, per_page=5)
    
    # Format timestamps with AM/PM for each record
    for record in history_data['records']:
        if 'created_at' in record:
            record['created_at'] = format_philippines_time_ampm(record['created_at'])
        
        # Prepare cleaned note (strip legacy embedded patient info)
        raw_notes = (record.get('notes') or '').strip()
        cleaned = raw_notes
        if raw_notes.lower().startswith('patient:'):
            temp = raw_notes
            # Remove labeled prefixes in order if present at the beginning, tolerant to '.' or ',' as terminators
            for label in ('Patient:', 'Age:', 'Gender:'):
                if temp.lower().startswith(label.lower()):
                    sub = temp[len(label):]
                    # find next '.' or ',' as end of the labeled value
                    dot_idx = sub.find('.')
                    comma_idx = sub.find(',')
                    end_idx = -1
                    if dot_idx == -1 and comma_idx == -1:
                        end_idx = -1
                    elif dot_idx == -1:
                        end_idx = comma_idx
                    elif comma_idx == -1:
                        end_idx = dot_idx
                    else:
                        end_idx = min(dot_idx, comma_idx)
                    if end_idx != -1:
                        temp = sub[end_idx + 1:].lstrip()
                    else:
                        temp = sub.lstrip()
            cleaned = temp.strip()
        record['note_clean'] = cleaned
    
    return render_template('history.html', history_data=history_data)


@app.route('/classify', methods=['POST'])
@login_required
def classify():
    """Process the hemoglobin data and make a prediction."""
    form = CBCForm()
    
    if form.validate_on_submit():
        hemoglobin = form.hemoglobin.data
        notes = form.notes.data
        
        # Get prediction from model
        result = anemia_model.predict(hemoglobin)
        
        # Save the record to database
        record_id = db.add_classification_record(
            user_id=current_user.id,
            hemoglobin=hemoglobin,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            recommendation=result['recommendation'],
            notes=notes
        )
        
        # Emit real-time update via WebSocket
        classification_data = {
            'id': record_id,
            'hemoglobin': hemoglobin,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'recommendation': result['recommendation'],
            'created_at': get_philippines_time_for_db(),
            'notes': notes,
            'user_id': current_user.id,
            'username': current_user.username
        }
        
        # Emit to the user's room
        socketio.emit('new_classification', classification_data, room=str(current_user.id))
        
        # Also emit to admin room if this is a normal user
        if not current_user.is_admin:
            socketio.emit('admin_new_classification', classification_data, room='admin_room')
        
        # Redirect to result page
        return redirect(url_for('result', record_id=record_id))
    
    return redirect(url_for('index'))

@app.route('/rfcclasify', methods=['POST'])
@login_required
def rfcclasify():
    """Process the hemoglobin data and make a prediction."""
    #try:
    form = CBCForm()
    
    submit = request.form.get('submit')
    
    if submit is None:
        return "Error: Submit button not clicked or missing in the form data.", 400

    wbc = request.form.get("wbc")
    rbc = request.form.get("rbc")
    hgb = request.form.get("hgb")
    hct = request.form.get("hct")
    mcv = request.form.get("mcv")
    mch = request.form.get("mch")
    mchc = request.form.get("mchc")
    plt = request.form.get("plt")
    notes = request.form.get("notes")  # Corrected from "mcv" to "notes"

    # # Example birth date as a string
    # birth_date_str = current_user.date_of_birth

    # # Convert string to date object
    # birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()

    # # Get today's date
    # today = datetime.today().date()

    # # Calculate age
    # age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    # gender = 0 if current_user.gender.lower() == "female" else 0
    
    model = joblib.load('best_rf_anemia_model.joblib')

    input_data = pd.DataFrame([{
        'WBC': float(wbc),
        'RBC': float(rbc),
        'HGB': float(hgb),
        'HCT': float(hct),
        'MCV': float(mcv),
        'MCH': float(mch),
        'MCHC': float(mchc),
        'PLT': float(plt)
    }])

    # Predict class and probability
    probabilities = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]

    # Label and confidence
    label_mapping = {0: 'Anemia', 1: 'Normal'}
    predicted_label = label_mapping[prediction]
    confidence = round(probabilities[prediction] * 100, 2)

    # Recommendations dictionary
    recommendations = {
        'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
        'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
                "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
        'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                    "Further testing might be needed to determine the underlying cause.",
        'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                "provider is necessary as severe anemia can lead to serious complications."
    }

    # Determine severity if Anemia
    if predicted_label == 'Normal':
        final_recommendation = recommendations['Normal']
        severity = 'None'
    else:
        if confidence >= 90:
            severity = 'Severe'
        elif confidence >= 75:
            severity = 'Moderate'
        else:
            severity = 'Mild'
        predicted_label = severity
        final_recommendation = recommendations[severity]

    # Output
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence}%")
    print(f"Recommendation: {final_recommendation}")


        # Save the record to database
    record_id = db.add_classification_record(
        user_id=current_user.id,
        wbc=float(wbc),
        rbc=float(rbc),
        hgb=float(hgb),
        hct=float(hct),
        mcv=float(mcv),
        mch=float(mch),
        mchc=float(mchc),
        plt=float(plt),
        predicted_class=predicted_label,
        confidence=confidence/100,
        recommendation=final_recommendation,
        notes=notes
    )

    
    

        # Emit real-time update via WebSocket
    classification_data = {
        'id': record_id,
        'user_id': current_user.id,
        'username': current_user.username,
        'wbc': wbc,
        'rbc': rbc,
        'hgb': hgb,
        'hct': hct,
        'mcv': mcv,
        'mch': mch,
        'mchc': mchc,
        'plt': plt,
        'predicted_class': predicted_label,
        'confidence': confidence / 100,
        'recommendation': final_recommendation,
        'created_at': get_philippines_time_for_db(),
        'notes': notes
    }

    # Emit to the user's room
    socketio.emit('new_classification', classification_data, room=str(current_user.id))
    
    # Also emit to admin room if this is a normal user
    if not current_user.is_admin:
        socketio.emit('admin_new_classification', classification_data, room='admin_room')
    
    # Redirect to result page
    return redirect(url_for('result', record_id=record_id))


@app.route('/xgbclasify', methods=['POST'])
@login_required
def xgbclasify():
    """Process the hemoglobin data and make a prediction."""
    #try:
    form = CBCForm()
    
    submit = request.form.get('submit')
    
    if submit is None:
        return "Error: Submit button not clicked or missing in the form data.", 400
    
    wbc = float(request.form.get("WBC"))
    rbc = float(request.form.get("RBC"))
    hgb = float(request.form.get("HEMOGLOBIN"))
    hct = float(request.form.get("HEMATOCRIT"))
    mcv = float(request.form.get("MCV"))
    mch = float(request.form.get("MCH"))
    mchc = float(request.form.get("MCHC"))
    plt = float(request.form.get("PLATELET"))
    neutrophils = float(request.form.get("NEUTROPHILS"))
    lymphocytes = float(request.form.get("LYMPHOCYTES"))
    monocytes = float(request.form.get("MONOCYTES"))
    eosinophils = float(request.form.get("EUSONIPHILS"))
    basophil = float(request.form.get("BASOPHIL"))
    # Default to median value from training data (0.8) if not provided
    # Using 0.0 was causing different predictions because training data median is 0.8
    # Explicitly check if field is empty vs 0 - empty should default to 0.8, but 0 should stay 0
    immature_granulocytes_input = request.form.get("IMMATURE_GRANULYTES", "").strip()
    if immature_granulocytes_input == "":
        immature_granulocytes = 0.8  # Default when field is left empty
    else:
        immature_granulocytes = float(immature_granulocytes_input)  # Use actual value (including 0)
    notes = request.form.get("notes")  # Only user's notes

    # Check if classifying for another person via toggle only
    classify_other_person = request.form.get("classify_other_person") == "on"

    # Debug logging of incoming data
    logger.info("/xgbclasify classify_other_person=%s", classify_other_person)
    logger.info("other_person_name=%s", request.form.get("other_person_name"))
    logger.info("other_person_age=%s", request.form.get("other_person_age"))
    logger.info("other_person_gender=%s", request.form.get("other_person_gender"))

    if classify_other_person:
        # Use alternative person's information
        other_person_age = request.form.get("other_person_age")
        other_person_gender = request.form.get("other_person_gender")
        other_person_name = request.form.get("other_person_name", "").strip()
        
        if not other_person_age or not other_person_gender:
            flash("Age and gender are required when classifying for another person.", "error")
            return redirect(url_for('dashboard'))
        
        age = int(float(other_person_age))
        gender = 1 if other_person_gender.lower() == "female" else 0
        
        # Append person's name, age, and gender to notes so history/admin can display it
        patient_info = f"Patient: {other_person_name}. Age: {other_person_age}. Gender: {other_person_gender.capitalize()}."
        if notes:
            notes = patient_info + " " + notes
        else:
            notes = patient_info
    else:
        # Use logged-in user's information (default behavior)
        birth_date_str = current_user.date_of_birth
        
        if not birth_date_str:
            flash("Please update your date of birth in your profile to use this feature.", "error")
            return redirect(url_for('dashboard'))

        # Convert string to date object
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()

        # Get today's date
        today = datetime.today().date()

        # Calculate age
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        gender = 1 if current_user.gender and current_user.gender.lower() == "female" else 0

    user_input = [
        age,
        gender,
        wbc,
        rbc,
        hgb,
        hct,
        mcv,
        mch,
        mchc,
        plt,
        neutrophils,
        lymphocytes,
        monocytes,
        eosinophils,
        basophil,
        immature_granulocytes
    ]

    print(user_input)
    predicted_label,confidence_scores = xgboost_predict(user_input)
    # Recommendations dictionary
    recommendations = {
        'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
        'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
                "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
        'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                    "Further testing might be needed to determine the underlying cause.",
        'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                "provider is necessary as severe anemia can lead to serious complications."
    }

    final_recommendation = recommendations[predicted_label]

    # Output
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {round(confidence_scores*100, 2)}%")
    print(f"Recommendation: {final_recommendation}")


        # Save the record to database
    record_id = db.add_classification_record(
        user_id=current_user.id,
        wbc = float(wbc),
        rbc = float(rbc),
        hgb = float(hgb),
        hct = float(hct),
        mcv = float(mcv),
        mch = float(mch),
        mchc = float(mchc),
        plt = float(plt),
        neutrophils = float(neutrophils),
        lymphocytes = float(lymphocytes),
        monocytes = float(monocytes),
        eosinophils = float(eosinophils),
        basophil = float(basophil),
        immature_granulocytes = float(immature_granulocytes),
        predicted_class=predicted_label,
        confidence=float(confidence_scores),
        recommendation=final_recommendation,
        notes=notes
    )

    
    

        # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        return obj
    
    # Emit real-time update via WebSocket
    classification_data = {
        'id': record_id,
        'user_id': current_user.id,
        'username': current_user.username,
        'wbc': convert_numpy_types(wbc),
        'rbc': convert_numpy_types(rbc),
        'hgb': convert_numpy_types(hgb),
        'hct': convert_numpy_types(hct),
        'mcv': convert_numpy_types(mcv),
        'mch': convert_numpy_types(mch),
        'mchc': convert_numpy_types(mchc),
        'plt': convert_numpy_types(plt),
        'neutrophils': convert_numpy_types(neutrophils),
        'lymphocytes': convert_numpy_types(lymphocytes),
        'monocytes': convert_numpy_types(monocytes),
        'eosinophils': convert_numpy_types(eosinophils),
        'basophil': convert_numpy_types(basophil),
        'immature_granulocytes': convert_numpy_types(immature_granulocytes),
        'predicted_class': predicted_label,
        'confidence': convert_numpy_types(confidence_scores),
        'recommendation': final_recommendation,
        'created_at': get_philippines_time_for_db(),
        'notes': notes,
        'age': age,
        'gender': current_user.gender
    }

    # Emit to the user's room
    socketio.emit('new_classification', classification_data, room=str(current_user.id))
    
    # Also emit to admin room if this is a normal user
    if not current_user.is_admin:
        socketio.emit('admin_new_classification', classification_data, room='admin_room')
    
    # Automatically send email with results
    try:
        user_data = db.get_user_by_id(current_user.id)
        if user_data:
            # Prepare record data for email
            record_data = {
                'predicted_class': predicted_label,
                'confidence': float(confidence_scores),
                'wbc': float(wbc),
                'rbc': float(rbc),
                'hgb': float(hgb),
                'hct': float(hct),
                'mcv': float(mcv),
                'mch': float(mch),
                'mchc': float(mchc),
                'plt': float(plt),
                'created_at': get_philippines_time_for_db(),
                'notes': notes
            }
            
            # Send email using Brevo API
            success, message = send_result_email_brevo(
                record_id, 
                user_data['email'], 
                f"{user_data['first_name']} {user_data['last_name']}".strip() or user_data['username'],
                record_data
            )
            
            if success:
                logger.info(f"Auto-email sent successfully to {user_data['email']}")
            else:
                logger.warning(f"Auto-email failed: {message}")
    except Exception as e:
        logger.error(f"Error sending auto-email: {str(e)}")
    
    # Redirect to result page
    return redirect(url_for('result', record_id=record_id))            

@app.route('/api/classification-stats')
@login_required
def get_classification_stats():
    """Get gender and age statistics for visualization."""
    try:
        # Get all classification records
        records = db.get_all_classification_history(limit=1000)
        
        # Process data for visualization
        gender_stats = {'Male': 0, 'Female': 0, 'Other': 0}
        age_groups = {'0-18': 0, '19-30': 0, '31-50': 0, '51-70': 0, '70+': 0}
        classification_stats = {'Normal': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0}
        
        for record in records:
            # Get user data
            user_data = db.get_user_by_id(record['user_id'])
            if user_data:
                # Gender statistics (normalize to title-case buckets)
                raw_gender = (user_data.get('gender') or 'other').strip().lower()
                if raw_gender in ('male', 'm'):
                    bucket = 'Male'
                elif raw_gender in ('female', 'f'):
                    bucket = 'Female'
                else:
                    bucket = 'Other'
                gender_stats[bucket] += 1
                
                # Age calculation (if date_of_birth is available)
                if user_data.get('date_of_birth'):
                    try:
                        birth_date = datetime.strptime(user_data['date_of_birth'], "%Y-%m-%d").date()
                        today = datetime.today().date()
                        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                        
                        # Age groups
                        if age <= 18:
                            age_groups['0-18'] += 1
                        elif age <= 30:
                            age_groups['19-30'] += 1
                        elif age <= 50:
                            age_groups['31-50'] += 1
                        elif age <= 70:
                            age_groups['51-70'] += 1
                        else:
                            age_groups['70+'] += 1
                    except:
                        pass
            
            # Classification statistics
            predicted_class = record.get('predicted_class', 'Normal')
            if predicted_class in classification_stats:
                classification_stats[predicted_class] += 1
        
        return jsonify({
            'success': True,
            'data': {
                'gender_stats': gender_stats,
                'age_groups': age_groups,
                'classification_stats': classification_stats
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting classification stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/xgbclasifytry', methods=['POST'])

def xgb_try_clasify():
    """Process the hemoglobin data and make a prediction."""
    #try:
    form = CBCForm()
    
    submit = request.form.get('submit')
    
    if submit is None:
        return "Error: Submit button not clicked or missing in the form data.", 400
    age = int(request.form.get("age"))
    wbc = float(request.form.get("WBC"))
    rbc = float(request.form.get("RBC"))
    hgb = float(request.form.get("HEMOGLOBIN"))
    hct = float(request.form.get("HEMATOCRIT"))
    mcv = float(request.form.get("MCV"))
    mch = float(request.form.get("MCH"))
    mchc = float(request.form.get("MCHC"))
    plt = float(request.form.get("PLATELET"))
    neutrophils = float(request.form.get("NEUTROPHILS"))
    lymphocytes = float(request.form.get("LYMPHOCYTES"))
    monocytes = float(request.form.get("MONOCYTES"))
    eosinophils = float(request.form.get("EUSONIPHILS"))
    basophil = float(request.form.get("BASOPHIL"))
    # Default to median value from training data (0.8) if not provided
    # Using 0.0 was causing different predictions because training data median is 0.8
    # Explicitly check if field is empty vs 0 - empty should default to 0.8, but 0 should stay 0
    immature_granulocytes_input = request.form.get("IMMATURE_GRANULYTES", "").strip()
    if immature_granulocytes_input == "":
        immature_granulocytes = 0.8  # Default when field is left empty
    else:
        immature_granulocytes = float(immature_granulocytes_input)  # Use actual value (including 0)
    #notes = request.form.get("notes")  # Corrected from "mcv" to "notes"

    
    gender = 1 if request.form.get("gender").lower() == "female" else 0

    user_input = [
        age,
        gender,
        wbc,
        rbc,
        hgb,
        hct,
        mcv,
        mch,
        mchc,
        plt,
        neutrophils,
        lymphocytes,
        monocytes,
        eosinophils,
        basophil,
        immature_granulocytes
    ]
    print(user_input)
    predicted_label,confidence_scores = xgboost_predict(user_input)
    # Recommendations dictionary
    recommendations = {
        'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
        'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
                "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
        'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                    "Further testing might be needed to determine the underlying cause.",
        'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                "provider is necessary as severe anemia can lead to serious complications."
    }

    

    final_recommendation = recommendations[predicted_label]

    

    # Output
    
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {round(confidence_scores*100, 2)}%")
    print(f"Recommendation: {final_recommendation}")

    cbc_results_summary = {
        "Normal": "Normal: Your CBC results, including your hemoglobin, hematocrit, red blood cell count, and other related values, are all within the normal range. This means your blood is healthy and able to carry oxygen properly throughout your body.",
        
        "Mild Anemia": "Mild Anemia: Some of your blood test results, such as your hemoglobin or red blood cell count, are just slightly below normal. While you might not feel many symptoms yet, these early changes suggest your blood isn't carrying oxygen quite as efficiently, so we'll monitor it and take steps if needed.",
        
        "Moderate Anemia": "Moderate Anemia: Several parts of your CBC, including hemoglobin, hematocrit, and red cell indices like MCV or MCH, show moderate changes. These results explain why you might be feeling more tired, weak, or short of breath, and we'll need to start treatment to correct it.",
        
        "Severe Anemia": "Severe Anemia: Your CBC shows multiple markers—like hemoglobin, red blood cell count, and hematocrit—are well below normal. This means your body isn't getting enough oxygen, which can cause serious symptoms, so we need to act quickly to manage and treat the cause."
    }
    if predicted_label != "Normal": 
        predicted_label += " Anemia"


    record_id = 0

        # Emit real-time update via WebSocket
    record = {
        'id': record_id,
        'wbc': wbc,
        'rbc': rbc,
        'hgb': hgb,
        'hct': hct,
        'mcv': mcv,
        'mch': mch,
        'mchc': mchc,
        'plt': plt,
        'neutrophils': neutrophils,
        'lymphocytes': lymphocytes,
        'monocytes': monocytes,
        'eosinophils': eosinophils,
        'basophil': basophil,
        'immature_granulocytes': immature_granulocytes,
        'predicted_class': predicted_label,
        'confidence': confidence_scores,
        'recommendation': final_recommendation,
        'definition': cbc_results_summary[predicted_label],
        'created_at': format_philippines_time_ampm(get_philippines_time_for_db()),
    }

    
    return render_template(
        'result_trial.html',
        record=record,
    )
    



@app.route('/result/<int:record_id>')
@login_required
def result(record_id):
    """Display classification result."""
    # Get the classification record
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM classification_history WHERE id = ? AND user_id = ?",
        (record_id, current_user.id)
    )
    
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        flash('Record not found.')
        return redirect(url_for('dashboard'))
    
    # Convert to dict
    record = dict(record)
    print(record)
    # Generate visualization if enabled
    visualization = None
    if db.get_system_setting('visualization_enabled') == 'true':
        # Get tree visualization
        visualization = anemia_model.get_tree_visualization()
    

    cbc_results_summary = {
        "Normal": "Normal: Your CBC results, including your hemoglobin, hematocrit, red blood cell count, and other related values, are all within the normal range. This means your blood is healthy and able to carry oxygen properly throughout your body.",
        
        "Mild Anemia": "Mild Anemia: Some of your blood test results, such as your hemoglobin or red blood cell count, are just slightly below normal. While you might not feel many symptoms yet, these early changes suggest your blood isn't carrying oxygen quite as efficiently, so we'll monitor it and take steps if needed.",
        
        "Moderate Anemia": "Moderate Anemia: Several parts of your CBC, including hemoglobin, hematocrit, and red cell indices like MCV or MCH, show moderate changes. These results explain why you might be feeling more tired, weak, or short of breath, and we'll need to start treatment to correct it.",
        
        "Severe Anemia": "Severe Anemia: Your CBC shows multiple markers—like hemoglobin, red blood cell count, and hematocrit—are well below normal. This means your body isn't getting enough oxygen, which can cause serious symptoms, so we need to act quickly to manage and treat the cause."
    }
    predicted_label = record["predicted_class"]
    if predicted_label != "Normal": 
        predicted_label += " Anemia"

    record["predicted_class"] = predicted_label

    record["definition"] = cbc_results_summary[predicted_label]
    
    # Format the created_at timestamp with AM/PM
    if "created_at" in record:
        record["created_at"] = format_philippines_time_ampm(record["created_at"])
    
    # Extract patient details from legacy notes format
    def _extract_after(label: str, text: str) -> str | None:
        idx = text.find(label)
        if idx == -1:
            return None
        sub = text[idx + len(label):]
        # take up to next '.'
        dot = sub.find('.')
        if dot == -1:
            val = sub.strip()
        else:
            val = sub[:dot].strip()
        return val or None

    patient_name_display = None
    patient_age_display = None
    patient_gender_display = None
    patient_note_remainder = (record.get('notes') or '').strip()

    note_text = record.get('notes') or ''
    if note_text.startswith('Patient:'):
        patient_name_display = _extract_after('Patient:', note_text)
        patient_age_display = _extract_after('Age:', note_text)
        patient_gender_display = _extract_after('Gender:', note_text)

        # Remove leading labeled parts in order, if present
        temp = note_text
        for label in ('Patient:', 'Age:', 'Gender:'):
            idx = temp.find(label)
            if idx == 0:
                # cut off up to and including the next dot+space
                sub = temp[len(label):]
                dot = sub.find('.')
                if dot != -1:
                    temp = sub[dot+1:].lstrip()
                else:
                    temp = sub.lstrip()
        patient_note_remainder = temp.strip()

    record['patient_name_display'] = patient_name_display
    record['patient_age_display'] = patient_age_display
    record['patient_gender_display'] = patient_gender_display
    record['patient_note_remainder'] = patient_note_remainder

    return render_template(
        'result.html',
        record=record,
        visualization=visualization
    )


@app.route('/api/classify', methods=['POST'])
@login_required
def api_classify():
    """API endpoint for anemia classification."""
    try:
        data = request.get_json()
        if not data or 'hemoglobin' not in data:
            return jsonify({'error': 'Missing hemoglobin value'}), 400
        
        hemoglobin = float(data['hemoglobin'])
        notes = data.get('notes', '')
        
        # Validate hemoglobin range
        if hemoglobin < 1 or hemoglobin > 25:
            return jsonify({'error': 'Hemoglobin value out of valid range (1-25 g/dL)'}), 400
        
        # Get prediction from model
        result = anemia_model.predict(hemoglobin)
        
        # Save the record to database
        record_id = db.add_classification_record(
            user_id=current_user.id,
            hemoglobin=hemoglobin,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            recommendation=result['recommendation'],
            notes=notes
        )
        
        # Add record_id to result
        result['record_id'] = record_id
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API classification error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Admin routes
@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get page parameter for pagination
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    
    # Get system statistics
    stats = db.get_statistics()
    
    # Get recent classifications with pagination
    recent_data = db.get_recent_classifications(page=page, per_page=3)
    
    # Format timestamps with AM/PM
    for record in recent_data['records']:
        if 'created_at' in record:
            record['created_at'] = format_philippines_time_ampm(record['created_at'])
    
    # Get chart data with imported data
    charts_data = get_combined_charts_data()
    
    return render_template('admin/dashboard.html', stats=stats, recent_data=recent_data, charts_data=charts_data)


@app.route('/admin/users')
@login_required
def admin_users():
    """Admin user management page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    users_data = db.get_users_paginated(page=page, per_page=5)
    
    # Format timestamps with AM/PM for each user record
    for user in users_data['records']:
        if 'created_at' in user:
            user['created_at'] = format_philippines_time_ampm(user['created_at'])
        if 'last_login' in user and user['last_login']:
            user['last_login'] = format_philippines_time_ampm(user['last_login'])
    
    return render_template('admin/users.html', users_data=users_data)


@app.route('/admin/api/username-exists')
@login_required
def admin_api_username_exists():
    """Admin API: check if a username exists, optionally excluding a user id."""
    if not current_user.is_admin:
        return jsonify({ 'success': False, 'error': 'Access denied' }), 403
    username = (request.args.get('username') or '').strip()
    exclude_id = (request.args.get('exclude_id') or '').strip()
    if not username:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_username(username)
    if not user:
        return jsonify({ 'success': True, 'exists': False })
    if exclude_id and str(user.get('id')) == str(exclude_id):
        return jsonify({ 'success': True, 'exists': False })
    return jsonify({ 'success': True, 'exists': True })


@app.route('/admin/api/email-exists')
@login_required
def admin_api_email_exists():
    """Admin API: check if an email exists, optionally excluding a user id."""
    if not current_user.is_admin:
        return jsonify({ 'success': False, 'error': 'Access denied' }), 403
    email = (request.args.get('email') or '').strip()
    exclude_id = (request.args.get('exclude_id') or '').strip()
    if not email:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_email(email)
    if not user:
        return jsonify({ 'success': True, 'exists': False })
    if exclude_id and str(user.get('id')) == str(exclude_id):
        return jsonify({ 'success': True, 'exists': False })
    return jsonify({ 'success': True, 'exists': True })

@app.route('/admin/api/medical-id-exists')
@login_required
def admin_api_medical_id_exists():
    """Admin API: check if a medical ID exists, optionally excluding a user id."""
    if not current_user.is_admin:
        return jsonify({ 'success': False, 'error': 'Access denied' }), 403
    medical_id = (request.args.get('medical_id') or '').strip()
    exclude_id = (request.args.get('exclude_id') or '').strip()
    if not medical_id:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_medical_id(medical_id)
    if not user:
        return jsonify({ 'success': True, 'exists': False })
    if exclude_id and str(user.get('id')) == str(exclude_id):
        return jsonify({ 'success': True, 'exists': False })
    return jsonify({ 'success': True, 'exists': True })


@app.route('/admin/user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def admin_edit_user(user_id):
    """Admin edit user page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get user data
    user_data = db.get_user(user_id)
    if not user_data:
        flash('User not found.')
        return redirect(url_for('admin_users'))
    
    form = AdminUserForm()
    
    if request.method == 'GET':
        # Pre-populate form with user data
        form.username.data = user_data['username']
        form.email.data = user_data['email']
        form.first_name.data = user_data['first_name']
        form.last_name.data = user_data['last_name']
        form.gender.data = user_data['gender']
        form.date_of_birth.data = user_data['date_of_birth']
        form.medical_id.data = user_data['medical_id']
        form.is_admin.data = bool(user_data['is_admin'])
        form.user_id.data = user_id
    
    if form.validate_on_submit():
        # Update user
        update_data = {
            'username': form.username.data,
            'email': form.email.data,
            'first_name': form.first_name.data,
            'last_name': form.last_name.data,
            'gender': form.gender.data,
            'date_of_birth': form.date_of_birth.data,
            'medical_id': form.medical_id.data,
            'is_admin': 1 if form.is_admin.data else 0
        }
        
        # Add password if provided
        if form.password.data:
            update_data['password'] = form.password.data
        
        success, result = db.update_user(user_id, **update_data)
        
        if success:
            flash('User updated successfully!')
            return redirect(url_for('admin_users'))
        else:
            flash(f'User update failed: {result}')
    
    return render_template('admin/edit_user.html', form=form, user=user_data)


@app.route('/admin/user/<int:user_id>/details')
@login_required
def admin_user_details(user_id):
    """Get detailed user information for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get pagination parameters for classification history
    class_page = request.args.get('class_page', 1, type=int)
    if class_page < 1:
        class_page = 1
    
    # Get user data
    user_data = db.get_user(user_id)
    if not user_data:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    
    # Get user's medical data
    medical_data = db.get_medical_data(user_id)
    
    # Get user's classification history with pagination (3 per page)
    classification_data = db.get_user_classification_history_paginated(user_id, page=class_page, per_page=3)
    classification_history = classification_data['records']
    
    # Get user's chat conversations (last 3 for pagination)
    all_conversations = simple_chat.get_user_conversations(user_id, is_admin=False)
    conversations = all_conversations[:3] if all_conversations else []
    
    # Format timestamps with AM/PM
    formatted_created_at = format_philippines_time_ampm(user_data['created_at'])
    formatted_last_login = format_philippines_time_ampm(user_data['last_login']) if user_data['last_login'] else None
    
    # Format classification history timestamps
    for record in classification_history:
        if 'created_at' in record:
            record['created_at'] = format_philippines_time_ampm(record['created_at'])
    
    # Format conversation timestamps
    for conversation in conversations:
        if 'last_message_time' in conversation and conversation['last_message_time']:
            conversation['last_message_time'] = format_philippines_time_ampm(conversation['last_message_time'])
    
    # Prepare response data
    user_details = {
        'user': {
            'id': user_data['id'],
            'username': user_data['username'],
            'email': user_data['email'],
            'first_name': user_data['first_name'],
            'last_name': user_data['last_name'],
            'gender': user_data['gender'],
            'date_of_birth': user_data['date_of_birth'],
            'medical_id': user_data['medical_id'],
            'is_admin': user_data['is_admin'],
            'created_at': formatted_created_at,
            'last_login': formatted_last_login
        },
        'medical_data': medical_data,
        'classification_history': classification_history,
        'classification_pagination': {
            'page': classification_data['page'],
            'per_page': classification_data['per_page'],
            'total': classification_data['total'],
            'total_pages': classification_data['total_pages'],
            'has_prev': classification_data['has_prev'],
            'has_next': classification_data['has_next'],
            'prev_num': classification_data['prev_num'],
            'next_num': classification_data['next_num']
        },
        'conversations': conversations
    }
    
    return jsonify({'success': True, 'data': user_details})


@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    """Admin delete user."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Prevent self-deletion
    if user_id == current_user.id:
        return jsonify({'success': False, 'error': 'Cannot delete your own account'}), 400
    
    # Delete the user
    success, message = db.delete_user(user_id)
    
    if success:
        logger.info(f"Admin {current_user.username} deleted user ID {user_id}")
        return jsonify({'success': True, 'message': message})
    else:
        logger.error(f"Failed to delete user ID {user_id}: {message}")
        return jsonify({'success': False, 'error': message}), 400


@app.route('/admin/history')
@login_required
def admin_history():
    """Admin classification history page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get page parameter for pagination
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    
    # Get paginated classification history
    history_data = db.get_classification_history_paginated(page=page, per_page=5)
    
    # Fetch classifications submitted for another person (with pagination)
    op_page = request.args.get('op_page', 1, type=int)
    if op_page < 1:
        op_page = 1
    other_person_data = db.get_other_person_classifications_paginated(page=op_page, per_page=5)

    # Format timestamps with AM/PM for each record
    for record in history_data['records']:
        if 'created_at' in record:
            record['created_at'] = format_philippines_time_ampm(record['created_at'])

    for rec in other_person_data['records']:
        if 'created_at' in rec:
            rec['created_at'] = format_philippines_time_ampm(rec['created_at'])

    # Get system statistics (same as dashboard)
    stats = db.get_statistics()
    
    # Calculate additional statistics for the history page
    total_records = history_data['total']
    anemic_cases = stats['anemic_cases']
    normal_cases = stats['normal_cases']
    
    # Calculate anemia rate
    if total_records > 0:
        anemia_rate = (anemic_cases / total_records) * 100
    else:
        anemia_rate = 0.0
    
    # Get slide parameter to preserve carousel state
    active_slide = request.args.get('slide', 0, type=int)
    
    return render_template('admin/history.html', 
                         history_data=history_data,
                         other_person_data=other_person_data,
                         total_records=total_records,
                         anemic_cases=anemic_cases,
                         normal_cases=normal_cases,
                         anemia_rate=anemia_rate,
                         stats=stats,
                         active_slide=active_slide)


@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    """Admin system settings page."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    form = SystemSettingsForm()
    
    if request.method == 'GET':
        # Pre-populate form with current settings
        form.site_name.data = db.get_system_setting('site_name') or 'AnemoCheck'
        form.site_description.data = db.get_system_setting('site_description') or 'Anemia Detection System'
        form.max_users.data = int(db.get_system_setting('max_users') or 1000)
        form.session_timeout.data = int(db.get_system_setting('session_timeout') or 30)
        form.model_confidence_threshold.data = float(db.get_system_setting('model_confidence_threshold') or 0.8)
        form.model_version.data = db.get_system_setting('model_version') or '1.0.0'
        form.enable_auto_retrain.data = db.get_system_setting('enable_auto_retrain') == 'true'
        # Brevo API settings
        form.brevo_api_key.data = db.get_system_setting('brevo_api_key') or ''
        form.brevo_sender_email.data = db.get_system_setting('brevo_sender_email') or ''
        form.brevo_sender_name.data = db.get_system_setting('brevo_sender_name') or 'AnemoCheck'
        
        # Check if API key exists and set appropriate placeholder
        existing_api_key = db.get_system_setting('brevo_api_key')
        if existing_api_key:
            form.brevo_api_key.render_kw = {'placeholder': 'API key is saved - enter new key to change'}
        else:
            form.brevo_api_key.render_kw = {'placeholder': 'Enter Brevo API key'}
        
        form.enable_email_notifications.data = db.get_system_setting('enable_email_notifications') == 'true'
        form.password_min_length.data = int(db.get_system_setting('password_min_length') or 8)
        form.max_login_attempts.data = int(db.get_system_setting('max_login_attempts') or 5)
        form.enable_two_factor.data = db.get_system_setting('enable_two_factor') == 'true'
    
    if form.validate_on_submit():
        logger.info("Admin settings form submitted")
        logger.info(f"Brevo API Key provided: {bool(form.brevo_api_key.data)}")
        logger.info(f"Brevo Sender Email: {form.brevo_sender_email.data}")
        logger.info(f"Brevo Sender Name: {form.brevo_sender_name.data}")
        logger.info(f"Enable Email Notifications: {form.enable_email_notifications.data}")
        
        # Update settings
        db.update_system_setting('site_name', form.site_name.data, current_user.id)
        db.update_system_setting('site_description', form.site_description.data, current_user.id)
        db.update_system_setting('max_users', str(form.max_users.data), current_user.id)
        db.update_system_setting('session_timeout', str(form.session_timeout.data), current_user.id)
        db.update_system_setting('model_confidence_threshold', str(form.model_confidence_threshold.data), current_user.id)
        db.update_system_setting('model_version', form.model_version.data, current_user.id)
        db.update_system_setting('enable_auto_retrain', 'true' if form.enable_auto_retrain.data else 'false', current_user.id)
        
        # Brevo API settings
        db.update_system_setting('brevo_sender_email', form.brevo_sender_email.data, current_user.id)
        db.update_system_setting('brevo_sender_name', form.brevo_sender_name.data, current_user.id)
        if form.brevo_api_key.data:
            logger.info(f"Attempting to save Brevo API key")
            logger.info(f"API key length: {len(form.brevo_api_key.data)}")
            logger.info(f"User ID: {current_user.id}")
            
            try:
                result = db.update_system_setting('brevo_api_key', form.brevo_api_key.data, current_user.id)
                logger.info(f"API key save result: {result}")
                
                if result:
                    logger.info("Brevo API Key updated successfully")
                else:
                    logger.error("Failed to save Brevo API key - database returned False")
            except Exception as e:
                logger.error(f"Exception during API key save: {str(e)}")
        else:
            logger.info("Brevo API Key not provided - keeping existing key")
        db.update_system_setting('enable_email_notifications', 'true' if form.enable_email_notifications.data else 'false', current_user.id)
        
        db.update_system_setting('password_min_length', str(form.password_min_length.data), current_user.id)
        db.update_system_setting('max_login_attempts', str(form.max_login_attempts.data), current_user.id)
        db.update_system_setting('enable_two_factor', 'true' if form.enable_two_factor.data else 'false', current_user.id)
        
        # Verify settings were saved
        saved_brevo_sender_email = db.get_system_setting('brevo_sender_email')
        saved_brevo_sender_name = db.get_system_setting('brevo_sender_name')
        saved_brevo_api_key = db.get_system_setting('brevo_api_key')
        saved_enable_notifications = db.get_system_setting('enable_email_notifications')
        logger.info(f"Saved Brevo Sender Email: {saved_brevo_sender_email}")
        logger.info(f"Saved Brevo Sender Name: {saved_brevo_sender_name}")
        logger.info(f"Saved Brevo API Key exists: {bool(saved_brevo_api_key)}")
        logger.info(f"Saved Enable Notifications: {saved_enable_notifications}")
        
        flash('Settings updated successfully!')
        return redirect(url_for('admin_settings'))
    
    return render_template('admin/settings.html', form=form)


# Email sending functionality - now handled by Brevo API service


# Email routes
@app.route('/send-result-email/<int:record_id>', methods=['POST'])
@login_required
def send_result_email(record_id):
    """Send anemia test result email to current user."""
    try:
        logger.info(f"Attempting to send email for record {record_id} to user {current_user.id}")
        
        # Get record data
        record_data = db.get_classification_record(record_id)
        if not record_data:
            logger.error(f"Record {record_id} not found")
            return jsonify({'success': False, 'error': 'Record not found'})
        
        logger.info(f"Found record: {record_data}")
        
        # Check if user owns this record
        if record_data['user_id'] != current_user.id:
            logger.error(f"User {current_user.id} does not own record {record_id}")
            return jsonify({'success': False, 'error': 'Access denied'})
        
        # Get user data
        user_data = db.get_user_by_id(current_user.id)
        if not user_data:
            logger.error(f"User {current_user.id} not found")
            return jsonify({'success': False, 'error': 'User not found'})
        
        logger.info(f"Found user: {user_data['email']}")
        
        # Send email
        success, message = send_result_email_brevo(record_id, user_data['email'], 
                                           f"{user_data['first_name']} {user_data['last_name']}".strip() or user_data['username'],
                                           record_data)
        
        if success:
            logger.info(f"Email sent successfully to {user_data['email']}")
            return jsonify({'success': True, 'message': message, 'email': user_data['email']})
        else:
            logger.error(f"Email sending failed: {message}")
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        logger.error(f"Error in send_result_email route: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'})


@app.route('/admin/send-result-email/<int:record_id>', methods=['POST'])
@login_required
def admin_send_result_email(record_id):
    """Send anemia test result email (admin function)."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Admin access required'})
    
    try:
        logger.info(f"Admin {current_user.id} attempting to send email for record {record_id}")
        
        # Get record data
        record_data = db.get_classification_record(record_id)
        if not record_data:
            logger.error(f"Record {record_id} not found")
            return jsonify({'success': False, 'error': 'Record not found'})
        
        # Get user data
        user_data = db.get_user_by_id(record_data['user_id'])
        if not user_data:
            logger.error(f"User {record_data['user_id']} not found")
            return jsonify({'success': False, 'error': 'User not found'})
        
        logger.info(f"Found user: {user_data['email']}")
        
        # Send email
        success, message = send_result_email_brevo(record_id, user_data['email'], 
                                           f"{user_data['first_name']} {user_data['last_name']}".strip() or user_data['username'],
                                           record_data)
        
        if success:
            logger.info(f"Email sent successfully to {user_data['email']}")
            return jsonify({'success': True, 'message': message, 'email': user_data['email']})
        else:
            logger.error(f"Email sending failed: {message}")
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        logger.error(f"Error in admin_send_result_email route: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'})


# Export routes
@app.route('/admin/export/dashboard.csv')
@login_required
def export_dashboard():
    """Export dashboard statistics as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get system statistics
    stats = db.get_statistics()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Metric', 'Value'])
    
    # Write statistics
    writer.writerow(['Total Users', stats['total_users']])
    writer.writerow(['Total Classifications', stats['total_classifications']])
    writer.writerow(['Anemic Cases', stats['anemic_cases']])
    writer.writerow(['Normal Cases', stats['normal_cases']])
    writer.writerow(['New Users (Last 7 Days)', stats['new_user_count']])
    writer.writerow(['Active Users (Last 7 Days)', stats['active_user_count']])
    
    # Add empty row
    writer.writerow([])
    writer.writerow(['Classification Distribution'])
    
    # Add class distribution
    for class_name, count in stats['class_distribution'].items():
        writer.writerow([class_name, count])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=dashboard_statistics.csv'}
    )
    return response


@app.route('/admin/export/classification_stats.csv')
@login_required
def export_classification_stats():
    """Export classification statistics as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get system statistics and combined chart data (original + imported)
    stats = db.get_statistics()
    charts_data = get_combined_charts_data()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Classification Statistics Export'])
    # Format current time with AM/PM and add tab to force Excel to display as text
    current_time_str = get_philippines_time().strftime('%Y-%m-%d %H:%M:%S')
    writer.writerow(['Generated on', '\t' + format_philippines_time_ampm(current_time_str)])
    writer.writerow(['Data includes', 'Original system data + Imported CSV data'])
    writer.writerow([])
    
    # Write overall statistics
    writer.writerow(['Overall Statistics'])
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Total Users', stats['total_users']])
    writer.writerow(['Total Classifications', stats['total_classifications']])
    writer.writerow(['Anemic Cases', stats['anemic_cases']])
    writer.writerow(['Normal Cases', stats['normal_cases']])
    writer.writerow([])
    
    # Add information about imported datasets
    try:
        imported_files = db.get_imported_files()
        if imported_files:
            writer.writerow(['Imported Datasets Information'])
            writer.writerow(['File Name', 'Date Imported', 'Total Records', 'Status'])
            for file_info in imported_files:
                status = 'Applied' if file_info['is_applied'] else 'Unapplied'
                imported_at_formatted = '\t' + format_philippines_time_ampm(file_info['imported_at']) if file_info.get('imported_at') else ''
                writer.writerow([
                    file_info['filename'],
                    imported_at_formatted,
                    file_info['total_records'],
                    status
                ])
            writer.writerow([])
    except Exception as e:
        # If there's an error getting imported files, continue without them
        pass
    
    # Write age group distribution
    writer.writerow(['Age Group Distribution'])
    writer.writerow(['Age Group', 'Count'])
    for age_group, count in charts_data['age_groups'].items():
        writer.writerow([age_group, count])
    writer.writerow([])
    
    # Write gender distribution
    writer.writerow(['Gender Distribution'])
    writer.writerow(['Gender', 'Count'])
    for gender, count in charts_data['gender_stats'].items():
        writer.writerow([gender, count])
    writer.writerow([])
    
    # Write severity classification distribution
    writer.writerow(['Severity Classification Distribution'])
    writer.writerow(['Severity', 'Count'])
    for severity, count in charts_data['severity_stats'].items():
        writer.writerow([severity, count])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=classification_statistics.csv'}
    )
    return response


@app.route('/admin/import/classification_data', methods=['POST'])
@csrf.exempt
@login_required
def import_classification_data():
    """Import classification data from CSV file."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied. Administrator privileges required.'})
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'File must be a CSV file'})
        
        import csv
        import io
        
        # Read CSV content
        csv_content = file.read().decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        # Expected columns: age, gender, category
        required_columns = ['age', 'gender', 'category']
        if not all(col in csv_reader.fieldnames for col in required_columns):
            return jsonify({
                'success': False, 
                'error': f'CSV must contain columns: {", ".join(required_columns)}'
            })
        
        imported_count = 0
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        # Ensure the table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_import_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                category TEXT NOT NULL,
                imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create imported file record
        print(f"Creating imported file record for: {file.filename}")
        file_id = db.create_imported_file(
            filename=file.filename,
            original_filename=file.filename,
            total_records=0,  # Will be updated after counting
            imported_by=current_user.id
        )
        print(f"Created file record with ID: {file_id}")
        
        # Use batch insert for better performance
        from timezone_utils import get_philippines_time_for_db
        ph_timestamp = get_philippines_time_for_db()  # Get timestamp once for all rows
        
        # Check if file_id column exists (only check once)
        has_file_id = False
        try:
            import database as db_module
            if db_module.USE_POSTGRES:
                # For PostgreSQL, check information_schema
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'classification_import_data' AND column_name = 'file_id'
                """)
                has_file_id = cursor.fetchone() is not None
            else:
                # For SQLite, use PRAGMA
                cursor.execute("PRAGMA table_info(classification_import_data)")
                columns = [col[1] for col in cursor.fetchall()]
                has_file_id = 'file_id' in columns
        except:
            # Assume file_id exists if check fails
            has_file_id = True
        
        # Prepare batch data
        batch_data = []
        for row in csv_reader:
            try:
                age = int(row['age'])
                gender = row['gender'].strip().title()  # Normalize gender to Title Case
                category = normalize_severity_category(row['category'].strip())  # Normalize category
                
                if has_file_id:
                    batch_data.append((age, gender, category, file_id, ph_timestamp))
                else:
                    batch_data.append((age, gender, category, ph_timestamp))
                
                imported_count += 1
            except (ValueError, KeyError) as e:
                continue  # Skip invalid rows
        
        # Batch insert all rows at once for much better performance
        if batch_data:
            try:
                if has_file_id:
                    cursor.executemany('''
                        INSERT INTO classification_import_data (age, gender, category, file_id, imported_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', batch_data)
                else:
                    cursor.executemany('''
                        INSERT INTO classification_import_data (age, gender, category, imported_at)
                        VALUES (?, ?, ?, ?)
                    ''', batch_data)
            except Exception as e:
                # If batch insert fails, fall back to individual inserts
                logger.warning(f"Batch insert failed, using individual inserts: {str(e)}")
                for row_data in batch_data:
                    try:
                        if has_file_id:
                            cursor.execute('''
                                INSERT INTO classification_import_data (age, gender, category, file_id, imported_at)
                                VALUES (?, ?, ?, ?, ?)
                            ''', row_data)
                        else:
                            cursor.execute('''
                                INSERT INTO classification_import_data (age, gender, category, imported_at)
                                VALUES (?, ?, ?, ?)
                            ''', row_data)
                    except:
                        continue
        
        # Update the total records count
        cursor.execute('''
            UPDATE imported_files 
            SET total_records = ?
            WHERE id = ?
        ''', (imported_count, file_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'imported_count': imported_count,
            'message': f'Successfully imported {imported_count} records'
        })
        
    except Exception as e:
        logger.error(f"Import error: {str(e)}")
        return jsonify({'success': False, 'error': f'Error processing file: {str(e)}'})


@app.route('/admin/api/charts_data')
@login_required
def get_charts_data_api():
    """API endpoint to get updated chart data."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied'})
        
        charts_data = get_combined_charts_data()
        return jsonify({'success': True, 'data': charts_data})
        
    except Exception as e:
        logger.error(f"Charts data API error: {str(e)}")
        return jsonify({'success': False, 'error': f'Error fetching chart data: {str(e)}'})


@app.route('/admin/api/imported_data')
@login_required
def get_imported_data():
    """API endpoint to get imported data list."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied'})
        
        print("Fetching imported files...")
        imported_files = db.get_imported_files()
        print(f"Retrieved {len(imported_files)} imported files")
        
        return jsonify({'success': True, 'data': imported_files})
        
    except Exception as e:
        logger.error(f"Error fetching imported data: {str(e)}")
        print(f"Error fetching imported data: {str(e)}")
        return jsonify({'success': False, 'error': f'Error fetching imported data: {str(e)}'})


@app.route('/admin/api/apply_dataset/<int:file_id>', methods=['POST'])
@csrf.exempt
@login_required
def apply_dataset(file_id):
    """Apply an imported dataset to the charts."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied'})
        
        print(f"Applying dataset {file_id}")
        db.update_file_status(file_id, True)
        return jsonify({'success': True, 'message': 'Dataset applied successfully'})
        
    except Exception as e:
        logger.error(f"Error applying dataset: {str(e)}")
        print(f"Error applying dataset: {str(e)}")
        return jsonify({'success': False, 'error': f'Error applying dataset: {str(e)}'})


@app.route('/admin/api/unapply_dataset/<int:file_id>', methods=['POST'])
@csrf.exempt
@login_required
def unapply_dataset(file_id):
    """Unapply an imported dataset from the charts."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied'})
        
        print(f"Unapplying dataset {file_id}")
        db.update_file_status(file_id, False)
        return jsonify({'success': True, 'message': 'Dataset unapplied successfully'})
        
    except Exception as e:
        logger.error(f"Error unapplying dataset: {str(e)}")
        print(f"Error unapplying dataset: {str(e)}")
        return jsonify({'success': False, 'error': f'Error unapplying dataset: {str(e)}'})


@app.route('/admin/api/delete_dataset/<int:file_id>', methods=['DELETE'])
@csrf.exempt
@login_required
def delete_dataset(file_id):
    """Delete an imported dataset permanently."""
    try:
        if not current_user.is_admin:
            return jsonify({'success': False, 'error': 'Access denied'})
        
        print(f"Deleting dataset {file_id}")
        db.delete_imported_file(file_id)
        return jsonify({'success': True, 'message': 'Dataset deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        print(f"Error deleting dataset: {str(e)}")
        return jsonify({'success': False, 'error': f'Error deleting dataset: {str(e)}'})


def get_combined_charts_data():
    """Get chart data combining original data with imported data."""
    try:
        # Get base chart data
        charts_data = db.get_admin_dashboard_charts()
        
        # Remove 'Check' categories from severity stats as they're not meaningful
        if 'Check' in charts_data['severity_stats']:
            del charts_data['severity_stats']['Check']
        
        # Normalize original gender data to standard format (Male/Female)
        normalized_gender_stats = {}
        for gender, count in charts_data['gender_stats'].items():
            # Standardize gender labels
            if gender.lower() in ['m', 'male']:
                normalized_gender = 'Male'
            elif gender.lower() in ['f', 'female']:
                normalized_gender = 'Female'
            else:
                normalized_gender = gender.title()
            normalized_gender_stats[normalized_gender] = normalized_gender_stats.get(normalized_gender, 0) + count
        charts_data['gender_stats'] = normalized_gender_stats
        
        # Get applied imported data and merge with existing data
        imported_data = db.get_applied_imported_data()
        imported_age_groups = imported_data['age_groups']
        imported_gender_stats = imported_data['gender_stats']
        imported_severity_stats = imported_data['severity_stats']
        
        # Merge imported data with existing data (add to existing counts)
        for age_group, count in imported_age_groups.items():
            charts_data['age_groups'][age_group] = charts_data['age_groups'].get(age_group, 0) + count
        
        # Merge gender data with proper normalization
        for gender, count in imported_gender_stats.items():
            # Standardize gender labels to avoid duplicates
            if gender.lower() in ['m', 'male']:
                normalized_gender = 'Male'
            elif gender.lower() in ['f', 'female']:
                normalized_gender = 'Female'
            else:
                normalized_gender = gender.title()
            charts_data['gender_stats'][normalized_gender] = charts_data['gender_stats'].get(normalized_gender, 0) + count
        
        # Merge severity data with proper normalization
        for category, count in imported_severity_stats.items():
            # Normalize severity category to standard format
            normalized_category = normalize_severity_category(category)
            # Skip 'Check' categories as they're not meaningful for severity classification
            if normalized_category != 'Check':
                charts_data['severity_stats'][normalized_category] = charts_data['severity_stats'].get(normalized_category, 0) + count
        
        return charts_data
        
    except Exception as e:
        logger.error(f"Error getting combined charts data: {str(e)}")
        # Return original data if there's an error
        return db.get_admin_dashboard_charts()


def normalize_severity_category(category):
    """Normalize severity category names to standard format."""
    if not category:
        return 'Other'
    
    category_lower = category.lower().strip()
    
    if 'normal' in category_lower:
        return 'Normal'
    elif 'mild' in category_lower and 'anemia' in category_lower:
        return 'Mild Anemia'
    elif 'mild' in category_lower:
        return 'Mild Anemia'
    elif 'moderate' in category_lower and 'anemia' in category_lower:
        return 'Moderate Anemia'
    elif 'moderate' in category_lower:
        return 'Moderate Anemia'
    elif 'severe' in category_lower and 'anemia' in category_lower:
        return 'Severe Anemia'
    elif 'severe' in category_lower:
        return 'Severe Anemia'
    else:
        return 'Other'  # Default to 'Other' for unknown categories


@app.route('/admin/export/users.csv')
@login_required
def export_users():
    """Export users data as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get all users
    users = db.get_all_users()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Username', 'Email', 'First Name', 'Last Name', 'Gender', 'Date of Birth', 'Medical ID', 'Is Admin', 'Created At', 'Last Login'])
    
    # Write user data
    for user in users:
        # Format timestamps with AM/PM and add tab to force Excel to display as text
        created_at = '\t' + format_philippines_time_ampm(user['created_at'])
        last_login = '\t' + format_philippines_time_ampm(user['last_login']) if user['last_login'] else ''
        writer.writerow([
            user['id'],
            user['username'],
            user['email'],
            user['first_name'] or '',
            user['last_name'] or '',
            user['gender'] or '',
            user['date_of_birth'] or '',
            user['medical_id'] or '',
            'Yes' if user['is_admin'] else 'No',
            created_at,
            last_login
        ])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=users_export.csv'}
    )
    return response


@app.route('/admin/export/classification_history.csv')
@login_required
def export_classification_history():
    """Export classification history as CSV."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get all classification history with extended user fields
    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ch.*,
               u.username, u.first_name, u.last_name, u.gender as user_gender, u.date_of_birth
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        ORDER BY ch.created_at DESC
    """)
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Create CSV content with proper formatting
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Helper: Immature granulocytes value with defaults
    def _ig_val(raw):
        if raw is None: return 0.8
        if isinstance(raw, str) and raw.strip() == '': return 0.8
        try: return float(raw)
        except Exception: return 0.8
    
    # Helper: parse legacy patient info from notes and return (name, age, gender, cleaned_notes)
    def _parse_legacy(raw):
        raw = (raw or '').strip()
        name = age = gender = ''
        cleaned = raw
        if raw.lower().startswith('patient:'):
            temp = raw
            def _extract(label, text):
                lbl = label.lower()
                if not text.lower().startswith(lbl): return None, text
                sub = text[len(label):]
                dot = sub.find('.'); comma = sub.find(',')
                end = -1
                if dot == -1 and comma == -1: end = -1
                elif dot == -1: end = comma
                elif comma == -1: end = dot
                else: end = min(dot, comma)
                val = sub[:end].strip() if end != -1 else sub.strip()
                rest = sub[end+1:].lstrip() if end != -1 else ''
                return (val or None), rest
            n, rem = _extract('Patient:', temp)
            if n is not None: name, temp = n, rem
            a, rem = _extract('Age:', temp)
            if a is not None: age, temp = a, rem
            g, rem = _extract('Gender:', temp)
            if g is not None: gender, temp = g, rem
            cleaned = (temp or '').strip()
        return name, age, gender, cleaned
    
    # Split into self vs other
    self_records = []
    other_records = []
    for rec in records:
        notes = (rec.get('notes') or '').strip()
        if rec.get('patient_name') or rec.get('patient_age') or rec.get('patient_gender') or notes.startswith('Patient:'):
            other_records.append(rec)
        else:
            self_records.append(rec)
    
    # Section 1: Self classifications
    writer.writerow(['Classifications (Self)'])
    writer.writerow([
        'ID', 'User ID', 'Username', 'Full Name', 'Age', 'Gender', 'Date',
        'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT',
        'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophil', 'Immature Granulocytes',
        'Predicted Class', 'Confidence', 'Recommendation', 'Notes'
    ])
    from datetime import datetime
    for r in self_records:
        full_name = f"{(r.get('first_name') or '').strip()} {(r.get('last_name') or '').strip()}".strip()
        # Age computed at classification time
        age_val = ''
        try:
            dob_str = r.get('date_of_birth')
            created_at = r.get('created_at')
            if dob_str and created_at:
                dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
                created_dt = datetime.strptime(str(created_at).split('.')[0], "%Y-%m-%d %H:%M:%S")
                d = created_dt.date()
                age_val = d.year - dob.year - ((d.month, d.day) < (dob.month, dob.day))
        except Exception:
            age_val = ''
        formatted_date = '\t' + format_philippines_time_ampm(r['created_at'])
        writer.writerow([
            r['id'], r['user_id'], r['username'], full_name, age_val, r.get('user_gender') or '',
            formatted_date,
            r['wbc'], r['rbc'], r['hgb'], r['hct'], r['mcv'], r['mch'], r['mchc'], r['plt'],
            r.get('neutrophils') or '', r.get('lymphocytes') or '', r.get('monocytes') or '',
            r.get('eosinophils') or '', r.get('basophil') or '', _ig_val(r.get('immature_granulocytes')),
            r['predicted_class'],
            f"{float(r.get('confidence', 0))*100:.2f}%" if r.get('confidence') is not None else '',
            r.get('recommendation') or '',
            (r.get('notes') or '').strip()
        ])
    
    # Blank line separator
    writer.writerow([])
    # Section 2: Another person
    writer.writerow(['Classifications for Another Person'])
    writer.writerow([
        'ID', 'User ID', 'Username', 'Patient Name', 'Patient Age', 'Patient Gender', 'Date',
        'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT',
        'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophil', 'Immature Granulocytes',
        'Predicted Class', 'Confidence', 'Recommendation', 'Notes'
    ])
    for r in other_records:
        p_name = r.get('patient_name') or ''
        p_age = r.get('patient_age') or ''
        p_gender = r.get('patient_gender') or ''
        raw_notes = (r.get('notes') or '').strip()
        cleaned_notes = raw_notes
        if not (p_name or p_age or p_gender):
            n, a, g, cleaned = _parse_legacy(raw_notes)
            p_name, p_age, p_gender = n or '', a or '', g or ''
            cleaned_notes = cleaned
        else:
            # Clean any legacy prefix from notes even if explicit fields exist
            _, _, _, cleaned = _parse_legacy(raw_notes)
            if cleaned:
                cleaned_notes = cleaned
        formatted_date = '\t' + format_philippines_time_ampm(r['created_at'])
        writer.writerow([
            r['id'], r['user_id'], r['username'],
            p_name, p_age, p_gender,
            formatted_date,
            r['wbc'], r['rbc'], r['hgb'], r['hct'], r['mcv'], r['mch'], r['mchc'], r['plt'],
            r.get('neutrophils') or '', r.get('lymphocytes') or '', r.get('monocytes') or '',
            r.get('eosinophils') or '', r.get('basophil') or '', _ig_val(r.get('immature_granulocytes')),
            r['predicted_class'],
            f"{float(r.get('confidence', 0))*100:.2f}%" if r.get('confidence') is not None else '',
            r.get('recommendation') or '',
            cleaned_notes
        ])
    
    csv_content = output.getvalue()
    output.close()
    
    # Create response with BOM for proper Excel compatibility
    from flask import Response
    # Add BOM for proper UTF-8 handling in Excel
    csv_content_with_bom = '\ufeff' + csv_content
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename=classification_history.csv'}
    )
    return response


# Classification History Actions
@app.route('/admin/classification/<int:record_id>/details')
@login_required
def admin_classification_details(record_id):
    """Get detailed classification information for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get classification record
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ch.*, u.username, u.first_name, u.last_name, u.email
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE ch.id = ?
    """, (record_id,))
    
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    # Convert to dict
    record = dict(record)
    
    # Format timestamp with AM/PM
    if 'created_at' in record:
        record['created_at'] = format_philippines_time_ampm(record['created_at'])
    
    # Ensure immature_granulocytes defaults to 0 if None
    if record.get('immature_granulocytes') is None:
        record['immature_granulocytes'] = 0
    
    # Prepare response data
    classification_details = {
        'record': record,
        'user': {
            'username': record['username'],
            'first_name': record['first_name'],
            'last_name': record['last_name'],
            'email': record['email']
        }
    }
    
    return jsonify({'success': True, 'data': classification_details})


@app.route('/admin/classification/<int:record_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_edit_classification(record_id):
    """Edit classification record."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('admin_history'))
    
    # Get classification record
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE ch.id = ?
    """, (record_id,))
    
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        flash('Record not found.')
        return redirect(url_for('admin_history'))
    
    record = dict(record)
    
    if request.method == 'POST':
        # Update the record
        wbc = request.form.get('wbc', type=float)
        rbc = request.form.get('rbc', type=float)
        hgb = request.form.get('hgb', type=float)
        hct = request.form.get('hct', type=float)
        mcv = request.form.get('mcv', type=float)
        mch = request.form.get('mch', type=float)
        mchc = request.form.get('mchc', type=float)
        plt = request.form.get('plt', type=float)
        neutrophils = request.form.get('neutrophils', type=float)
        lymphocytes = request.form.get('lymphocytes', type=float)
        monocytes = request.form.get('monocytes', type=float)
        eosinophils = request.form.get('eosinophils', type=float)
        basophil = request.form.get('basophil', type=float)
        immature_granulocytes = request.form.get('immature_granulocytes', type=float)
        predicted_class = request.form.get('predicted_class')
        confidence_percentage = request.form.get('confidence', type=float)
        
        # Validate confidence percentage
        if confidence_percentage is None or confidence_percentage < 0 or confidence_percentage > 100:
            flash('Confidence must be between 0 and 100%.')
            return render_template('admin/edit_classification.html', record=record)
        
        # Convert percentage (0-100) to decimal (0-1) for storage
        confidence = confidence_percentage / 100.0
        recommendation = request.form.get('recommendation')
        notes = request.form.get('notes')
        
        # Update the record in database
        conn = db.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE classification_history 
            SET wbc = ?, rbc = ?, hgb = ?, hct = ?, mcv = ?, mch = ?, mchc = ?, plt = ?,
                neutrophils = ?, lymphocytes = ?, monocytes = ?, eosinophils = ?, basophil = ?, immature_granulocytes = ?,
                predicted_class = ?, confidence = ?, recommendation = ?, notes = ?
            WHERE id = ?
        """, (wbc, rbc, hgb, hct, mcv, mch, mchc, plt, neutrophils, lymphocytes, monocytes, 
              eosinophils, basophil, immature_granulocytes, predicted_class, confidence, 
              recommendation, notes, record_id))
        
        conn.commit()
        conn.close()
        
        flash('Classification record updated successfully!')
        return redirect(url_for('admin_history'))
    
    return render_template('admin/edit_classification.html', record=record)


@app.route('/admin/classification/<int:record_id>/delete', methods=['POST'])
@login_required
def admin_delete_classification(record_id):
    """Delete classification record."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get record details for confirmation
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE ch.id = ?
    """, (record_id,))
    
    record = cursor.fetchone()
    
    if not record:
        conn.close()
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    # Delete the record
    cursor.execute("DELETE FROM classification_history WHERE id = ?", (record_id,))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Admin {current_user.username} deleted classification record ID {record_id}")
    return jsonify({'success': True, 'message': f'Classification record for {record["username"]} deleted successfully'})


@app.route('/admin/classification/delete-all', methods=['POST'])
@login_required
def admin_delete_all_classifications():
    """Delete ALL classification records. Admin-only, returns JSON."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM classification_history")
        conn.commit()
        conn.close()
        logger.warning(f"Admin {current_user.username} deleted ALL classification records")
        return jsonify({'success': True, 'message': 'All classification records have been deleted.'})
    except Exception as e:
        logger.error(f"Failed to delete all classification records: {e}")
        return jsonify({'success': False, 'error': 'Failed to delete all classification records.'}), 500

@app.route('/admin/classification/filtered-data')
@login_required
def admin_classification_filtered_data():
    """Get filtered classification history data for AJAX requests."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get filter parameters
    user_filter = request.args.get('user', '')
    result_filter = request.args.get('result', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    # Build query
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    # Base query
    query = """
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE 1=1
    """
    params = []
    
    # Apply user filter
    if user_filter:
        query += " AND u.username = ?"
        params.append(user_filter)
    
    # Apply result filter
    if result_filter:
        if result_filter == 'Anemic':
            # Filter for all anemic types (Mild, Moderate, Severe)
            query += " AND ch.predicted_class IN ('Mild', 'Moderate', 'Severe')"
        else:
            query += " AND ch.predicted_class = ?"
            params.append(result_filter)
    
    # Apply date filters
    if date_from:
        query += " AND DATE(ch.created_at) >= ?"
        params.append(date_from)
    
    if date_to:
        query += " AND DATE(ch.created_at) <= ?"
        params.append(date_to)
    
    # Order by date descending
    query += " ORDER BY ch.created_at DESC"
    
    cursor.execute(query, params)
    records = cursor.fetchall()
    
    # Also fetch filtered "another person" entries
    op_query = """
        SELECT ch.*, u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        WHERE ch.notes LIKE 'Patient:%'
    """
    op_params = []
    if user_filter:
        op_query += " AND u.username = ?"
        op_params.append(user_filter)
    if result_filter:
        if result_filter == 'Anemic':
            op_query += " AND ch.predicted_class IN ('Mild', 'Moderate', 'Severe')"
        else:
            op_query += " AND ch.predicted_class = ?"
            op_params.append(result_filter)
    if date_from:
        op_query += " AND DATE(ch.created_at) >= ?"
        op_params.append(date_from)
    if date_to:
        op_query += " AND DATE(ch.created_at) <= ?"
        op_params.append(date_to)
    op_query += " ORDER BY ch.created_at DESC"
    cursor2 = db.get_db_connection().cursor()
    cursor2.execute(op_query, op_params)
    other_records = cursor2.fetchall()
    cursor2.connection.close()
    conn.close()
    
    # Convert to list of dicts
    filtered_records = []
    for record in records:
        record_dict = dict(record)
        # Format confidence as percentage
        record_dict['confidence_percentage'] = round(record_dict['confidence'] * 100, 2)
        # Format timestamp with AM/PM
        if 'created_at' in record_dict:
            record_dict['created_at'] = format_philippines_time_ampm(record_dict['created_at'])
        filtered_records.append(record_dict)
    
    # Prepare other person list
    other_filtered = []
    for rec in other_records:
        r = dict(rec)
        if 'created_at' in r:
            r['created_at'] = format_philippines_time_ampm(r['created_at'])
        r['confidence_percentage'] = round((r.get('confidence') or 0) * 100, 2)
        other_filtered.append(r)
    
    return jsonify({
        'success': True,
        'data': filtered_records,
        'total_count': len(filtered_records),
        'other_person': other_filtered,
        'other_total_count': len(other_filtered)
    })


@app.route('/admin/classification/available-users')
@login_required
def admin_classification_available_users():
    """Get list of users who have classification records."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    conn = db.get_db_connection()
    cursor = conn.cursor()
    
    # Get unique usernames from classification history
    cursor.execute("""
        SELECT DISTINCT u.username
        FROM classification_history ch
        LEFT JOIN users u ON ch.user_id = u.id
        ORDER BY u.username
    """)
    
    users = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({
        'success': True,
        'users': users
    })


# Simple Chat routes
simple_chat.init_chat_tables()

@app.route('/chat')
@login_required
def chat():
    """User messenger interface with admin chat."""
    # Get user conversations
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=False)
    
    # Format timestamps with AM/PM
    for conversation in conversations:
        if 'last_message_time' in conversation and conversation['last_message_time']:
            conversation['last_message_time'] = format_philippines_time_ampm(conversation['last_message_time'])
    
    return render_template('user_messenger.html', 
                         conversations=conversations,
                         current_user=current_user)

@app.route('/admin/messenger')
@login_required
def admin_messenger():
    """Admin messenger interface with individual chat windows."""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.')
        return redirect(url_for('dashboard'))
    
    # Get all users
    all_users = simple_chat.get_all_users()
    logger.info(f"Admin messenger: Found {len(all_users)} users")
    
    # Get admin conversations for history
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=True)
    
    # Format timestamps with AM/PM
    for conversation in conversations:
        if 'last_message_time' in conversation and conversation['last_message_time']:
            conversation['last_message_time'] = format_philippines_time_ampm(conversation['last_message_time'])
    
    logger.info(f"Admin messenger: Found {len(conversations)} conversations")
    
    return render_template('admin/messenger.html', 
                         all_users=all_users,
                         conversations=conversations,
                         current_user=current_user)

@app.route('/admin/chat/start', methods=['POST'])
@login_required
def admin_start_chat():
    """Admin starts chat with user."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    user_id = data.get('user_id')
    logger.info(f"Admin start chat: user_id={user_id}")
    
    if not user_id:
        return jsonify({'success': False, 'error': 'User ID required'}), 400
    
    # Create conversation
    success, conversation_id = simple_chat.create_conversation(user_id, admin_id=current_user.id)
    logger.info(f"Admin start chat: success={success}, conversation_id={conversation_id}")
    
    if success:
        return jsonify({
            'success': True,
            'conversation_id': conversation_id
        })
    else:
        return jsonify({'success': False, 'error': conversation_id}), 500

@app.route('/admin/chat/conversation/<int:user_id>')
@login_required
def admin_get_conversation(user_id):
    """Get conversation between admin and user."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Find existing conversation
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=True)
    conversation = next((c for c in conversations if c['user_id'] == user_id), None)
    
    if conversation:
        return jsonify({
            'success': True,
            'conversation_id': conversation['id']
        })
    else:
        return jsonify({'success': False, 'error': 'No conversation found'})

@app.route('/admin/chat/messages/<int:conversation_id>')
@login_required
def admin_get_messages(conversation_id):
    """Get messages for conversation."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    logger.info(f"Admin get messages: conversation_id={conversation_id}, admin_id={current_user.id}")
    
    messages = simple_chat.get_conversation_messages(conversation_id)
    
    # Format timestamps with AM/PM
    for message in messages:
        if 'created_at' in message:
            message['created_at'] = format_philippines_time_ampm(message['created_at'])
    
    logger.info(f"Admin get messages result: {len(messages)} messages found")
    
    return jsonify({
        'success': True,
        'messages': messages
    })

@app.route('/admin/chat/delete-message', methods=['POST'])
@login_required
def admin_delete_message():
    """Delete a specific message for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    message_id = data.get('message_id')
    
    logger.info(f"Admin delete message: message_id={message_id}, admin_id={current_user.id}")
    
    if not message_id:
        return jsonify({'success': False, 'error': 'Message ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Check if message belongs to current admin
        cursor.execute('SELECT sender_id FROM chat_messages WHERE id = ?', (message_id,))
        message = cursor.fetchone()
        
        if not message:
            return jsonify({'success': False, 'error': 'Message not found'}), 404
        
        if message['sender_id'] != current_user.id:
            return jsonify({'success': False, 'error': 'You can only delete your own messages'}), 403
        
        # Delete the message
        cursor.execute('DELETE FROM chat_messages WHERE id = ?', (message_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Admin message deleted: message_id={message_id}")
        return jsonify({'success': True, 'message': 'Message deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting admin message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/chat/delete-conversation', methods=['POST'])
@login_required
def admin_delete_conversation():
    """Delete entire conversation for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    logger.info(f"Admin delete conversation: conversation_id={conversation_id}, admin_id={current_user.id}")
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Delete all messages from this conversation
        cursor.execute('DELETE FROM chat_messages WHERE conversation_id = ?', (conversation_id,))
        
        # Delete the conversation itself
        cursor.execute('DELETE FROM chat_conversations WHERE id = ?', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Admin conversation deleted: conversation_id={conversation_id}")
        return jsonify({'success': True, 'message': 'Conversation deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting admin conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/chat/send', methods=['POST'])
@login_required
def admin_send_message():
    """Admin sends message."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    message_text = data.get('message', '').strip()
    
    logger.info(f"Admin send message: conversation_id={conversation_id}, message_text='{message_text}', admin_id={current_user.id}")
    
    if not message_text:
        return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    # Send message
    success, message_id = simple_chat.send_message(conversation_id, current_user.id, message_text)
    logger.info(f"Admin send message result: success={success}, message_id={message_id}")
    
    if success:
        return jsonify({'success': True, 'message_id': message_id})
    else:
        return jsonify({'success': False, 'error': message_id}), 500

@app.route('/user/chat/start', methods=['POST'])
@login_required
def user_start_chat():
    """User starts chat with admin."""
    data = request.get_json()
    admin_id = data.get('admin_id', 1)  # Default to admin ID 1 if not specified
    
    # Check if conversation already exists with this admin
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=False)
    existing_conversation = None
    
    for conv in conversations:
        if conv.get('admin_id') == admin_id:
            existing_conversation = conv
            break
    
    if existing_conversation:
        # Return existing conversation
        return jsonify({
            'success': True,
            'conversation_id': existing_conversation['id']
        })
    else:
        # Create new conversation with specific admin
        success, conversation_id = simple_chat.create_conversation(current_user.id, admin_id)
        
        if success:
            return jsonify({
                'success': True,
                'conversation_id': conversation_id
            })
        else:
            return jsonify({'success': False, 'error': conversation_id}), 500

@app.route('/user/chat/conversation')
@login_required
def user_get_conversation():
    """Get user's conversation with admin."""
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=False)
    
    if conversations:
        return jsonify({
            'success': True,
            'conversation_id': conversations[0]['id']
        })
    else:
        return jsonify({'success': False, 'error': 'No conversation found'})

@app.route('/user/chat/messages/<int:conversation_id>')
@login_required
def user_get_messages(conversation_id):
    """Get messages for user conversation."""
    logger.info(f"User get messages: conversation_id={conversation_id}, user_id={current_user.id}")
    
    messages = simple_chat.get_conversation_messages(conversation_id)
    
    # Format timestamps with AM/PM
    for message in messages:
        if 'created_at' in message:
            message['created_at'] = format_philippines_time_ampm(message['created_at'])
    
    logger.info(f"User get messages result: {len(messages)} messages found")
    
    return jsonify({
        'success': True,
        'messages': messages
    })

@app.route('/user/chat/clear-history', methods=['POST'])
@login_required
def user_clear_chat_history():
    """Clear chat history for user."""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    
    logger.info(f"User clear chat history: conversation_id={conversation_id}, user_id={current_user.id}")
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Delete all messages from this conversation
        cursor.execute('DELETE FROM chat_messages WHERE conversation_id = ?', (conversation_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User chat history cleared: conversation_id={conversation_id}")
        return jsonify({'success': True, 'message': 'Chat history cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/user/chat/delete-message', methods=['POST'])
@login_required
def user_delete_message():
    """Delete a specific message for user."""
    data = request.get_json()
    message_id = data.get('message_id')
    
    logger.info(f"User delete message: message_id={message_id}, user_id={current_user.id}")
    
    if not message_id:
        return jsonify({'success': False, 'error': 'Message ID required'}), 400
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Check if message belongs to current user
        cursor.execute('SELECT sender_id FROM chat_messages WHERE id = ?', (message_id,))
        message = cursor.fetchone()
        
        if not message:
            return jsonify({'success': False, 'error': 'Message not found'}), 404
        
        if message['sender_id'] != current_user.id:
            return jsonify({'success': False, 'error': 'You can only delete your own messages'}), 403
        
        # Delete the message
        cursor.execute('DELETE FROM chat_messages WHERE id = ?', (message_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User message deleted: message_id={message_id}")
        return jsonify({'success': True, 'message': 'Message deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/user/chat/send', methods=['POST'])
@login_required
def user_send_message():
    """User sends message."""
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    message_text = data.get('message', '').strip()
    
    logger.info(f"User send message: conversation_id={conversation_id}, message_text='{message_text}', user_id={current_user.id}")
    
    if not message_text:
        return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
    
    if not conversation_id:
        return jsonify({'success': False, 'error': 'Conversation ID required'}), 400
    
    # Send message
    success, message_id = simple_chat.send_message(conversation_id, current_user.id, message_text)
    logger.info(f"User send message result: success={success}, message_id={message_id}")
    
    if success:
        return jsonify({'success': True, 'message_id': message_id})
    else:
        return jsonify({'success': False, 'error': message_id}), 500

@app.route('/admin/chat/clear-data', methods=['POST'])
@login_required
def admin_clear_chat_data():
    """Clear all chat data (for testing purposes)."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    try:
        conn = simple_chat.get_db_connection()
        cursor = conn.cursor()
        
        # Clear all messages
        cursor.execute('DELETE FROM chat_messages')
        
        # Clear all conversations
        cursor.execute('DELETE FROM chat_conversations')
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Chat data cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/chat/check-new-messages')
@login_required
def admin_check_new_messages():
    """Check for new messages for admin."""
    if not current_user.is_admin:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    # Get conversations where admin is involved
    conversations = simple_chat.get_user_conversations(current_user.id, is_admin=True)
    
    new_messages = []
    for conv in conversations:
        # Get latest message
        messages = simple_chat.get_conversation_messages(conv['id'])
        if messages:
            latest_message = messages[-1]
            # Check if message is from user (not admin) and recent
            if latest_message['sender_id'] != current_user.id:
                # Check if message is within last 5 minutes
                from timezone_utils import parse_philippines_time
                message_time = parse_philippines_time(latest_message['created_at'])
                if message_time and (get_philippines_time() - message_time).seconds < 300:  # 5 minutes
                    new_messages.append({
                        'user_id': conv['user_id'],
                        'username': conv['username'],
                        'message': latest_message['message_text'][:50] + '...' if len(latest_message['message_text']) > 50 else latest_message['message_text']
                    })
    
    return jsonify({
        'success': True,
        'new_messages': new_messages
    })

@app.route('/chat/unread-count')
@login_required
def get_unread_count():
    """Get unread message count."""
    # For simple chat, we'll return 0 as we don't track unread status
    return jsonify({
        'success': True,
        'unread_count': 0
    })


# WebSocket events
@socketio.on('connect')
def handle_connect(auth=None):
    """Handle client connection to WebSocket."""
    if current_user.is_authenticated:
        # Join a room specific to this user
        join_room(str(current_user.id))
        logger.info(f"User {current_user.username} connected to WebSocket")
        
        # If admin, also join admin room
        if current_user.is_admin:
            join_room('admin_room')
            logger.info(f"Admin {current_user.username} joined admin room")
    else:
        logger.info("Anonymous user connected to WebSocket")


@socketio.on('request_update')
def handle_update_request():
    """Handle client request for updates."""
    if current_user.is_authenticated:
        # Get user's recent classification history
        history = db.get_user_classification_history(current_user.id, limit=5)
        emit('history_update', history)


@socketio.on('join_conversation')
def handle_join_conversation(data):
    """Handle user joining a conversation room."""
    if current_user.is_authenticated:
        conversation_id = data.get('conversation_id')
        if conversation_id:
            # Verify user has access to this conversation
            conversations = simple_chat.get_user_conversations(current_user.id, current_user.is_admin)
            if any(conv['id'] == conversation_id for conv in conversations):
                join_room(f'conversation_{conversation_id}')
                logger.info(f"User {current_user.username} joined conversation {conversation_id}")


@socketio.on('leave_conversation')
def handle_leave_conversation(data):
    """Handle user leaving a conversation room."""
    if current_user.is_authenticated:
        conversation_id = data.get('conversation_id')
        if conversation_id:
            leave_room(f'conversation_{conversation_id}')
            logger.info(f"User {current_user.username} left conversation {conversation_id}")


@socketio.on('typing')
def handle_typing(data):
    """Handle typing indicator."""
    if current_user.is_authenticated:
        conversation_id = data.get('conversation_id')
        is_typing = data.get('is_typing', False)
        
        if conversation_id:
            typing_data = {
                'user_id': current_user.id,
                'username': current_user.username,
                'is_typing': is_typing
            }
            emit('user_typing', typing_data, room=f'conversation_{conversation_id}', include_self=False)


# OTP Configuration
OTP_EXPIRY_MINUTES = 10  # OTP expires in 10 minutes

def generate_otp():
    """Generate a 6-digit OTP code."""
    return ''.join(random.choices(string.digits, k=6))

# OTP email function moved to email_service.py (Brevo API)

# Initialize the database and model when the app starts
with app.app_context():
    # Initialize database if it doesn't exist (only for SQLite)
    if not db.USE_POSTGRES and db.DB_PATH and not os.path.exists(db.DB_PATH):
        db.init_db()
    else:
        # Ensure new patient columns exist on existing databases (SQLite or Postgres)
        try:
            db.ensure_patient_columns()
        except Exception as _e:
            pass
    
    # Initialize anemia model with system settings
    threshold_normal = float(db.get_system_setting('threshold_normal') or 12.0)
    threshold_mild = float(db.get_system_setting('threshold_mild') or 10.0)
    threshold_moderate = float(db.get_system_setting('threshold_moderate') or 8.0)
    model_type = db.get_system_setting('model_type') or 'decision_tree'
    
    anemia_model.update_thresholds(
        threshold_normal=threshold_normal,
        threshold_mild=threshold_mild,
        threshold_moderate=threshold_moderate
    )
    anemia_model.set_model_type(model_type)
    #anemia_model.initialize()


@app.route('/export/history.csv')
@login_required
def export_my_classification_history():
    """Export current user's classification history as CSV."""
    # Fetch current user's records
    records = db.get_user_classification_history(current_user.id, limit=100000)

    # Build CSV
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Split records into self vs another person
    self_records = []
    other_records = []
    for r in records:
        notes = (r.get('notes') or '').strip()
        if r.get('patient_name') or r.get('patient_age') or r.get('patient_gender') or notes.lower().startswith('patient:'):
            other_records.append(r)
        else:
            self_records.append(r)

    # Section 1: Self classifications
    writer.writerow(['Classifications (Self)'])
    writer.writerow([
        'Date',
        'Full Name', 'Age', 'Gender',
        'WBC', 'RBC', 'HGB (g/dL)', 'HCT (%)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)', 'PLT',
        'NEU (%)', 'LYM (%)', 'MON (%)', 'EOS (%)', 'BAS (%)', 'IGR (%)',
        'Classification', 'Confidence', 'Notes'
    ])
    from datetime import datetime
    full_name = f"{(getattr(current_user, 'first_name', '') or '').strip()} {(getattr(current_user, 'last_name', '') or '').strip()}".strip()
    user_gender = (getattr(current_user, 'gender', '') or '')
    user_dob = getattr(current_user, 'date_of_birth', None)
    for r in self_records:
        formatted_date = '\t' + format_philippines_time_ampm(r.get('created_at', ''))
        age_val = ''
        try:
            if user_dob and r.get('created_at'):
                dob = datetime.strptime(user_dob, "%Y-%m-%d").date()
                created_dt = datetime.strptime(str(r.get('created_at')).split('.')[0], "%Y-%m-%d %H:%M:%S")
                d = created_dt.date()
                age_val = d.year - dob.year - ((d.month, d.day) < (dob.month, dob.day))
        except Exception:
            age_val = ''
        writer.writerow([
            formatted_date,
            full_name, age_val, user_gender,
            r.get('wbc', ''), r.get('rbc', ''), r.get('hgb', ''), r.get('hct', ''), r.get('mcv', ''), r.get('mch', ''), r.get('mchc', ''), r.get('plt', ''),
            r.get('neutrophils') if r.get('neutrophils') is not None else '',
            r.get('lymphocytes') if r.get('lymphocytes') is not None else '',
            r.get('monocytes') if r.get('monocytes') is not None else '',
            r.get('eosinophils') if r.get('eosinophils') is not None else '',
            r.get('basophil') if r.get('basophil') is not None else '',
            r.get('immature_granulocytes') if r.get('immature_granulocytes') is not None else '',
            r.get('predicted_class', ''),
            f"{float(r.get('confidence', 0))*100:.2f}%" if r.get('confidence') is not None else '',
            (r.get('notes') or '').strip()
        ])

    # Section 2: Another Person
    writer.writerow([])
    writer.writerow(['Classifications for Another Person'])
    writer.writerow([
        'Date',
        'Patient Name', 'Patient Age', 'Patient Gender',
        'WBC', 'RBC', 'HGB (g/dL)', 'HCT (%)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)', 'PLT',
        'NEU (%)', 'LYM (%)', 'MON (%)', 'EOS (%)', 'BAS (%)', 'IGR (%)',
        'Classification', 'Confidence', 'Notes'
    ])
    def _parse_legacy(raw):
        raw = (raw or '').strip()
        name = age = gender = ''
        cleaned = raw
        if raw.lower().startswith('patient:'):
            temp = raw
            def _extract(label, text):
                lbl = label.lower()
                if not text.lower().startswith(lbl):
                    return None, text
                sub = text[len(label):]
                dot = sub.find('.')
                comma = sub.find(',')
                end = -1
                if dot == -1 and comma == -1:
                    end = -1
                elif dot == -1:
                    end = comma
                elif comma == -1:
                    end = dot
                else:
                    end = min(dot, comma)
                value = sub[:end].strip() if end != -1 else sub.strip()
                remainder = sub[end + 1:].lstrip() if end != -1 else ''
                return value or None, remainder
            n, rem = _extract('Patient:', temp)
            if n is not None:
                name = n; temp = rem
            a, rem = _extract('Age:', temp)
            if a is not None:
                age = a; temp = rem
            g, rem = _extract('Gender:', temp)
            if g is not None:
                gender = g; temp = rem
            cleaned = (temp or '').strip()
        return name, age, gender, cleaned
    for r in other_records:
        formatted_date = '\t' + format_philippines_time_ampm(r.get('created_at', ''))
        p_name = r.get('patient_name') or ''
        p_age = r.get('patient_age') or ''
        p_gender = r.get('patient_gender') or ''
        raw_notes = (r.get('notes') or '').strip()
        cleaned_notes = raw_notes
        if not (p_name or p_age or p_gender):
            n, a, g, cleaned = _parse_legacy(raw_notes)
            p_name, p_age, p_gender = n or '', a or '', g or ''
            cleaned_notes = cleaned
        else:
            # remove any legacy preface
            _, _, _, cleaned = _parse_legacy(raw_notes)
            if cleaned:
                cleaned_notes = cleaned
        writer.writerow([
            formatted_date,
            p_name, p_age, p_gender,
            r.get('wbc', ''), r.get('rbc', ''), r.get('hgb', ''), r.get('hct', ''), r.get('mcv', ''), r.get('mch', ''), r.get('mchc', ''), r.get('plt', ''),
            r.get('neutrophils') if r.get('neutrophils') is not None else '',
            r.get('lymphocytes') if r.get('lymphocytes') is not None else '',
            r.get('monocytes') if r.get('monocytes') is not None else '',
            r.get('eosinophils') if r.get('eosinophils') is not None else '',
            r.get('basophil') if r.get('basophil') is not None else '',
            r.get('immature_granulocytes') if r.get('immature_granulocytes') is not None else '',
            r.get('predicted_class', ''),
            f"{float(r.get('confidence', 0))*100:.2f}%" if r.get('confidence') is not None else '',
            cleaned_notes
        ])

    csv_content = output.getvalue()
    output.close()

    # Response with BOM for Excel compatibility
    from flask import Response
    csv_content_with_bom = '\ufeff' + csv_content
    filename = f"classification_history_{current_user.username}.csv"
    response = Response(
        csv_content_with_bom,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )
    return response


@app.route('/api/profile/email-exists')
@login_required
def api_profile_email_exists():
    """Check if an email already exists (excluding current user's email)."""
    email = (request.args.get('email') or '').strip()
    if not email:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    user = db.get_user_by_email(email)
    if not user:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    # If found but it's the same as current user, allow it
    is_current = (str(user.get('id')) == str(current_user.id))
    return jsonify({ 'success': True, 'exists': not is_current, 'isCurrent': is_current })

@app.route('/api/profile/medical-id-exists')
@login_required
def api_profile_medical_id_exists():
    """Check if a medical ID already exists (excluding current user's own)."""
    mid = (request.args.get('medical_id') or '').strip()
    # Empty medical_id is allowed (treated as NULL) — never considered a duplicate
    if not mid:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    user = db.get_user_by_medical_id(mid)
    if not user:
        return jsonify({ 'success': True, 'exists': False, 'isCurrent': False })
    is_current = (str(user.get('id')) == str(current_user.id))
    return jsonify({ 'success': True, 'exists': not is_current, 'isCurrent': is_current })


@app.route('/api/register/username-exists')
def api_register_username_exists():
    """Public endpoint: check if a username already exists for registration page."""
    username = (request.args.get('username') or '').strip()
    if not username:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_username(username)
    return jsonify({ 'success': True, 'exists': bool(user) })

@app.route('/api/register/email-exists')
def api_register_email_exists():
    """Public endpoint: check if an email already exists for registration page."""
    email = (request.args.get('email') or '').strip()
    if not email:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_email(email)
    return jsonify({ 'success': True, 'exists': bool(user) })

@app.route('/api/register/medical-id-exists')
def api_register_medical_id_exists():
    """Public endpoint: check if a medical ID already exists for registration page."""
    medical_id = (request.args.get('medical_id') or '').strip()
    if not medical_id:
        return jsonify({ 'success': True, 'exists': False })
    user = db.get_user_by_medical_id(medical_id)
    return jsonify({ 'success': True, 'exists': bool(user) })


@app.route('/forgot-password', methods=['POST'])
@csrf.exempt
def forgot_password():
    """Send password reset OTP to user's email."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'success': False, 'error': 'Email is required'})
        
        # Check if user exists
        user = db.get_user_by_email(email)
        if not user:
            return jsonify({'success': False, 'error': 'Looks like this email isn\'t linked to an account yet.'})
        
        # Generate OTP
        otp_code = generate_otp()
        expires_at = get_philippines_time_plus_minutes(10)
        
        # Store OTP
        if db.store_password_reset_otp(email, otp_code, expires_at):
            # Send OTP email
            if send_password_reset_otp_email(email, otp_code):
                return jsonify({'success': True, 'message': 'Password reset code sent to your email'})
            else:
                return jsonify({'success': False, 'error': 'Failed to send email. Please try again.'})
        else:
            return jsonify({'success': False, 'error': 'Failed to generate reset code. Please try again.'})
            
    except Exception as e:
        logger.error(f"Error in forgot_password: {str(e)}")
        return jsonify({'success': False, 'error': 'An error occurred. Please try again.'})


@app.route('/verify-otp', methods=['POST'])
@csrf.exempt
def verify_otp():
    """Verify OTP code for password reset."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        otp_code = data.get('otp_code', '').strip()
        
        if not all([email, otp_code]):
            return jsonify({'success': False, 'error': 'Email and OTP code are required'})
        
        # Verify OTP
        if db.verify_password_reset_otp(email, otp_code):
            return jsonify({'success': True, 'message': 'OTP verified successfully'})
        else:
            return jsonify({'success': False, 'error': 'Invalid or expired verification code'})
            
    except Exception as e:
        logger.error(f"Error in verify_otp: {str(e)}")
        return jsonify({'success': False, 'error': 'An error occurred. Please try again.'})


@app.route('/reset-password', methods=['POST'])
@csrf.exempt
def reset_password():
    """Reset user password after OTP verification."""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        new_password = data.get('new_password', '').strip()
        
        if not all([email, new_password]):
            return jsonify({'success': False, 'error': 'Email and new password are required'})
        
        if len(new_password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters long'})
        
        # Update password
        password_hash = generate_password_hash(new_password)
        if db.update_user_password_by_email(email, password_hash):
            # Clean up OTP
            db.cleanup_password_reset_otp()
            return jsonify({'success': True, 'message': 'Password reset successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to update password. Please try again.'})
            
    except Exception as e:
        logger.error(f"Error in reset_password: {str(e)}")
        return jsonify({'success': False, 'error': 'An error occurred. Please try again.'})


def send_password_reset_otp_email(email, otp_code):
    """Send password reset OTP email using Brevo API."""
    try:
        # Get Brevo service
        from email_service import get_brevo_service
        brevo_service = get_brevo_service()
        
        if not brevo_service:
            logger.warning("Brevo email service not configured. Using development fallback")
            print(f"\n{'='*60}")
            print(f"DEVELOPMENT MODE - PASSWORD RESET OTP")
            print(f"{'='*60}")
            print(f"To: {email}")
            print(f"Subject: Password Reset - AnemoCheck")
            print(f"")
            print(f"Your password reset code is: {otp_code}")
            print(f"This code will expire in 10 minutes.")
            print(f"{'='*60}\n")
            return True
        
        # Create email content
        subject = "Password Reset - AnemoCheck"
        
        # Create HTML email content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Password Reset</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #c62828; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .otp-box {{ background-color: white; padding: 30px; margin: 20px 0; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .otp-code {{ font-size: 36px; font-weight: bold; color: #c62828; letter-spacing: 8px; margin: 20px 0; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Password Reset Request</h1>
                </div>
                <div class="content">
                    <h2>Hello,</h2>
                    <p>You have requested to reset your password for your AnemoCheck account.</p>
                    
                    <div class="otp-box">
                        <h3>Your Password Reset Code</h3>
                        <div class="otp-code">{otp_code}</div>
                        <p>Enter this code in the password reset page to continue.</p>
                    </div>
                    
                    <div class="warning">
                        <strong>Important:</strong>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>This code will expire in 10 minutes</li>
                            <li>Do not share this code with anyone</li>
                            <li>If you didn't request this password reset, please ignore this email</li>
                        </ul>
                    </div>
                    
                    <p>If you have any questions, please contact our support team.</p>
                </div>
                <div class="footer">
                    <p>This is an automated message from AnemoCheck. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = f"""
        Password Reset Request
        
        Hello,
        
        You have requested to reset your password for your AnemoCheck account.
        
        Your Password Reset Code: {otp_code}
        
        Enter this code in the password reset page to continue.
        
        Important:
        - This code will expire in 10 minutes
        - Do not share this code with anyone
        - If you didn't request this password reset, please ignore this email
        
        If you have any questions, please contact our support team.
        
        This is an automated message from AnemoCheck. Please do not reply to this email.
        """
        
        # Send email using Brevo
        success, message = brevo_service.send_email(
            to_email=email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            to_name=email.split('@')[0]
        )
        
        if success:
            logger.info(f"Password reset OTP sent successfully to {email}")
            return True
        else:
            logger.error(f"Failed to send password reset OTP: {message}")
            # Fallback to development mode
            print(f"\n{'='*60}")
            print(f"EMAIL SENDING FAILED - DEVELOPMENT FALLBACK")
            print(f"{'='*60}")
            print(f"To: {email}")
            print(f"Subject: Password Reset - AnemoCheck")
            print(f"")
            print(f"Your password reset code is: {otp_code}")
            print(f"This code will expire in 10 minutes.")
            print(f"{'='*60}\n")
            return True
        
    except Exception as e:
        logger.error(f"Error sending password reset OTP email: {str(e)}")
        # Fallback to development mode
        print(f"\n{'='*60}")
        print(f"EMAIL SENDING FAILED - DEVELOPMENT FALLBACK")
        print(f"{'='*60}")
        print(f"To: {email}")
        print(f"Subject: Password Reset - AnemoCheck")
        print(f"")
        print(f"Your password reset code is: {otp_code}")
        print(f"This code will expire in 10 minutes.")
        print(f"{'='*60}\n")
        return True


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)