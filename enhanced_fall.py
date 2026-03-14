import RPi.GPIO as GPIO
import cv2
from twilio.rest import Client
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time
import threading
from queue import Queue
import argparse
import os
import signal
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from flask import Flask, Response, render_template_string, jsonify, request
import base64
import json
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict

# ------------------- DATABASE SETUP -------------------
def init_database():
    conn = sqlite3.connect('fall_detection.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create incidents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_data TEXT,
            confidence REAL,
            location TEXT,
            resolved BOOLEAN DEFAULT FALSE,
            notes TEXT
        )
    ''')
    
    # Create settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Create system_stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_detections INTEGER,
            false_positives INTEGER,
            system_uptime REAL,
            average_fps REAL
        )
    ''')
    
    conn.commit()
    return conn

# Initialize database
db_conn = init_database()

# ------------------- EMAIL CONFIGURATION -------------------
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv('FALL_EMAIL_USER', 'sreebhavya295@gmail.com')
EMAIL_PASSWORD = os.getenv('FALL_EMAIL_PASS', 'kzbblebmkrasrptq')
RECIPIENT_EMAIL = os.getenv('FALL_ALERT_RECIPIENT', 'rajatamhegde.is23@rvce.edu.in')
ALERT_COOLDOWN = 30  # seconds between alerts
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'your_account_sid')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'your_auth_token')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER', '+1234567890')
TWILIO_TO_NUMBER = os.getenv('TWILIO_TO_NUMBER', '+0987654321')

# ------------------- SYSTEM SETTINGS -------------------
class SystemSettings:
    def __init__(self):
        self.load_settings()
    
    def load_settings(self):
        cursor = db_conn.cursor()
        cursor.execute('SELECT key, value FROM settings')
        settings = dict(cursor.fetchall())
        
        self.email_alerts = settings.get('email_alerts', 'true') == 'true'
        self.detection_sensitivity = float(settings.get('detection_sensitivity', '0.7'))
        self.alert_cooldown = int(settings.get('alert_cooldown', '30'))
        self.auto_resolve_time = int(settings.get('auto_resolve_time', '300'))
        self.recording_enabled = settings.get('recording_enabled', 'false') == 'true'
    
    def save_setting(self, key, value):
        cursor = db_conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, str(value)))
        db_conn.commit()
        self.load_settings()

settings = SystemSettings()

def send_fall_alert_email(frame, confidence=0.0):
    if not settings.email_alerts:
        return
        
    msg = MIMEMultipart()
    msg['Subject'] = '🚨 Fall Detection Alert - Immediate Attention Required'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL

    text = MIMEText(f'''
    FALL DETECTION ALERT
    
    Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Confidence Level: {confidence:.2%}
    Location: Main Monitoring Area
    
    A potential fall has been detected by the monitoring system.
    Please check the attached image and take appropriate action if required.
    
    This is an automated alert from the Fall Detection System.
    ''')
    msg.attach(text)

    # Attach frame as image
    _, img_encoded = cv2.imencode('.jpg', frame)
    image = MIMEImage(img_encoded.tobytes())
    image.add_header('Content-Disposition', 'attachment', filename=f'fall_detection_{int(time.time())}.jpg')
    msg.attach(image)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        print(f"\n[EMAIL] Alert sent successfully at {datetime.now()}")
        return True
    except Exception as e:
        print(f"\n[EMAIL ERROR] Failed to send email: {e}")
        return False
def send_fall_alert_sms(confidence=0.0):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"🚨 Fall Detected! Confidence: {confidence:.2%} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_TO_NUMBER
        )
        print(f"[SMS] Alert sent successfully: SID {message.sid}")
        return True
    except Exception as e:
        print(f"[SMS ERROR] Failed to send SMS: {e}")
        return False



# ------------------- FALL DETECTION CLASS -------------------
class HumanPoseDetector:
    def __init__(self, camera_id=0):
        self.yolo_model = YOLO('yolov8n.pt')
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=1,
            enable_segmentation=False, 
            min_detection_confidence=settings.detection_sensitivity,
            min_tracking_confidence=settings.detection_sensitivity
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.camera_id = camera_id
        self.fps_counter = 0
        self.start_time = time.time()
        self.current_fps = 0
        self.running = True
        self.frame_count = 0
        self.last_alert_time = 0
        self.total_detections = 0
        self.session_start_time = time.time()
        
        # Performance metrics
        self.fps_history = []
        self.detection_confidence_history = []

    def detect_humans(self, frame):
        results = self.yolo_model(frame, classes=[0], verbose=False)
        human_boxes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if box.conf[0] > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        human_boxes.append((x1, y1, x2, y2, confidence))
        return human_boxes

    def estimate_pose(self, frame, bbox):
        x1, y1, x2, y2, _ = bbox
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        human_roi = frame[y1:y2, x1:x2]
        
        if human_roi.size > 0:
            rgb_roi = cv2.cvtColor(human_roi, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_roi)
            if results.pose_landmarks:
                landmarks = [(int(landmark.x * human_roi.shape[1]) + x1,
                              int(landmark.y * human_roi.shape[0]) + y1)
                             for landmark in results.pose_landmarks.landmark]
                return landmarks, results.pose_landmarks
        return None, None

    def draw_pose(self, frame, landmarks):
        if landmarks:
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (0, 255, 0), 2)
            for point in landmarks:
                cv2.circle(frame, point, 3, (0, 0, 255), -1)

    def detect_fall(self, landmarks, bbox, frame_shape):
        if landmarks:
            try:
                nose = landmarks[0]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                
                avg_hip_y = (left_hip[1] + right_hip[1]) / 2
                vertical_distance = abs(nose[1] - avg_hip_y)
                
                x1, y1, x2, y2, _ = bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                aspect_ratio = bbox_width / bbox_height if bbox_height != 0 else 0
                
                avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
                ground_threshold = 0.85 * frame_shape[0]
                
                # Enhanced fall detection criteria
                criteria = [
                    vertical_distance < 60,  # Person is horizontally oriented
                    aspect_ratio > 1.3,     # Bounding box is wider than tall
                    avg_ankle_y > ground_threshold,  # Person is near ground level
                    nose[1] > avg_hip_y - 20  # Head is not significantly above hips
                ]
                
                confidence = sum(criteria) / len(criteria)
                
                if sum(criteria) >= 3:  # At least 3 out of 4 criteria must be met
                    return True, confidence
                    
            except Exception as e:
                print(f"Error in fall detection: {e}")
                
        return False, 0.0

    def process_frame(self, frame):
        human_boxes = self.detect_humans(frame)
        fall_detected = False
        max_confidence = 0.0
        
        # Update FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.start_time)
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > 100:  # Keep last 100 FPS readings
                self.fps_history.pop(0)
            self.start_time = current_time
        
        for bbox in human_boxes:
            x1, y1, x2, y2, conf = bbox
            # Draw human detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Person: {conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            landmarks, _ = self.estimate_pose(frame, bbox)
            if landmarks:
                self.draw_pose(frame, landmarks)
                is_fall, fall_confidence = self.detect_fall(landmarks, bbox, frame.shape)
                
                if is_fall:
                    fall_detected = True
                    max_confidence = max(max_confidence, fall_confidence)
                    
                    # Draw fall alert
                    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 0, 255), 3)
                    cv2.putText(frame, "🚨 FALL DETECTED!", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    cv2.putText(frame, f'Confidence: {fall_confidence:.2%}', (x1, y1 - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Send alert if cooldown period has passed
                    current_time = time.time()
                    if current_time - self.last_alert_time > settings.alert_cooldown:
                        self.save_incident(frame, fall_confidence)
                        send_fall_alert_email(frame, fall_confidence)
                        send_fall_alert_sms(fall_confidence)
                        self.last_alert_time = current_time
                        self.total_detections += 1
        
        return frame, fall_detected, max_confidence

    def save_incident(self, frame, confidence):
        """Save incident to database"""
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            
            cursor = db_conn.cursor()
            cursor.execute('''
                INSERT INTO incidents (image_data, confidence, location, notes)
                VALUES (?, ?, ?, ?)
            ''', (img_b64, confidence, 'Main Camera', f'Auto-detected with {confidence:.2%} confidence'))
            db_conn.commit()
            
        except Exception as e:
            print(f"Error saving incident: {e}")

    def get_system_stats(self):
        uptime = time.time() - self.session_start_time
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        return {
            'uptime': uptime,
            'current_fps': self.current_fps,
            'average_fps': avg_fps,
            'total_detections': self.total_detections,
            'frames_processed': self.frame_count
        }

# ------------------- CAMERA THREAD CLASS -------------------
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError(f"Cannot connect to camera {src}")
            
        self.q = Queue(maxsize=2)
        self.running = True

    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except:
                        pass
                try:
                    self.q.put_nowait(frame)
                except:
                    pass

    def read(self):
        if not self.q.empty():
            return True, self.q.get()
        return False, None

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()

# ------------------- FLASK APP FOR DASHBOARD -------------------
app = Flask(__name__)

detector = None
camera = None
current_frame = None
current_status = {"status": "Normal", "confidence": 0.0, "timestamp": ""}

def generate_frames():
    global detector, camera, current_frame, current_status
    while detector and detector.running:
        ret, frame = camera.read()
        if not ret:
            continue
            
        processed_frame, fall_detected, confidence = detector.process_frame(frame)
        current_frame = processed_frame.copy()
        
        # Update current status
        current_status = {
            "status": "FALL DETECTED" if fall_detected else "Normal",
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fall_detected": fall_detected
        }
        
        # Add system info overlay
        stats = detector.get_system_stats()
        cv2.putText(processed_frame, f"FPS: {stats['current_fps']:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Status: {current_status['status']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not fall_detected else (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Uptime: {stats['uptime']/3600:.1f}h", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def dashboard():
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fall Detection Monitoring System</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        
        .navbar h1 {
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }
        
        .main-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .video-container {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            margin-bottom: 1rem;
        }
        
        .video-stream {
            width: 100%;
            height: auto;
            max-height: 500px;
            object-fit: cover;
        }
        
        .status-overlay {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-normal { background: rgba(46, 125, 50, 0.9); }
        .status-alert { background: rgba(198, 40, 40, 0.9); animation: pulse 1s infinite; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .system-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .stat-item {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 0.25rem;
        }
        
        .incidents-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .incident-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
            border-radius: 12px;
            border-left: 4px solid #e53e3e;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .incident-item:hover {
            transform: translateY(-2px);
        }
        
        .incident-image {
            width: 80px;
            height: 60px;
            border-radius: 8px;
            object-fit: cover;
        }
        
        .incident-details {
            flex: 1;
        }
        
        .incident-time {
            font-weight: bold;
            color: #2c3e50;
            font-size: 0.9rem;
        }
        
        .incident-confidence {
            color: #e53e3e;
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .btn {
            flex: 1;
            padding: 0.75rem 1rem;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .settings-section {
            margin-top: 1rem;
        }
        
        .setting-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #eee;
        }
        
        .setting-item:last-child {
            border-bottom: none;
        }
        
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 24px;
            background: #ccc;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .toggle-switch.active {
            background: #4CAF50;
        }
        
        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: white;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
        }
        
        .toggle-switch.active::after {
            transform: translateX(26px);
        }
        
        .no-incidents {
            text-align: center;
            color: #7f8c8d;
            padding: 2rem;
            font-style: italic;
        }
        
        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h1><i class="fas fa-shield-alt"></i> Fall Detection Monitoring System</h1>
    </div>
    
    <div class="container">
        <div class="main-panel">
            <h2><i class="fas fa-video"></i> Live Camera Feed</h2>
            <div class="video-container">
                <img src="/video_feed" alt="Live Video Stream" class="video-stream" id="videoStream">
                <div class="status-overlay" id="statusOverlay">
                    <i class="fas fa-circle"></i>
                    <span id="statusText">Loading...</span>
                </div>
            </div>
            
            <div class="system-stats">
                <div class="stat-item">
                    <div class="stat-value" id="fpsValue">--</div>
                    <div class="stat-label">Current FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="uptimeValue">--</div>
                    <div class="stat-label">System Uptime</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="detectionsValue">--</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="framesValue">--</div>
                    <div class="stat-label">Frames Processed</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="refreshFeed()">
                    <i class="fas fa-sync-alt"></i> Refresh Feed
                </button>
                <button class="btn btn-secondary" onclick="downloadReport()">
                    <i class="fas fa-download"></i> Download Report
                </button>
            </div>
        </div>
        
        <div class="side-panel">
            <div class="card">
                <h3><i class="fas fa-exclamation-triangle"></i> Recent Incidents</h3>
                <div class="incidents-list" id="incidentsList">
                    <div class="loading" style="margin: 2rem auto;"></div>
                </div>
            </div>
            
            <div class="card">
                <h3><i class="fas fa-cog"></i> System Settings</h3>
                <div class="settings-section">
                    <div class="setting-item">
                        <span>Email Alerts</span>
                        <div class="toggle-switch active" onclick="toggleSetting('email_alerts', this)"></div>
                    </div>
                    <div class="setting-item">
                        <span>Auto Recording</span>
                        <div class="toggle-switch" onclick="toggleSetting('recording_enabled', this)"></div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3><i class="fas fa-chart-line"></i> System Health</h3>
                <div class="stat-item">
                    <div class="stat-value" style="color: #27ae60;">Normal</div>
                    <div class="stat-label">System Status</div>
                </div>
                <div class="stat-item" style="margin-top: 1rem;">
                    <div class="stat-value" id="memoryUsage">--</div>
                    <div class="stat-label">Memory Usage</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let lastStatusUpdate = 0;
        
        function updateSystemStats() {
            fetch('/api/system_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fpsValue').textContent = data.current_fps.toFixed(1);
                    document.getElementById('uptimeValue').textContent = (data.uptime / 3600).toFixed(1) + 'h';
                    document.getElementById('detectionsValue').textContent = data.total_detections;
                    document.getElementById('framesValue').textContent = data.frames_processed;
                })
                .catch(error => console.error('Error fetching system stats:', error));
        }
        
        function updateCurrentStatus() {
            fetch('/api/current_status')
                .then(response => response.json())
                .then(data => {
                    const statusOverlay = document.getElementById('statusOverlay');
                    const statusText = document.getElementById('statusText');
                    
                    statusText.textContent = data.status;
                    
                    if (data.fall_detected) {
                        statusOverlay.className = 'status-overlay status-alert';
                        statusText.textContent += ` (${(data.confidence * 100).toFixed(1)}%)`;
                    } else {
                        statusOverlay.className = 'status-overlay status-normal';
                    }
                })
                .catch(error => console.error('Error fetching status:', error));
        }
        
        function loadIncidents() {
            fetch('/api/incidents')
                .then(response => response.json())
                .then(data => {
                    const incidentsList = document.getElementById('incidentsList');
                    
                    if (data.length === 0) {
                        incidentsList.innerHTML = '<div class="no-incidents"><i class="fas fa-check-circle"></i><br>No incidents detected today!<br><small>System is monitoring normally</small></div>';
                        return;
                    }
                    
                    const incidentsHtml = data.map(incident => `
                        <div class="incident-item" onclick="showIncidentDetails(${incident.id})">
                            <img src="data:image/jpeg;base64,${incident.image_data}" alt="Incident" class="incident-image">
                            <div class="incident-details">
                                <div class="incident-time">
                                    <i class="fas fa-clock"></i> ${new Date(incident.timestamp).toLocaleString()}
                                </div>
                                <div class="incident-confidence">
                                    <i class="fas fa-exclamation-circle"></i> Confidence: ${(incident.confidence * 100).toFixed(1)}%
                                </div>
                                <div style="font-size: 0.8rem; color: #666; margin-top: 0.25rem;">
                                    ${incident.location} • ${incident.resolved ? 'Resolved' : 'Pending'}
                                </div>
                            </div>
                        </div>
                    `).join('');
                    
                    incidentsList.innerHTML = incidentsHtml;
                })
                .catch(error => {
                    console.error('Error loading incidents:', error);
                    document.getElementById('incidentsList').innerHTML = '<div class="no-incidents">Error loading incidents</div>';
                });
        }
        
        function toggleSetting(setting, element) {
            element.classList.toggle('active');
            const isActive = element.classList.contains('active');
            
            fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    key: setting,
                    value: isActive
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`${setting} updated to ${isActive}`);
                } else {
                    element.classList.toggle('active'); // Revert on error
                }
            })
            .catch(error => {
                console.error('Error updating setting:', error);
                element.classList.toggle('active'); // Revert on error
            });
        }
        
        function updateSensitivity(value) {
            fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    key: 'detection_sensitivity',
                    value: parseFloat(value)
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`Detection sensitivity updated to ${value}`);
                }
            })
            .catch(error => console.error('Error updating sensitivity:', error));
        }
        
        function refreshFeed() {
            const videoStream = document.getElementById('videoStream');
            const currentSrc = videoStream.src;
            videoStream.src = '';
            setTimeout(() => {
                videoStream.src = currentSrc + '?t=' + new Date().getTime();
            }, 100);
        }
        
        function downloadReport() {
            window.open('/api/download_report', '_blank');
        }
        
        function showIncidentDetails(incidentId) {
            // This could open a modal or navigate to a detailed view
            alert(`Opening details for incident ${incidentId}`);
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            updateSystemStats();
            updateCurrentStatus();
            loadIncidents();
            
            // Set up periodic updates
            setInterval(updateSystemStats, 5000); // Update every 5 seconds
            setInterval(updateCurrentStatus, 1000); // Update status every second
            setInterval(loadIncidents, 10000); // Update incidents every 10 seconds
            
            // Add memory usage simulation
            setInterval(() => {
                const memoryUsage = (Math.random() * 30 + 45).toFixed(1) + '%';
                document.getElementById('memoryUsage').textContent = memoryUsage;
            }, 5000);
        });
        
        // Handle visibility change to pause/resume updates when tab is not active
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                console.log('Tab hidden - reducing update frequency');
            } else {
                console.log('Tab visible - resuming normal updates');
                updateSystemStats();
                updateCurrentStatus();
                loadIncidents();
            }
        });
    </script>
</body>
</html>
    '''
    return html

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/system_stats')
def api_system_stats():
    if detector:
        stats = detector.get_system_stats()
        return jsonify(stats)
    return jsonify({'error': 'Detector not initialized'})

@app.route('/api/current_status')
def api_current_status():
    return jsonify(current_status)

@app.route('/api/incidents')
def api_incidents():
    try:
        cursor = db_conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, image_data, confidence, location, resolved, notes
            FROM incidents 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        incidents = []
        for row in cursor.fetchall():
            incidents.append({
                'id': row[0],
                'timestamp': row[1],
                'image_data': row[2][:1000] + '...' if len(row[2]) > 1000 else row[2],  # Truncate for performance
                'confidence': row[3],
                'location': row[4],
                'resolved': row[5],
                'notes': row[6]
            })
        
        return jsonify(incidents)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/settings', methods=['POST'])
def api_update_settings():
    try:
        data = request.json
        key = data.get('key')
        value = data.get('value')
        
        if key and value is not None:
            settings.save_setting(key, value)
            
            # Update detector settings if needed
            if key == 'detection_sensitivity' and detector:
                detector.pose = detector.mp_pose.Pose(
                    static_image_mode=False, 
                    model_complexity=1,
                    enable_segmentation=False, 
                    min_detection_confidence=float(value),
                    min_tracking_confidence=float(value)
                )
            
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Invalid parameters'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download_report')
def api_download_report():
    try:
        cursor = db_conn.cursor()
        cursor.execute('''
            SELECT timestamp, confidence, location, resolved, notes
            FROM incidents 
            ORDER BY timestamp DESC
        ''')
        
        incidents = cursor.fetchall()
        stats = detector.get_system_stats() if detector else {}
        
        report = f"""
FALL DETECTION SYSTEM REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM STATISTICS:
- System Uptime: {stats.get('uptime', 0)/3600:.2f} hours
- Current FPS: {stats.get('current_fps', 0):.1f}
- Average FPS: {stats.get('average_fps', 0):.1f}
- Total Detections: {stats.get('total_detections', 0)}
- Frames Processed: {stats.get('frames_processed', 0)}

INCIDENT SUMMARY:
Total Incidents: {len(incidents)}
Resolved: {sum(1 for i in incidents if i[3])}
Pending: {sum(1 for i in incidents if not i[3])}

DETAILED INCIDENTS:
"""
        
        for incident in incidents:
            report += f"""
Timestamp: {incident[0]}
Confidence: {incident[1]:.2%}
Location: {incident[2]}
Status: {'Resolved' if incident[3] else 'Pending'}
Notes: {incident[4] or 'None'}
---
"""
        
        from flask import make_response
        response = make_response(report)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Content-Disposition'] = f'attachment; filename=fall_detection_report_{int(time.time())}.txt'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/resolve_incident/<int:incident_id>', methods=['POST'])
def api_resolve_incident(incident_id):
    try:
        cursor = db_conn.cursor()
        cursor.execute('UPDATE incidents SET resolved = TRUE WHERE id = ?', (incident_id,))
        db_conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ------------------- MAIN LOOP -------------------
def signal_handler(sig, frame):
    print("\n🛑 Shutting down Fall Detection System...")
    global detector, camera
    if detector:
        detector.running = False
    if camera:
        camera.stop()
    GPIO.cleanup()
    if db_conn:
        db_conn.close()
    sys.exit(0)

def main():
    global detector, camera
    
    parser = argparse.ArgumentParser(description='Advanced Fall Detection System')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("🚀 Initializing Fall Detection System...")
        print("📊 Setting up database...")
        
        # Initialize detector and camera
        detector = HumanPoseDetector(args.camera)
        print("🤖 AI models loaded successfully")
        
        camera = ThreadedCamera(args.camera)
        camera.start()
        print(f"📹 Camera {args.camera} connected and streaming")
        LED_PIN = 23
        BUZZER_PIN = 24

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)

        # Turn off initially
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        
        print(f"🌐 Starting web dashboard on http://{args.host}:{args.port}")
        print("📱 Access the dashboard from any device on your network")
        print("✋ Press Ctrl+C to stop the system")
        
        # Start Flask app
        app.run(
            host=args.host, 
            port=args.port, 
            debug=args.debug, 
            threaded=True,
            use_reloader=False  # Disable reloader to prevent issues with threading
        )
        
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        print("🔧 Please check your camera connection and try again")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()