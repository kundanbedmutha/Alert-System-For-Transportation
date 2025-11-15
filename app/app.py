from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import pandas as pd
import os
import sqlite3
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from flask_dance.contrib.google import make_google_blueprint, google
from dotenv import load_dotenv
try:
    import resend
except Exception:
    resend = None
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------------------------------
# ✅ 1️⃣ Load environment variables first
# -------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# -------------------------------------------------
# ✅ 2️⃣ Initialize Flask app
# -------------------------------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.secret_key = os.getenv("SECRET_KEY", "super_secret_key_123")

def snap_to_road(lat, lon):
    try:
        url = f"http://router.project-osrm.org/nearest/v1/driving/{lon},{lat}"
        res = requests.get(url).json()
        if res and "waypoints" in res and len(res["waypoints"]) > 0:
            snapped = res["waypoints"][0]["location"]
            return snapped[1], snapped[0]  # (lat, lon)
    except Exception as e:
        print("⚠️ Road snap failed:", e)
    return lat, lon  # fallback to original

# Configure Google OAuth
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET"),
    redirect_to="google_login"
)
app.register_blueprint(google_bp, url_prefix="/login")

from functools import wraps

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                flash("Please sign in with Google.", "warning")
                return redirect(url_for("clerk_login"))  # ⬅️ now goes to Clerk
            if role and session.get("role") != role:
                flash("Access denied: insufficient permissions.", "danger")
                return redirect(url_for("home"))
            return f(*args, **kwargs)
        return wrapped
    return decorator


# -------------------------------------------------
# 1️⃣ Initialize Flask
# -------------------------------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.secret_key = "super_secret_key_123"

# ⚙️ Configure Google OAuth
google_bp = make_google_blueprint(
    client_id="YOUR_GOOGLE_CLIENT_ID",
    client_secret="YOUR_GOOGLE_CLIENT_SECRET",
    redirect_to="google_login"
)
app.register_blueprint(google_bp, url_prefix="/login")


# -------------------------------------------------
# 2️⃣ Auto-create dummy models if missing
# -------------------------------------------------
os.makedirs("app/models", exist_ok=True)
alert_model_path = "app/models/alert_model.joblib"
accident_model_path = "app/models/accident_model.joblib"
severity_model_path = "app/models/severity_model.joblib"

if not all(os.path.exists(p) for p in [alert_model_path, accident_model_path, severity_model_path]):
    print("⚙️ No trained models found — creating dummy models for testing...")
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    dummy_model = RandomForestClassifier()
    dummy_model.fit(X, y)
    joblib.dump(dummy_model, alert_model_path)
    joblib.dump(dummy_model, accident_model_path)
    joblib.dump(dummy_model, severity_model_path)
    print("✅ Dummy models created successfully.")

# -------------------------------------------------
# 3️⃣ Load models safely
# -------------------------------------------------
try:
    alert_model = joblib.load(alert_model_path)
    accident_model = joblib.load(accident_model_path)
    severity_model = joblib.load(severity_model_path)
    feature_list = ["speed", "temperature", "humidity", "rain", "visibility"]
    print("✅ Models loaded successfully.")
except Exception as e:
    print("❌ Error loading models:", e)
    alert_model = accident_model = severity_model = None
    feature_list = []

# -------------------------------------------------
# 🧩 Database schema auto-fix (ensures correct columns)
# -------------------------------------------------
def ensure_database_schema():
    conn = sqlite3.connect("alerts.db")

    # ✅ Ensure alerts table exists and has correct columns
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            timestamp TEXT,
            vehicle_id TEXT,
            message TEXT,
            severity TEXT,
            latitude REAL,
            longitude REAL
        )
    """)
    alert_cols = [c[1] for c in conn.execute("PRAGMA table_info(alerts);")]
    if "latitude" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN latitude REAL;")
    if "longitude" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN longitude REAL;")

    # ✅ Ensure notifications table exists and has correct columns
        # ✅ Ensure users table exists (for Clerk / Google sign-ins)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            vehicle_id TEXT,
            password TEXT,
            role TEXT DEFAULT 'driver'
        )
    """)
    print("✅ Users table verified.")

    notif_cols = [c[1] for c in conn.execute("PRAGMA table_info(notifications);")]
    required_notif_cols = ["timestamp", "target", "message", "level", "extra_info"]
    for col in required_notif_cols:
        if col not in notif_cols:
            # Add missing notification columns if any
            if col == "id":
                continue
            conn.execute(f"ALTER TABLE notifications ADD COLUMN {col} TEXT;")

    conn.commit()
    conn.close()
    print("✅ Database schema verified & updated.")


# ✅ Run automatically on startup
ensure_database_schema()

# -------------------------------------------------
# 🧠 Helper: Log alerts into SQLite (with GPS support)
# -------------------------------------------------
def log_alert(vehicle_id, message, severity="Normal", latitude=None, longitude=None):
    conn = sqlite3.connect("alerts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            timestamp TEXT,
            vehicle_id TEXT,
            message TEXT,
            severity TEXT,
            latitude REAL,
            longitude REAL
        )
    """)
    conn.execute(
        """
        INSERT INTO alerts (timestamp, vehicle_id, message, severity, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(), vehicle_id, message, severity, latitude, longitude)
    )
    conn.commit()
    conn.close()

# -------------------------------------------------
# 🧠 Helper: Snap coordinates to nearest road (simple approximation)
# -------------------------------------------------
def snap_to_road(lat, lon):
    """
    Simulates snapping GPS coordinates to nearest road.
    In production, this can be replaced with Google Roads API.
    """
    try:
        # For demo: round to 4 decimal places (approx 11m accuracy)
        rounded_lat = round(float(lat), 4)
        rounded_lon = round(float(lon), 4)
        return rounded_lat, rounded_lon
    except Exception as e:
        print("⚠️ Error in snap_to_road:", e)
        return lat, lon
# -------------------------------------------------
# 🧠 Helper: Log notifications for driver/police
# -------------------------------------------------
def log_notification(target, message, level="info", extra_info=None):
    """
    Logs notifications for drivers, police, or system alerts
    """
    conn = sqlite3.connect("alerts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            target TEXT,
            message TEXT,
            level TEXT,
            extra_info TEXT
        )
    """)
    conn.execute(
        """
        INSERT INTO notifications (timestamp, target, message, level, extra_info)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(), target, message, level, str(extra_info) if extra_info else None)
    )
    conn.commit()
    conn.close()

# -------------------------------------------------
# 🧠 Helper: Log notifications (Driver / Police)
# -------------------------------------------------
def log_notification(target, message, level="info", extra_info=None):
    """
    Logs a notification event for driver, police, or system.
    - target: "driver", "police", or "system"
    - message: Notification message text
    - level: "info", "warning", or "critical"
    - extra_info: optional JSON/text for details
    """
    conn = sqlite3.connect("alerts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            target TEXT,
            message TEXT,
            level TEXT,
            extra_info TEXT
        )
    """)
    conn.execute(
        """
        INSERT INTO notifications (timestamp, target, message, level, extra_info)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(), target, message, level, str(extra_info) if extra_info else None)
    )
    conn.commit()
    conn.close()


# -------------------------------------------------
# 5️⃣ ROUTES
# -------------------------------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/alerts')
def alerts():
    try:
        conn = sqlite3.connect("alerts.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC").fetchall()
        conn.close()

        now = datetime.now()
        live_alerts = [r for r in rows if (now - datetime.fromisoformat(r["timestamp"])).seconds < 3600]
        past_alerts = [r for r in rows if (now - datetime.fromisoformat(r["timestamp"])).seconds >= 3600]

        return render_template(
            "alerts.html",
            live_alerts=live_alerts,
            past_alerts=past_alerts
        )
    except Exception as e:
        print("⚠️ Error reading alerts:", e)
        return "Error loading alerts", 500

@app.route('/map')
def map_page():
    return render_template("map.html")

# -------------------------------------------------
# 6️⃣ SENSOR ROUTE
# -------------------------------------------------
@app.route('/sensor', methods=['POST'])
def sensor():
    """Receive live sensor data from simulator"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    try:
        # Extract core data
        X = pd.DataFrame([{f: data.get(f, 0) for f in feature_list}])
        vehicle_id = data.get("vehicle_id", "UNKNOWN")
        lat = data.get("latitude", 18.5204)
        lon = data.get("longitude", 73.8567)

        # ✅ Snap to nearest road for realism
        lat, lon = snap_to_road(lat, lon)

        # Predict from models
        pred = alert_model.predict(X)[0]
        anomaly = bool(pred == 1)
        accident_prob = float(accident_model.predict_proba(X)[0][1])
        high_risk = accident_prob > 0.6

        # 🔹 Use simulator severity if provided
        if "simulated_severity" in data:
            severity = data["simulated_severity"]
        else:
            if accident_prob < 0.3:
                severity = "Low"
            elif accident_prob < 0.6:
                severity = "Medium"
            elif accident_prob < 0.8:
                severity = "High"
            else:
                severity = "Critical"

        # ✅ Determine recommendation and messages
        if severity == "Low":
            recommendation = "✅ All clear. Maintain safe driving speed."
            driver_msg = "You’re driving safely. Keep your speed steady."
            police_msg = None

        elif severity == "Medium":
            recommendation = "⚠️ Caution! Slightly risky conditions detected."
            driver_msg = "Slow down slightly and stay alert."
            police_msg = None

        elif severity == "High":
            recommendation = "🚨 High accident risk detected!"
            driver_msg = "Reduce speed immediately! Check visibility and maintain distance."
            police_msg = f"⚠️ High-risk detected near {vehicle_id}. Possible traffic hazard."

        else:  # Critical
            recommendation = "🚑 CRITICAL ALERT! Possible accident imminent!"
            driver_msg = "Stop the vehicle safely! Turn on hazard lights and await assistance."
            police_msg = f"🚨 CRITICAL ALERT: Possible major accident detected! Immediate dispatch required for {vehicle_id}."

        # ✅ Log alert into DB
        log_alert(
            data.get("vehicle_id", "UNKNOWN"),
            recommendation,
            severity,
            data.get("latitude"),
            data.get("longitude")
        )

        # ✅ Log notifications
        log_notification("Driver", vehicle_id, driver_msg, severity)
        if police_msg:
            log_notification("Police", vehicle_id, police_msg, severity)

        # ✅ Emit live alerts for dashboard
        socketio.emit('new_alert', {
            "timestamp": datetime.now().isoformat(),
            "vehicle_id": vehicle_id,
            "latitude": lat,
            "longitude": lon,
            "message": recommendation,
            "severity": severity
        })

        # 🔥 Emit map alerts only for High/Critical
        if severity in ["High", "Critical"]:
            socketio.emit('map_alert', {
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": vehicle_id,
                "latitude": lat,
                "longitude": lon,
                "message": recommendation,
                "severity": severity
            })

        # 🔔 Emit driver and police notifications
        send_driver_alert(vehicle_id, driver_msg, severity)
        if police_msg:
            socketio.emit('police_notification', {
                "vehicle_id": vehicle_id,
                "message": police_msg,
                "severity": severity
            })

        # ✅ Send JSON response to simulator
        return jsonify({
            "status": "ok",
            "vehicle_id": vehicle_id,
            "accident_risk": round(accident_prob, 2),
            "severity": severity,
            "latitude": lat,
            "longitude": lon,
            "recommendation": recommendation
        })

    except Exception as e:
        print("⚠️ ERROR in /sensor route:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route("/dashboard")
def dashboard():
    """System overview dashboard"""
    try:
        # Load your dataset for analysis
        dataset_path = "E:\\College\\Sem 5\\SPM\\Project 2\\data\\Synthetic_Transportation_Dataset_Expanded_v2.csv"

        if not os.path.exists(dataset_path):
            return "⚠️ Dataset not found. Please check the path.", 404

        df = pd.read_csv(dataset_path)

        # Calculate anomaly / accident counts
        if "Accident_Occurred" in df.columns:
            df["Accident_Occurred"] = df["Accident_Occurred"].astype(str).str.lower()
            anomalies = df[df["Accident_Occurred"].isin(["1", "true", "yes"])].shape[0]
            normal = len(df) - anomalies
        else:
            anomalies = 0
            normal = len(df)

        total = len(df)
        percent = round((anomalies / total) * 100, 2) if total > 0 else 0

        # Severity distribution
        severity_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        if "Accident_Severity" in df.columns:
            for level in ["Low", "Medium", "High", "Critical"]:
                severity_counts[level] = df[df["Accident_Severity"].str.lower() == level.lower()].shape[0]

        # ✅ Render dashboard template
        return render_template(
            "dashboard.html",
            total=total,
            normal=normal,
            anomalies=anomalies,
            percent=percent,
            severity_counts=severity_counts
        )

    except Exception as e:
        print("⚠️ Dashboard error:", e)
        return "Error loading dashboard.", 500

@app.route('/notifications')
def notifications():
    """Display driver and police notifications"""
    try:
        conn = sqlite3.connect("alerts.db")
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM notifications ORDER BY timestamp DESC").fetchall()
        conn.close()
        return render_template("notifications.html", notifications=rows)
    except Exception as e:
        print("⚠️ Error reading notifications:", e)
        return "Error loading notifications", 500

@app.route("/logout")
def logout():
    session.clear()
    flash("👋 You have been logged out.", "info")
    return redirect(url_for("clerk_login"))

@app.route("/user_dashboard")
@login_required(role="driver")
def user_dashboard():
    # driver-only dashboard
    vehicle_id = session.get("vehicle_id")
    conn = sqlite3.connect("alerts.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM alerts WHERE vehicle_id=? ORDER BY timestamp DESC LIMIT 10", (vehicle_id,)).fetchall()
    conn.close()
    return render_template("user_dashboard.html", alerts=rows, name=session.get("name"))

    return render_template("user_dashboard.html", alerts=alerts, name=session.get("name"))

@app.route("/google_login")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    user_info = resp.json()
    email = user_info["email"]
    name = user_info.get("name", "Google User")

    conn = sqlite3.connect("alerts.db")
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    if not user:
        # Auto-register new user
        conn.execute("INSERT INTO users (name, email, vehicle_id, password) VALUES (?, ?, ?, ?)",
                     (name, email, "UNKNOWN", generate_password_hash("oauth")))
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    conn.close()

    # Log in session
    session["user_id"] = user["id"]
    session["name"] = user["name"]
    session["vehicle_id"] = user["vehicle_id"]
    session["role"] = user["role"]

    flash(f"Welcome, {user['name']} (Google)!", "success")
    return redirect("/user_dashboard" if user["role"] == "driver" else "/dashboard")

import requests, sqlite3
from werkzeug.security import generate_password_hash

CLERK_SECRET_KEY = "sk_test_vjB7KdOAS4rEJdVG0ZNMT4aEHDrEsc4eo7hsoX1DsZ"

def send_driver_alert(vehicle_id, message, severity="Info"):
    """
    Sends a new alert message to the driver.
    Logs it in DB + emits via Socket.IO.
    """
    try:
        # ✅ Log in database
        conn = sqlite3.connect("alerts.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                target TEXT,
                message TEXT,
                level TEXT,
                extra_info TEXT
            )
        """)
        conn.execute("""
            INSERT INTO notifications (timestamp, target, message, level, extra_info)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            "Driver",
            message,
            severity,
            vehicle_id
        ))
        conn.commit()
        conn.close()

        # ✅ Emit to connected dashboards via SocketIO
        socketio.emit('driver_notification', {
            "timestamp": datetime.now().isoformat(),
            "vehicle_id": vehicle_id,
            "message": message,
            "severity": severity
        })

        print(f"🚗 Driver Alert Sent [{severity}] → {vehicle_id}: {message}")

    except Exception as e:
        print("⚠️ Error sending driver alert:", e)

@app.route("/send_driver_alert", methods=["POST"])
def send_driver_alert_route():
    """
    API endpoint to trigger an alert manually.
    Example: POST /send_driver_alert { "vehicle_id": "VH1", "message": "Stop immediately!" }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    vehicle_id = data.get("vehicle_id", "UNKNOWN")
    message = data.get("message", "Alert!")
    severity = data.get("severity", "Info")

    send_driver_alert(vehicle_id, message, severity)
    return jsonify({"status": "sent", "vehicle_id": vehicle_id})

@app.route("/login")
def login():
    return redirect(url_for("clerk_login"))

@app.route("/register")
def register():
    return redirect(url_for("clerk_login"))

@app.route("/clerk_login")
def clerk_login():
    from dotenv import load_dotenv
    load_dotenv()
    clerk_key = os.getenv("CLERK_PUBLISHABLE_KEY")
    return render_template("clerk_login.html", clerk_key=clerk_key)

@app.route('/alert', methods=['POST'])
def alert():
    data = request.get_json()
    severity = data.get('severity')
    location = data.get('location')     # "lat,lon" OR dict {"lat": 19.15, "lon": 72.89}

    # If severity is high or critical, return map flag
    show_map = severity.lower() in ["high", "critical"]

    return jsonify({
        "severity": severity,
        "show_map": show_map,
        "location": location
    })

# -------------------------------------------------
# 7️⃣ RUN APP
# -------------------------------------------------
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)
