# 🚨 Accident Severity Prediction & Real-Time Alert Dashboard
### *AI-Powered Smart Transportation System using Flask, ML & Live Vehicle Streaming*

This project is a complete **AI/ML-based transportation safety system** designed to detect accident risk, classify severity levels, and send real-time alerts to drivers and authorities.

It combines:

- 🧠 **Machine Learning Models**
- 📡 **Live IoT Vehicle Data Simulation**
- 🗺️ **Real-Time Leaflet Map Visualizations**
- 🔔 **Driver & Police Alert Notifications**
- 📊 **Admin Dashboard with Analytics**
- 🔐 **Google/Clerk Authentication**
- ⚡ **Socket.IO Live Event Streaming**

This system is ideal for projects, portfolios, research, or real-world smart transport prototypes.

---

# 🧩 Features

## 🚗 1. **Live Vehicle Data Streaming**
A Python simulator sends continuous real-time sensor data:

- Speed  
- Temperature  
- Humidity  
- Rain  
- Visibility  
- Latitude & Longitude  

These values are processed by the backend to detect risks.

---

## 🤖 2. **Accident Severity Prediction (ML Models)**

The backend uses trained ML models:

- `alert_model.joblib`
- `accident_model.joblib`
- `severity_model.joblib`

Predictions include:

- Accident Probability  
- Severity Categorization (Low, Medium, High, Critical)  
- Recommended Actions  

---

## 🔴 3. **Real-Time Alerts Dashboard**

Live alerts automatically appear via **Socket.IO** inside the dashboard:

- Includes timestamp, vehicle ID, severity, location
- High and critical alerts flash & play warning sounds
- Integrated “View Map” button for quick location navigation

---

## 🗺️ 4. **Live Map With Danger Zones (Leaflet JS)**

High & Critical severity alerts:

- Show markers on the map  
- Draw colored hazard circles  
- Auto-remove after 2 minutes  
- Can be clicked to remove manually  
- Supports direct navigation from alerts table  

---

## 🔐 5. **Authentication System**
Supports both:

- Google OAuth  
- Clerk authentication  

User roles:
- **Admin** → Dashboard, Alerts, Notifications, Map  
- **Driver** → Personalized driver dashboard  

---

## 📊 6. **AI/ML Dashboard**

Shows:

- Total dataset records  
- Normal vs abnormal events  
- Anomaly percentage  
- Severity distribution chart (Chart.js)  
- Weekly accident trends  

---

## 📁 Project Structure