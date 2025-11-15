import requests
import time
import random

# 🌐 Flask API endpoint
API_URL = "http://127.0.0.1:8000/sensor"

# 🚗 Example vehicle IDs
VEHICLES = ["VH1", "VH2", "VH3", "VH4", "VH5", "VH6", "VH7", "VH8", "VH9", "VH10"]
SEVERITY_OPTIONS = ["Low", "Medium", "High", "Critical"]

# 🗺️ Simulated GPS ranges (around Pune)
BASE_LAT, BASE_LON = 18.5204, 73.8567

print("🚗 Starting Live Vehicle Data Stream Simulator...")
print("Press Ctrl + C to stop.\n")

while True:
    try:
        vehicle_id = random.choice(VEHICLES)
        speed = round(random.uniform(40, 120), 2)
        temperature = round(random.uniform(20, 55), 2)
        humidity = round(random.uniform(30, 90), 2)
        rain = random.choice([0, 1])
        visibility = round(random.uniform(0.2, 1.0), 2)

        # 🎯 Random location offset
        lat = BASE_LAT + random.uniform(-0.05, 0.05)
        lon = BASE_LON + random.uniform(-0.05, 0.05)

        # Random severity for diversity
        simulated_severity = random.choices(
            SEVERITY_OPTIONS, weights=[0.5, 0.25, 0.15, 0.1], k=1
        )[0]

        payload = {
            "vehicle_id": vehicle_id,
            "speed": speed,
            "temperature": temperature,
            "humidity": humidity,
            "rain": rain,
            "visibility": visibility,
            "latitude": lat,
            "longitude": lon,
            "simulated_severity": simulated_severity
        }

        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"[{vehicle_id}] 🛰️ {lat:.4f},{lon:.4f} | Risk={data.get('accident_risk')} | "
                  f"Severity={data.get('severity')} | Msg={data.get('recommendation')}")
        else:
            print(f"❌ Server returned status {response.status_code}: {response.text}")

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")

    time.sleep(random.randint(3, 6))
