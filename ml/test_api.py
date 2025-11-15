import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "speed": 90,
    "temperature": 47,
    "humidity": 70,
    "rain": 1,
    "visibility": 0.3
}

response = requests.post(url, json=data)
print(response.json())
