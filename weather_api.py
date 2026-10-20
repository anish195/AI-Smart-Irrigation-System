"""Optional Open-Meteo forecast, no API key required. Gracefully handles unavailable network."""
import requests

def forecast(latitude=17.59, longitude=78.12):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude,
              "daily": "temperature_2m_max,relative_humidity_2m_mean,precipitation_sum",
              "timezone": "auto", "forecast_days": 5}
    response = requests.get(url, params=params, timeout=8)
    response.raise_for_status()
    daily = response.json()["daily"]
    return [{"Date": daily["time"][i], "Temperature": daily["temperature_2m_max"][i],
             "Humidity": daily["relative_humidity_2m_mean"][i],
             "Rainfall": daily["precipitation_sum"][i]}
            for i in range(len(daily["time"]))]
