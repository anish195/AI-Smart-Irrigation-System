"""Water-balance simulator and irrigation policies. Units: millimetres of water."""
import numpy as np
import pandas as pd

CAPACITY = 100.0

def water_loss(temperature, humidity):
    """Simple assumed daily loss; not a calibrated crop evapotranspiration model."""
    return max(1.0, 3.0 + 0.12 * (temperature - 30.0) - 0.02 * (humidity - 50.0))

def next_storage(current, rain, irrigation, temperature, humidity, capacity=CAPACITY):
    loss = water_loss(temperature, humidity)
    return float(np.clip(current + max(0.0, rain) + max(0.0, irrigation) - loss, 0, capacity))

def generate_weather(days=180, seed=42):
    """Weather scenario used by both policies; fully synthetic."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(1, days + 1):
        # Modest day-to-day temperature variations and occasional rainy days.
        temp = float(np.clip(31 + 4 * np.sin(day / 15) + rng.normal(0, 2), 20, 42))
        humidity = float(np.clip(58 - 0.8 * (temp - 31) + rng.normal(0, 7), 25, 95))
        rain = float(rng.uniform(2, 10)) if rng.random() < 0.20 else 0.0
        rows.append({"Day": day, "Temperature": temp, "Humidity": humidity, "Rainfall": rain})
    return pd.DataFrame(rows)

def fixed_policy(day, current, rain, temp, humidity, target=65, minimum=40, max_daily=15):
    """Conventional example: 5 mm every third day."""
    return 5.0 if day % 3 == 0 else 0.0

def smart_policy(day, current, rain, temp, humidity, target=65, minimum=40, max_daily=15):
    """Use predicted *no-irrigation* end-of-day storage to decide today's water."""
    expected_without = next_storage(current, rain, 0, temp, humidity)
    if expected_without >= minimum:
        return 0.0
    return float(np.clip(target - expected_without, 0, max_daily))

def simulate(weather, policy, initial=60.0, target=65, minimum=40, max_daily=15):
    current = float(initial)
    output = []
    for item in weather.itertuples(index=False):
        irrigation = float(policy(item.Day, current, item.Rainfall, item.Temperature,
                                  item.Humidity, target, minimum, max_daily))
        after = next_storage(current, item.Rainfall, irrigation, item.Temperature, item.Humidity)
        output.append({"Day": item.Day, "Temperature": item.Temperature,
                       "Humidity": item.Humidity, "Rainfall": item.Rainfall,
                       "Irrigation": irrigation, "Water_Loss": water_loss(item.Temperature, item.Humidity),
                       "Current_Soil_Water": current, "Next_Day_Soil_Water": after,
                       "Below_Minimum": after < minimum})
        current = after
    return pd.DataFrame(output)

def summary(df):
    return {"water_mm": round(float(df["Irrigation"].sum()), 2),
            "stress_days": int(df["Below_Minimum"].sum()),
            "average_storage_mm": round(float(df["Next_Day_Soil_Water"].mean()), 2)}
