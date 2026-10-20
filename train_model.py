"""Train a transparent ML demonstration on synthetic water-balance observations."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from irrigation import generate_weather, simulate, fixed_policy, next_storage

ROOT = Path(__file__).resolve().parent
FEATURES = ["Current_Soil_Water", "Temperature", "Humidity", "Rainfall", "Irrigation"]

def train(days=180, seed=42):
    weather = generate_weather(days, seed)
    data = simulate(weather, fixed_policy)
    split = int(len(data) * 0.8)
    X_train, X_test = data[FEATURES].iloc[:split], data[FEATURES].iloc[split:]
    y_train, y_test = data["Next_Day_Soil_Water"].iloc[:split], data["Next_Day_Soil_Water"].iloc[split:]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=9, min_samples_leaf=2, random_state=seed)
    }
    predictions = pd.DataFrame({"Day": data["Day"].iloc[split:].to_numpy(),
                                "Actual": y_test.to_numpy()})
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        metrics[name] = {"MAE_mm": float(mean_absolute_error(y_test, predicted)),
                         "RMSE_mm": float(np.sqrt(mean_squared_error(y_test, predicted))),
                         "R2": float(r2_score(y_test, predicted))}
        joblib.dump(model, ROOT / ("linear_model.joblib" if name == "Linear Regression" else "forest_model.joblib"))
    # Persistence and exact simulator are vital comparators on synthetic data.
    persistence = X_test["Current_Soil_Water"].to_numpy()
    physics = np.array([next_storage(row.Current_Soil_Water, row.Rainfall, row.Irrigation,
                                     row.Temperature, row.Humidity) for row in X_test.itertuples(index=False)])
    for name, estimate in [("Persistence baseline", persistence), ("Known water-balance model", physics)]:
        predictions[name] = estimate
        metrics[name] = {"MAE_mm": float(mean_absolute_error(y_test, estimate)),
                         "RMSE_mm": float(np.sqrt(mean_squared_error(y_test, estimate))),
                         "R2": float(r2_score(y_test, estimate))}
    data.to_csv(ROOT / "synthetic_sensor_data.csv", index=False)
    predictions.to_csv(ROOT / "test_predictions.csv", index=False)
    (ROOT / "metrics.json").write_text(json.dumps({"dataset_days":days, "training_days":split,
         "test_days": days-split, "metrics":metrics}, indent=2), encoding="utf-8")
    print("Training finished. Results (synthetic data):")
    print(pd.DataFrame(metrics).T.round(4).to_string())
    return metrics

if __name__ == "__main__":
    train()
