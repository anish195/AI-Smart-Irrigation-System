import pandas as pd
import streamlit as st
from pathlib import Path
from irrigation import generate_weather, simulate, fixed_policy, smart_policy, summary, next_storage

st.set_page_config(page_title="My Smart Irrigation Project", page_icon="🌱")
st.title("🌱 Smart Irrigation Using Python and ML")
st.write("This is my student project to understand how irrigation can be planned using weather and soil water data.")
st.caption("Note: The sensor data is simulated. I have not tested this on a real farm.")

st.header("1. Compare two irrigation methods")
st.write("Fixed method: water the soil every 3 days. Smart method: water only when predicted storage goes below a chosen minimum.")

days = st.slider("Number of days", 30, 180, 90)
minimum = st.slider("Minimum soil water (mm)", 20, 60, 40)
target = st.slider("Target soil water (mm)", minimum, 90, max(65, minimum))
weather = generate_weather(days=days, seed=42)

fixed = simulate(weather, fixed_policy, minimum=minimum, target=target)
smart = simulate(weather, smart_policy, minimum=minimum, target=target)
fixed_result = summary(fixed)
smart_result = summary(smart)

col1, col2 = st.columns(2)
col1.metric("Water used: fixed schedule", f"{fixed_result['water_mm']} mm")
col2.metric("Water used: smart schedule", f"{smart_result['water_mm']} mm")
st.write("Days with soil water below the minimum:")
st.write(f"Fixed: {fixed_result['stress_days']} days | Smart: {smart_result['stress_days']} days")

if fixed_result["water_mm"] > 0:
    savings = (fixed_result["water_mm"] - smart_result["water_mm"]) / fixed_result["water_mm"] * 100
    st.write(f"Change in water used: {savings:.1f}% less with the smart method")
st.caption("These numbers are only from the simulation, not actual water savings.")

chart = pd.DataFrame({
    "Fixed method": fixed["Next_Day_Soil_Water"].to_numpy(),
    "Smart method": smart["Next_Day_Soil_Water"].to_numpy()
}, index=fixed["Day"])
st.line_chart(chart)
st.write("Water used each day:")
st.bar_chart(pd.DataFrame({
    "Fixed method": fixed["Irrigation"].to_numpy(),
    "Smart method": smart["Irrigation"].to_numpy()
}, index=fixed["Day"]))

st.header("2. Try your own values")
current = st.slider("Current soil water (mm)", 0, 100, 45)
temperature = st.slider("Temperature (°C)", 20, 45, 32)
humidity = st.slider("Humidity (%)", 20, 100, 60)
rain = st.slider("Expected rainfall (mm)", 0.0, 25.0, 0.0)

amount = smart_policy(1, current, rain, temperature, humidity, target, minimum, 15)
estimated = next_storage(current, rain, amount, temperature, humidity)
st.success(f"Suggested irrigation: {amount:.1f} mm")
st.write(f"Estimated soil water after today: {estimated:.1f} mm")
st.caption("The irrigation suggestion uses a simple rule based on the water-balance equation.")


st.header("3. Compare ML Models")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from train_model import FEATURES

# Train both models using simulated data
@st.cache_resource
def get_models():
    weather = generate_weather(days=180, seed=42)
    data = simulate(weather, fixed_policy)

    X = data[FEATURES]
    y = data["Next_Day_Soil_Water"]

    # First 80% for training
    X_train = X.iloc[:144]
    X_test = X.iloc[144:]
    y_train = y.iloc[:144]
    y_test = y.iloc[144:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        results.append({
            "Model": name,
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3)
        })

    return models, pd.DataFrame(results), X_test, y_test


models, results, X_test, y_test = get_models()

st.subheader("Model Accuracy")
st.dataframe(results, hide_index=True)

# Let the user select the ML model
selected_model = st.selectbox(
    "Choose ML Model",
    ["Linear Regression", "Random Forest"]
)

# Use values entered in Section 2
sample = pd.DataFrame([{
    "Current_Soil_Water": current,
    "Temperature": temperature,
    "Humidity": humidity,
    "Rainfall": rain,
    "Irrigation": amount
}])

# Predict using selected model
prediction = models[selected_model].predict(sample)[0]

st.write("Selected Model:", selected_model)
st.success(
    f"Predicted Soil Water: {prediction:.2f} mm"
)

# Compare predictions with actual test values
st.subheader("Actual vs Predicted")

test_predictions = models[selected_model].predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": test_predictions
})

st.line_chart(comparison)

st.caption(
    "Models are trained and evaluated using simulated data. "
    "These results do not represent real farm accuracy."
)

st.header("4. Dataset")
if st.checkbox("Show first 10 simulated readings"):
    st.dataframe(fixed.head(10))
st.download_button("Download simulated data", smart.to_csv(index=False), "irrigation_results.csv")

st.write("---")
st.write("Made using Python, pandas, NumPy, scikit-learn and Streamlit.")
