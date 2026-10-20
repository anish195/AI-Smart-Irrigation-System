from irrigation import generate_weather, simulate, fixed_policy, smart_policy, next_storage, summary

def test():
    weather = generate_weather(90, seed=42)
    assert len(weather) == 90
    assert (weather["Rainfall"] >= 0).all()
    fixed = simulate(weather, fixed_policy)
    smart = simulate(weather, smart_policy)
    for data in (fixed, smart):
        assert len(data) == 90
        assert data["Next_Day_Soil_Water"].between(0, 100).all()
        assert (data["Irrigation"] >= 0).all()
        assert data.isna().sum().sum() == 0
    assert next_storage(98, 10, 0, 30, 50) == 100.0
    print("PASS: simulation, bounds, deterministic weather, no missing values")
    print("Fixed:", summary(fixed))
    print("Smart:", summary(smart))

if __name__ == "__main__":
    test()
