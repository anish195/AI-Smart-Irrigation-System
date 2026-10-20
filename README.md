# Smart Irrigation Using Python and Machine Learning 🌱

I made this project to try using Python and machine learning for a basic irrigation problem.

## What it does

- Creates simulated weather and soil water data (no sensors used)
- Trains Linear Regression and Random Forest to predict next-day soil water
- Gives an irrigation suggestion using a simple water-balance rule
- Compares watering every third day with watering when the soil is getting dry
- Shows the results in a Streamlit dashboard

## How to run

You need Python installed.

Open a terminal inside this folder and run:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The dashboard should open in your browser.

## Files

- `app.py` - dashboard
- `irrigation.py` - generates weather, calculates soil water and irrigation
- `train_model.py` - ML training and evaluation
- `weather_api.py` - optional example for a weather forecast API (not used in the basic dashboard)
- `test_project.py` - a few basic checks

The training script also creates CSV result files and saved models.

## Basic idea

Next soil water = current soil water + rainfall + irrigation - water loss

I limit the result to between 0 and 100 mm in this example.

The ML model tries to learn this relationship from generated examples.
The irrigation recommendation is a separate rule, not an ML decision.

## Limitations

This is a learning prototype, not a working farm product. The weather, soil
capacity, water loss and crop thresholds are assumed. There are no physical
sensors and I have not validated it using real agricultural data. Any water
savings shown are simulated only.


##  Development Note

This project was developed as part of my learning in Python and Machine Learning, with guidance and assistance from LLM's during development. It uses simulated sensor data and has not yet been tested on real agricultural hardware.