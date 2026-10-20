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

## Online dashboard

Upload the source files to a public GitHub repository. In Streamlit Community Cloud, create an app using this repository, choose `app.py` as the main file, and deploy. The dashboard trains its Linear Regression model automatically on first load.

To generate model files and evaluation results separately, run `python train_model.py`.
