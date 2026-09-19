# 🚲 Bike Rental Demand Prediction

**Author:** Jawad Akbar | University of Management and Technology

## 📌 Project Overview

This project predicts bike rental demand based on environmental and time-related variables such as temperature, humidity, weather, season, and hour of the day.

The project uses a custom Multiple Linear Regression model built from scratch with NumPy. A Streamlit web application allows users to enter different conditions and generate a predicted number of bike rentals.

**Live Application:** [Bike Rental Demand Prediction](https://bike-rental-predictor.streamlit.app/)

## 📊 Dataset & Preprocessing

The dataset contains hourly bike rental data, including weather conditions, temperature, humidity, windspeed, and rental counts.

### Data Processing

- **Datetime Extraction:** The `datetime` column was converted into separate `year`, `month`, `day`, `hour`, and `dayofweek` features.
- **Missing Values & Duplicates:** Missing values were checked and duplicate rows were removed.
- **Feature Encoding:** One-Hot Encoding was applied to the `season` and `weather` columns using `pandas.get_dummies`.
- **Feature Alignment:** Training and testing datasets were aligned so that they contained the same encoded feature columns.
- **Feature Scaling:** Numerical features were standardized using `StandardScaler` before training the model.

## 📈 Exploratory Data Analysis

Several visualizations were created using Matplotlib and Seaborn to understand the relationship between rental demand and different features.

### Main Observations

1. **Weather:** Average rental demand is higher during clear weather and decreases during worse weather conditions.
2. **Time of Day:** Rental demand shows noticeable peaks during morning and evening hours on working days.
3. **Working Days:** The hourly demand pattern differs between working days and non-working days.
4. **Seasonality:** Rental demand varies across different seasons.

These patterns helped identify time and weather-related variables that could be useful for prediction.

## 🧠 Model Architecture & Training

The project uses a **Multiple Linear Regression model implemented from scratch using NumPy** rather than using a ready-made regression model from a machine learning library.

### Model Details

- **Algorithm:** Multiple Linear Regression
- **Optimization:** Gradient Descent
- **Learning Rate:** 0.01
- **Iterations:** 1000
- **Train/Test Split:** 80/20
- **Feature Scaling:** StandardScaler

During training, the model calculates predictions, computes the gradients for the weights and bias, and updates them over multiple iterations.

## 🎯 Model Evaluation

The model was evaluated using the 20% test split.

| Metric | Result |
|---|---:|
| MAE | 105.09 |
| RMSE | 140.66 |
| R² | 0.4005 |

The R² score of approximately 0.40 indicates that the current linear model explains around 40% of the variation in the test-set rental demand.

The results also show that bike rental demand is influenced by relationships that may not be fully captured by a simple linear model.

## 💻 Web Application

The trained model was integrated into a Streamlit application.

Users can enter:

- Hour
- Month
- Season
- Working day
- Weather
- Temperature
- Humidity
- Windspeed

The application processes these inputs in the same format used during model training and returns an estimated number of bike rentals.

## 🚲 Fleet Capacity Comparison

The application can also compare predicted demand with a user-defined fleet capacity.

For example, if the predicted demand is higher than the available fleet, the application can indicate that the available number of bikes may not be sufficient to meet the predicted demand.

This provides a simple example of how a demand prediction model could be connected to an operational decision.

## 🚀 Future Improvements

The current model predicts total bike rental demand at the system level.

Future versions could include:

- More advanced regression models
- Random Forest or Gradient Boosting
- Better handling of time-based patterns
- Hyperparameter tuning
- Cross-validation
- More detailed weather features
- Station-level demand prediction
- Geographic features such as station location or GPS coordinates

With station-level data, the model could be extended to predict demand for individual docking stations and help with localized fleet management.

## ⚙️ How to Run Locally
1. Clone the repository: `git clone https://github.com/JawadAkbar/Rental-Bike-Prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Launch the application: `streamlit run streamlit.py`
