import streamlit as st
import pandas as pd
from main import Linear_Regression
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_and_train():
    train = pd.read_csv("train_processed.csv")
    X = train.drop("count", axis=1)
    Y = train["count"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Linear_Regression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_scaled, Y)
    
    return model, scaler, X.columns

model, scaler, feature_columns = load_and_train()

st.set_page_config(page_title="Bike Demand Forecast", page_icon="📈", layout="centered")

with st.sidebar:
    st.header("⚙️ Operational Settings")
    st.markdown("Configure physical fleet constraints below.")
    fleet_capacity = st.number_input("Total System Bike Inventory", min_value=0, max_value=5000, value=500, step=50)
    st.divider()
    st.caption("Model Engine: Multivariate Linear Regression (Gradient Descent)")
    st.caption("Developed by Jawad Akbar")

st.title("Bike Rental Demand Prediction Dashboard")
st.markdown("Configure environmental and temporal variables below to forecast system-wide rental demand.")

with st.form("simulation_form"):
    st.subheader("📅 Simulation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Temporal Conditions**")
        hour = st.slider("Hour of the Day", 0, 23, 8, format="%d:00")
        month = st.selectbox("Month", range(1, 13), format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
        season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1:"Spring", 2:"Summer", 3:"Fall", 4:"Winter"}[x])
        workingday = st.radio("Day Classification", [1, 0], format_func=lambda x: "Working Day" if x == 1 else "Weekend / Holiday", horizontal=True)

    with col2:
        st.markdown("**Environmental Conditions**")
        weather = st.selectbox("Weather Forecast", [1, 2, 3, 4], format_func=lambda x: {1:"Clear / Few Clouds", 2:"Mist / Cloudy", 3:"Light Rain / Snow", 4:"Heavy Rain / Extreme"}[x])
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=45.0, value=22.0, step=0.5)
        
        col_h, col_w = st.columns(2)
        with col_h:
            humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        with col_w:
            windspeed = st.number_input("Windspeed", min_value=0.0, max_value=60.0, value=15.0)

    submit_button = st.form_submit_button("Generate Forecast", type="primary", use_container_width=True)

if submit_button:
    input_data = {col: 0 for col in feature_columns}
    
    input_data['hour'] = hour
    input_data['month'] = month
    input_data['workingday'] = workingday
    input_data['temp'] = temp
    input_data['atemp'] = temp
    input_data['humidity'] = humidity
    input_data['windspeed'] = windspeed
    input_data['year'] = 2011 
    input_data['day'] = 15 
    input_data['holiday'] = 0 if workingday == 1 else 1 
    input_data['dayofweek'] = 2 if workingday == 1 else 6 
    
    if f'season_{season}' in feature_columns:
        input_data[f'season_{season}'] = 1
    if f'weather_{weather}' in feature_columns:
        input_data[f'weather_{weather}'] = 1
        
    user_df = pd.DataFrame([input_data])[feature_columns]
    
    scaled_input = scaler.transform(user_df)
    raw_prediction = model.predict(scaled_input)[0]
    final_prediction = max(0, int(raw_prediction))
    
    st.divider()
    st.markdown("### 📊 Operational Forecast")
    
    met1, met2, met3 = st.columns(3)
    met1.metric("Predicted Demand", f"{final_prediction} Bikes")
    met2.metric("Available Fleet", f"{fleet_capacity} Bikes")
    
    utilization = (final_prediction / fleet_capacity) * 100 if fleet_capacity > 0 else 0
    met3.metric("Expected Utilization", f"{utilization:.1f}%")

    if final_prediction > fleet_capacity:
        st.error(f"🚨 **Critical Fleet Shortage:** Demand exceeds physical supply by **{final_prediction - fleet_capacity} bikes**. Initiate fleet rebalancing routing or activate dynamic surge pricing.")
    elif utilization > 80:
        st.warning(f"⚠️ **High Utilization Warning:** Network is projected to operate at **{utilization:.1f}% capacity**. Monitor dock levels closely to prevent local shortages.")
    else:
        st.success(f"✅ **Stable Operations:** Network has sufficient capacity. **{fleet_capacity - final_prediction} bikes** will remain in station reserve.")