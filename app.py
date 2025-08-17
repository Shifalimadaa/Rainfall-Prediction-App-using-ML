import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Set Streamlit page configuration
st.set_page_config(page_title="Rainfall Prediction", layout="wide")

# Title and subtitle with improved CSS
st.markdown("""
    <style>
    .main { 
        background-color: #F0F2F6; 
    }
    .title {
        font-size:50px !important;
        color: #2E86AB;
        text-align: center;
        font-weight: bold;
    }
    .subtitle {
        font-size:25px !important;
        color: #555;
        text-align: center;
    }
    .stAlert > div {
        text-align: left !important;
    }
    /* Prevent dulling effect */
    .stApp > div {
        opacity: 1 !important;
    }
    /* Smooth transitions */
    .stSlider > div > div > div {
        transition: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Rainfall Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Using Random Forest Classifier</div>', unsafe_allow_html=True)
st.markdown("---")

# Initialize ALL session state variables at once
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.data_loaded = False
    st.session_state.model_loaded = False
    st.session_state.prediction_made = False
    st.session_state.prediction_result = None
    st.session_state.prediction_proba = None

# Load the dataset automatically
@st.cache_data
def load_data():
    df = pd.read_csv("Rainfall.csv")
    df.columns = df.columns.str.strip()
    df.drop(columns=["day"], errors="ignore", inplace=True)
    
    if "winddirection" in df.columns:
        df["winddirection"] = df["winddirection"].fillna(df["winddirection"].mode()[0])
    if "windspeed" in df.columns:
        df["windspeed"] = df["windspeed"].fillna(df["windspeed"].median())
    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].map({"yes": 1, "no": 0})
    
    return df

# Load trained model
@st.cache_resource
def load_model():
    with open("rainfall_model.pkl", "rb") as f:
        return pickle.load(f)

# Load data and model only once
if not st.session_state.data_loaded:
    data = load_data()
    st.session_state.data = data
    st.session_state.data_loaded = True
else:
    data = st.session_state.data

if not st.session_state.model_loaded:
    model_data = load_model()
    st.session_state.model = model_data["model"]
    st.session_state.features = model_data["feature_names"]
    st.session_state.model_loaded = True

model = st.session_state.model
features = st.session_state.features

# Visualizations (cached to prevent recomputation)
with st.expander("Exploratory Data Analysis"):
    st.write("### Sample Data")
    st.dataframe(data.head())

    if 'fig1_cached' not in st.session_state:
        fig1, ax1 = plt.subplots()
        sns.countplot(x="rainfall", data=data, ax=ax1)
        st.session_state.fig1_cached = fig1
    
    st.write("### Rainfall Distribution")
    st.pyplot(st.session_state.fig1_cached)

    if 'fig2_cached' not in st.session_state:
        fig2, axs = plt.subplots(3, 3, figsize=(15, 10))
        for i, col in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp',
                                 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']):
            if col in data.columns:
                sns.histplot(data[col], kde=True, ax=axs[i//3, i%3])
                axs[i//3, i%3].set_title(col)
        st.session_state.fig2_cached = fig2
    
    st.write("### Feature Distributions")
    st.pyplot(st.session_state.fig2_cached)

    if 'fig3_cached' not in st.session_state:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
        st.session_state.fig3_cached = fig3
    
    st.write("### Correlation Heatmap")
    st.pyplot(st.session_state.fig3_cached)

# Preprocessing
df = data.copy()
df = df.drop(columns=['maxtemp', 'temparature', 'mintemp'], errors="ignore")

# Sidebar input with unique keys and no recomputation
st.sidebar.header("🌤️ Weather Parameters")

# Use session state for slider values to maintain state
if 'slider_values' not in st.session_state:
    st.session_state.slider_values = {
        "pressure": 1015,
        "dewpoint": 20,
        "humidity": 95,
        "cloud": 81,
        "sunshine": 0.0,
        "winddirection": 40,
        "windspeed": 13.7
    }

# Create sliders with session state
pressure = st.sidebar.slider("Pressure (hPa)", 980, 1050, 
                            st.session_state.slider_values["pressure"], 
                            key="pressure_slider")
dewpoint = st.sidebar.slider("Dew Point (°C)", 0, 30, 
                            st.session_state.slider_values["dewpoint"], 
                            key="dewpoint_slider")
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 
                            st.session_state.slider_values["humidity"], 
                            key="humidity_slider")
cloud = st.sidebar.slider("Cloud (%)", 0, 100, 
                         st.session_state.slider_values["cloud"], 
                         key="cloud_slider")
sunshine = st.sidebar.slider("Sunshine (hrs)", 0.0, 12.0, 
                            st.session_state.slider_values["sunshine"], 
                            key="sunshine_slider")
winddirection = st.sidebar.slider("Wind Direction (°)", 0, 360, 
                                 st.session_state.slider_values["winddirection"], 
                                 key="winddirection_slider")
windspeed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 30.0, 
                             st.session_state.slider_values["windspeed"], 
                             key="windspeed_slider")

# Update session state
st.session_state.slider_values.update({
    "pressure": pressure,
    "dewpoint": dewpoint,
    "humidity": humidity,
    "cloud": cloud,
    "sunshine": sunshine,
    "winddirection": winddirection,
    "windspeed": windspeed
})

# Create input dataframe
input_values = {
    "pressure": pressure,
    "dewpoint": dewpoint,
    "humidity": humidity,
    "cloud": cloud,
    "sunshine": sunshine,
    "winddirection": winddirection,
    "windspeed": windspeed
}

input_df = pd.DataFrame([input_values])
input_df = input_df.reindex(columns=features, fill_value=0)

# Add predict button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🌦️ Predict Rainfall", type="primary", use_container_width=True)

# Show current input values with smooth updates
st.subheader("🌡️ Current Weather Parameters")

# Create 3 columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡️ Pressure", f"{pressure} hPa")
    st.metric("💧 Humidity", f"{humidity}%")
    st.metric("☁️ Cloud Cover", f"{cloud}%")

with col2:
    st.metric("🌡️ Dew Point", f"{dewpoint}°C")
    st.metric("☀️ Sunshine", f"{sunshine} hrs")
    
with col3:
    st.metric("🧭 Wind Direction", f"{winddirection}°")
    st.metric("💨 Wind Speed", f"{windspeed} km/h")

# Handle prediction
if predict_button:
    with st.spinner("🔮 Making prediction..."):
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        st.session_state.prediction_made = True
        st.session_state.prediction_result = prediction
        st.session_state.prediction_proba = prediction_proba

# Display results
if st.session_state.prediction_made:
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    prediction = st.session_state.prediction_result
    prediction_proba = st.session_state.prediction_proba
    
    # Create result columns
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        if prediction == 1:
            st.success("🌧️ **Rainfall Expected**")
            confidence = prediction_proba[1] * 100
        else:
            st.success("☀️ **No Rainfall Expected**")
            confidence = prediction_proba[0] * 100
    
    with result_col2:
        st.info(f"🎯 Confidence: {confidence:.1f}%")
    
    # Weather tips
    st.subheader("💡 Weather Tips")
    if prediction == 1:
        st.markdown("""
        **🌧️ Rainfall Expected:**
        - 🌂 Carry an umbrella
        - 🚗 Drive carefully on wet roads
        - 🏠 Plan indoor activities
        - 📱 Monitor weather updates
        """)
    else:
        st.markdown("""
        **☀️ No Rainfall Expected:**
        - 🌞 Perfect for outdoor activities!
        - 🧺 Great day for picnics
        - 🚴 Ideal for sports and exercise
        - 🌻 Good time for gardening
        """)
    
    # Prediction probabilities
    st.subheader("📊 Prediction Probabilities")
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric("☀️ No Rainfall", f"{prediction_proba[0]*100:.1f}%")
    with prob_col2:
        st.metric("🌧️ Rainfall", f"{prediction_proba[1]*100:.1f}%")

else:
    st.info("👆 Adjust the weather parameters in the sidebar and click **'Predict Rainfall'** to see the prediction!")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: gray; padding: 20px;">'
    '🌦️ Developed by Sakshi Shetty & Shifali Mada | ©️ 2025'
    '</div>', 
    unsafe_allow_html=True
)