import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set the title of the Streamlit app
st.title('Clustering Players')
st.markdown('ML model to cluster players based on similarity')

# Header for player features
st.header('Player Features')

# Create two columns for input
col1, col2 = st.columns(2)

# Column 1 for characteristics
with col1:
    st.text('Player Statistics')
    appearance = st.slider('Appearance', 0.0, 100.0, 10.0)
    goals = st.slider('Goals', 0.0, 0.5, 0.005)

# Column 2 for additional characteristics
with col2:
    st.text('More Player Statistics')
    award = st.slider('Award (0 or 1)', 0.0, 5.0, 1.0)
    height = st.slider('Height (cm)', 165.0, 195.0, 165.0)

# Load model and scaler once at the start
clf = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler_modell.joblib')  # Assuming you saved the scaler used in training

# Function to predict player cluster
def predict(data):
    data_scaled = scaler.transform(data)  # Scale the input data
    return clf.predict(data_scaled)

# When the button is pressed, predict the cluster
if st.button('Predict Player Cluster'):
    data = np.array([[appearance, goals, award, height]])
    result = predict(data)
    st.text(f'The player belongs to cluster: {result[0]}')

