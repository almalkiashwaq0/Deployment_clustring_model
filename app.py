import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.title('Clustering Players')
st.markdown('ML model to cluster players based on similarity')


st.header('Player Features')


col1, col2 = st.columns(2)


with col1:
    #st.text('Player Statistics')
    appearance = st.slider('Appearance', 0.0, 100.0, 10.0)
    goals = st.slider('Goals', 0.0, 0.5, 0.005)


with col2:
    #st.text('More Player Statistics')
    award = st.slider('Award (0 or 1)', min_value=0, max_value=1, step=1)
    height = st.slider('Height (cm)', 165.0, 195.0, 165.0)


clf = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler_modell.joblib') 


def predict(data):
    data_scaled = scaler.transform(data) 
    return clf.predict(data_scaled)


if st.button('Predict Player Cluster'):
    data = np.array([[appearance, goals, award, height]])
    result = predict(data)
    st.text(f'The player belongs to cluster: {result[0]}')

